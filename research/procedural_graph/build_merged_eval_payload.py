"""Build clean Kaggle payload for testing the updated procedural graph (new predictor) on Kaggle."""
from pathlib import Path
import hashlib
import json
import random
import shutil

REPO = Path('/home/alex/kaggriculture')
RUN_DIR = REPO / 'research/procedural_graph/runs/merged_eval_kaggle'
OUT = RUN_DIR / 'payload'
EVALUATION_ID = 'calibrated-merged-graph-vs-hazel-20260923-v1'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    # 1. Candidate: Updated procedural graph with new predictor (09-22)
    cand_dir = OUT / 'agents/candidate_merged_graph'
    runtime_src = REPO / 'research/procedural_graph/hazel_runtime'
    shutil.copytree(runtime_src, cand_dir / 'hazel_runtime',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    copy(REPO / 'research/procedural_graph/agent_graph.py', cand_dir / 'main.py')
    copy(REPO / 'research/procedural_graph/policy_graph.json', cand_dir / 'policy_graph.json')

    # 2. Opponent: Hazel Weir (last submission)
    opponents = []
    for n in ['hazel_weir']:
        src = REPO / 'shinka/champions/submissions' / n
        manifest = json.loads((src / 'MANIFEST.json').read_text())
        dst = OUT / 'agents' / ('submission_' + n)
        for f in manifest['archive_files']:
            copy(src / f, dst / f)
        copy(src / 'MANIFEST.json', dst / 'SOURCE_MANIFEST.json')
        opponents.append({
            'id': 'submission_' + n,
            'label': manifest['public_name'] + ' (submitted package)',
            'kind': 'submission',
            'submission_id': manifest['kaggle']['submission_id'],
        })

    # 3. RPC helper with network denial
    rpc = (REPO / 'research/frozen_eval/runs/iter27_trace_mining/payload/pool_upgrade_bundle_agent.py').read_text()
    (OUT / 'pool_upgrade_bundle_agent.py').write_text(rpc)

    # 4. Runner
    copy(REPO / 'research/frozen_eval/runner.py', OUT / 'runner.py')

    # 5. Seeds: 20 seeds per seat = 40 games against Hazel Weir
    # 4 shards x 5 seeds/shard x 2 seats = 40 games total (10 games per shard)
    rng = random.Random(202609231800)
    all_seeds = rng.sample(range(2_000_000, 2_000_000_000), 200)
    seeds_0 = all_seeds[:20]
    seeds_1 = all_seeds[100:120]

    manifest_data = {
        'evaluation_id': EVALUATION_ID,
        'candidate_id': 'candidate_merged_graph',
        'engine_version': '1.32.7',
        'opponents': opponents,
        'seeds': {'0': seeds_0, '1': seeds_1},
        'design': (
            'Benchmark calibrated procedural graph against last submission (Hazel Weir) across 40 games.'
        ),
        'files': {
            str(f.relative_to(OUT)): digest(f)
            for f in sorted(OUT.rglob('*'))
            if f.is_file() and f.name != 'manifest.json' and '__pycache__' not in f.parts
        },
    }
    (OUT / 'manifest.json').write_text(json.dumps(manifest_data, indent=2) + '\n')

    print(json.dumps({
        'payload': str(OUT),
        'evaluation_id': EVALUATION_ID,
        'opponents': len(opponents),
        'seeds_per_seat': len(seeds_0),
        'files': len(manifest_data['files']),
        'bytes': sum(f.stat().st_size for f in OUT.rglob('*') if f.is_file()),
    }, indent=2))


if __name__ == '__main__':
    main()
