"""Build a separate Hazel executable graph candidate, never overwrite frozen runs."""
from pathlib import Path
import hashlib
import json
import shutil
from graph_runtime import MARKET_STAGES, TURN_STAGES

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUT = HERE / 'payload'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def chain(nodes):
    ids = [n['id'] for n in nodes]
    return {'entry': ids[0], 'exit': ids[-1], 'nodes': nodes,
            'edges': [{'source': a, 'target': b, 'relation': 'NEXT'} for a, b in zip(ids, ids[1:])]}


def main():
    if OUT.exists():
        raise SystemExit('Refusing to overwrite existing payload')
    src = REPO / 'shinka/champions/submissions/hazel_weir'
    manifest = json.loads((src / 'MANIFEST.json').read_text())
    assert sha(src / 'champion.py') == manifest['champion']['sha256']
    assert sha(src / 'checkpoint/model.safetensors') == manifest['oracle']['model_safetensors_sha256']
    assert sha(src / 'kagg_oracle.py') == manifest['oracle']['kagg_oracle_sha256']
    assert sha(src / 'kagg_ttm_numpy.py') == manifest['oracle']['kagg_ttm_numpy_sha256']
    graph = {
        'schema_version': 1,
        'name': 'Hazel Weir complete executable policy import',
        'source': {'submission_id': manifest['kaggle']['submission_id'],
                   'champion_sha256': sha(src / 'champion.py'),
                   'oracle_model_sha256': sha(src / 'checkpoint/model.safetensors')},
        'semantics': 'Source-preserving ordered execution; original guards and early returns remain in bound routines. Not decorative prose. Atomic backbone retained intact. No graph-only efficacy claim.',
        'turn': chain([{'id': key, 'binding': key, 'summary': text} for key, text in TURN_STAGES]),
        'market': chain([{'id': key, 'binding': key, 'summary': text,
                          'source_lines': [start, MARKET_STAGES[i+1][1]-1 if i+1 < len(MARKET_STAGES) else 750]}
                         for i, (key, start, text) in enumerate(MARKET_STAGES)]),
    }
    (HERE / 'policy_graph.json').write_text(json.dumps(graph, indent=2) + '\n')
    opponents = []
    for name in ['hazel_weir', 'copper_weir']:
        source = REPO / 'shinka/champions/submissions' / name
        sm = json.loads((source / 'MANIFEST.json').read_text())
        target = OUT / 'agents' / ('submission_' + name)
        for relative in sm['archive_files']:
            dest = target / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / relative, dest)
        opponents.append({'id': 'submission_' + name, 'label': sm['public_name'],
                          'kind': 'submission', 'submission_id': sm['kaggle']['submission_id']})
    candidate = OUT / 'agents/candidate_hazel_graph'
    shutil.copytree(OUT / 'agents/submission_hazel_weir', candidate)
    for name in ('policy_graph.json', 'graph_runtime.py'):
        shutil.copy2(HERE / name, candidate / name)
    main_text = (candidate / 'main.py').read_text()
    main_text = main_text.replace('import champion as _champion  # noqa: E402  (loads the backbone and the oracle)',
        'import champion as _champion\nfrom graph_runtime import HazelGraph\n_GRAPH = HazelGraph(_champion, os.path.join(HERE, "policy_graph.json"))')
    main_text = main_text.replace('return _champion.agent(obs, configuration)', 'return _GRAPH.agent(obs, configuration)')
    (candidate / 'main.py').write_text(main_text)
    frozen = REPO / 'research/frozen_eval/runs/iter27_trace_mining/payload'
    shutil.copytree(frozen / 'agents/candidate_iter27', OUT / 'agents/candidate_iter27',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    opponents.append({'id': 'candidate_iter27', 'label': 'Frozen old iteration 27', 'kind': 'control'})
    shutil.copy2(frozen / 'pool_upgrade_bundle_agent.py', OUT / 'pool_upgrade_bundle_agent.py')
    shutil.copy2(REPO / 'research/frozen_eval/runner.py', OUT / 'runner.py')
    eval_manifest = {
        'evaluation_id': 'hazel-executable-graph-import-v1', 'candidate_id': 'candidate_hazel_graph',
        'engine_version': '1.32.7', 'opponents': opponents,
        'seeds': {'0': [1205926567], '1': [1900960840]},
        'design': 'Paired diagnostic regression seeds from prior trace tests; NOT a held-out efficacy benchmark. One seed per seat per opponent.',
        'files': {str(p.relative_to(OUT)): sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()},
    }
    (OUT / 'manifest.json').write_text(json.dumps(eval_manifest, indent=2) + '\n')
    print(json.dumps({'source_id': graph['source']['submission_id'], 'turn_nodes': len(TURN_STAGES),
                      'market_nodes': len(MARKET_STAGES), 'payload': str(OUT), 'files': len(eval_manifest['files'])}, indent=2))


if __name__ == '__main__':
    main()
