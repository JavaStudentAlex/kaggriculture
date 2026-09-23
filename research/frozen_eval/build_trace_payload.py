"""Build a focused trace-mining payload: iter-27 vs Hazel Weir + Copper Weir only.

400 fresh seeds per seat (800 total games across 2 opponents = 1600 games).
The runner will dump full env.steps traces for every match.
"""
from pathlib import Path
import hashlib
import json
import random
import shutil

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
RUN_DIR = HERE / 'runs' / 'iter27_trace_mining'
OUT = RUN_DIR / 'payload'

EVALUATION_ID = 'iter27-trace-mining-20260923-400x2-v1'
CANDIDATE_GRAPH_SHA = '5abf926b81653557c9023ca60eb9f1656eb2e359c4641e70b36f7c8f3417ee49'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def tree(src, dst):
    for f in sorted(src.rglob('*')):
        if f.is_file() and '__pycache__' not in f.parts and f.suffix != '.pyc':
            copy(f, dst / f.relative_to(src))


def common(dst):
    evo = REPO / 'shinka/evolution'
    for n in ['initial.py', 'kagg_oracle.py', 'kagg_ttm_numpy.py']:
        copy(evo / n, dst / n)
    for n in ['config.json', 'labels.json', 'model.safetensors', 'scaler.npz']:
        copy(evo / 'checkpoint' / n, dst / 'checkpoint' / n)
    tree(REPO / 'shinka/champions/dependencies/mohui_v66', dst / 'mohui_v66')
    for n in ['features.py', 'mechanics.py']:
        copy(REPO / 'research/opponent_model' / n, dst / 'opponent_model' / n)
    copy(REPO / 'research/procedural_graph/graph_engine.py', dst / 'graph_engine.py')


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    # Verify candidate graph
    graph_src = REPO / 'research/procedural_graph/highcpu_evo_run/cand_iter_027_Island-Watering.json'
    graph = json.loads(graph_src.read_text())
    canonical_hash = hashlib.sha256(
        json.dumps(graph, sort_keys=True, separators=(',', ':')).encode()
    ).hexdigest()
    assert canonical_hash == CANDIDATE_GRAPH_SHA, f"Graph hash mismatch: {canonical_hash}"

    # Build candidate bundle
    cand = OUT / 'agents/candidate_iter27'
    common(cand)
    copy(graph_src, cand / 'policy_graph.json')
    copy(REPO / 'research/procedural_graph/agent_graph.py', cand / 'main.py')

    # Build opponent bundles — only the two submissions
    opponents = []
    for n in ['hazel_weir', 'copper_weir']:
        src = REPO / 'shinka/champions/submissions' / n
        manifest = json.loads((src / 'MANIFEST.json').read_text())
        checks = {
            'champion.py': manifest['champion']['sha256'],
            'kagg_oracle.py': manifest['oracle']['kagg_oracle_sha256'],
            'kagg_ttm_numpy.py': manifest['oracle']['kagg_ttm_numpy_sha256'],
            'checkpoint/model.safetensors': manifest['oracle']['model_safetensors_sha256'],
        }
        assert all(digest(src / f) == sha for f, sha in checks.items()), f"{n} hash mismatch"
        dst = OUT / 'agents' / ('submission_' + n)
        for f in manifest['archive_files']:
            copy(src / f, dst / f)
        copy(src / 'MANIFEST.json', dst / 'SOURCE_MANIFEST.json')
        opponents.append({
            'id': 'submission_' + n,
            'label': manifest['public_name'] + ' (submitted package)',
            'kind': 'submission',
            'submission_id': manifest['kaggle']['submission_id'],
            'identity': 'Saved extracted package; key hashes match submission manifest.',
        })

    # Freeze a private RPC helper with network denial
    rpc = (REPO / 'shinka/evolution/pool_upgrade_bundle_agent.py').read_text()
    rpc = rpc.replace("        bundle = Path(bundle_arg).resolve()", """        import socket
        network_attempts = [0]
        def deny_network(*args, **kwargs):
            network_attempts[0] += 1
            raise RuntimeError('Frozen evaluation denies network connections')
        socket.socket.connect = deny_network
        socket.socket.connect_ex = deny_network
        socket.create_connection = deny_network
        bundle = Path(bundle_arg).resolve()""")
    rpc = rpc.replace("'pid': os.getpid(), 'entrypoint': str(entry.relative_to(bundle)),",
                      "'network_guard': {'mode': 'python_socket_connect_denial', 'scope': 'agent_child', 'connection_attempts_at_startup': 0}, 'pid': os.getpid(), 'entrypoint': str(entry.relative_to(bundle)),")
    rpc = rpc.replace("        send({'ok': True, 'phase': 'ready', 'metadata': _metadata(bundle, entry)})",
                      "        if network_attempts[0]: raise RuntimeError('Network attempt during agent import')\n        send({'ok': True, 'phase': 'ready', 'metadata': _metadata(bundle, entry)})")
    rpc = rpc.replace("'action': action})", "'action': action, 'network_attempts': network_attempts[0]})")
    rpc = rpc.replace("                return reply['action']", """                if reply.get('network_attempts', 0):
                    raise BundleAgentError('Agent attempted network access')
                return reply['action']""")
    (OUT / 'pool_upgrade_bundle_agent.py').write_text(rpc)

    # Copy runner.py into payload so the Kaggle kernel can use it
    copy(HERE / 'runner.py', OUT / 'runner.py')

    # Generate 800 fresh disjoint seeds (400 seat-0 + 400 seat-1)
    rng = random.Random(202609230400)
    all_seeds = rng.sample(range(1_000_000, 2_000_000_000), 800)
    assert len(set(all_seeds)) == 800

    # Need exactly 2 opponents for this manifest, but runner validates ==15.
    # We'll pass --opponents to subset. Still need manifest structure valid.
    # Actually, let's patch: the validation requires 15 opponents. For this
    # trace-mining run we only need 2. Let's set opponent count == 2 and
    # use a separate validation path. Better: just skip the 15-opponent
    # assertion for this dedicated evaluation. We'll run with the runner
    # directly specifying opponents.

    manifest_data = {
        'evaluation_id': EVALUATION_ID,
        'candidate_id': 'candidate_iter27',
        'candidate_graph_sha256': CANDIDATE_GRAPH_SHA,
        'engine_version': '1.32.7',
        'opponents': opponents,
        'seeds': {'0': all_seeds[:400], '1': all_seeds[400:]},
        'design': (
            '400 disjoint fresh seeds per seat × 2 submission opponents = 1600 games. '
            'Full env.steps traces saved for procedural graph forensics. '
            'No evolution, no pool champions, no self-control.'
        ),
        'files': {
            str(f.relative_to(OUT)): digest(f)
            for f in sorted(OUT.rglob('*'))
            if f.is_file() and f.name != 'manifest.json' and '__pycache__' not in f.parts
        },
    }
    (OUT / 'manifest.json').write_text(json.dumps(manifest_data, indent=2) + '\n')

    total_games = len(opponents) * (len(all_seeds[:400]) + len(all_seeds[400:]))
    print(json.dumps({
        'payload': str(OUT),
        'evaluation_id': EVALUATION_ID,
        'opponents': len(opponents),
        'seeds_per_seat': 400,
        'total_games': total_games,
        'files': len(manifest_data['files']),
        'bytes': sum(f.stat().st_size for f in OUT.rglob('*') if f.is_file()),
        'graph_sha256': CANDIDATE_GRAPH_SHA,
    }, indent=2))


if __name__ == '__main__':
    main()
