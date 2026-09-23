"""Build a private, immutable iteration-27 evaluation payload; no cloud side effects."""
from pathlib import Path
import hashlib
import json
import random
import shutil

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
OUT = HERE / 'runs' / 'iter27_20260923' / 'payload'

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
    OUT.mkdir(parents=True, exist_ok=True)
    graph_src = REPO / 'research/procedural_graph/highcpu_evo_run/cand_iter_027_Island-Watering.json'
    graph = json.loads(graph_src.read_text())
    canonical = hashlib.sha256(json.dumps(graph, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    assert canonical == '5abf926b81653557c9023ca60eb9f1656eb2e359c4641e70b36f7c8f3417ee49'
    cand = OUT / 'agents/candidate_iter27'
    common(cand)
    copy(graph_src, cand / 'policy_graph.json')
    copy(REPO / 'research/procedural_graph/agent_graph.py', cand / 'main.py')
    codenames = json.loads((REPO / 'shinka/champions/CODENAMES.json').read_text())
    labels = {Path(x['file']).name: x['codename'] for x in codenames['champions']}
    opponents = []
    for f in sorted((REPO / 'shinka/champions/pool').glob('*.py')):
        if f.name == '__init__.py':
            continue
        dst = OUT / 'agents' / f.stem
        common(dst)
        copy(f, dst / 'main.py')
        graph_file = f.with_suffix('.graph.json')
        if graph_file.exists():
            copy(graph_file, dst / 'main.graph.json')
        kind = 'self_control' if 'evo_0027_' in f.name else 'pool'
        opponents.append({'id': f.stem, 'label': labels.get(f.name, f.stem), 'kind': kind,
                          'source': str(f.relative_to(REPO)), 'source_sha256': digest(f)})
    for n in ['hazel_weir', 'copper_weir']:
        src = REPO / 'shinka/champions/submissions' / n
        manifest = json.loads((src / 'MANIFEST.json').read_text())
        checks = {'champion.py': manifest['champion']['sha256'],
                  'kagg_oracle.py': manifest['oracle']['kagg_oracle_sha256'],
                  'kagg_ttm_numpy.py': manifest['oracle']['kagg_ttm_numpy_sha256'],
                  'checkpoint/model.safetensors': manifest['oracle']['model_safetensors_sha256']}
        assert all(digest(src / f) == sha for f, sha in checks.items())
        dst = OUT / 'agents' / ('submission_' + n)
        for f in manifest['archive_files']:
            copy(src / f, dst / f)
        copy(src / 'MANIFEST.json', dst / 'SOURCE_MANIFEST.json')
        opponents.append({'id': 'submission_' + n, 'label': manifest['public_name'] + ' (submitted package)',
                          'kind': 'submission', 'submission_id': manifest['kaggle']['submission_id'],
                          'identity': 'Saved extracted package; key policy/oracle/checkpoint hashes match submission manifest. Original archive unavailable.'})
    # Freeze a private RPC copy with socket attempts blocked in each child.
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
    rpc = rpc.replace("'pid': os.getpid(), 'entrypoint': str(entry.relative_to(bundle)),", "'network_guard': {'mode': 'python_socket_connect_denial', 'scope': 'agent_child', 'connection_attempts_at_startup': 0}, 'pid': os.getpid(), 'entrypoint': str(entry.relative_to(bundle)),")
    rpc = rpc.replace("        send({'ok': True, 'phase': 'ready', 'metadata': _metadata(bundle, entry)})", "        if network_attempts[0]: raise RuntimeError('Network attempt during agent import')\n        send({'ok': True, 'phase': 'ready', 'metadata': _metadata(bundle, entry)})")
    rpc = rpc.replace("'action': action})", "'action': action, 'network_attempts': network_attempts[0]})")
    rpc = rpc.replace("                return reply['action']", """                if reply.get('network_attempts', 0):
                    raise BundleAgentError('Agent attempted network access')
                return reply['action']""")
    (OUT / 'pool_upgrade_bundle_agent.py').write_text(rpc)
    rng = random.Random(202609230027)
    seeds = rng.sample(range(1_000_000, 2_000_000_000), 200)
    assert len(set(seeds)) == 200
    manifest = {'evaluation_id': 'iter27-frozen-20260923-100x2-v1', 'candidate_id': 'candidate_iter27',
                'candidate_graph_sha256': canonical, 'engine_version': '1.32.7',
                'opponents': opponents, 'seeds': {'0': seeds[:100], '1': seeds[100:]},
                'design': '100 disjoint fresh seeds per seat per opponent; same seat seed blocks across opponents. No evolution, no induction. Self-control reported separately.',
                'submission_caveat': 'Copper pool policy and submitted Copper package deliberately retained as separate deployment variants.',
                'files': {str(f.relative_to(OUT)): digest(f) for f in sorted(OUT.rglob('*')) if f.is_file() and f.name != 'manifest.json' and '__pycache__' not in f.parts}}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'payload': str(OUT), 'opponents': len(opponents), 'matches': len(opponents)*200,
                      'per_shard': len(opponents)*40, 'files': len(manifest['files']),
                      'bytes': sum(f.stat().st_size for f in OUT.rglob('*') if f.is_file()),
                      'graph_sha256': canonical}, indent=2))

if __name__ == '__main__':
    main()
