"""Freeze CURRENT merged entrypoint for regression games; never mutate prior payload."""
import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
PG = HERE.parent
OUT = HERE / 'merged_payload'


def main():
    if OUT.exists(): raise SystemExit('Refusing overwrite')
    old = HERE / 'payload'
    manifest = json.loads((old / 'manifest.json').read_text())
    for opponent in manifest['opponents']:
        shutil.copytree(old / 'agents' / opponent['id'], OUT / 'agents' / opponent['id'],
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    candidate = OUT / 'agents/candidate_merged_graph'
    shutil.copytree(PG / 'hazel_runtime', candidate / 'hazel_runtime',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(PG / 'agent_graph.py', candidate / 'main.py')
    shutil.copy2(PG / 'policy_graph.json', candidate / 'policy_graph.json')
    for name in ['runner.py', 'pool_upgrade_bundle_agent.py']:
        shutil.copy2(old / name, OUT / name)
    manifest['evaluation_id'] = 'current-graph-hazel-merged-v1'
    manifest['candidate_id'] = 'candidate_merged_graph'
    manifest['files'] = {str(p.relative_to(OUT)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted(OUT.rglob('*')) if p.is_file()}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('Frozen current merged entrypoint:', OUT)


if __name__ == '__main__': main()
