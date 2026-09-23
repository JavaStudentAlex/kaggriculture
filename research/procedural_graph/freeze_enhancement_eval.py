"""Freeze current graph for local regression without touching archived payloads."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

PG = Path(__file__).resolve().parent
BASELINE = PG / 'runs/merged_eval_kaggle/payload'


def freeze(output: Path) -> Path:
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite frozen payload: {output}')
    manifest = json.loads((BASELINE / 'manifest.json').read_text())
    ignore = shutil.ignore_patterns('__pycache__', '*.pyc', '*.pyo', '*.bak')
    for opponent in manifest['opponents']:
        shutil.copytree(BASELINE / 'agents' / opponent['id'],
                        output / 'agents' / opponent['id'], ignore=ignore)
    candidate = output / 'agents/candidate_merged_graph'
    shutil.copytree(PG / 'hazel_runtime', candidate / 'hazel_runtime', ignore=ignore)
    shutil.copy2(PG / 'agent_graph.py', candidate / 'main.py')
    shutil.copy2(PG / 'policy_graph.json', candidate / 'policy_graph.json')
    for filename in ('runner.py', 'pool_upgrade_bundle_agent.py'):
        shutil.copy2(BASELINE / filename, output / filename)
    manifest['evaluation_id'] = 'graph-enhancement-local-validation-' + output.name
    manifest['validation_scope'] = (
        'Same held-fixed disjoint-seat seed blocks as archived baseline; '
        'small local engineering check, not a promotion tournament or paired-seat superiority proof.'
    )
    manifest['files'] = {
        str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(output.rglob('*')) if path.is_file()
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'payload': str(output), 'hashed_files': len(manifest['files']),
                      'evaluation_id': manifest['evaluation_id']}))
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    freeze(parser.parse_args().output.resolve())
