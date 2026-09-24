"""Re-pin policy_graph.json to the runtime files as they are on disk.

agent_graph.py refuses to run when a pinned hazel_runtime/ file or the entrypoint
changed, so every runtime edit must be followed by this script (and by the tests).
The file list itself is not changed: only the listed files are re-hashed.

    python research/procedural_graph/refresh_provenance.py [--check]
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GRAPH = ROOT / 'policy_graph.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(check=False):
    graph = json.loads(GRAPH.read_text())
    provenance = graph['provenance']
    stale = {}
    for relative, pinned in provenance['runtime_bundle_hashes'].items():
        actual = sha(ROOT / 'hazel_runtime' / relative)
        if actual != pinned:
            stale[relative] = actual
    entry = sha(ROOT / 'agent_graph.py')
    if provenance.get('entrypoint_sha256') != entry:
        stale['<entrypoint agent_graph.py>'] = entry
    model = provenance['runtime_bundle_hashes'].get('checkpoint/model.safetensors')
    for relative in sorted(stale):
        print(f'stale: {relative}')
    if check:
        return 1 if stale else 0
    for relative, actual in stale.items():
        if relative.startswith('<'):
            provenance['entrypoint_sha256'] = actual
        else:
            provenance['runtime_bundle_hashes'][relative] = actual
    new_model = provenance['runtime_bundle_hashes'].get('checkpoint/model.safetensors')
    if new_model != model:
        graph['source']['oracle_model_sha256'] = new_model
        graph['model_checkpoint']['model_sha256'] = new_model
    GRAPH.write_text(json.dumps(graph, indent=2) + '\n')
    print(f'{len(stale)} pin(s) refreshed')
    return 0


if __name__ == '__main__':
    sys.exit(main('--check' in sys.argv[1:]))
