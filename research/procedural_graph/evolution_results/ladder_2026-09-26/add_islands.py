#!/usr/bin/env python3
"""One-off (2026-09-27): add islands that evolve on another production engine to run ladder1.

    python add_islands.py RUN_DIR SEED_GRAPH --island "NAME=FOCUS" [--island ...] [--apply]

Each new island starts from SEED_GRAPH (e.g. a make_ladder_graph.py graph on the public engine's
09-27 version) and keeps it as its own `seed` (the loop's seed_of), so its mixing and its changes are
measured against that seed, and it mixes only with islands on the same engine. The seed's settings key
joins "seen". Without --apply it prints what it would add. Run it only while the loop is stopped.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]  # research/procedural_graph
sys.path.insert(0, str(HERE))

import graph_edits  # noqa: E402
from pool_upgrade_state import atomic_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir', type=Path)
    ap.add_argument('seed_graph', type=Path)
    ap.add_argument('--island', action='append', required=True, metavar='NAME=FOCUS')
    ap.add_argument('--apply', action='store_true')
    args = ap.parse_args()
    checkpoint = args.run_dir / 'checkpoint.json'
    state = json.loads(checkpoint.read_text())
    seed = json.loads(args.seed_graph.read_text())
    if seed.get('runtime') != 'hazel_merged_v1':
        raise SystemExit('the seed graph must declare runtime hazel_merged_v1')
    names = {i['name'] for i in state['islands']}
    constants = graph_edits.catalog()
    for spec in args.island:
        name, focus = spec.split('=', 1)
        if name in names:
            raise SystemExit(f'island {name} exists')
        state['islands'].append({'name': name, 'focus': focus, 'stages': [f'engine {graph_edits.engine_name(seed)}'],
                                 'graph': copy.deepcopy(seed), 'seed': copy.deepcopy(seed),
                                 'history': [], 'rejections': []})
        print(f'+ {name} on {graph_edits.engine_name(seed)}: {focus[:120]}')
    key = graph_edits.settings_key(seed, constants)
    if key not in state['seen']:
        state['seen'].append(key)
    print(f"{len(state['islands'])} islands: " + ', '.join(i['name'] for i in state['islands']))
    if args.apply:
        atomic_json(checkpoint, state)
        print(f'written {checkpoint}')


if __name__ == '__main__':
    main()
