#!/usr/bin/env python3
"""Move an island of run ladder1 onto another island's production engine (2026-09-27), or give it a new
name and focus (2026-09-28).

    python convert_island.py RUN_DIR --replace NAME --name NEW_NAME --from-island OTHER --focus TEXT
                             [--stage AREA ...] [--apply]
    python convert_island.py RUN_DIR --replace NAME --name NEW_NAME --refocus --focus TEXT [--stage AREA ...]
                             [--apply]

NEW_NAME takes NAME's place in the rotation, so it plays at the iterations NAME would have played. With
--from-island it starts from OTHER's champion and keeps OTHER's seed as its own `seed` (the loop's seed_of):
its gains and its mixing are measured against that seed, and it mixes only with islands on the same engine.
Its history and rejections start empty, since NAME's belong to another engine. With --refocus it keeps
NAME's champion, seed, history and rejections, and only the name, the focus and the areas named in the
prompt change. Without --apply it prints what it would change. Run it only while the loop is stopped.
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
    ap.add_argument('--replace', required=True, help='the island to replace')
    ap.add_argument('--name', required=True, help='the new island')
    source_args = ap.add_mutually_exclusive_group(required=True)
    source_args.add_argument('--from-island', help='the island whose champion and seed it starts from')
    source_args.add_argument('--refocus', action='store_true', help='keep the champion, seed and history')
    ap.add_argument('--focus', required=True)
    ap.add_argument('--stage', action='append', default=None, help='areas named in the prompt (default: the engine)')
    ap.add_argument('--apply', action='store_true')
    args = ap.parse_args()
    checkpoint = args.run_dir / 'checkpoint.json'
    state = json.loads(checkpoint.read_text())
    names = [i['name'] for i in state['islands']]
    if args.name in names:
        raise SystemExit(f'island {args.name} exists')
    if args.replace not in names or (args.from_island and args.from_island not in names):
        raise SystemExit(f'unknown island; the run has {names}')
    mixing = state.get('mixing') or {}
    if mixing and mixing.get('after') != state.get('mixed_after'):
        raise SystemExit(f"a mixing after iteration {mixing.get('after')} is unfinished; convert after it")
    old = state['islands'][names.index(args.replace)]
    if args.refocus:
        new = dict(copy.deepcopy(old), name=args.name, focus=args.focus, stages=args.stage or old['stages'])
        seed = old.get('seed') or state['seed_graph']
        origin = f"{old['name']}'s champion, history and rejections"
    else:
        source = state['islands'][names.index(args.from_island)]
        seed = source.get('seed') or state['seed_graph']
        new = {'name': args.name, 'focus': args.focus, 'graph': copy.deepcopy(source['graph']),
               'seed': copy.deepcopy(seed), 'history': [], 'rejections': []}
        new['stages'] = args.stage or [f"engine {graph_edits.engine_name(source['graph'])}"]
        origin = f"{args.from_island}'s champion"
    engine = graph_edits.engine_name(new['graph'])
    state['islands'][names.index(args.replace)] = new
    print(f"- {old['name']} on {graph_edits.engine_name(old['graph'])}: {len(old['history'])} promotions, "
          f"{len(old['rejections'])} rejections")
    print(f"+ {args.name} on {engine}, from {origin}: {args.focus[:160]}")
    print(f"areas: {new['stages']}")
    print(f"changes of its champion over its seed: {graph_edits.diff(seed, new['graph'], graph_edits.catalog())}")
    print(f"next iteration {state['next_iteration']}: "
          f"{state['islands'][(state['next_iteration'] - 1) % len(state['islands'])]['name']}; "
          f"{args.name} plays at iterations n with n % {len(names)} == {(names.index(args.replace) + 1) % len(names)}")
    print(f"{len(state['islands'])} islands: " + ', '.join(i['name'] for i in state['islands']))
    if args.apply:
        atomic_json(checkpoint, state)
        print(f'written {checkpoint}')


if __name__ == '__main__':
    main()
