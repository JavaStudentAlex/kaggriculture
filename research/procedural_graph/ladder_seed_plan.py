#!/usr/bin/env python3
"""The gauntlet's job plan for ladder evolution: our lost ladder seeds against the public agent
that plays like the opponent who beat us there, plus random seeds against the pool.

    python research/procedural_graph/ladder_seed_plan.py \
        --losses shinka/champions/evidence/ladder_pool_20260926/rowan_glen_index.json=replays/ours/rowan_glen \
        --losses shinka/champions/evidence/ladder_pool_20260926/linden_brook_index.json=replays/ours/linden_brook \
        --evidence shinka/champions/evidence/ladder_pool_20260926/ladder_match.jsonl \
        --random-opponents haideptry_2965,haideptry_shepherd,abo_v57,... --random-seeds 10 --out plan.json

Each lost game (an index from fetch_games.py; its seed is the replay's info.seed) gets an
opponent: the ladder bundle that reproduced the opponent's recorded actions longest
(match_ladder_games.py, at least --min-steps equal steps), else the bundle of its step-0 opening
(OPENINGS), else --fallback. The candidate plays each lost seed from both seats. Random seeds
are the gauntlet's evolution seeds (disjoint from the arena validation seeds), the candidate's
seat alternating. Every entry: {"tag": bundle, "seed", "a_seat", "set": "lost"|"random", ...}.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from graph_gauntlet import seed_list  # noqa: E402

# the opponent's step-0 wheat orders -> the pool bundle with that opening (2026-09-26 ladder)
OPENINGS = (
    ((('BUY_PRODUCT', 'WHEAT', 13),), 'abo_v57_open13'),
    ((('BUY_PRODUCT', 'WHEAT', 14),), 'abo_v57_open13'),
    ((('BUY_PRODUCT', 'WHEAT', 13), ('SELL', 'WHEAT', 13)), 'abo_v57_open13'),
    ((('BUY_PRODUCT', 'WHEAT', 5),), 'tetsutani_demand'),
    ((('BUY_PRODUCT', 'WHEAT', 8), ('SELL', 'WHEAT', 3)), 'haideptry_shepherd'),
    ((('BUY_PRODUCT', 'WHEAT', 20), ('SELL', 'WHEAT', 15)), 'haideptry_2965'),
    ((('BUY_PRODUCT', 'WHEAT', 6), ('SELL', 'WHEAT', 5)), 'hanifnoerrofiq_pioneers'),
)


def wheat_opening(action):
    return tuple(tuple([o[0], o[1], int(o[2])]) for o in (action or {}).get('market') or []
                 if isinstance(o, (list, tuple)) and len(o) >= 3 and o[1] == 'WHEAT' and o[0] in ('BUY_PRODUCT', 'SELL'))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--losses', action='append', required=True, metavar='INDEX=REPLAY_DIR')
    ap.add_argument('--evidence', action='append', default=[], help='match_ladder_games.py output (JSON lines)')
    ap.add_argument('--min-steps', type=int, default=100)
    ap.add_argument('--fallback', default='haideptry_2965')
    ap.add_argument('--random-opponents', default='')
    ap.add_argument('--random-seeds', type=int, default=10)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()

    best = {}
    for path in args.evidence:
        for line in Path(path).read_text().splitlines():
            row = json.loads(line)
            if row['equal_steps'] >= args.min_steps and row['equal_steps'] > best.get(row['episode'], (0, ''))[0]:
                best[row['episode']] = (row['equal_steps'], row['bundle'])
    plan, why = [], collections.Counter()
    for spec in args.losses:
        index, replays = spec.split('=', 1)
        for game in json.loads(Path(index).read_text()):
            if game.get('res', game.get('result')) != 'L':
                continue
            replay = json.loads((Path(replays) / f"episode-{game['id']}-replay.json").read_text())
            seed = int(replay['info']['seed'])
            opening = wheat_opening(replay['steps'][1][1 - game['seat']].get('action'))
            if game['id'] in best:
                tag, how = best[game['id']][1], f"match {best[game['id']][0]} steps"
            else:
                tag = next((b for o, b in OPENINGS if opening == o), args.fallback)
                how = 'opening' if tag != args.fallback or any(opening == o for o, _ in OPENINGS) else 'fallback'
            why[how.split()[0]] += 1
            for seat in (0, 1):
                plan.append({'tag': tag, 'seed': seed, 'a_seat': seat, 'set': 'lost', 'episode': game['id'],
                             'opponent': game.get('team') or game.get('opponent'), 'kaggle_seat': game['seat'],
                             'assigned_by': how})
    seeds = seed_list(args.random_seeds)
    for tag in [t for t in args.random_opponents.split(',') if t]:
        plan += [{'tag': tag, 'seed': s, 'a_seat': i % 2, 'set': 'random'} for i, s in enumerate(seeds)]
    seen, unique = set(), []
    for entry in plan:
        key = (entry['tag'], entry['seed'], entry['a_seat'])
        if key not in seen:
            seen.add(key)
            unique.append(entry)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(unique, indent=1) + '\n')
    per = collections.Counter((e['set'], e['tag']) for e in unique)
    print(json.dumps({'jobs': len(unique), 'lost_games_assigned_by': why,
                      'per_set_and_opponent': {f'{s}:{t}': n for (s, t), n in sorted(per.items())}}, indent=1))


if __name__ == '__main__':
    main()
