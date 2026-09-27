#!/usr/bin/env python3
"""The gauntlet's job plan for ladder evolution: our lost ladder seeds against the public agent
that plays like the opponent who beat us there, plus random seeds against the pool.

    python research/procedural_graph/ladder_seed_plan.py \
        --losses shinka/champions/evidence/ladder_pool_20260926/rowan_glen_index.json=replays/ours/rowan_glen \
        --losses shinka/champions/evidence/ladder_pool_20260926/linden_brook_index.json=replays/ours/linden_brook \
        --evidence shinka/champions/evidence/ladder_pool_20260926/ladder_match.jsonl \
        --random-opponents haideptry_2965,haideptry_shepherd,abo_v57,... --random-seeds 10 --out plan.json

    # a running evolution's plan grows by a submission's newer losses (its entries are kept, except
    # that a lost game the new evidence matches longer gets that bundle)
    python research/procedural_graph/ladder_seed_plan.py --extend plan.json --with-ties \
        --losses shinka/champions/evidence/alder_ford_20260926/index.json=replays/ours/alder_ford \
        --evidence shinka/champions/evidence/alder_ford_20260926/ladder_match.jsonl \
        --fallback tetsutani_demand --out plan.json

Each lost game (an index from fetch_games.py; its seed is the replay's info.seed) gets an
opponent: the ladder-pool bundle that reproduced the opponent's recorded actions longest
(match_ladder_games.py, at least --min-steps equal steps; among equally long ones the bundle of
the opening if it is one of them, else the first by name), else the bundle of its step-0
opening (OPENINGS), else --fallback. Evidence rows of bundles outside the pool (candidates
that were only matched) are ignored. With --extend, an earlier lost game that the given evidence
matches longer than its assignment did is moved to that bundle (`reassigned_from` keeps the old
one), with or without --losses. The candidate plays each lost seed from both seats. With
--replay-opponents it also plays the seat we played against the rival's recorded moves
(make_replay_opponents.py bundles replay_<episode>), for every lost game that has one, or for every
game whose result is in --replay-results (e.g. W,L,T: won games too, against their replays only).
Random seeds are the gauntlet's evolution seeds (disjoint from the arena validation seeds), the
candidate's seat alternating. Every entry: {"tag": bundle, "seed", "a_seat", "set":
"lost"|"random", ...}.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LADDER = HERE.parents[1] / 'shinka' / 'champions' / 'ladder'
sys.path.insert(0, str(HERE))

from graph_gauntlet import seed_list  # noqa: E402

# the opponent's step-0 wheat orders -> the pool bundle with that opening (2026-09-26 ladder). Changed
# in the evening for Alder Ford's losses; the plan's earlier entries were made with abo_v57_open13 for
# buy 13, haideptry_shepherd for buy 8 / sell 3 and haideptry_2965 for buy 20 / sell 15.
OPENINGS = (
    # tetsutani's "Shape the Shop" opens exactly so and plays three 13-wheat rivals move for move
    ((('BUY_PRODUCT', 'WHEAT', 13),), 'tetsutani_shape_shop'),
    ((('BUY_PRODUCT', 'WHEAT', 14),), 'abo_v57_open13'),
    ((('BUY_PRODUCT', 'WHEAT', 13), ('SELL', 'WHEAT', 13)), 'abo_v57_open13'),
    ((('BUY_PRODUCT', 'WHEAT', 5),), 'tetsutani_demand'),
    # the Four-Turn Forecast reproduces these rivals at least as long as the Shepherd, in every game
    ((('BUY_PRODUCT', 'WHEAT', 8), ('SELL', 'WHEAT', 3)), 'leoprovorov_forecast'),
    # the 2965 notebook's version of 09-26 13:54 UTC, which ladder players ran that afternoon
    ((('BUY_PRODUCT', 'WHEAT', 20), ('SELL', 'WHEAT', 15)), 'haideptry_2965_0926'),
    ((('BUY_PRODUCT', 'WHEAT', 6), ('SELL', 'WHEAT', 5)), 'hanifnoerrofiq_pioneers'),
)


def wheat_opening(action):
    return tuple(tuple([o[0], o[1], int(o[2])]) for o in (action or {}).get('market') or []
                 if isinstance(o, (list, tuple)) and len(o) >= 3 and o[1] == 'WHEAT' and o[0] in ('BUY_PRODUCT', 'SELL'))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--losses', action='append', default=[], metavar='INDEX=REPLAY_DIR')
    ap.add_argument('--evidence', action='append', default=[], help='match_ladder_games.py output (JSON lines)')
    ap.add_argument('--min-steps', type=int, default=100)
    ap.add_argument('--fallback', default='haideptry_2965')
    ap.add_argument('--random-opponents', default='')
    ap.add_argument('--random-seeds', type=int, default=10)
    ap.add_argument('--with-ties', action='store_true', help='tied games count as lost ones')
    ap.add_argument('--replay-opponents', type=Path, default=None, metavar='DIR',
                    help='directory of replay_<episode> bundles (shinka/champions/replay_opponents)')
    ap.add_argument('--replay-results', default='', metavar='W,L,T',
                    help='ladder results whose games get a replay job (default: the lost ones, as for the stand-ins)')
    ap.add_argument('--extend', type=Path, default=None,
                    help='an existing plan: its entries are kept as they are, the new ones appended')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()

    best = {}  # episode -> (equal steps, the pool bundles that reached them)
    for path in args.evidence:
        for line in Path(path).read_text().splitlines():
            row = json.loads(line)
            if row['equal_steps'] < args.min_steps or not (LADDER / row['bundle'] / 'SOURCE.json').is_file():
                continue
            steps, tied = best.get(row['episode'], (0, set()))
            if row['equal_steps'] > steps:
                best[row['episode']] = (row['equal_steps'], {row['bundle']})
            elif row['equal_steps'] == steps:
                tied.add(row['bundle'])
    plan = json.loads(args.extend.read_text()) if args.extend else []
    why = collections.Counter()
    for entry in plan:
        if entry.get('set') != 'lost' or entry.get('episode') not in best:
            continue
        steps, tied = best[entry['episode']]
        how = entry.get('assigned_by', '')
        before = int(how.split()[1]) if how.startswith('match ') else 0
        if steps > before and entry['tag'] not in tied:
            entry['reassigned_from'] = entry['tag']
            entry.update(tag=sorted(tied)[0], assigned_by=f'match {steps} steps')
            why['reassigned'] += 1
    results = ('L', 'T') if args.with_ties else ('L',)
    replay_results = tuple(args.replay_results.split(',')) if args.replay_results else results
    for spec in args.losses:
        index, replays = spec.split('=', 1)
        for game in json.loads(Path(index).read_text()):
            result = game.get('res', game.get('result'))
            source = args.replay_opponents / f"replay_{game['id']}" / 'SOURCE.json' if args.replay_opponents else None
            if source and result in replay_results and source.is_file():
                meta = json.loads(source.read_text())
                plan.append({'tag': meta['name'], 'seed': int(meta['seed']), 'a_seat': int(meta['our_seat']),
                             'set': 'replay', 'episode': game['id'], 'opponent': game.get('team') or game.get('opponent'),
                             'kaggle_seat': game['seat'], 'assigned_by': 'recorded moves', 'result': result})
                why['replay'] += 1
            if result not in results:
                continue
            replay = json.loads((Path(replays) / f"episode-{game['id']}-replay.json").read_text())
            seed = int(replay['info']['seed'])
            opening = wheat_opening(replay['steps'][1][1 - game['seat']].get('action'))
            if game['id'] in best:
                steps, tied = best[game['id']]
                by_opening = next((b for o, b in OPENINGS if opening == o), None)
                tag, how = (by_opening if by_opening in tied else sorted(tied)[0]), f'match {steps} steps'
            else:
                tag = next((b for o, b in OPENINGS if opening == o), args.fallback)
                how = 'opening' if tag != args.fallback or any(opening == o for o, _ in OPENINGS) else 'fallback'
            why[how.split()[0]] += 1
            for seat in (0, 1):
                plan.append({'tag': tag, 'seed': seed, 'a_seat': seat, 'set': 'lost', 'episode': game['id'],
                             'opponent': game.get('team') or game.get('opponent'), 'kaggle_seat': game['seat'],
                             'assigned_by': how, **({'result': 'T'} if game.get('res') == 'T' else {})})
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
