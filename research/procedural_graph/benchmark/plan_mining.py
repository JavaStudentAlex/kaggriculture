#!/usr/bin/env python3
"""Plan mining: the production plans of the ladder's best players, per world (runs in a Kaggle CPU notebook).

    python plan_mining.py [--inputs /kaggle/input] [--out /kaggle/working/plans.jsonl] [--top N] [--workers 4]
                          [--census-hour H]

Kaggle's daily datasets (kaggle/kaggriculture-episodes-<day>, attached to the notebook) hold the games between
the ladder's best players. For every game (or the `--top` highest-rated per day, manifest.csv) and each seat this
writes one line: the world (the first two shops, known from step 144, which is when the public engine picks its
route tape), the result, and the plan, as the same three views plan_report.py builds for our route tapes:
- census: the seat's public farm at hour CENSUS_HOUR of days CENSUS_DAYS: money, hands (hired by the day, so
  none at hour 0), land quadrants, animals by kind, planted tiles by crop (the first run, 09-28, took it at hour 0);
- buys: its market orders summed up to those days: BUY_ANIMAL units by animal, BUY_SEED units by crop, HIRE,
  BUY_LAND, BUY_PRODUCT units (wheat, fertilizer);
- sells: SELL units submitted over the game by product (arena.py's ledger counts the same for our games).
Kaggle stores the action taken from observation t at steps[t + 1], and that is how it is read here.
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import sys
import time
from pathlib import Path

CENSUS_DAYS = (3, 6, 9, 12, 15, 20, 25, 29)
CENSUS_HOUR = 12
ROUTE_STEP = 144


def buy_key(order):
    """The plan key a market order counts under, with its units, or None."""
    if not isinstance(order, (list, tuple)) or not order:
        return None
    kind = order[0]
    if kind == 'BUY_ANIMAL' and len(order) >= 3:
        return f'ANIMAL:{order[1]}', int(order[2] or 0)
    if kind == 'BUY_SEED' and len(order) >= 3:
        return f'SEED:{order[1]}', int(order[2] or 0)
    if kind == 'BUY_PRODUCT' and len(order) >= 3:
        return f'PRODUCT:{order[1]}', int(order[2] or 0)
    if kind == 'HIRE':
        return 'HIRE', 1
    if kind == 'BUY_LAND':
        return 'LAND', 1
    return None


def plan_of(actions, days=CENSUS_DAYS):
    """buys (cumulative at the start of each census day) and sells (whole game) of a list of actions by step."""
    buys, sells, running = {}, {}, {}
    marks = {24 * d: d for d in days}
    for step, action in enumerate(actions):
        if step in marks:
            buys[str(marks[step])] = dict(running)
        market = (action or {}).get('market') if isinstance(action, dict) else None
        for order in market or []:
            key = buy_key(order)
            if key is not None:
                running[key[0]] = running.get(key[0], 0) + key[1]
            elif isinstance(order, (list, tuple)) and len(order) >= 3 and order[0] == 'SELL':
                sells[order[1]] = sells.get(order[1], 0) + int(order[2] or 0)
    for d in days:
        buys.setdefault(str(d), dict(running))
    return buys, sells


def census_of(farm):
    animals, crops = {}, {}
    for row in farm.get('tiles') or []:
        for tile in row:
            if not isinstance(tile, dict):
                continue
            if tile.get('animal'):
                animals[tile['animal']] = animals.get(tile['animal'], 0) + 1
            elif tile.get('kind') == 'PLANT' and tile.get('crop'):
                crops[tile['crop']] = crops.get(tile['crop'], 0) + 1
    return {'money': farm.get('money'), 'hands': len(farm.get('hands') or []),
            'land': len(farm.get('unlocked_quadrants') or []), 'animals': animals, 'crops': crops}


def mine(job):
    path, day, row, hour = job
    try:
        replay = json.loads(Path(path).read_text())
    except Exception as exc:  # a damaged file must not stop the run
        return [{'episode': row.get('episode_id'), 'error': repr(exc)[:200]}]
    steps = replay['steps']
    rewards = replay.get('rewards') or [s.get('reward') for s in steps[-1]]
    info = replay.get('info') or {}
    teams = info.get('TeamNames') or [a.get('Name') for a in info.get('Agents') or []] or [None, None]
    public = [s[0].get('observation') or {} for s in steps]
    shops = ((public[min(ROUTE_STEP, len(public) - 1)].get('town') or {}).get('unlocked_shops') or [])
    final_shops = ((public[-1].get('town') or {}).get('unlocked_shops') or [])
    out = []
    for seat in (0, 1):
        actions = [steps[t + 1][seat].get('action') for t in range(len(steps) - 1)]
        buys, sells = plan_of(actions)
        census = {str(d): census_of(public[24 * d + hour]['farms'][seat]) for d in CENSUS_DAYS
                  if 24 * d + hour < len(public)}
        mine_, theirs = rewards[seat], rewards[1 - seat]
        out.append({'episode': int(row['episode_id']), 'day': day, 'seat': seat, 'team': teams[seat],
                    'rival_team': teams[1 - seat], 'avg_score': float(row.get('avg_score') or 0),
                    'min_score': float(row.get('min_score') or 0), 'seed': (info or {}).get('seed'),
                    'reward': mine_, 'rival_reward': theirs,
                    'result': 'W' if mine_ > theirs else 'L' if mine_ < theirs else 'T',
                    'world': shops[:2], 'shops': final_shops, 'opening': (actions[0] or {}).get('market'),
                    'census': census, 'buys': buys, 'sells': sells})
    return out


def games(inputs: Path, top: int, hour: int = CENSUS_HOUR):
    jobs = []
    for manifest in sorted(inputs.rglob('manifest.csv')):
        rows = sorted(csv.DictReader(manifest.open()), key=lambda r: -float(r.get('avg_score') or 0))
        if top:
            rows = rows[:top]
        for row in rows:
            path = manifest.parent / f"{row['episode_id']}.json"
            if path.exists():
                jobs.append((str(path), manifest.parent.name, row, hour))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', default='/kaggle/input')
    ap.add_argument('--out', default='/kaggle/working/plans.jsonl')
    ap.add_argument('--top', type=int, default=0, help='games per day by average rating (0: all)')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--census-hour', type=int, default=CENSUS_HOUR, help='hour of the census days (0-23)')
    args = ap.parse_args()
    jobs = games(Path(args.inputs), args.top, args.census_hour)
    print(f'{len(jobs)} games', flush=True)
    start, done, errors = time.time(), 0, 0
    with open(args.out, 'w') as out, multiprocessing.Pool(args.workers) as pool:
        for rows in pool.imap_unordered(mine, jobs, chunksize=4):
            for r in rows:
                errors += 'error' in r
                out.write(json.dumps(r, separators=(',', ':')) + '\n')
            done += 1
            if done % 200 == 0:
                print(f'{done}/{len(jobs)} games, {time.time() - start:.0f} s', flush=True)
    print(f'PLAN_MINING_DONE games={done} errors={errors} seconds={time.time() - start:.0f}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
