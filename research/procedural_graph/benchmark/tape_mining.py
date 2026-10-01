#!/usr/bin/env python3
"""Tape mining: the full action tapes of chosen top teams, per world (runs in a Kaggle CPU notebook).

    python tape_mining.py [--inputs /kaggle/input] [--out /kaggle/working/tapes.jsonl.gz] [--team NAME ...]

The engine we play (tetsutani_demand_0927) replays route tapes: one opening tape for steps 0-143, then from step
144 the tape of the world (the first two shops), all taken from one bot's games so that every world tape continues
the same opening. Our farm is therefore the same in every game, and smaller by day 10 than the ladder's best
(shinka/champions/evidence/aspen_vale_20260929). This extracts, for the given teams (default: the four top teams of
the plan mining with at most two openings), every seat's 719 actions as the engine's tapes store them, so a route
set can be built from one bot's games. Kaggle stores the action taken from observation t at steps[t + 1].

One gzip JSON line per (game, seat) of a chosen team: episode, day, seat, team, rival_team, reward, rival_reward,
result, world (the first two shops at step 144), shops (all by the end), avg_score, census (the seat's farm at hour 12
of days 3, 6, 9, 12, 20) and actions (719 action dicts).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import multiprocessing
import sys
import time
from pathlib import Path

ROUTE_STEP = 144
CENSUS_DAYS = (3, 6, 9, 12, 20)
TEAMS = ('DSM', 'DECEM', 'Vadim Vasilenko', 'Unknown Mother-Goose')


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
    path, day, row, teams = job
    try:
        replay = json.loads(Path(path).read_text())
    except Exception as exc:  # a damaged file must not stop the run
        return [{'episode': row.get('episode_id'), 'error': repr(exc)[:200]}]
    info = replay.get('info') or {}
    names = info.get('TeamNames') or [a.get('Name') for a in info.get('Agents') or []] or [None, None]
    if not any(n in teams for n in names):
        return []
    steps = replay['steps']
    rewards = replay.get('rewards') or [s.get('reward') for s in steps[-1]]
    public = [s[0].get('observation') or {} for s in steps]
    shops = (public[min(ROUTE_STEP, len(public) - 1)].get('town') or {}).get('unlocked_shops') or []
    final_shops = (public[-1].get('town') or {}).get('unlocked_shops') or []
    out = []
    for seat in (0, 1):
        if names[seat] not in teams:
            continue
        actions = [steps[t + 1][seat].get('action') for t in range(len(steps) - 1)]
        census = {str(d): census_of(public[24 * d + 12]['farms'][seat]) for d in CENSUS_DAYS
                  if 24 * d + 12 < len(public)}
        mine_, theirs = rewards[seat], rewards[1 - seat]
        out.append({'episode': int(row['episode_id']), 'day': day, 'seat': seat, 'team': names[seat],
                    'rival_team': names[1 - seat], 'reward': mine_, 'rival_reward': theirs,
                    'result': 'W' if mine_ > theirs else 'L' if mine_ < theirs else 'T',
                    'world': shops[:2], 'shops': final_shops, 'avg_score': float(row.get('avg_score') or 0),
                    'seed': info.get('seed'), 'census': census, 'actions': actions})
    return out


def games(inputs: Path, teams):
    jobs = []
    for manifest in sorted(inputs.rglob('manifest.csv')):
        for row in csv.DictReader(manifest.open()):
            path = manifest.parent / f"{row['episode_id']}.json"
            if path.exists():
                jobs.append((str(path), manifest.parent.name, row, teams))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', default='/kaggle/input')
    ap.add_argument('--out', default='/kaggle/working/tapes.jsonl.gz')
    ap.add_argument('--team', action='append', help='a team name (repeatable; default: %s)' % ', '.join(TEAMS))
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    teams = tuple(args.team or TEAMS)
    jobs = games(Path(args.inputs), teams)
    print(f'{len(jobs)} games, teams {teams}', flush=True)
    start, done, seats, errors = time.time(), 0, {}, 0
    with gzip.open(args.out, 'wt') as out, multiprocessing.Pool(args.workers) as pool:
        for rows in pool.imap_unordered(mine, jobs, chunksize=4):
            for r in rows:
                if 'error' in r:
                    errors += 1
                    continue
                seats[r['team']] = seats.get(r['team'], 0) + 1
                out.write(json.dumps(r, separators=(',', ':')) + '\n')
            done += 1
            if done % 500 == 0:
                print(f'{done}/{len(jobs)} games, {time.time() - start:.0f} s, seats {seats}', flush=True)
    print(f'TAPE_MINING_DONE games={done} seats={seats} errors={errors} seconds={time.time() - start:.0f}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
