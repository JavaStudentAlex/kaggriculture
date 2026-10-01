#!/usr/bin/env python3
"""Top-player benchmark, the part that runs in a Kaggle CPU notebook (make_top_bench.py ships it).

    python top_bench.py --shard K --shards N --top T --graphs a,b,c [--workers 4] [--deadline SECONDS]

Kaggle's daily datasets (kaggle/kaggriculture-episodes-<day>, attached to the notebook) hold the games
between the ladder's best players. For the T games of each attached day with the highest average rating
(manifest.csv), this shard's share (index % N == K) becomes replay opponents: `replay_<episode>_s<seat>` plays
that seat's recorded moves, and every graph plays the other seat on the game's seed. So each graph takes
the place of a top player against the recorded moves of the other; the job carries what the replaced player
earned in the real game. Results go to /kaggle/working/results.jsonl (arena.py's lines, one per game).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPLAY_MAIN = '''"""Replay opponent (top_bench.py): plays one seat of a recorded top game, move for move."""
import gzip
import json
from pathlib import Path

_ACTIONS = json.loads(gzip.decompress((Path(__file__).resolve().parent / 'actions.json.gz').read_bytes()))


def agent(observation, configuration=None):
    step = int(observation['step'])
    return (_ACTIONS[step] if 0 <= step < len(_ACTIONS) else None) or {}
'''


def top_games(inputs: Path, top: int):
    """(day directory, manifest row) of the `top` highest-rated games of every attached day."""
    games = []
    for manifest in sorted(inputs.rglob('manifest.csv')):
        rows = sorted(csv.DictReader(manifest.open()), key=lambda r: -float(r['avg_score']))[:top]
        games += [(manifest.parent, row) for row in rows]
    return games


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard', type=int, required=True)
    ap.add_argument('--shards', type=int, required=True)
    ap.add_argument('--top', type=int, required=True)
    ap.add_argument('--graphs', required=True, help='bundle names under bundles/')
    ap.add_argument('--inputs', default='/kaggle/input')
    ap.add_argument('--out', default='/kaggle/working')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--deadline', type=float, default=0.0)
    args = ap.parse_args()
    graphs = args.graphs.split(',')
    out = Path(args.out)
    games = top_games(Path(args.inputs), args.top)
    mine = [g for i, g in enumerate(games) if i % args.shards == args.shard]
    print(f'{len(games)} top games attached, {len(mine)} in this shard', flush=True)
    jobs = []
    for day_dir, row in mine:
        episode = row['episode_id']
        replay = json.loads((day_dir / f'{episode}.json').read_text())
        steps = replay['steps']
        seed = int(replay['info']['seed'])
        rewards = replay.get('rewards') or [s.get('reward') for s in steps[-1]]
        teams = (replay.get('info') or {}).get('TeamNames')
        for seat in (0, 1):
            actions = [steps[t + 1][seat].get('action') for t in range(len(steps) - 1)]
            name = f'replay_{episode}_s{seat}'
            bundle = HERE / 'bundles' / name
            bundle.mkdir(parents=True, exist_ok=True)
            (bundle / 'actions.json.gz').write_bytes(
                gzip.compress(json.dumps(actions, separators=(',', ':')).encode(), mtime=0))
            (bundle / 'main.py').write_text(REPLAY_MAIN)
            ours = 1 - seat
            for graph in graphs:
                jobs.append({'tag': f'{graph}@{name}', 'a': f'bundles/{graph}', 'b': f'bundles/{name}',
                             'seed': seed, 'a_seat': ours, 'graph': graph, 'episode': int(episode),
                             'day': day_dir.name, 'avg_score': float(row['avg_score']),
                             'min_score': float(row['min_score']), 'recorded_rewards': rewards,
                             'replaced_team': teams[ours] if teams else None,
                             'replayed_team': teams[seat] if teams else None})
    (out / 'jobs_top.json').write_text(json.dumps({'jobs': jobs}))
    (HERE / 'jobs_top.json').write_text(json.dumps({'jobs': jobs}))
    print(f'{len(jobs)} games: {len(mine)} top games x 2 seats x {len(graphs)} graphs', flush=True)
    command = [sys.executable, 'arena.py', '--jobs', 'jobs_top.json', '--root', '.', '--workers', str(args.workers),
               '--out', str(out / 'results.jsonl')]
    if args.deadline:
        command += ['--deadline', str(args.deadline)]
    return subprocess.run(command, cwd=HERE).returncode


if __name__ == '__main__':
    sys.exit(main())
