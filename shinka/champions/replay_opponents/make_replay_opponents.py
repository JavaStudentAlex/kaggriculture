#!/usr/bin/env python3
"""Replay opponents: one seat of a recorded ladder game, played back move for move.

    python shinka/champions/replay_opponents/make_replay_opponents.py INDEX REPLAY_DIR [--results L,T]

INDEX is a loss-audit index (fetch_games.py: fields id, seat = our seat, res, team, osub); replays are
REPLAY_DIR/episode-<id>-replay.json. For each game whose result is in --results, the rival's seat
becomes the bundle replay_<id>/ next to this file:

    main.py              returns, at observation step t, the action the rival took from that observation
                         (stored at steps[t+1] of the replay)
    actions.json.gz      those 719 actions, exactly as recorded
    SOURCE.json          episode, seats, rival, seed, recorded rewards, sha256 of actions.json.gz

The bundle does not react to the other seat. Played against the agent that played our seat, from the
same seed and seat, it reproduces the ladder game (`payload.py` opponents: the name replay_<id>); against
a changed agent it keeps the rival's moves, which approximates the rival while the change is small (the
engine refuses an order the rival can no longer pay for or fill).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

MAIN = '''"""Replay opponent built by make_replay_opponents.py: plays one seat of a recorded ladder game.

At observation step t it returns the action the recorded player took from that observation. It does not
react to the other seat; SOURCE.json names the game.
"""
import gzip
import json
from pathlib import Path

_ACTIONS = json.loads(gzip.decompress((Path(__file__).resolve().parent / 'actions.json.gz').read_bytes()))


def agent(observation, configuration=None):
    step = int(observation['step'])
    return (_ACTIONS[step] if 0 <= step < len(_ACTIONS) else None) or {}
'''


def build(game, replay, out):
    steps = replay['steps']
    seat = 1 - int(game['seat'])
    actions = [steps[t + 1][seat].get('action') for t in range(len(steps) - 1)]
    data = gzip.compress(json.dumps(actions, separators=(',', ':')).encode(), mtime=0)
    dst = out / f"replay_{game['id']}"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / 'actions.json.gz').write_bytes(data)
    (dst / 'main.py').write_text(MAIN)
    record = {'name': dst.name, 'episode': game['id'], 'seat': seat, 'our_seat': int(game['seat']),
              'opponent': game.get('team'), 'opponent_submission': game.get('osub'),
              'seed': int(replay['info']['seed']), 'recorded_rewards': replay.get('rewards'),
              'recorded_margin_for_us': game.get('diff'), 'steps': len(actions), 'entry': 'main.py',
              'files': {'actions.json.gz': hashlib.sha256(data).hexdigest()}}
    (dst / 'SOURCE.json').write_text(json.dumps(record, indent=1, ensure_ascii=False) + '\n')
    return record


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('index', type=Path)
    ap.add_argument('replays', type=Path)
    ap.add_argument('--results', default='L,T', help='results to build (index field res)')
    ap.add_argument('--out', type=Path, default=HERE)
    args = ap.parse_args()
    wanted = set(args.results.split(','))
    for game in json.loads(args.index.read_text()):
        if game.get('res') not in wanted:
            continue
        replay = json.loads((args.replays / f"episode-{game['id']}-replay.json").read_text())
        record = build(game, replay, args.out)
        print(f"{record['name']}: seat {record['seat']} of seed {record['seed']}, {game.get('team')}, "
              f"recorded margin for us {record['recorded_margin_for_us']}")


if __name__ == '__main__':
    main()
