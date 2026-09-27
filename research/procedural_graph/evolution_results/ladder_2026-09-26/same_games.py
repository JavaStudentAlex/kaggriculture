"""Game by game: do two ladder_validate runs play a graph to the same result?

    same_games.py RUN_DIR NAME:LABEL NAME:LABEL [--expect N]

Compares the games of LABEL in <RUN_DIR>/games/<NAME>.jsonl with those of the second label, by opponent, seed
and seat (both seats' final money). Exits 1 unless every game of the second label has an identical reference
(and, with --expect, there are at least N of them). Used before a runtime change carries the cached margins
over (carry_fingerprint.py): a graph without the new stage, rebuilt with the new runtime, must replay its games.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def games(run_dir: Path, spec: str) -> dict:
    name, label = spec.split(':', 1)
    out = {}
    for line in (run_dir / 'games' / f'{name}.jsonl').read_text().splitlines():
        r = json.loads(line)
        if r.get('errors') or r.get('statuses') != ['DONE', 'DONE']:
            continue
        tag, opponent = r['tag'].split('@', 1)
        if tag == label:
            seat = r['a_seat']
            out[(opponent, r['seed'], seat)] = (r['rewards'][seat], r['rewards'][1 - seat])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir', type=Path)
    ap.add_argument('reference')
    ap.add_argument('check')
    ap.add_argument('--expect', type=int, default=1)
    a = ap.parse_args()
    ref, new = games(a.run_dir, a.reference), games(a.run_dir, a.check)
    same = diff = missing = 0
    for key, val in sorted(new.items()):
        if key not in ref:
            missing += 1
        elif ref[key] == val:
            same += 1
        else:
            diff += 1
            print('DIFF', *key, val, ref[key])
    print(f'{a.check} vs {a.reference}: {same} identical, {diff} different, {missing} without a reference')
    return 0 if diff == missing == 0 and same >= a.expect else 1


if __name__ == '__main__':
    sys.exit(main())
