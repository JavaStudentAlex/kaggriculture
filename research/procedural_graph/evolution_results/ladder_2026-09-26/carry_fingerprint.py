#!/usr/bin/env python3
"""One-off (2026-09-27): keep run ladder1's cached pool margins across the rival_counter runtime change.

    python carry_fingerprint.py RUN_DIR PLAN --old <fingerprint> [--apply]

The gauntlet fingerprint hashes every file under hazel_runtime/, so adding rival_model.py, the public
engine's 09-27 version and the rival_counter stage makes every cached margin stale. A graph without the
rival_counter node runs exactly the code it ran before (checked by replaying cached games before the
restart), so its margins stay valid. With --apply this prepares the gauntlet for PLAN with the code on
disk (building new opponents) and rewrites the scores files that carry --old with the new fingerprint.
Run it only while the loop is stopped.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]  # research/procedural_graph
sys.path.insert(0, str(HERE))

from graph_gauntlet import ColabPoolExecutor, Gauntlet  # noqa: E402
from pool_upgrade_state import atomic_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir', type=Path)
    ap.add_argument('plan', type=Path)
    ap.add_argument('--old', required=True, help='the fingerprint the cached margins carry now')
    ap.add_argument('--seeds_per_opponent', type=int, default=20)
    ap.add_argument('--pool_dir', type=Path, default=Path.home() / 'kagg-evo' / 'pool')
    ap.add_argument('--apply', action='store_true')
    args = ap.parse_args()
    run_dir = args.run_dir.resolve()
    gauntlet = Gauntlet(run_dir, ColabPoolExecutor(args.pool_dir), args.seeds_per_opponent,
                        plan=json.loads(args.plan.read_text()))
    new = gauntlet.prepare()
    carried = other = 0
    for path in sorted((run_dir / 'scores').glob('*.json')):
        record = json.loads(path.read_text())
        if record.get('fingerprint') != args.old:
            other += 1
            continue
        carried += 1
        if args.apply:
            record['fingerprint'] = new
            atomic_json(path, record)
    print(f'old {args.old[:12]} -> new {new[:12]}: {carried} scores files carried over, {other} with another '
          f'fingerprint left alone' + ('' if args.apply else ' (report only, no --apply)'))


if __name__ == '__main__':
    main()
