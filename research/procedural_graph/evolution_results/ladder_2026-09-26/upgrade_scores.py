#!/usr/bin/env python3
"""One-off (2026-09-26): carry run ladder1's cached pool margins over to the gauntlet's per-job cache.

    python upgrade_scores.py RUN_DIR OLD_PLAN NEW_PLAN [--apply]

graph_gauntlet's fingerprint used to cover the plan and its opponent list, so a plan that grew made
every cached margin stale and each island champion replayed all its pool games. The fingerprint now
covers the runtime, the harness, the head-to-head seeds and the engine version, and each cached margin
is checked against the digest of its opponent's bundle. This script recomputes the old fingerprint from
OLD_PLAN and reports which scores files carry it. With --apply it prepares the gauntlet for NEW_PLAN
(building its new opponents) and rewrites those files with the new fingerprint and their opponents'
digests, so the graphs play only NEW_PLAN's new jobs. Other files are left alone (their graphs replay).
Run it only while the loop is stopped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]  # research/procedural_graph
sys.path.insert(0, str(HERE))

from graph_gauntlet import ENGINE_VERSION, ColabPoolExecutor, Gauntlet, _tree_digest, seed_list  # noqa: E402
from pool_upgrade_state import atomic_json  # noqa: E402


def old_fingerprint(run_dir, plan, seeds_per_opponent):
    """graph_gauntlet.Gauntlet.prepare()'s fingerprint before 2026-09-26 19:30 UTC."""
    plan = [dict(e) for e in plan]
    opponents = tuple(sorted({e['tag'] for e in plan}))
    files = [p for name in opponents for p in (run_dir / 'bundles' / name).rglob('*') if p.is_file()]
    files += [p for p in (HERE / 'hazel_runtime').rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    files += [HERE / 'agent_graph.py', run_dir / 'arena.py', run_dir / 'bundle_agent.py']
    return hashlib.sha256(json.dumps({
        'files': _tree_digest(files), 'seeds': seed_list(seeds_per_opponent), 'opponents': opponents,
        'plan': plan, 'engine': ENGINE_VERSION}, sort_keys=True).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run_dir', type=Path)
    ap.add_argument('old_plan', type=Path)
    ap.add_argument('new_plan', type=Path)
    ap.add_argument('--seeds_per_opponent', type=int, default=20)
    ap.add_argument('--pool_dir', type=Path, default=Path.home() / 'kagg-evo' / 'pool')
    ap.add_argument('--apply', action='store_true')
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    old = old_fingerprint(run_dir, json.loads(args.old_plan.read_text()), args.seeds_per_opponent)
    gauntlet = None
    if args.apply:
        gauntlet = Gauntlet(run_dir, ColabPoolExecutor(args.pool_dir), args.seeds_per_opponent,
                            plan=json.loads(args.new_plan.read_text()))
        gauntlet.prepare()
    carried = 0
    for path in sorted((run_dir / 'scores').glob('*.json')):
        record = json.loads(path.read_text())
        if 'opponents' in record:
            status = 'already per job'
        elif record.get('fingerprint') != old:
            status = f"other fingerprint {str(record.get('fingerprint'))[:12]}: its graph replays"
        else:
            carried += 1
            status = f"{len(record['margins'])} margins carried over"
            if gauntlet:
                tags = sorted({key.split('|')[0] for key in record['margins']})
                atomic_json(path, {'fingerprint': gauntlet.fingerprint,
                                   'opponents': {t: gauntlet.opponent_digests[t] for t in tags},
                                   'margins': record['margins']})
        print(f'{path.name}: {status}')
    print(f'old fingerprint {old[:12]}; {carried} files carried over'
          + (f'; new fingerprint {gauntlet.fingerprint[:12]}' if gauntlet else ' (report only, no --apply)'))


if __name__ == '__main__':
    main()
