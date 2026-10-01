#!/usr/bin/env python3
"""Carry the seed's cached games into the land run, after checking that the runtime with the land_plot stage
replays them (2026-09-29).

    python inert_land.py RUN_DIR OLD_RUN_DIR SEED_GRAPH PLAN --old_bundle g_... [--check 24] [--pool_dir DIR]

The gauntlet fingerprint hashes every runtime file and a bundle's name hashes its files, so the new runtime gives
the seed graph a new bundle and a new fingerprint. A graph without the land_plot node runs the code it ran
before. This copies the old run's frozen opponent bundles, plays --check of the seed's cached jobs (one per
opponent, both seats among them) with the new bundle through the Colab pool, requires every margin to equal the
cached one, and only then writes the cached margins under the new bundle name and fingerprint.
Prints INERT_LAND_EXIT=0 on success.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]  # research/procedural_graph
sys.path.insert(0, str(HERE))

from graph_gauntlet import ColabPoolExecutor, Gauntlet, job_key, margins  # noqa: E402
from pool_upgrade_state import atomic_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir', type=Path)
    ap.add_argument('old_run', type=Path)
    ap.add_argument('seed_graph', type=Path)
    ap.add_argument('plan', type=Path)
    ap.add_argument('--old_bundle', required=True)
    ap.add_argument('--check', type=int, default=24)
    ap.add_argument('--pool_dir', type=Path, default=Path.home() / 'kagg-evo' / 'pool')
    a = ap.parse_args()
    run = a.run_dir.resolve()
    (run / 'bundles').mkdir(parents=True, exist_ok=True)
    for src in sorted((a.old_run / 'bundles').iterdir()):   # the frozen opponents, not the old run's graphs
        if src.is_dir() and not src.name.startswith(('g_', '.')) and not (run / 'bundles' / src.name).exists():
            shutil.copytree(src, run / 'bundles' / src.name, symlinks=True)
    old = json.loads((a.old_run / 'scores' / f'{a.old_bundle}.json').read_text())
    g = Gauntlet(run, ColabPoolExecutor(a.pool_dir), 20, plan=json.loads(a.plan.read_text()))
    fingerprint = g.prepare()
    bundle = g.bundle(json.loads(a.seed_graph.read_text()))
    valid = {k for k, v in old['opponents'].items() if g.opponent_digests.get(k) == v}
    jobs = [j for j in g.jobs(bundle) if job_key(j) in old['margins'] and j['tag'] in valid]
    random.Random(20260929).shuffle(jobs)
    picked, tags = [], set()
    for j in jobs:
        if j['tag'] not in tags:
            picked.append(j)
            tags.add(j['tag'])
        if len(picked) >= a.check:
            break
    print(f'new fingerprint {fingerprint[:12]}, seed bundle {bundle} (was {a.old_bundle}); {len(valid)} of '
          f"{len(old['opponents'])} opponents unchanged; checking {len(picked)} games", flush=True)
    rows = g._play(f'inert_{bundle}', picked, sorted({bundle, *tags}))
    found, errors = margins(rows, bundle)
    diffs = [(job_key(j), old['margins'][job_key(j)], found.get(job_key(j))) for j in picked
             if found.get(job_key(j)) != old['margins'][job_key(j)]]
    for d in diffs[:10]:
        print('  differs:', d)
    print(f'{len(picked) - len(diffs)} of {len(picked)} games equal, {len(errors)} errors', flush=True)
    if errors or diffs:
        print('INERT_LAND_EXIT=1')
        return 1
    atomic_json(run / 'scores' / f'{bundle}.json',
                {'fingerprint': fingerprint, 'opponents': old['opponents'], 'margins': old['margins']})
    print(f"carried {len(old['margins'])} margins to scores/{bundle}.json")
    print('INERT_LAND_EXIT=0')
    return 0


if __name__ == '__main__':
    sys.exit(main())
