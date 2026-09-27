#!/usr/bin/env python3
"""What a staged gauntlet would have done with run ladder1's played candidates (read-only).

    python stage_replay.py RUN_DIR [--fraction 0.3 ...] [--verbose]

For every played candidate (candidates.jsonl, stages "gauntlet" and "mixing") it takes the candidate's
margins (scores/<bundle>.json and its head-to-head games) and its incumbent's (scores/<incumbent>.json, the
incumbent named by games/<candidate>_vs_<incumbent>.jsonl), applies Gauntlet.stop_after_first_stage to the
first-stage jobs of each --fraction, and prints how many candidates would have stopped, how many of them were
promoted, and the share of the games that saves.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]  # research/procedural_graph
sys.path.insert(0, str(HERE))

from graph_gauntlet import HEAD_TO_HEAD, Gauntlet, compare, first_stage, margins  # noqa: E402


def played_candidates(run):
    """(record, candidate margins, incumbent baseline, the jobs both have) for every played candidate."""
    for line in (run / 'candidates.jsonl').read_text().splitlines():
        r = json.loads(line) if line.strip() else {}
        if r.get('stage') not in ('gauntlet', 'mixing') or not r.get('bundle'):
            continue
        cand = r['bundle']
        games = sorted((run / 'games').glob(f'{cand}_vs_*.jsonl'))
        scores = run / 'scores' / f'{cand}.json'
        if not games or not scores.exists():
            continue
        inc_scores = run / 'scores' / f"{games[-1].stem.split('_vs_')[1]}.json"
        if not inc_scores.exists():
            continue
        found = dict(json.loads(scores.read_text())['margins'])
        h2h, _ = margins([json.loads(x) for x in games[-1].read_text().splitlines() if x.strip()], cand)
        found.update({k: v for k, v in h2h.items() if k.startswith(HEAD_TO_HEAD + '|')})
        base = {'margins': json.loads(inc_scores.read_text())['margins']}
        keys = [k for k in found if k.startswith(HEAD_TO_HEAD + '|') or k in base['margins']]
        yield r, {k: found[k] for k in keys}, base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir', type=Path)
    ap.add_argument('--fraction', type=float, action='append')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()
    stopper = Gauntlet.__new__(Gauntlet)
    stopper.alpha = 0.05
    candidates = list(played_candidates(args.run_dir))
    for fraction in args.fraction or [0.3]:
        total = saved = stops = promoted_stops = 0
        for r, found, base in candidates:
            first = {k: v for k, v in found.items() if first_stage(k, fraction)}
            stop = stopper.stop_after_first_stage(first, base)
            promoted = bool((r.get('verdict') or {}).get('promote'))
            total += len(found)
            if stop:
                stops += 1
                promoted_stops += promoted
                saved += len(found) - len(first)
            if args.verbose:
                full = compare({'margins': found}, base)
                print(f"{r.get('iteration'):>3} {r.get('island', '')[:18]:18s} "
                      f"{'PROMOTED' if promoted else 'rejected':8s} full {full['wins']}W-{full['losses']}L "
                      f"${full['mean_change']:+.0f} | first stage {len(first)} games: "
                      + (f"STOP ({stop['reason']}: {stop['wins']}W-{stop['losses']}L)" if stop else 'plays on')
                      + f" | {'; '.join(r.get('changes') or [])[:60]}")
        print(f'fraction {fraction}: {stops} of {len(candidates)} candidates stopped, {promoted_stops} of them '
              f'promoted; {saved / max(total, 1):.0%} of the games saved')


if __name__ == '__main__':
    main()
