#!/usr/bin/env python3
"""What promotion rules that count game results would have done with run ladder1's candidates (read-only).

    python flip_replay.py RUN_DIR [--verbose]

The ladder rates results, not dollars. For every played candidate (stage_replay.played_candidates) a game's
result is worth 1 for a win, 0.5 for a tie and 0 for a loss; a pool game's change is the candidate's result
minus the incumbent's on the same job, and a head-to-head game's is its result minus 0.5 (a graph against
itself draws). It prints how often results changed and which verdicts each rule changes:
- dollars: the rule in use (graph_gauntlet.compare);
- veto: the same, but never promote a candidate whose results got worse on balance;
- results: the veto, and also promote a candidate whose results improved significantly (sign test over the
  games whose result changed) even when the dollar test fails.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]  # research/procedural_graph
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from arena.report import sign_test  # noqa: E402
from graph_gauntlet import HEAD_TO_HEAD, compare, result_points as points  # noqa: E402
from stage_replay import played_candidates  # noqa: E402


def result_changes(found, base):
    """(games whose result improved, got worse, net points) of a candidate against its incumbent."""
    up = down = 0
    net = 0.0
    for key, margin in found.items():
        if key.startswith(HEAD_TO_HEAD + '|'):
            change = points(margin) - 0.5
        elif key in base['margins']:
            change = points(margin) - points(base['margins'][key])
        else:
            continue
        up += change > 0
        down += change < 0
        net += change
    return up, down, net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir', type=Path)
    ap.add_argument('--alpha', type=float, default=0.05)
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()
    rows = []
    for r, found, base in played_candidates(args.run_dir):
        stats = compare({'margins': found}, base, args.alpha)  # its 'promote' is the "results" rule below
        dollars = bool(stats['wins'] > stats['losses'] and stats['mean_change'] > 0 and stats['p'] <= args.alpha)
        up, down, net = result_changes(found, base)
        veto = dollars and net >= 0
        results = veto or (net > 0 and sign_test(up, down) <= args.alpha)
        rows.append((r, stats, dollars, veto, results, up, down, net, (r.get('verdict') or {}).get('promote')))
    for name, i in (('dollars', 2), ('veto', 3), ('results', 4)):
        print(f'{name:8s}: {sum(row[i] for row in rows)} of {len(rows)} candidates promoted')
    assert all(row[4] == row[1]['promote'] for row in rows), 'compare() no longer applies the results rule'
    changed = [row for row in rows if row[2] != row[4] or row[2] != row[3]]
    print('verdicts that differ:')
    for r, stats, dollars, veto, results, up, down, net, logged in (rows if args.verbose else changed):
        print(f"  {r.get('iteration')!s:>3} {r.get('island', '')[:22]:22s} {r.get('stage'):8s} "
              f"dollars {stats['wins']}W-{stats['losses']}L ${stats['mean_change']:+.0f} p={stats['p']:.2g} | "
              f"results +{up}/-{down} net {net:+.1f} | dollars {'P' if dollars else '-'} veto {'P' if veto else '-'} "
              f"results {'P' if results else '-'} (logged {'P' if logged else '-'}) | "
              f"{'; '.join(r.get('changes') or [])[:70]}")


if __name__ == '__main__':
    main()
