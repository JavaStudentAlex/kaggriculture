#!/usr/bin/env python3
"""Summarize arena results (results_*.jsonl under the given dirs/files).

    python research/procedural_graph/arena/report.py runs/arena/ID/results [--traces]

Per tag: games, W-L-T of bundle a, win% (ties count half), mean cash margin a-b with
its standard error, exact two-sided sign-test p over decided games, mean cash, games
with errors (never counted as results) and seconds per game. --traces adds, from the
saved traces, the median first step where a's action differs from b's and the mean
cash gap at the end of selected days.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path


def load(paths):
    rows = {}
    for path in map(Path, paths):
        for f in [path] if path.is_file() else sorted(path.rglob('results_*.jsonl')):
            for line in f.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    rows[(r['tag'], r['seed'], r['a_seat'])] = r
    return list(rows.values())


def sign_test(wins, losses):
    n = wins + losses
    if not n:
        return 1.0
    pmf = [math.comb(n, i) / 2 ** n for i in range(n + 1)]
    return min(1.0, sum(p for p in pmf if p <= pmf[wins] + 1e-15))


def summary(rows):
    games, errors = defaultdict(list), defaultdict(int)
    for r in rows:
        rewards = r.get('rewards')
        if r.get('errors') or not rewards or None in rewards or r.get('statuses') != ['DONE', 'DONE']:
            errors[r['tag']] += 1
            continue
        a, b = rewards[r['a_seat']], rewards[1 - r['a_seat']]
        games[r['tag']].append((a - b, a, b, r.get('total_s', 0)))
    print(f"{'tag':22s} {'n':>4s} {'W-L-T':>9s} {'win%':>6s} {'margin':>8s} {'SE':>6s} {'p':>7s} "
          f"{'cash a':>8s} {'cash b':>8s} {'err':>4s} {'s/game':>6s}")
    for tag in sorted(set(games) | set(errors)):
        g = games.get(tag, [])
        n = len(g)
        w = sum(m > 0 for m, *_ in g)
        l = sum(m < 0 for m, *_ in g)
        mean = sum(m for m, *_ in g) / n if n else 0.0
        se = (math.sqrt(sum((m - mean) ** 2 for m, *_ in g) / (n - 1)) / math.sqrt(n)) if n > 1 else 0.0
        avg = [sum(x[k] for x in g) / n if n else 0.0 for k in (1, 2, 3)]
        print(f"{tag:22s} {n:4d} {f'{w}-{l}-{n - w - l}':>9s} {100 * (w + 0.5 * (n - w - l)) / max(1, n):5.1f}% "
              f"{mean:+8.0f} {se:6.0f} {sign_test(w, l):7.2g} {avg[0]:8.0f} {avg[1]:8.0f} "
              f"{errors.get(tag, 0):4d} {avg[2]:6.0f}")


def divergence(paths):
    by = defaultdict(list)
    for path in map(Path, paths):
        for f in sorted(path.rglob('*.json.gz')):
            tag = f.name.split('_seed')[0]
            seat = int(f.name.split('_aseat')[1].split('.')[0])
            trace = json.load(gzip.open(f))
            first = next((t for t, row in enumerate(trace)
                          if row['seats'][seat]['action'] != row['seats'][1 - seat]['action']), None)
            gaps = [trace[min(len(trace) - 1, d * 24 + 23)]['money'] for d in range(30)]
            by[tag].append((first, [g[seat] - g[1 - seat] for g in gaps]))
    days = (5, 10, 15, 20, 25, 29)
    print(f"\n{'tag':22s} {'first diff':>10s}  mean cash gap a-b at end of day " + ' '.join(f'd{d:<6d}' for d in days))
    for tag, rs in sorted(by.items()):
        firsts = sorted(f for f, _ in rs if f is not None)
        median = firsts[len(firsts) // 2] if firsts else None
        gap = [sum(g[d] for _, g in rs) / len(rs) for d in days]
        print(f"{tag:22s} {str(median):>10s}  {'':32s}" + ' '.join(f'{x:+7.0f}' for x in gap))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('paths', nargs='+')
    ap.add_argument('--traces', action='store_true')
    a = ap.parse_args()
    summary(load(a.paths))
    if a.traces:
        divergence(a.paths)


if __name__ == '__main__':
    main()
