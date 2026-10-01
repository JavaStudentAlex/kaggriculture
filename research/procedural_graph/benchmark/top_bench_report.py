#!/usr/bin/env python3
"""Report of the top-player benchmark (make_top_bench.py): how our graphs do in a top player's place.

    python top_bench_report.py RESULTS.jsonl [RESULTS.jsonl ...] [--json OUT]

Per graph, over the games it played:
- W-L-T and mean margin against the other top player's recorded moves (they do not react to us);
- our cash against the cash the replaced top player earned in the real game from the same seat ("gap": ours
  minus theirs; the recorded moves reacted to that player, not to us, so this is an estimate);
- the same split by the replaced player's rating band;
- per product, the units we submitted to sell and their value at the quote (arena.py's ledger), next to the
  other graphs'.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]


def rows(paths):
    seen = {}
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get('rewards') and not r.get('errors'):
                seen[(r['tag'], r['seed'], r['a_seat'])] = r
    return list(seen.values())


def summary(games):
    ours = [g['rewards'][g['a_seat']] for g in games]
    margins = [g['rewards'][g['a_seat']] - g['rewards'][1 - g['a_seat']] for g in games]
    gaps = [g['rewards'][g['a_seat']] - g['recorded_rewards'][g['a_seat']] for g in games if g.get('recorded_rewards')]
    w = sum(m > 0 for m in margins)
    l = sum(m < 0 for m in margins)
    return {'games': len(games), 'W': w, 'L': l, 'T': len(games) - w - l,
            'mean_margin': round(statistics.mean(margins)) if margins else None,
            'our_cash': round(statistics.mean(ours)) if ours else None,
            'gap_to_top_player': round(statistics.mean(gaps)) if gaps else None,
            'gap_median': round(statistics.median(gaps)) if gaps else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('results', nargs='+')
    ap.add_argument('--json')
    args = ap.parse_args()
    games = rows(args.results)
    by_graph = defaultdict(list)
    for g in games:
        by_graph[g['graph']].append(g)
    report = {}
    for graph, gs in sorted(by_graph.items()):
        entry = {'all': summary(gs)}
        for low, high in ((0, 2900), (2900, 3000), (3000, 9999)):
            band = [g for g in gs if low <= g['min_score'] < high]
            if band:
                entry[f'min rating {low}-{high}'] = summary(band)
        sold = defaultdict(float)
        value = defaultdict(float)
        for g in gs:
            ledger = (g.get('ledger') or [None, None])[g['a_seat']] or {}
            for p in PRODUCTS:
                sold[p] += (ledger.get('submitted_sell_units') or {}).get(p, 0)
                value[p] += (ledger.get('submitted_sell_value_at_quote') or {}).get(p, 0)
        entry['per_game_sell_value'] = {p: round(value[p] / len(gs)) for p in PRODUCTS if value[p]}
        entry['per_game_sell_units'] = {p: round(sold[p] / len(gs), 1) for p in PRODUCTS if sold[p]}
        report[graph] = entry
    for graph, entry in report.items():
        a = entry['all']
        print(f"{graph}: {a['W']}W-{a['L']}L-{a['T']}T of {a['games']}, mean margin ${a['mean_margin']:+,}, "
              f"our cash ${a['our_cash']:,}, vs the replaced top player ${a['gap_to_top_player']:+,} a game "
              f"(median ${a['gap_median']:+,})")
        for band, s in entry.items():
            if band.startswith('min rating'):
                print(f"   {band}: {s['W']}W-{s['L']}L-{s['T']}T of {s['games']}, gap ${s['gap_to_top_player']:+,}")
        print('   sold per game (value at quote): ' + ', '.join(f'{p} ${v:,}' for p, v in
                                                            entry['per_game_sell_value'].items()))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1) + '\n')


if __name__ == '__main__':
    main()
