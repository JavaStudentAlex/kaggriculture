#!/usr/bin/env python3
"""Compare the top players' plans with our route tapes, world by world, and propose tape changes (cliproxyapi).

    python benchmark/plan_report.py --plans plans.jsonl [--plans ...] --routes route_plans.json
        [--engine tetsutani_demand] [--out report.md] [--candidates candidates.json]

plans.jsonl comes from plan_mining.py (a Kaggle notebook over the daily top-game datasets), route_plans.json from
route_plans.py. A plan is compared as its purchases summed at the start of days DAYS (animals by kind, seeds by crop,
hires, land; log1p units, mean absolute difference): every top seat gets its nearest tape. Per world (the first two
shops) the report shows how many top seats play close to some tape at all, which tape our engine plays there, and
for each tape the seats nearest to it: their number, wins and mean margin. A world is a candidate when the seats
nearest to another tape won clearly more than those nearest to ours (at least --min-seats of them, within
--max-distance of that tape): the queue can then test that tape there through the engine's `_V92_TABLE`, which
overrides the route of any first-two-shop pair. It is evidence from other players' games, not from ours: the
gauntlet decides.

With --ours (plan_mining.py run over our own ladder replays) and --team, it also compares the farm census (hands,
land, animals, planted tiles, money at the start of a day) of the top players' winners and losers with our games',
overall and per world.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

DAYS = ('6', '12', '20', '29')
KEYS = ('ANIMAL:COW', 'ANIMAL:GOOSE', 'ANIMAL:SHEEP', 'SEED:WHEAT', 'SEED:CARROT', 'SEED:TOMATO', 'SEED:STRAWBERRY',
        'SEED:MELON', 'HIRE', 'LAND')
SHORT = {'ANIMAL:COW': 'cow', 'ANIMAL:GOOSE': 'goose', 'ANIMAL:SHEEP': 'sheep', 'HIRE': 'hire', 'LAND': 'land'}


def vector(buys):
    return [math.log1p(max(0, (buys.get(d) or {}).get(k, 0))) for d in DAYS for k in KEYS]


def distance(a, b):
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)


def margin(row):
    return float(row['reward']) - float(row['rival_reward'])


def profile(buys, day='29'):
    b = buys.get(day) or {}
    return ' '.join(f"{SHORT[k]} {b.get(k, 0):g}" for k in SHORT)


def mean_profile(rows, day='29'):
    if not rows:
        return '-'
    return ' '.join(f"{SHORT[k]} {statistics.mean((r['buys'].get(day) or {}).get(k, 0) for r in rows):.1f}"
                    for k in SHORT)


CENSUS = (('money', lambda c: c.get('money') or 0), ('hands', lambda c: c.get('hands', 0)),
          ('land', lambda c: c.get('land', 0))) + tuple(
    (a.lower(), (lambda a: lambda c: (c.get('animals') or {}).get(a, 0))(a)) for a in ('COW', 'GOOSE', 'SHEEP')) + tuple(
    (k.lower(), (lambda k: lambda c: (c.get('crops') or {}).get(k, 0))(k))
    for k in ('WHEAT', 'CARROT', 'TOMATO', 'STRAWBERRY', 'MELON'))


CENSUS_DAYS = (3, 6, 9, 12, 15, 20, 25, 29)   # plan_mining.py's


def hires_per_day(row, day):
    """HIRE orders a day since the census day before `day` (hands are hired by the day)."""
    before = max((d for d in CENSUS_DAYS if d < int(day)), default=0)
    now = (row['buys'].get(day) or {}).get('HIRE', 0)
    then = (row['buys'].get(str(before)) or {}).get('HIRE', 0) if before else 0
    return (now - then) / max(1, int(day) - before)


def census_row(label, rows, day):
    rows = [r for r in rows if day in (r.get('census') or {})]
    if not rows:
        return f'| {label} | 0 |' + ' |' * (len(CENSUS) + 1)
    cs = [r['census'][day] for r in rows]
    cells = [f'{statistics.mean(f(c) for c in cs):,.0f}' if name == 'money' else f'{statistics.mean(f(c) for c in cs):.1f}'
             for name, f in CENSUS]
    cells.append(f'{statistics.mean(hires_per_day(r, day) for r in rows):.1f}')
    return f'| {label} | {len(cs)} | ' + ' | '.join(cells) + ' |'


def census_section(wins, losses, ours, by_world, days=('9', '15', '20', '25')):
    head = '| seats | n | ' + ' | '.join(name for name, _ in CENSUS) + ' | hires a day |'
    rule = '|' + '---|' * (len(CENSUS) + 3)
    lines = ['', '## Farm census at the start of a day (mean per seat)', '',
             f"Top players' winners and losers, and our {len(ours)} games (our own ladder replays). Animals and "
             'planted tiles are counts on the farm; hands are hired by the day, so "hires a day" counts the HIRE '
             'orders a day since the census day before (a census at hour 0 shows no hands).']
    for day in days:
        lines += ['', f'Day {day}:', '', head, rule, census_row('top winners', wins, day),
                  census_row('top losers', losses, day), census_row('ours', ours, day)]
    ours_by_world = defaultdict(list)
    for r in ours:
        if len(r.get('world') or []) == 2:
            ours_by_world['|'.join(r['world'])].append(r)
    common = sorted(ours_by_world, key=lambda w: -len(ours_by_world[w]))[:10]
    lines += ['', "Day 15 in the worlds of most of our games (top winners of that world against ours):", '', 
              '| world | seats | n | ' + ' | '.join(name for name, _ in CENSUS) + ' | hires a day |',
              '|' + '---|' * (len(CENSUS) + 4)]
    for world in common:
        top_w = [r for r in by_world.get(world, []) if r['result'] == 'W']
        lines.append(census_row(f'{world} | top winners', top_w, '15'))
        lines.append(census_row(f'{world} | ours', ours_by_world[world], '15'))
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plans', action='append', required=True)
    ap.add_argument('--routes', required=True)
    ap.add_argument('--engine', default='tetsutani_demand')
    ap.add_argument('--max-distance', type=float, default=0.3, help='a seat plays like a tape within this')
    ap.add_argument('--min-seats', type=int, default=10)
    ap.add_argument('--min-gain', type=float, default=0.15, help='win-rate lead over the seats near our tape')
    ap.add_argument('--ours', action='append', default=[], help="plan_mining.py lines of our own games")
    ap.add_argument('--team', default='Sunshine through fog', help='our team name in --ours')
    ap.add_argument('--out', default=None)
    ap.add_argument('--candidates', default=None)
    args = ap.parse_args()
    rows = [json.loads(line) for p in args.plans for line in open(p) if line.strip()]
    errors = [r for r in rows if 'error' in r]
    rows = [r for r in rows if 'error' not in r and len(r.get('world') or []) == 2]
    engine = json.loads(Path(args.routes).read_text())[args.engine]
    tapes = {int(k): vector(v['buys']) for k, v in engine['routes'].items()}
    ours_of = {k: int(v) for k, v in engine['world_route'].items()}
    for r in rows:
        v = vector(r['buys'])
        dists = {t: distance(v, tv) for t, tv in tapes.items()}
        r['near'] = min(dists, key=dists.get)
        r['d_near'] = dists[r['near']]
        r['ours'] = ours_of.get('|'.join(r['world']))
        r['d_ours'] = dists.get(r['ours'], float('inf'))
    lines = [f"# Top players' plans against our route tapes ({args.engine})", '',
             f"{len(rows)} seats of {len({r['episode'] for r in rows})} games, days "
             f"{', '.join(sorted({r['day'] for r in rows}))}; {len(errors)} unreadable. Distance: mean absolute "
             f"difference of log1p purchases at the start of days {', '.join(DAYS)} "
             f"({', '.join(KEYS)}); a seat plays like a tape within {args.max_distance}.", '']
    close = [r for r in rows if r['d_near'] <= args.max_distance]
    lines.append(f"Seats within {args.max_distance} of some tape: {len(close)} of {len(rows)} "
                 f"({100 * len(close) / max(1, len(rows)):.0f}%); median distance to the nearest tape "
                 f"{statistics.median(r['d_near'] for r in rows):.3f}, to our tape of their world "
                 f"{statistics.median(r['d_ours'] for r in rows):.3f}.")
    wins = [r for r in rows if r['result'] == 'W']
    losses = [r for r in rows if r['result'] == 'L']
    lines += ['', 'Purchases by the end (day 29 start), mean per seat: winners ' + mean_profile(wins) +
              '; losers ' + mean_profile(losses) + '.', '',
              '| world | seats | our tape (its buys by day 29) | near our tape: seats, W, mean margin | '
              'best other tape: seats, W, mean margin | winners buy (day 29) |', '|---|---|---|---|---|---|']
    by_world = defaultdict(list)
    for r in rows:
        by_world['|'.join(r['world'])].append(r)
    candidates = {}
    for world, rs in sorted(by_world.items(), key=lambda kv: -len(kv[1])):
        ours = ours_of.get(world)
        groups = defaultdict(list)
        for r in rs:
            if r['d_near'] <= args.max_distance:
                groups[r['near']].append(r)

        def stats(g):
            if not g:
                return 0, 0.0, 0.0
            return len(g), sum(x['result'] == 'W' for x in g) / len(g), statistics.mean(margin(x) for x in g)
        n_ours, w_ours, m_ours = stats(groups.get(ours, []))
        others = [(t, stats(g)) for t, g in groups.items() if t != ours and len(g) >= args.min_seats]
        best = max(others, key=lambda x: (x[1][1], x[1][2]), default=None)
        ours_plan = profile(engine['routes'][str(ours)]['buys']) if ours is not None else '-'
        best_text = (f"tape {best[0]}: {best[1][0]}, {100 * best[1][1]:.0f}%, {best[1][2]:+,.0f}" if best else '-')
        lines.append(f"| {world} | {len(rs)} | {ours}: {ours_plan} | {n_ours}, {100 * w_ours:.0f}%, {m_ours:+,.0f} | "
                     f"{best_text} | {mean_profile([r for r in rs if r['result'] == 'W'])} |")
        if best and best[1][1] >= w_ours + args.min_gain and best[1][2] > m_ours and best[1][1] > 0.5:
            candidates[world] = {'tape': best[0], 'ours': ours, 'seats': best[1][0], 'win_rate': best[1][1],
                                 'mean_margin': best[1][2], 'ours_seats': n_ours, 'ours_win_rate': w_ours,
                                 'ours_mean_margin': m_ours}
    if args.ours:
        ours_rows = [json.loads(line) for p in args.ours for line in open(p) if line.strip()]
        ours_rows = [r for r in ours_rows if r.get('team') == args.team and 'error' not in r]
        lines += census_section(wins, losses, ours_rows, by_world)
    table = dict(engine.get('v92_table') or {})
    table.update({w: c['tape'] for w, c in candidates.items()})
    lines += ['', f"Candidates (the seats near another tape won at least {100 * args.min_gain:.0f} points more often "
              f"than those near ours, at least {args.min_seats} seats): {len(candidates)}"]
    lines += [f"- {w}: tape {c['tape']} instead of {c['ours']} ({c['seats']} seats, {100 * c['win_rate']:.0f}% won, "
              f"{c['mean_margin']:+,.0f} a game; near ours {c['ours_seats']} seats, {100 * c['ours_win_rate']:.0f}%, "
              f"{c['ours_mean_margin']:+,.0f})" for w, c in candidates.items()]
    if candidates:
        lines += ['', 'Queue edit (the complete table): ' + json.dumps({'engine_parameters': {'_V92_TABLE': table}})]
    text = '\n'.join(lines) + '\n'
    if args.out:
        Path(args.out).write_text(text)
    print(text)
    if args.candidates:
        Path(args.candidates).write_text(json.dumps({'engine': args.engine, 'candidates': candidates,
                                                     'v92_table': table}, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
