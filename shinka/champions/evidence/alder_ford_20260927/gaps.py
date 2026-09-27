"""Where Alder Ford's lost and tied games lost money: per-category gaps (ours - rival) from resim stats.

usage: python gaps.py [stats.json]   (stats.json: resim_games.py output; default: the one next to this file)
"""
import json, collections, sys
from pathlib import Path
G = json.load(open(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent / 'stats.json'))
P = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]

def cats(side):
    c = collections.Counter()
    for p in P:
        c['sell:' + p] += side['sold'].get(p, [0, 0])[1]
    for k, v in side['spend'].items():
        kind, _, what = k.partition(':')
        if kind == 'product':
            c['sell:' + what] -= v          # buying a product to resell: net it against that product's sales
        elif kind == 'animal':
            c['animals'] -= v
        elif kind == 'seed':
            c['seeds'] -= v
        else:
            c[kind] -= v                    # hire, land, feed, ...
    c['shed_end'] += side.get('shed_end_value', 0)
    return c

def herd(side):
    farm = side.get('farm') or {}
    last = farm[max(farm, key=int)] if farm else {}
    a = last.get('animals', {}) if isinstance(last, dict) else {}
    return '/'.join(f"{a.get(k, 0)}{k[0]}" for k in ('COW', 'SHEEP', 'GOOSE'))

rows = []
tot = {'close': collections.Counter(), 'large': collections.Counter()}
for g in G:
    me, op = g['me'], g['opp']
    d = cats(me); d.subtract(cats(op))
    final = me['money_by_day'][-1] - op['money_by_day'][-1]
    gap = [a - b for a, b in zip(me['money_by_day'], op['money_by_day'])]
    kind = 'close' if abs(final) <= 600 else 'large'
    tot[kind].update(d)
    top = sorted(((k, v) for k, v in d.items() if abs(v) >= 1), key=lambda kv: -abs(kv[1]))[:4]
    op0 = ' '.join(f"{o[0][:4]}{o[2] if len(o) > 2 else ''}" for o in (op.get('opening') or [])[:2] if o)
    rows.append((final, g['id'], g['res'], (g['team'] or '')[:16], g.get('orate'), gap, herd(me), herd(op), top, op0))
rows.sort()
print('episode    res rival            rate   final   d10   d15   d20   d25 | herd us  them   | rival step0  | biggest gaps (ours - theirs)')
for final, i, res, team, rate, gap, h1, h2, top, op0 in rows:
    dd = lambda k: f"{gap[k]:+6.0f}" if k < len(gap) else '     -'
    print(f"{i} {res}   {team:16s} {str(rate or '')[:4]:4s} {final:+7.0f} {dd(10)} {dd(15)} {dd(20)} {dd(25)} | {h1:8s} {h2:8s} | {op0:12s} | "
          + ', '.join(f"{k.replace('sell:', '')} {v:+.0f}" for k, v in top))
for kind in ('close', 'large'):
    n = sum(1 for r in rows if (abs(r[0]) <= 600) == (kind == 'close'))
    print(f"\n{kind} ({n} games): summed gaps by category:", ', '.join(f"{k.replace('sell:', '')} {v:+.0f}" for k, v in sorted(tot[kind].items(), key=lambda kv: kv[1]) if abs(v) >= 100))
