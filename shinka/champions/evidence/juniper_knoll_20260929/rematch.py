"""Paired comparison of two gauntlet score caches (<run>/scores/<bundle>.json: margins by 'opponent|seed|seat'):
the games both played, how many got better or worse, the mean change, the sign test and the results that changed.

    python rematch.py <new scores.json> <old scores.json>

Used for the KAD run's seed (Juniper Knoll + KAD advice with both levers off + the 09-28 refit, calibrated) against
Juniper's graph with the 09-13 predictor (the land run's seed, g_d46eaa6322231c85): with the levers off KAD changes no
action, so this is the refit's rematch on 763 games.
"""
import json
import math
import sys


def result(m):
    return 1.0 if m > 0 else 0.5 if m == 0 else 0.0


new, old = (json.load(open(p))['margins'] for p in sys.argv[1:3])
keys = sorted(set(new) & set(old))
d = [new[k] - old[k] for k in keys]
better, worse = sum(x > 0 for x in d), sum(x < 0 for x in d)
n = better + worse
p = sum(math.comb(n, i) for i in range(0, min(better, worse) + 1)) / 2 ** n * 2 if n else 1.0
up = sum(result(new[k]) > result(old[k]) for k in keys)
down = sum(result(new[k]) < result(old[k]) for k in keys)
wins = lambda s: sum(s[k] > 0 for k in keys)
print(f'{len(keys)} common games (new {len(new)}, old {len(old)}): {better} better, {worse} worse, '
      f'{len(keys) - n} same; mean change ${sum(d) / max(len(d), 1):+,.1f} a game; sign test p={min(p, 1.0):.3g}')
print(f'results: +{up} / -{down}; wins {wins(old)} -> {wins(new)}')
by = {}
for k, x in zip(keys, d):
    o = k.split('|')[0]
    b = by.setdefault(o, [0, 0, 0, 0.0])
    b[0] += x > 0
    b[1] += x < 0
    b[2] += 1
    b[3] += x
print('by opponent (better/worse of games, mean change), largest totals first:')
for o, (b, w, c, s) in sorted(by.items(), key=lambda kv: -abs(kv[1][3]))[:15]:
    print(f'  {o[:40]:40s} {b:3d}/{w:3d} of {c:3d}  {s / c:+8,.0f}')
