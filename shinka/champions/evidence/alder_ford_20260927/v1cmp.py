"""validate1 per opponent, with the validation seeds and the held-out lost seeds apart.

usage: v1cmp.py LABEL1 LABEL2 [LABEL3 ...]   (labels of ladder_validate.py --graph LABEL=...)

A markdown table: each graph's W-L-T, then the last graph's change a game against each earlier one
(better / worse games, exact two-sided sign test). Run it on cliproxyapi, where the games are.
"""
import collections
import json
import math
import sys

R = '/home/alex/kagg-evo/runs/ladder1'
labels = sys.argv[1:]
key = lambda j: f"{j['tag']}|{j['seed']}|{j['a_seat']}"
jobs = {key(j): j for j in json.load(open(f'{R}/jobs/validate1.json'))['jobs']}
m = collections.defaultdict(dict)
for line in open(f'{R}/games/validate1.jsonl'):
    r = json.loads(line)
    k = key(r)
    if k not in jobs or r.get('errors') or r.get('statuses') != ['DONE', 'DONE']:
        continue
    label, opp = r['tag'].split('@', 1)
    s = r['a_seat']
    m[label][(jobs[k].get('set') or 'validation', opp, r['seed'], s)] = r['rewards'][s] - r['rewards'][1 - s]


def sign_p(b, w):
    n = b + w
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, w) + 1)) / 2 ** n) if n else 1.0


def rec(v):
    return f"{sum(x > 0 for x in v)}-{sum(x < 0 for x in v)}-{sum(x == 0 for x in v)}"


rows = collections.defaultdict(lambda: [[] for _ in labels])
for k in m[labels[0]]:
    if all(k in m[l] for l in labels):
        for g in (f"{k[0]}: {k[1]}", f"{k[0]}: all", 'all'):
            for i, l in enumerate(labels):
                rows[g][i].append(m[l][k])
last = labels[-1]
print('| games | n | ' + ' | '.join(labels) + ' | ' + ' | '.join(f"{last} vs {l}" for l in labels[:-1]) + ' |')
print('|---' * (2 + 2 * len(labels) - 1) + '|')
for g in sorted(rows, key=lambda g: (g == 'all', g.endswith(': all'), g)):
    v = rows[g]
    cells = []
    for i in range(len(labels) - 1):
        d = [y - x for x, y in zip(v[i], v[-1])]
        b, w = sum(x > 0 for x in d), sum(x < 0 for x in d)
        cells.append(f"{sum(d) / len(d):+.0f} ({b} / {w}, p {sign_p(b, w):.2g})")
    print(f"| {g} | {len(v[0])} | " + ' | '.join(rec(x) for x in v) + ' | ' + ' | '.join(cells) + ' |')
