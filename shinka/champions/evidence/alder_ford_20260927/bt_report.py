"""Replay backtest report: each graph against the champion on the same recorded games.

usage: python bt_report.py <run name> [base label] [families.json]
Per graph: record on the replays, result flips vs the base (L->W etc.), mean change, exact sign test,
and the mean change per rival family (families.json from families.py).
"""
import collections, json, math, sys
from pathlib import Path
run = Path('/home/alex/kagg-evo/runs/ladder1')
name = sys.argv[1]
base = sys.argv[2] if len(sys.argv) > 2 else 'champion'
fam = json.load(open(sys.argv[3])) if len(sys.argv) > 3 else {}
short = lambda f: f.split(' (')[0]

m = collections.defaultdict(dict)
for line in open(run / 'games' / f'{name}.jsonl'):
    r = json.loads(line)
    rw = r.get('rewards')
    if r.get('errors') or not rw or None in rw or r.get('statuses') != ['DONE', 'DONE']:
        continue
    label, opp = r['tag'].split('@', 1)
    if opp.startswith('replay_'):
        s = r['a_seat']
        m[label][opp] = rw[s] - rw[1 - s]

def sign_p(b, w):
    n = b + w
    if not n:
        return 1.0
    k = min(b, w)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)

res = lambda x: 'W' if x > 0 else 'L' if x < 0 else 'T'
B = m[base]
rec = lambda d: (lambda d: f"{sum(v > 0 for v in d)}-{sum(v < 0 for v in d)}-{sum(v == 0 for v in d)}")(list(d))
print(f"{base}: {len(B)} replays, {rec(B.values())}, mean {sum(B.values()) / len(B):+.0f}")
families = sorted({short(v[0]) for v in fam.values()}) if fam else []
print('graph'.ljust(14), 'n', ' record', '  mean chg', ' better/worse', '    p   ', 'L->W T->W W->L W->T T->L', '|', ' '.join(f[:10].rjust(10) for f in families))
for label in sorted(m, key=lambda l: -sum(m[l][g] - B[g] for g in m[l] if g in B) / max(1, len([g for g in m[l] if g in B]))):
    if label == base:
        continue
    common = [g for g in m[label] if g in B]
    d = {g: m[label][g] - B[g] for g in common}
    b, w = sum(v > 0 for v in d.values()), sum(v < 0 for v in d.values())
    fl = collections.Counter(f'{res(B[g])}->{res(m[label][g])}' for g in common)
    per = []
    for f in families:
        ds = [d[g] for g in common if g in fam and short(fam[g][0]) == f]
        per.append(f"{sum(ds) / len(ds):+10.0f}" if ds else ' ' * 10)
    print(f"{label:14s} {len(common):2d} {rec(m[label][g] for g in common):>8s} {sum(d.values()) / len(d):+9.0f}  {b:4d}/{w:<4d}  {sign_p(b, w):8.2g}  "
          f"{fl['L->W']:4d} {fl['T->W']:4d} {fl['W->L']:4d} {fl['W->T']:4d} {fl['T->L']:4d} | " + ' '.join(per))
