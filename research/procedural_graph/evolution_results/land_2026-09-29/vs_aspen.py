"""Paired comparison of cached gauntlet margins against Aspen Vale's bundle (run on cliproxyapi in ~/kagg-evo)."""
import json
from collections import Counter, defaultdict
from math import comb

ASPEN = 'g_16bb4af65a704f00'
L1 = 'runs/ladder1/scores/'


def load(path):
    return json.load(open(path))['margins']


def sign_p(u, d):
    n = u + d
    if n == 0:
        return 1.0
    k = min(u, d)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def res(m):
    return 1.0 if m > 0 else 0.5 if m == 0 else 0.0


plan = json.load(open('oracle-20260929/repo/research/procedural_graph/evolution_results/oracle_2026-09-29/plan.json'))
kind = {}
for j in plan:
    key = f"{j['tag']}|{j['seed']}|{j['a_seat']}"
    s = j.get('set', '?')
    if j['tag'].startswith('replay_'):
        s = 'replay:' + s
    kind[key] = s
print('plan sets:', dict(Counter(kind.values())))

aspen = load(L1 + ASPEN + '.json')
cands = {
    'Next-Counter it93 (Aspen + _S809_LOOK 4 + _CA_MARGIN -22)': L1 + 'g_99508e7525447d95.json',
    'oracle/land seed (+ _SR_MARGIN 14 + guard)': 'oracle-20260929/run/scores/g_2e6e1664c98cc2a9.json',
    'Next-Market (0927, SR14, LOOK4, guard)': L1 + 'g_4eaa091d91d1fdd1.json',
    'Next-Production (0927, LOOK4)': L1 + 'g_25c379af0098dd6d.json',
    'Opening (old engine, Cedar line)': L1 + 'g_a359ef853bf580cf.json',
    'Crops (old engine)': L1 + 'g_e096a89b8a65c907.json',
    'Herd (old engine)': L1 + 'g_eb41d7064104da08.json',
    'Tactics (old engine)': L1 + 'g_9717dfe1c3034317.json',
    'Endgame (old engine)': L1 + 'g_777d89527bc510d5.json',
}


def compare(name, a, b, detail=False):
    keys = sorted(set(a) & set(b))
    up = sum(1 for k in keys if b[k] > a[k] + 0.5)
    down = sum(1 for k in keys if b[k] < a[k] - 0.5)
    mean = sum(b[k] - a[k] for k in keys) / max(1, len(keys))
    wa = Counter('W' if a[k] > 0 else 'T' if a[k] == 0 else 'L' for k in keys)
    wb = Counter('W' if b[k] > 0 else 'T' if b[k] == 0 else 'L' for k in keys)
    ru = sum(1 for k in keys if res(b[k]) > res(a[k]))
    rd = sum(1 for k in keys if res(b[k]) < res(a[k]))
    print(f"\n{name}: {len(keys)} shared games | better {up} worse {down} | mean {mean:+.0f} $/game p={sign_p(up, down):.2g}"
          f" | results +{ru}/-{rd} (p={sign_p(ru, rd):.2g}) | Aspen {wa['W']}-{wa['L']}-{wa['T']} -> {wb['W']}-{wb['L']}-{wb['T']}")
    if not detail:
        return
    groups = defaultdict(list)
    for k in keys:
        tag = k.split('|')[0]
        fam = 'replay (recorded ladder rivals)' if tag.startswith('replay_') else tag
        groups[fam].append(k)
        groups['set=' + kind.get(k, '?')].append(k)
    for g in sorted(groups):
        ks = groups[g]
        u = sum(1 for k in ks if b[k] > a[k] + 0.5)
        d = sum(1 for k in ks if b[k] < a[k] - 0.5)
        m = sum(b[k] - a[k] for k in ks) / len(ks)
        wa = Counter('W' if a[k] > 0 else 'T' if a[k] == 0 else 'L' for k in ks)
        wb = Counter('W' if b[k] > 0 else 'T' if b[k] == 0 else 'L' for k in ks)
        print(f"   {g:42s} n={len(ks):3d} better {u:3d} worse {d:3d} mean {m:+7.0f}  W-L-T {wa['W']}-{wa['L']}-{wa['T']} -> {wb['W']}-{wb['L']}-{wb['T']}")


for i, (name, path) in enumerate(cands.items()):
    compare(name, aspen, load(path), detail=i < 2)

seed = load('oracle-20260929/run/scores/g_2e6e1664c98cc2a9.json')
compare('seed vs its parent (Next-Counter it93): the guard + _SR_MARGIN 14 alone', load(L1 + 'g_99508e7525447d95.json'), seed, detail=False)
reg = [k for k in seed if kind.get(k) == 'replay:aspen_win_regression']
w = Counter('W' if seed[k] > 0 else 'T' if seed[k] == 0 else 'L' for k in reg)
print(f"\nAspen's 36 recorded ladder wins replayed with the seed: {w['W']}-{w['L']}-{w['T']}; lost ones:",
      sorted((k, seed[k]) for k in reg if seed[k] <= 0))
