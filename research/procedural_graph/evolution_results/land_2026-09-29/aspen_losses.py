"""Candidates on the plan jobs added for Aspen Vale's ladder losses (plan_before_av -> plan), run in ~/kagg-evo."""
import json
from collections import Counter
old = json.load(open('runs/ladder1/plan_before_av.json'))
new = json.load(open('repo/research/procedural_graph/evolution_results/ladder_2026-09-26/plan.json'))
k = lambda j: f"{j['tag']}|{j['seed']}|{j['a_seat']}"
added = [j for j in new if k(j) not in {k(o) for o in old}]
print('jobs added for Aspen Vale:', len(added), Counter('replay' if j['tag'].startswith('replay_') else 'stand-in' for j in added))
m = lambda p: json.load(open(p))['margins']
A = m('runs/ladder1/scores/g_16bb4af65a704f00.json')
C = {'it93': m('runs/ladder1/scores/g_99508e7525447d95.json'), 'seed': m('oracle-20260929/run/scores/g_2e6e1664c98cc2a9.json')}
for part in ('replay', 'stand-in'):
    ks = [k(j) for j in added if (j['tag'].startswith('replay_')) == (part == 'replay')]
    wa = Counter('W' if A[x] > 0 else 'T' if A[x] == 0 else 'L' for x in ks)
    line = f"{part:8s} n={len(ks)} Aspen {wa['W']}-{wa['L']}-{wa['T']} mean {sum(A[x] for x in ks)/len(ks):+.0f}"
    for n, c in C.items():
        wb = Counter('W' if c[x] > 0 else 'T' if c[x] == 0 else 'L' for x in ks)
        up = sum(c[x] > A[x] + .5 for x in ks); dn = sum(c[x] < A[x] - .5 for x in ks)
        line += f" | {n} {wb['W']}-{wb['L']}-{wb['T']} better {up} worse {dn} mean {sum(c[x]-A[x] for x in ks)/len(ks):+.0f}"
    print(line)
    if part == 'replay':
        for x in ks:
            if (C['seed'][x] > 0) != (A[x] > 0) or (C['it93'][x] > 0) != (A[x] > 0):
                print('   flip', x, 'Aspen', A[x], 'it93', C['it93'][x], 'seed', C['seed'][x])
c = json.load(open('runs/ladder1/checkpoint.json'))
for isl in c['islands']:
    for r in isl['history']:
        if r['iteration'] in (84, 92) and isl['name'] == 'Island-Next-Counter':
            po = r['verdict'].get('per_opponent', {})
            print(r['iteration'], {o: v for o, v in po.items() if 'head' in o or 'mirror' in o or 'champ' in o or o.startswith('g_')})
