"""Counts from counterfactual.py results (cliproxyapi, ~/kagg-evo/juniper_review): for each set of recorded games
(Juniper Knoll's / Aspen Vale's, lost or tied / won) and each bundle, in how many games the bundle played differently
from the recorded agent at all, how many results it changes (against the rival's recorded moves) and the mean change
of the margin.

    python cf_summary.py
"""
import collections
import glob
import json

res_score = {'W': 1.0, 'T': 0.5, 'L': 0.0}
idx = {g['id']: g for g in json.load(open('index_all.json'))}
rows = collections.defaultdict(dict)
for f in glob.glob('cf/*.jsonl'):
    for line in open(f):
        d = json.loads(line)
        rows[d['episode']][d['bundle']] = d
sets = {('juniper', 'lost or tied'): lambda g: g['sub'] == 'juniper' and g['res'] != 'W',
        ('juniper', 'won'): lambda g: g['sub'] == 'juniper' and g['res'] == 'W',
        ('aspen', 'lost or tied'): lambda g: g['sub'] == 'aspen' and g['res'] != 'W',
        ('aspen', 'won'): lambda g: g['sub'] == 'aspen' and g['res'] == 'W'}
bundles = sorted({b for r in rows.values() for b in r})
fidelity_bad = [(e, b) for e, r in rows.items() for b, d in r.items()
                if b == idx[e]['sub'] and (d['equal_steps'] < 719 or d['cf_diff'] != d['diff'])]
print('fidelity (the submitted bundle replays its own games):', 'all exact' if not fidelity_bad else fidelity_bad)
for (sub, kind), pick in sets.items():
    games = [e for e in rows if pick(idx[e])]
    print(f'\n{sub}, {kind}: {len(games)} games replayed')
    for b in bundles:
        ds = [rows[e][b] for e in games if b in rows[e]]
        if not ds or b == sub:
            continue
        acted = sum(d['equal_steps'] < 719 for d in ds)
        flips = collections.Counter(f"{d['res']}->{d['cf_res']}" for d in ds if d['cf_res'] != d['res'])
        net = sum(res_score[d['cf_res']] - res_score[d['res']] for d in ds)
        dm = sum(d['cf_diff'] - d['diff'] for d in ds) / len(ds)
        firsts = sorted(d['equal_steps'] for d in ds if d['equal_steps'] < 719)
        print(f"  {b:10s} {len(ds):3d} games: plays differently in {acted:3d} (first at step {firsts[0] if firsts else '-'}, "
              f"median {firsts[len(firsts) // 2] if firsts else '-'}); results changed {dict(flips) or 'none'} "
              f"(net {net:+.1f}); margin {dm:+,.0f} a game")
