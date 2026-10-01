"""Table of counterfactual.py results (cliproxyapi, in ~/kagg-evo/juniper_review): one row per recorded game, one
column per bundle: the result and margin it reaches against the rival's recorded moves, '=' when it plays the recorded
game move for move, '@<step>' where it first plays differently.

    python cf_table.py [bundle ...]
"""
import collections
import glob
import json
import sys

order = sys.argv[1:] or ['juniper', 'aspen', 'j_noguard', 'j_look3', 'j_ca15', 'j_sr12']
rows = collections.defaultdict(dict)
for f in glob.glob('cf/*.jsonl'):
    for line in open(f):
        d = json.loads(line)
        rows[d['episode']][d['bundle']] = d
idx = {g['id']: g for g in json.load(open('index_all.json'))}
print('episode   sub     team            res   margin | ' + ' | '.join(f'{b:>18s}' for b in order))
for e, bs in sorted(rows.items(), key=lambda kv: (idx[kv[0]]['sub'], idx[kv[0]]['diff'])):
    g = idx[e]
    cells = []
    for b in order:
        d = bs.get(b)
        if not d:
            cells.append(' ' * 18)
            continue
        eq = d['equal_steps']
        tag = '=' if eq >= 719 else f'@{eq}'
        cells.append(f"{d['cf_res']} {d['cf_diff']:+8.0f} {tag:>7s}")
    print(f"{e} {g['sub']:7s} {g['team'][:15]:15s} {g['res']} {g['diff']:+8.0f} | " + ' | '.join(cells))
