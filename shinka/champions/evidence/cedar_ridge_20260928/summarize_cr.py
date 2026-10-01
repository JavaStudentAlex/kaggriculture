"""Cedar Ridge's lost games against strong rivals: the rival's class (rival_counter's MirrorTracker on the recorded
public state) and the public bundles that reproduced its recorded moves longest (match_ladder_games.py), on
cliproxyapi. Writes classes_crL.json and best_match_crL.txt, prints the counts."""
import collections
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, '/home/alex/kagg-evo/repo/research/procedural_graph')
from hazel_runtime.rival_model import MirrorTracker  # noqa: E402

CR = Path('/home/alex/kagg-evo/cedar_ridge')
EXACT = 700
index = json.load(open(CR / 'index_crL.json'))
classes = {}
for g in index:
    f = json.load(open(CR / 'features' / f"{g['id']}.json"))
    mt = MirrorTracker()
    for t, (r, o) in enumerate(zip(f['rival'], f['ours'])):
        if mt.observe(t, r[:-2], o[:-2]) is not None:
            break
    classes[g['id']] = mt.decision
best = collections.defaultdict(list)
for path in glob.glob(str(CR / 'match' / 'crL' / '*.jsonl')):
    for line in open(path):
        if line.strip():
            m = json.loads(line)
            best[m['episode']].append((m['equal_steps'], m['bundle']))
rows, by_class, by_bundle = [], collections.Counter(), collections.Counter()
for g in index:
    ms = sorted(best.get(g['id'], []), reverse=True)
    top = ms[0] if ms else (0, None)
    ties = [b for s, b in ms if s == top[0]]
    who = top[1] if top[0] >= 100 else 'no public agent'
    kind = 'exact' if top[0] >= EXACT else 'long' if top[0] >= 100 else '-'
    rows.append((classes[g['id']], g['diff'], g['team'], g['orate'], top[0], ', '.join(ties[:3]) if top[0] >= 100 else ''))
    by_class[classes[g['id']]] += 1
    by_bundle[(kind, ties[0] if top[0] >= 100 else 'no public agent')] += 1
json.dump({str(g['id']): {'class': classes[g['id']], 'diff': g['diff'], 'team': g['team'], 'orate': g['orate'],
                          'matches': sorted(best.get(g['id'], []), reverse=True)[:3]} for g in index},
          open(CR / 'classes_crL.json', 'w'), indent=1, ensure_ascii=False)
with open(CR / 'best_match_crL.txt', 'w') as out:
    for r in rows:
        out.write(repr(r) + '\n')
print('losses by class:', dict(by_class))
print('losses by best match (exact = 700+ equal moves, long = 100+):')
for (kind, b), n in by_bundle.most_common():
    ds = [r[1] for r, g in zip(rows, index) if (('exact' if r[4] >= EXACT else 'long' if r[4] >= 100 else '-'), (r[5].split(', ')[0] if r[4] >= 100 else 'no public agent')) == (kind, b)]
    print(f'  {n:3d} {kind:5s} {b:28s} median margin {sorted(ds)[len(ds) // 2]:+,.0f}')
