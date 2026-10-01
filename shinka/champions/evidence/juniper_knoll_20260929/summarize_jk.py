"""Juniper Knoll's games: the rival's class (rival_counter's MirrorTracker on the recorded public state) and the
public bundles that reproduced its recorded moves longest (match_ladder_games.py), on cliproxyapi
(summarize_cr.py of ../cedar_ridge_20260928 for this submission; also run for the won games with `avW`).
Writes classes_<batch>.json and best_match_<batch>.txt, prints one line per game and the counts.

    python summarize_jk.py <batch>
"""
import collections
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, '/home/alex/kagg-evo/repo/research/procedural_graph')
from hazel_runtime.rival_model import MirrorTracker  # noqa: E402

AV = Path('/home/alex/kagg-evo/juniper_review')
BATCH = sys.argv[1] if len(sys.argv) > 1 else 'avL'
EXACT = 700
index = json.load(open(AV / f'index_{BATCH}.json'))
classes = {}
for g in index:
    f = json.load(open(AV / 'features' / f"{g['id']}.json"))
    mt = MirrorTracker()
    for t, (r, o) in enumerate(zip(f['rival'], f['ours'])):
        if mt.observe(t, r[:-2], o[:-2]) is not None:
            break
    classes[g['id']] = mt.decision
best = collections.defaultdict(list)
for path in glob.glob(str(AV / 'match' / BATCH / '*.jsonl')):
    for line in open(path):
        if line.strip():
            m = json.loads(line)
            best[m['episode']].append((m['equal_steps'], m['bundle']))


def kind(steps):
    return 'exact' if steps >= EXACT else 'long' if steps >= 100 else '-'


rows = []
for g in sorted(index, key=lambda g: g['diff']):
    ms = sorted(best.get(g['id'], []), reverse=True)
    top = ms[0] if ms else (0, None)
    ties = [b for s, b in ms if s == top[0]]
    rows.append({'id': g['id'], 'seat': g['seat'], 'class': classes[g['id']], 'diff': g['diff'], 'team': g['team'],
                 'orate': g['orate'], 'steps': top[0], 'kind': kind(top[0]),
                 'agent': ties[0] if top[0] >= 100 else 'no public agent', 'ties': ties[:4] if top[0] >= 100 else [],
                 'runner_up': [m for m in ms if m[1] not in ties][:2]})
json.dump({str(r['id']): r for r in rows}, open(AV / f'classes_{BATCH}.json', 'w'), indent=1, ensure_ascii=False)
with open(AV / f'best_match_{BATCH}.txt', 'w') as out:
    for r in rows:
        line = (f"{r['id']} seat {r['seat']} {r['diff']:+9,.0f} {str(r['orate'] or '-'):>7s} {r['team'][:22]:22s} "
                f"{r['class']:14s} {r['steps']:3d} {r['kind']:5s} {', '.join(r['ties']) or 'no public agent'}")
        out.write(line + '\n')
        print(line)
print(f'\n{BATCH}: by class:', dict(collections.Counter(r['class'] for r in rows)))
print('by best match (exact = 700+ equal moves, long = 100+):')
for (k, a), n in collections.Counter((r['kind'], r['agent']) for r in rows).most_common():
    ds = sorted(r['diff'] for r in rows if (r['kind'], r['agent']) == (k, a))
    print(f'  {n:3d} {k:5s} {a:28s} median margin {ds[len(ds) // 2]:+,.0f}')
