import json, sys
from pathlib import Path
sys.path.insert(0, '/home/alex/kagg-evo/dev/research/procedural_graph')
sys.path.insert(0, '/home/alex/kagg-evo/dev/research/procedural_graph/rival')
from hazel_runtime.rival_model import MirrorTracker
out = {}
for p in sorted(Path('/home/alex/kagg-evo/rival/features').glob('*.json')):
    g = json.load(open(p))
    mt = MirrorTracker()
    decided = None
    for t, (r, o) in enumerate(zip(g['rival'], g['ours'])):
        if mt.observe(t, r[:-2], o[:-2]) is not None and decided is None:
            decided = t
            break
    out[p.stem] = {'class': mt.decision, 'decided_at': decided, 'diverged_at': mt.diverged_at,
                   'gap': mt.gap, 'team': g['team'], 'res': g['res'], 'diff': g['diff'], 'orate': g.get('orate')}
json.dump(out, open('/home/alex/kagg-evo/rival/classes.json', 'w'), indent=1, ensure_ascii=False)
print(len(out))
