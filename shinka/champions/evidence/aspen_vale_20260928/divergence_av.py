"""Where the rivals that ran a public engine for a while left it (cliproxyapi): for every lost game whose best match is
a public bundle, the step and the first differing action of that bundle against the rival's recorded move, with the
rival's action and the bundle's side by side.

    python divergence_av.py [avL|avW]
"""
import json
import sys
from pathlib import Path

AV = Path('/home/alex/kagg-evo/aspen_vale')
BATCH = sys.argv[1] if len(sys.argv) > 1 else 'avL'
classes = json.load(open(AV / f'classes_{BATCH}.json'))
for eid, c in sorted(classes.items(), key=lambda kv: -kv[1]['steps']):
    if c['steps'] < 100:
        continue
    rows = [json.loads(line) for line in open(AV / 'match' / BATCH / f"{c['agent']}.jsonl") if line.strip()]
    m = next(r for r in rows if str(r['episode']) == eid)
    d = m.get('first_difference') or {}
    step = m['equal_steps']
    day, hour = divmod(step, 24)
    print(f"{eid} {c['team'][:16]:16s} {c['diff']:+7,.0f} {c['agent']:22s} left at step {step} (day {day} h{hour})")
    b, r = d.get('bundle') or {}, d.get('recorded') or {}
    if b.get('farmer') != r.get('farmer'):
        print(f"    farmer: engine {b.get('farmer')} | rival {r.get('farmer')}")
    bh, rh = b.get('hands') or [], r.get('hands') or []
    for i in range(max(len(bh), len(rh))):
        x, y = (bh[i] if i < len(bh) else None), (rh[i] if i < len(rh) else None)
        if x != y:
            print(f"    hand {i}: engine {x} | rival {y}")
    if b.get('market') != r.get('market'):
        print(f"    market: engine {json.dumps(b.get('market'))[:260]}")
        print(f"            rival  {json.dumps(r.get('market'))[:260]}")
