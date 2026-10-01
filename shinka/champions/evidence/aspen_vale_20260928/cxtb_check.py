"""The 09-27 engine's tomato gate in recorded games (cliproxyapi): its revenue projection at step 432 (counter T-B,
main.py:6700-6800: invest when _cxtb_expected_revenue >= _CXTB_MIN_REVENUE, 9,000) and the author's original gate
(shops and price, _CXTB_BASE_QUALIFIES), next to what tomatoes earned each seat (flows_av.py's exact accounting).

    python cxtb_check.py <index.json> <replay dir> <flows.json>
"""
import importlib.util
import json
import sys

ENGINE = '/home/alex/kagg-evo/repo/shinka/champions/ladder/tetsutani_demand_0927/agent'
sys.path.insert(0, ENGINE)
spec = importlib.util.spec_from_file_location('eng0927', ENGINE + '/main.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

index, replays, flows = json.load(open(sys.argv[1])), sys.argv[2], json.load(open(sys.argv[3]))
print(f"{'episode':9s} {'margin':>7s} {'team':15s} | proj rev  cxtb base | tomato $ ours / rival (units) | quads d20")
for g in sorted(index, key=lambda g: g['diff']):
    steps = json.load(open(f"{replays}/episode-{g['id']}-replay.json"))['steps']
    ours = g['seat']
    # seat 0's recorded observation carries every shared field (seat 1's lacks `step`); ours brings the private part
    o = dict(steps[432][0]['observation'], player=ours, private=steps[432][ours]['observation']['private'])
    revenue = m._cxtb_expected_revenue(o)
    cxtb, base = m._cxtb_qualifies(o, None), m._CXTB_BASE_QUALIFIES(o, None)
    f = flows[str(g['id'])]
    t_o, t_r = f['ours']['sell'].get('TOMATO', [0, 0]), f['rival']['sell'].get('TOMATO', [0, 0])
    o20 = steps[480][0]['observation']['farms']
    print(f"{g['id']} {g['diff']:+7,.0f} {g['team'][:15]:15s} | {revenue:8,.0f} {str(cxtb):5s} {str(base):5s} | "
          f"{t_o[1]:7,.0f} / {t_r[1]:7,.0f} ({t_o[0]}/{t_r[0]}) | {len(o20[ours]['unlocked_quadrants'])}/"
          f"{len(o20[1 - ours]['unlocked_quadrants'])}", flush=True)
