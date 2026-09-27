"""The games of an index by the rival's step-0 wheat orders (read from the replay opponent bundles).

usage: python families.py <index.json> [replay_opponents dir] [out.json]

Prints one line per game (family, result, margin, rival) and a table by family, and writes out.json
({replay_<id>: [family, margin, rival]}, the input of ../alder_ford_20260927/bt_report.py). The families
are those of ../alder_ford_20260927/families.py.
"""
import collections
import gzip
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE.parents[1] / 'replay_opponents'


def family(a0):
    m = [o for o in (a0 or {}).get('market') or [] if o]
    buy = sum(int(o[2]) for o in m if o[0] == 'BUY_PRODUCT' and o[1] == 'WHEAT' and len(o) > 2)
    sell = sum(int(o[2]) for o in m if o[0] == 'SELL' and o[1] == 'WHEAT' and len(o) > 2)
    seed = any(o[0] == 'BUY_SEED' for o in m)
    if buy == 5 and seed and not sell:
        return 'tetsutani (buy 5 + seed)'
    if buy == 20 and sell == 15:
        return '2965 (buy 20 / sell 15)'
    if buy == 8 and sell == 3:
        return 'forecast (buy 8 / sell 3)'
    if buy == 13 and not sell:
        return 'shape shop (buy 13)'
    return f'other (buy {buy} / sell {sell}{" + seed" if seed else ""})'


games = json.load(open(sys.argv[1]))
by = collections.defaultdict(list)
out = {}
for g in games:
    b = R / f"replay_{g['id']}"
    if not (b / 'actions.json.gz').exists():
        print(g['id'], 'no replay bundle')
        continue
    a = json.loads(gzip.decompress((b / 'actions.json.gz').read_bytes()))
    f = family(a[0])
    by[f].append(g)
    out[f"replay_{g['id']}"] = [f, g['diff'], g['team']]
    print(f"{g['id']} {g['res']} {g['diff']:+8.0f}  {f:34s} {(g['team'] or '')[:24]:24s} {g.get('orate') or ''}")
print(f"\n{'rival family':34s} games  W-L-T    mean margin  losses > $600")
for f, gs in sorted(by.items(), key=lambda kv: -len(kv[1])):
    ms = [g['diff'] for g in gs]
    w, l, t = sum(m > 0 for m in ms), sum(m < 0 for m in ms), sum(m == 0 for m in ms)
    print(f"{f:34s} {len(ms):3d}   {w}-{l}-{t}".ljust(50) + f"{sum(ms) / len(ms):+9.0f}   {sum(m < -600 for m in ms)}")
if len(sys.argv) > 3:
    json.dump(out, open(sys.argv[3], 'w'), indent=1, ensure_ascii=False)
