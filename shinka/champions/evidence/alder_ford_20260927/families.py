"""Alder Ford's recorded ladder games by the rival's step-0 wheat orders (from the replay bundles).

usage: python families.py [replay_opponents dir] [out.json]
"""
import collections, gzip, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
R = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parents[1] / 'replay_opponents'
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / 'families.json'

def family(a0):
    m = [o for o in (a0 or {}).get('market') or [] if o]
    buy = sum(int(o[2]) for o in m if o[0] == 'BUY_PRODUCT' and o[1] == 'WHEAT' and len(o) > 2)
    sell = sum(int(o[2]) for o in m if o[0] == 'SELL' and o[1] == 'WHEAT' and len(o) > 2)
    seed = any(o[0] == 'BUY_SEED' for o in m)
    if buy == 5 and seed and not sell: return 'tetsutani (buy 5 + seed)'
    if buy == 20 and sell == 15: return '2965 (buy 20 / sell 15)'
    if buy == 8 and sell == 3: return 'forecast (buy 8 / sell 3)'
    if buy == 13 and not sell: return 'shape shop (buy 13)'
    return f'other (buy {buy} / sell {sell}{" + seed" if seed else ""})'

fam = {}
for src in sorted(R.glob('replay_*/SOURCE.json')):
    s = json.loads(src.read_text())
    a = json.loads(gzip.decompress((src.parent / 'actions.json.gz').read_bytes()))
    fam[s['name']] = (family(a[0]), s['recorded_margin_for_us'], s['opponent'])
json.dump(fam, open(OUT, 'w'), indent=1, ensure_ascii=False)
by = collections.defaultdict(list)
for name, (f, m, _) in fam.items():
    by[f].append(m)
print('rival family                  games  W-L-T    mean margin  losses > $600')
for f, ms in sorted(by.items(), key=lambda kv: -len(kv[1])):
    w, l, t = sum(m > 0 for m in ms), sum(m < 0 for m in ms), sum(m == 0 for m in ms)
    print(f'{f:30s} {len(ms):3d}   {w}-{l}-{t}'.ljust(44) + f'{sum(ms)/len(ms):+8.0f}   {sum(m < -600 for m in ms)}')
