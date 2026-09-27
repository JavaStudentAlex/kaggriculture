"""Recover the agents of the notebooks pulled on 2026-09-27 (static decoding, no notebook code runs) as
match bundles in cands2/, and name any that equals an existing bundle's agent."""
import hashlib
import json
import shutil
import sys
from pathlib import Path

REPO = Path('/home/alex/kagg-evo/repo/shinka/champions/ladder')
sys.path.insert(0, str(REPO))
import build_ladder_pool as B  # noqa: E402

D = Path('/home/alex/kagg-evo/alder_ford2')
NAMES = {  # notebook dir -> (bundle name, selector: '' = the first decodable blob)
    'tetsutani__demand-preserving-turn-sale-timing': ('tetsutani_demand_0927', 'ARCHIVE_B64'),
    'tetsutani__shape-the-shop-work-the-pasture-kaggriculture': ('tetsutani_shape_shop_0927', 'MAIN_B64'),
    'leoprovorov__four-turn-forecast-notebook-version-2': ('leoprovorov_forecast_0927', 'SOURCE'),
    'evgendvorkin__kaggriculture-version-31-26-09-bronze-going-up': ('dvorkin_v31', ''),
    'lynnsakurai__farmer-john-and-the-idle-seller': ('lynn_idle', ''),
    'lynnsakurai__farmer-john-and-the-wheat-seller': ('lynn_wheat_0927', ''),
    'haodou092__kaggriculture-harvest-ledger': ('haodou_ledger', ''),
    'guruprasaathas111__kaggriculture-top-2-master-engine-v4': ('guru_master_v4', ''),
    'leoprovorov__god-s-mode-hacked-stores': ('leo_gods_mode', ''),
    'leoprovorov__a-song-of-ice-and-fire-fixed-flexible': ('leo_ice_fire', ''),
    'georgymamarin__kaggriculture-what-2600-farms-do-differently': ('mamarin_2600', ''),
}
known = {}
for src in list(REPO.glob('*/SOURCE.json')) + list((D.parent / 'alder_ford' / 'cands').glob('*/SOURCE.json')):
    s = json.loads(src.read_text())
    known.setdefault(tuple(sorted(s['files'].items())), []).append(s['name'])
out = D / 'cands2'
out.mkdir(exist_ok=True)
for nbdir, (name, selector) in NAMES.items():
    nb = next((D / 'nb0927' / nbdir).glob('*.ipynb'), None)
    try:
        files, how = B.recover(nb, selector)
    except SystemExit as ex:
        print(f"{name:26s} not recovered: {str(ex)[-90:]}")
        continue
    entry = 'main.py' if 'main.py' in files else 'agent.py' if 'agent.py' in files else sorted(files)[0]
    record = {'name': name, 'notebook': nbdir.replace('__', '/', 1), 'decoded': f"{selector or 'first blob'} ({how})",
              'entry': entry, 'files': {k: hashlib.sha256(v).hexdigest() for k, v in sorted(files.items())},
              'pulled_utc': '2026-09-27T10:00Z'}
    same = known.get(tuple(sorted(record['files'].items())), [])
    dst = out / name
    if dst.exists():
        shutil.rmtree(dst)
    (dst / 'agent').mkdir(parents=True)
    for rel, data in files.items():
        (dst / 'agent' / rel).parent.mkdir(parents=True, exist_ok=True)
        (dst / 'agent' / rel).write_bytes(data)
    host = (REPO / 'host_main.py').read_text()
    (dst / 'main.py').write_text(host)
    (dst / 'SOURCE.json').write_text(json.dumps(record, indent=1) + '\n')
    size = sum(len(v) for v in files.values())
    print(f"{name:26s} {record['decoded']:34s} entry {entry:8s} {size:8d} bytes  " + (f"SAME AS {', '.join(same)}" if same else 'NEW'))
