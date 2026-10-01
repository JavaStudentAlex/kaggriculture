"""Recover the agents of the notebooks pulled on 2026-09-28 (static decoding, no notebook code runs) as match
bundles in cands3/, and name any that equals an existing bundle's agent (run on cliproxyapi)."""
import hashlib
import json
import shutil
import sys
from pathlib import Path

REPO = Path('/home/alex/kagg-evo/repo/shinka/champions/ladder')
sys.path.insert(0, str(REPO))
import build_ladder_pool as B  # noqa: E402

D = Path('/home/alex/kagg-evo/cedar_ridge')
NAMES = {  # notebook dir -> (bundle name, selector: '' = the first decodable blob)
    'leoprovorov__a-song-of-ice-and-fire-fixed-flexible': ('leo_ice_fire_0928', ''),
    'leoprovorov__god-s-mode-hacked-stores': ('leo_gods_mode_0928', ''),
    'guruprasaathas111__kaggriculture-master-engine-v53e01d74d8f': ('guru_master_v5', ''),
    'haodou092__kaggriculture-harvest-ledger': ('haodou_ledger_0928', ''),
    'haideptry__the-2965-master-hybrid-engine': ('haideptry_2965_0928', 'AGENT_B64'),
    'guruprasaathas111__kaggriculture-top-2-master-engine-v4': ('guru_master_v4_0928', ''),
    'leoprovorov__31415926535897932384626433832795058202884197169399': ('leo_pi', ''),
    'haideptry__the-shepherds-ledger-herd-safe-sovereign': ('haideptry_shepherd_0928', 'AGENT_B64'),
    'evgendvorkin__kaggriculture-version-31-26-09-bronze-going-up': ('dvorkin_v31_0928', ''),
    'kunaldesale2408__kaggriculture-ttv1': ('kunal_ttv1', ''),
    'georgymamarin__kaggriculture-what-2600-farms-do-differently': ('mamarin_2600_0928', ''),
}
known = {}
for src in (list(REPO.glob('*/SOURCE.json')) + list(Path('/home/alex/kagg-evo/alder_ford/cands').glob('*/SOURCE.json'))
            + list(Path('/home/alex/kagg-evo/alder_ford2/cands2').glob('*/SOURCE.json'))):
    s = json.loads(src.read_text())
    known.setdefault(tuple(sorted(s['files'].items())), []).append(s['name'])
out = D / 'cands3'
out.mkdir(exist_ok=True)
for nbdir, (name, selector) in NAMES.items():
    nb = next((D / 'nb0928' / nbdir).glob('*.ipynb'), None)
    try:
        files, how = B.recover(nb, selector)
    except SystemExit as ex:
        print(f"{name:26s} not recovered: {str(ex)[-90:]}")
        continue
    entry = 'main.py' if 'main.py' in files else 'agent.py' if 'agent.py' in files else sorted(files)[0]
    record = {'name': name, 'notebook': nbdir.replace('__', '/', 1), 'decoded': f"{selector or 'first blob'} ({how})",
              'entry': entry, 'files': {k: hashlib.sha256(v).hexdigest() for k, v in sorted(files.items())},
              'pulled_utc': '2026-09-28T08:00Z'}
    same = known.get(tuple(sorted(record['files'].items())), [])
    dst = out / name
    if dst.exists():
        shutil.rmtree(dst)
    if same:   # an agent we already have: no new bundle
        print(f"{name:26s} SAME AS {', '.join(same)}")
        continue
    (dst / 'agent').mkdir(parents=True)
    for rel, data in files.items():
        (dst / 'agent' / rel).parent.mkdir(parents=True, exist_ok=True)
        (dst / 'agent' / rel).write_bytes(data)
    (dst / 'main.py').write_text((REPO / 'host_main.py').read_text())
    (dst / 'SOURCE.json').write_text(json.dumps(record, indent=1) + '\n')
    size = sum(len(v) for v in files.values())
    print(f"{name:26s} {record['decoded']:34s} entry {entry:8s} {size:8d} bytes  NEW")
