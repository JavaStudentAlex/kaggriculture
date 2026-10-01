"""Recover the agents of the notebooks that ran after the 09-28 08:04 UTC pull (pulled again at 20:26 UTC; static
decoding, no notebook code runs) as match bundles in cands4/, and name any that equals an agent we already have
(recover_0928.py of ../cedar_ridge_20260928 with this pull's notebooks; run on cliproxyapi).

Three notebooks changed their packaging, so they get their own decoders, each checked against the SHA-256 values the
notebook itself asserts:
- tetsutani's engine: ARCHIVE_B85, base85 with line breaks (removed as the notebook does) of a tar.gz;
- flexonafft's agent: _PAYLOAD, base64 + zlib of main.py (the generic decoder, under the new label);
- leoprovorov's God's mode: PAYLOAD_B85, base85 + lzma of a file container (2-byte count, then per file a 2-byte name
  size, the name, an 8-byte size and the bytes); only its .py files go into the bundle.
"""
import ast
import base64
import hashlib
import io
import json
import lzma
import shutil
import sys
import tarfile
from pathlib import Path

REPO = Path('/home/alex/kagg-evo/repo/shinka/champions/ladder')
sys.path.insert(0, str(REPO))
import build_ladder_pool as B  # noqa: E402

D = Path('/home/alex/kagg-evo/aspen_vale')


def constants(nb, marker):
    """{name: literal value} of the simple assignments in the code cell that contains `marker`."""
    for cell in json.loads(nb.read_text())['cells']:
        code = ''.join(cell.get('source', []))
        if cell.get('cell_type') == 'code' and marker in code:
            out = {}
            for node in ast.walk(ast.parse(code)):
                if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                    try:
                        out[node.targets[0].id] = ast.literal_eval(node.value)
                    except ValueError:
                        pass
            return out
    raise SystemExit(f'{nb}: no code cell with {marker}')


def tetsutani_b85(nb):
    c = constants(nb, 'ARCHIVE_B85')
    archive = base64.b85decode(''.join(c['ARCHIVE_B85'].split()).encode('ascii'))
    assert hashlib.sha256(archive).hexdigest() == c['EXPECTED_ARCHIVE_SHA256'], 'archive sha256'
    with tarfile.open(fileobj=io.BytesIO(archive), mode='r:gz') as tf:
        files = {m.name: tf.extractfile(m).read() for m in tf.getmembers() if m.isfile()}
    assert sorted(files) == sorted(c['EXPECTED_MEMBERS']), sorted(files)
    assert hashlib.sha256(files['main.py']).hexdigest() == c['EXPECTED_MAIN_SHA256'], 'main.py sha256'
    return files, f"ARCHIVE_B85 (b85+gzip+tar, variant {c.get('VARIANT')})"


def flexon_payload(nb):
    files, how = B.recover(nb, '_PAYLOAD')
    expected = constants(nb, 'EXPECTED_SHA256')['EXPECTED_SHA256']
    assert hashlib.sha256(files['agent.py']).hexdigest() == expected, 'main.py sha256'
    return {'main.py': files['agent.py']}, f'_PAYLOAD ({how})'


def gods_mode_container(nb):
    c = constants(nb, 'PAYLOAD_B85')
    blob = lzma.decompress(base64.b85decode(c['PAYLOAD_B85']))
    cursor, files = 2, {}
    for _ in range(int.from_bytes(blob[:2], 'big')):
        size = int.from_bytes(blob[cursor:cursor + 2], 'big')
        name = blob[cursor + 2:cursor + 2 + size].decode('utf-8')
        cursor += 2 + size
        length = int.from_bytes(blob[cursor:cursor + 8], 'big')
        data = blob[cursor + 8:cursor + 8 + length]
        cursor += 8 + length
        assert hashlib.sha256(data).hexdigest() == c['EXPECTED_SHA256'][name], name
        files[name] = data
    assert cursor == len(blob)
    return {k: v for k, v in files.items() if k.endswith('.py')}, 'PAYLOAD_B85 (b85+lzma+container)'


NAMES = {  # notebook dir -> (bundle name, selector: '' = the first decodable blob, or a decoder)
    'tetsutani__demand-preserving-turn-sale-timing': ('tetsutani_demand_0928', tetsutani_b85),
    'evgendvorkin__kaggriculture-version-31-26-09-bronze-going-up': ('dvorkin_v31_0928b', ''),
    'leoprovorov__god-s-mode-hacked-stores': ('leo_gods_mode_0928b', gods_mode_container),
    'leoprovorov__a-song-of-ice-and-fire-fixed-flexible': ('leo_ice_fire_0928b', ''),
    'haodou092__kaggriculture-harvest-ledger': ('haodou_ledger_0928b', ''),
    'flexonafft__kaggriculture-multi-route-farming-agent': ('flexon_multiroute_0928', flexon_payload),
}
known = {}
for src in (list(REPO.glob('*/SOURCE.json')) + list(Path('/home/alex/kagg-evo/alder_ford/cands').glob('*/SOURCE.json'))
            + list(Path('/home/alex/kagg-evo/alder_ford2/cands2').glob('*/SOURCE.json'))
            + list(Path('/home/alex/kagg-evo/cedar_ridge/cands3').glob('*/SOURCE.json'))):
    s = json.loads(src.read_text())
    known.setdefault(tuple(sorted(s['files'].items())), []).append(s['name'])
out = D / 'cands4'
out.mkdir(exist_ok=True)
for nbdir, (name, selector) in NAMES.items():
    nb = next((D / 'nb0928b' / nbdir).glob('*.ipynb'), None)
    try:
        files, how = selector(nb) if callable(selector) else B.recover(nb, selector)
    except (SystemExit, AssertionError, KeyError) as ex:
        print(f"{name:26s} not recovered: {type(ex).__name__} {str(ex)[-90:]}")
        continue
    if not callable(selector):
        how = f"{selector or 'first blob'} ({how})"
    entry = 'main.py' if 'main.py' in files else 'agent.py' if 'agent.py' in files else sorted(files)[0]
    record = {'name': name, 'notebook': nbdir.replace('__', '/', 1), 'decoded': how, 'entry': entry,
              'files': {k: hashlib.sha256(v).hexdigest() for k, v in sorted(files.items())},
              'pulled_utc': '2026-09-28T20:26Z'}
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
    print(f"{name:26s} {how:60s} entry {entry:8s} {size:8d} bytes  NEW")
