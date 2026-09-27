"""Static recovery of the notebooks whose archive recover_new.py missed: the public engine's new
ARCHIVE_B85 (triple-quoted, multi-line) and lynn's ARCHIVE_PARTS spread over many cells. No notebook
code runs: the literals are read with ast and decoded here."""
import ast
import base64
import glob
import gzip
import hashlib
import io
import json
import lzma
import shutil
import tarfile
import zlib
from pathlib import Path

D = Path('/home/alex/kagg-evo/alder_ford2')
REPO = Path('/home/alex/kagg-evo/repo/shinka/champions/ladder')


def literals(code, name):
    """String literals assigned to `name` or appended to it."""
    try:  # a multi-line literal may have lines starting with ! or %: filter magics only if needed
        tree = ast.parse(code)
    except SyntaxError:
        tree = ast.parse('\n'.join(l for l in code.splitlines() if not l.lstrip().startswith(('%', '!'))))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(getattr(t, 'id', None) == name for t in node.targets) \
                and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            out.append(node.value.value)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'append' \
                and getattr(node.func.value, 'id', None) == name and node.args \
                and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
            out.append(node.args[0].value)
    return out


def unpack(raw):
    for f in (lambda b: b, gzip.decompress, zlib.decompress, lzma.decompress):
        try:
            data = f(raw)
            with tarfile.open(fileobj=io.BytesIO(data)) as tar:
                return {m.name.lstrip('./'): tar.extractfile(m).read() for m in tar.getmembers() if m.isfile()}
        except Exception:
            continue
    raise SystemExit('not a (compressed) tar archive')


def bundle(name, nbdir, files, how):
    entry = 'main.py' if 'main.py' in files else 'agent.py'
    record = {'name': name, 'notebook': nbdir.replace('__', '/', 1), 'decoded': how, 'entry': entry,
              'files': {k: hashlib.sha256(v).hexdigest() for k, v in sorted(files.items())},
              'pulled_utc': '2026-09-27T10:00Z'}
    dst = D / 'cands2' / name
    if dst.exists():
        shutil.rmtree(dst)
    (dst / 'agent').mkdir(parents=True)
    for rel, data in files.items():
        (dst / 'agent' / rel).write_bytes(data)
    (dst / 'main.py').write_text((REPO / 'host_main.py').read_text())
    (dst / 'SOURCE.json').write_text(json.dumps(record, indent=1) + '\n')
    print(name, how, {k: (len(v), hashlib.sha256(v).hexdigest()[:12]) for k, v in files.items()})


for nbdir, name in (('tetsutani__demand-preserving-turn-sale-timing', 'tetsutani_demand_0927'),
                    ('guruprasaathas111__kaggriculture-top-2-master-engine-v4', 'guru_master_v4')):
    nb = json.load(open(glob.glob(str(D / 'nb0927' / nbdir / '*.ipynb'))[0]))
    code = next(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code' and 'ARCHIVE_B85' in ''.join(c['source']))
    raw = base64.b85decode(''.join(literals(code, 'ARCHIVE_B85')[0].split()))
    bundle(name, nbdir, unpack(raw), 'ARCHIVE_B85 (b85+tar)')

nbdir = 'lynnsakurai__farmer-john-and-the-idle-seller'
nb = json.load(open(glob.glob(str(D / 'nb0927' / nbdir / '*.ipynb'))[0]))
parts = [p for c in nb['cells'] if c['cell_type'] == 'code' for p in literals(''.join(c['source']), 'ARCHIVE_PARTS')]
bundle('lynn_idle', nbdir, unpack(base64.b85decode(''.join(parts).encode('ascii'))), f'ARCHIVE_PARTS x{len(parts)} (b85+tar)')

old = json.loads((REPO / 'tetsutani_demand' / 'SOURCE.json').read_text())['files']
print('pool tetsutani_demand main.py', old)
