"""Package only explicitly frozen, credential-free evaluation assets."""
from pathlib import Path
import hashlib,json,shutil,tarfile
HERE=Path(__file__).resolve().parent
RUN=HERE/'runs/iter27_20260923'
PAYLOAD=RUN/'payload'
for name in ['smoke.py','runner.py']:
    src=HERE/name
    if src.exists():shutil.copy2(src,PAYLOAD/name)
manifest=json.loads((PAYLOAD/'manifest.json').read_text())
manifest['files']={str(f.relative_to(PAYLOAD)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(PAYLOAD.rglob('*')) if f.is_file() and f.name!='manifest.json' and '__pycache__' not in f.parts and f.suffix!='.pyc'}
(PAYLOAD/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
DATASET=RUN/'dataset';DATASET.mkdir(exist_ok=True)
archive=DATASET/'frozen_payload.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
    for rel in [*manifest['files'],'manifest.json']:
        tar.add(PAYLOAD/rel,arcname='payload/'+rel,recursive=False)
meta={'title':'Kaggriculture frozen iter27 evaluation','id':'sunshinethroughfog/kagg-frozen-iter27-20260923','licenses':[{'name':'other'}]}
(DATASET/'dataset-metadata.json').write_text(json.dumps(meta,indent=2))
print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'files':len(manifest['files']),'has_runner':(PAYLOAD/'runner.py').exists()},indent=2))
