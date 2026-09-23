"""Package the trace-mining payload into a tar.gz for Kaggle dataset upload."""
from pathlib import Path
import hashlib
import json
import shutil
import tarfile

HERE = Path(__file__).resolve().parent
RUN_DIR = HERE / 'runs' / 'iter27_trace_mining'
PAYLOAD = RUN_DIR / 'payload'
DATASET = RUN_DIR / 'dataset'


def main():
    DATASET.mkdir(parents=True, exist_ok=True)
    archive = DATASET / 'frozen_payload.tar.gz'

    with tarfile.open(archive, 'w:gz', compresslevel=6) as tar:
        tar.add(PAYLOAD, arcname='payload')

    h = hashlib.sha256()
    with open(archive, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)

    files_count = sum(1 for _ in PAYLOAD.rglob('*') if _.is_file())
    manifest = json.loads((PAYLOAD / 'manifest.json').read_text())

    # Include runner.py in the archive check
    has_runner = (PAYLOAD / 'runner.py').exists()

    # Dataset metadata
    meta = {
        'title': 'kagg-iter27-trace-mining',
        'id': 'sunshinethroughfog/kagg-iter27-trace-mining',
        'licenses': [{'name': 'CC0-1.0'}],
    }
    (DATASET / 'dataset-metadata.json').write_text(json.dumps(meta, indent=2))

    result = {
        'archive': str(archive),
        'bytes': archive.stat().st_size,
        'sha256': h.hexdigest(),
        'files': files_count,
        'has_runner': has_runner,
        'evaluation_id': manifest['evaluation_id'],
        'opponents': len(manifest['opponents']),
        'seeds_per_seat': len(manifest['seeds']['0']),
    }
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
