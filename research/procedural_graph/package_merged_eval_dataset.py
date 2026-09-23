"""Package the merged graph evaluation payload and upload/update Kaggle dataset."""
from pathlib import Path
import hashlib
import json
import subprocess
import tarfile

REPO = Path('/home/alex/kaggriculture')
RUN_DIR = REPO / 'research/procedural_graph/runs/merged_eval_kaggle'
PAYLOAD = RUN_DIR / 'payload'
DATASET = RUN_DIR / 'dataset'


def main():
    DATASET.mkdir(parents=True, exist_ok=True)
    archive = DATASET / 'frozen_payload.tar.gz'
    print(f'Compressing payload into {archive}...')

    with tarfile.open(archive, 'w:gz', compresslevel=6) as tar:
        tar.add(PAYLOAD, arcname='payload')

    h = hashlib.sha256()
    with open(archive, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)

    files_count = sum(1 for _ in PAYLOAD.rglob('*') if _.is_file())
    manifest = json.loads((PAYLOAD / 'manifest.json').read_text())

    meta = {
        'title': 'kagg-merged-graph-eval',
        'id': 'sunshinethroughfog/kagg-merged-graph-eval',
        'licenses': [{'name': 'CC0-1.0'}],
    }
    (DATASET / 'dataset-metadata.json').write_text(json.dumps(meta, indent=2))

    result = {
        'archive': str(archive),
        'bytes': archive.stat().st_size,
        'sha256': h.hexdigest(),
        'files': files_count,
        'evaluation_id': manifest['evaluation_id'],
        'opponents': len(manifest['opponents']),
        'seeds_per_seat': len(manifest['seeds']['0']),
    }
    print(json.dumps(result, indent=2))

    # Check if dataset exists on Kaggle, create or update
    status_proc = subprocess.run(['kaggle', 'datasets', 'status', 'sunshinethroughfog/kagg-merged-graph-eval'],
                                 capture_output=True, text=True)
    print('Kaggle dataset status output:', status_proc.stdout.strip(), status_proc.stderr.strip())
    if 'ready' in status_proc.stdout.lower() or status_proc.returncode == 0:
        print('Updating existing Kaggle dataset...')
        subprocess.run(['kaggle', 'datasets', 'version', '-p', str(DATASET), '-m', 'Update merged procedural graph payload', '-r', 'zip'],
                       check=True)
    else:
        print('Creating new Kaggle dataset...')
        subprocess.run(['kaggle', 'datasets', 'create', '-p', str(DATASET), '-r', 'zip'], check=True)
    print('Dataset ready on Kaggle!')


if __name__ == '__main__':
    main()
