"""Generate 4 Kaggle CPU shard kernels for trace-mining evaluation."""
from pathlib import Path
import json

HERE = Path(__file__).resolve().parent
RUN_DIR = HERE / 'runs' / 'iter27_trace_mining'
EVALUATION_ID = 'iter27-trace-mining-20260923-400x2-v1'
DATASET_SOURCE = 'sunshinethroughfog/kagg-iter27-trace-mining'
SHARDS = 4


KERNEL_TEMPLATE = '''#!/usr/bin/env python3
"""Frozen iter-27 trace-mining shard {shard}/4. CPU-only, no evolution, no uploads."""
import hashlib, json, os, shutil, subprocess, sys, tarfile, time
from pathlib import Path

EVALUATION_ID = '{evaluation_id}'
SHARD = {shard}
SHARDS = {shards}

os.environ.update(
    OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
    NUMEXPR_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='', KAGG_ORACLE_BACKEND='numpy',
    KAGG_ORACLE_DEVICE='cpu', PYTHONDONTWRITEBYTECODE='1',
)

work = Path('/kaggle/working')
print(f'SHARD {{SHARD}}/{{SHARDS}} — {{EVALUATION_ID}}', flush=True)
print(f'Hardware: {{os.cpu_count()}} CPUs, {{len(os.sched_getaffinity(0))}} affinity', flush=True)

# Locate and extract payload
archives = list(Path('/kaggle/input').rglob('frozen_payload.tar.gz'))
if archives:
    with tarfile.open(archives[0]) as tar:
        tar.extractall('/kaggle/temp/frozen_eval', filter='data')
    root = Path('/kaggle/temp/frozen_eval/payload')
else:
    manifests = [p for p in Path('/kaggle/input').rglob('manifest.json')
                 if json.loads(p.read_text()).get('evaluation_id') == EVALUATION_ID]
    if not manifests:
        raise RuntimeError('Payload missing: ' + str([str(p) for p in Path('/kaggle/input').rglob('*')][:50]))
    root = Path('/kaggle/temp/frozen_eval/payload')
    shutil.copytree(manifests[0].parent, root, dirs_exist_ok=True)

# Verify manifest
m = json.loads((root / 'manifest.json').read_text())
assert m['evaluation_id'] == EVALUATION_ID, f"ID mismatch: {{m['evaluation_id']}}"
for rel, sha in m['files'].items():
    assert hashlib.sha256((root / rel).read_bytes()).hexdigest() == sha, f'Hash mismatch: {{rel}}'
print(f'Payload verified: {{len(m["files"])}} files', flush=True)

# Install engine
subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet',
                'kaggle-environments==1.32.7', 'safetensors'], check=True)

# Pilot: 1 seed per opponent per seat (4 games) to gate bulk dispatch
runner = str(root / 'runner.py')
common = [sys.executable, runner,
          '--manifest', str(root / 'manifest.json'),
          '--root', str(root),
          '--shard-index', str(SHARD), '--shards', str(SHARDS),
          '--workers', '4']

pilot_dir = work / 'pilot'
print(f'Running pilot (4 games)...', flush=True)
subprocess.run(common + ['--output', str(pilot_dir), '--smoke', '--save-traces',
                          '--deadline-seconds', '3600'], check=True)
pilot = json.loads((pilot_dir / 'summary.json').read_text())
if pilot.get('status') != 'complete' or pilot['overall']['invalid']:
    raise RuntimeError(f'Pilot failed: {{pilot["status"]}} invalid={{pilot["overall"]["invalid"]}}')
print(f'PILOT PASSED: {{pilot["overall"]}}', flush=True)

# Full run: 400 seeds, save traces, 9h deadline
print(f'Starting full run: 400 games with trace saving...', flush=True)
subprocess.run(common + ['--output', str(work / 'results'), '--seed-count', '400',
                          '--save-traces', '--deadline-seconds', '32400'], check=True)

summary = json.loads((work / 'results' / 'summary.json').read_text())
print(f'SHARD_FINISHED {{SHARD}} {{EVALUATION_ID}}', flush=True)
print(json.dumps(summary['overall'], indent=2), flush=True)

# Count traces
traces = list((work / 'results' / 'traces').glob('*.json.gz'))
print(f'Traces saved: {{len(traces)}}', flush=True)
print(f'Trace size: {{sum(t.stat().st_size for t in traces) / 1e6:.1f}} MB', flush=True)
'''


def main():
    for shard in range(SHARDS):
        d = RUN_DIR / 'shards' / str(shard)
        d.mkdir(parents=True, exist_ok=True)

        script = KERNEL_TEMPLATE.format(
            shard=shard,
            shards=SHARDS,
            evaluation_id=EVALUATION_ID,
        )
        (d / 'run.py').write_text(script)

        meta = {
            'id': f'sunshinethroughfog/kagg-iter27-trace-{shard}',
            'title': f'kagg-iter27-trace-{shard}',
            'code_file': 'run.py',
            'language': 'python',
            'kernel_type': 'script',
            'is_private': True,
            'enable_gpu': False,
            'enable_tpu': False,
            'enable_internet': True,
            'dataset_sources': [DATASET_SOURCE],
            'competition_sources': ['kaggriculture'],
        }
        (d / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2))

    print(f'Prepared {SHARDS} shard kernels')
    for shard in range(SHARDS):
        print(f'  Shard {shard}: {RUN_DIR}/shards/{shard}/')


if __name__ == '__main__':
    main()
