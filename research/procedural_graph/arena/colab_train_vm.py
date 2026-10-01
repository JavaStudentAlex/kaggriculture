"""KAD-HP-1 training on a Colab TPU VM: the VM side of colab_train.py, started detached by it.

It is the Kaggle TPU notebook's launcher (research/action_diffusion/make_kaggle_kernel.py, LAUNCHER)
with /kaggle/input replaced by what the runner uploaded to /content/kad:
- code.tar.gz: the training code (train_job.py, policy_train.py, ...);
- run/: the run to continue (last.pt, best.pt, metrics.jsonl, scores.json, config.json);
- links.json: {name: signed download link} of the encoded days (days/<day>.tar, the data notebooks'
  outputs); the links carry their own access token, so no Kaggle credential is on the VM. They are
  never printed;
- launch.json: {"job_args": [...], "train_args": [...]} for train_job.py.
The days are downloaded and unpacked into /content/kad/days, then train_job.py runs with --tpu on
every TPU core of the VM and resumes run/last.pt. Log lines: `KAD_DAYS ...`, train_job.py's
`JOB_STAGE ...`, and finally `KAD_COLAB_EXIT=<code>`.
"""
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path('/content/kad')


def log(message):
    print(f'{time.strftime("%H:%M:%S")} {message}', flush=True)


def fetch(name, url):
    """Download one days/<day>.tar and unpack it into <ROOT>/days; (name, status, bytes)."""
    day = Path(name).stem
    if (ROOT / 'days' / day / 'manifest.json').exists():
        return name, 'present', 0
    part = ROOT / 'download' / f'{day}.tar'
    code = None
    for attempt in range(1, 5):
        code = subprocess.run(['curl', '-sSfL', '--retry', '3', '--max-time', '3600', '-o', str(part), url],
                              capture_output=True).returncode
        if code == 0 and tarfile.is_tarfile(part):
            size = part.stat().st_size
            with tarfile.open(part) as tar:
                tar.extractall(ROOT / 'days', filter='data')
            part.unlink()
            if (ROOT / 'days' / day / 'manifest.json').exists():
                return name, 'ok', size
        time.sleep(15 * attempt)
    return name, f'failed (curl exit {code})', 0


def main():
    config = json.loads((ROOT / 'launch.json').read_text())
    code = ROOT / 'code'
    if not (code / 'train_job.py').exists():
        code.mkdir(parents=True, exist_ok=True)
        with tarfile.open(ROOT / 'code.tar.gz') as tar:
            tar.extractall(code, filter='data')
    # as on Kaggle: without a TPU, torch_xla fails here instead of quietly training on the CPU
    os.environ['PJRT_DEVICE'] = 'TPU'
    import torch
    import torch_xla
    log(f'torch {torch.__version__} | torch_xla {torch_xla.__version__} | TPU {os.environ.get("TPU_ACCELERATOR_TYPE")} '
        f'| CPUs {os.cpu_count()} | disk free GB {shutil.disk_usage("/content").free / 1e9:.0f}')

    (ROOT / 'days').mkdir(exist_ok=True)
    (ROOT / 'download').mkdir(exist_ok=True)
    links = json.loads((ROOT / 'links.json').read_text())
    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        results = list(pool.map(lambda item: fetch(*item), sorted(links.items())))
    failed = [name for name, status, _ in results if status not in ('ok', 'present')]
    log(f'KAD_DAYS {len(results) - len(failed)}/{len(results)} days in place, '
        f'{sum(size for *_, size in results) / 1e9:.1f} GB downloaded in {time.time() - started:.0f} s; '
        f'failed {failed}')
    if failed:
        log('KAD_COLAB_EXIT=2')
        return 2

    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'orjson'])
    run = ROOT / 'run'
    inits = [str(run / name) for name in ('best.pt', 'last.pt') if (run / name).exists()]
    command = [sys.executable, str(code / 'train_job.py'), '--root', str(ROOT), '--run-dir', str(run), '--tpu',
               *(['--init-from', *inits] if inits else []), *config['job_args'], '--', *config['train_args']]
    log('job: ' + ' '.join(command))
    exit_code = subprocess.run(command, cwd=code).returncode
    log(f'KAD_COLAB_EXIT={exit_code}')
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
