#!/usr/bin/env python3
"""The opponent model's refit on a Colab GPU VM (started by colab_refit.py; the recipe of AGENTS.md section 6, as the
Kaggle notebook research/opponent_model/ops/kaggle/train_5days.py runs it, on one GPU).

Runs in /content/refit, which the runner fills with:
  code.tar.gz   research/opponent_model's scripts (extract.py, extract_parallel.py, features.py, mechanics.py,
                metrics.py, ttm_dataset.py, train_ttm.py)
  base/         the checkpoint to refit (model.safetensors, config.json, scaler.npz, labels.json)
  launch.json   {"days": [oldest .. newest], "lr": "2e-5", "epochs": 30, "patience": 5}

Stages, each logged as `REFIT_STAGE <name>`:
  1. venv: Python 3.12 with kaggle-environments 1.32.7 (the replays' engine, which extract.py reads the rules from),
     granite-tsfm, accelerate and torch for the VM's CUDA;
  2. download: the days' public zips (no credential: Kaggle's dataset download URL redirects to a signed link);
  3. shards: extract_parallel.py, one worker per day, labels next_action (AGENTS.md 4.2-4.3);
  4. smoke: train_ttm.py for one epoch on 20 episodes of the newest day, so a library mismatch shows in minutes;
  5. refit: train_ttm.py with ops/finetune.sh's settings: the newest day split 90/10 by episode, the days before it
     as extra training at decay 2, window stride 4, lr 2e-5 (= 1 GPU x 64), plateau x0.5 after 2, early stop after 5,
     at most 30 epochs, epoch 0 = the base checkpoint on the same validation windows (the number to beat);
  6. result.tar.gz: best/ (weights, config, scaler, labels, val_episodes), scores.json, trainer_state.json (every
     evaluation), train.log.
Ends with REFIT_EXIT=<code>.
"""
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
import zipfile
from pathlib import Path

ROOT = Path('/content/refit')
CODE, BASE, RUN = ROOT / 'code', ROOT / 'base', ROOT / 'run'
VENV = ROOT / 'venv'
PY = VENV / 'bin' / 'python'
DATASET = 'https://www.kaggle.com/api/v1/datasets/download/kaggle/kaggriculture-episodes-{day}'
TORCH_INDEX = 'https://download.pytorch.org/whl/cu128'
PROBE = ('import torch, tsfm_public, kaggle_environments, transformers, accelerate; '
         'print(torch.__version__, transformers.__version__, torch.cuda.get_device_name(0) '
         'if torch.cuda.is_available() else None, torch.cuda.is_available())')


def log(message):
    print(f'{time.strftime("%H:%M:%S")} {message}', flush=True)


def stage(name):
    log(f'REFIT_STAGE {name}')


def sh(command, **kwargs):
    command = [str(c) for c in command]
    log('$ ' + ' '.join(command)[:400])
    done = subprocess.run(command, **kwargs)
    if done.returncode:
        raise RuntimeError(f'{Path(command[0]).name} {" ".join(command[1:3])} ... exit {done.returncode}')
    return done


def venv():
    probe = [str(PY), '-c', PROBE]
    if not (PY.exists() and subprocess.run(probe, capture_output=True).returncode == 0):
        started = time.time()
        sh([sys.executable, '-m', 'pip', 'install', '-q', 'uv'])
        uv = [sys.executable, '-m', 'uv']
        if subprocess.run(uv + ['venv', '--python', '3.12', str(VENV)]).returncode:
            raise RuntimeError('cannot make a Python 3.12 venv')
        sh(uv + ['pip', 'install', '-q', '--python', str(PY), 'kaggle-environments==1.32.7', 'granite-tsfm',
                 'accelerate', 'torch', '--extra-index-url', TORCH_INDEX, '--index-strategy', 'unsafe-best-match'])
        log(f'venv built in {time.time() - started:.0f} s')
    done = subprocess.run(probe, capture_output=True, text=True)
    log(f'venv: {done.stdout.strip()} {done.stderr[-800:] if done.returncode else ""}')
    if done.returncode or not done.stdout.strip().endswith('True'):
        raise RuntimeError('the venv cannot import the refit libraries, or it sees no CUDA')


def download(days):
    zips = ROOT / 'zips'
    zips.mkdir(exist_ok=True)
    for day in days:
        path = zips / f'kaggriculture-episodes-{day}.zip'
        if path.exists() and zipfile.is_zipfile(path):
            continue
        part = path.with_suffix('.part')
        for attempt in range(1, 6):
            code = subprocess.run(['curl', '-sSfL', '--retry', '3', '--max-time', '1800', '-o', str(part),
                                   DATASET.format(day=day)]).returncode
            if code == 0 and zipfile.is_zipfile(part):
                part.replace(path)
                break
            log(f'download {day}: attempt {attempt} failed (curl exit {code})')
            time.sleep(15 * attempt)
        else:
            raise RuntimeError(f'cannot download {day}')
        with zipfile.ZipFile(path) as z:
            members = sum(1 for n in z.namelist() if n.endswith('.json'))
        log(f'{day}: {path.stat().st_size / 1e9:.2f} GB, {members} episodes')
    return zips


def shards(zips, days):
    out = ROOT / 'shards'
    sh([PY, CODE / 'extract_parallel.py', '--replays', zips, '--out', out, '--workers', len(days),
        '--alignment', 'next_action'], cwd=CODE)
    missing = [d for d in days if not (out / f'kaggriculture-episodes-{d}.npz').exists()]
    if missing:
        raise RuntimeError(f'no shard for {missing}')
    day_dir, prev_dir = ROOT / 'stage' / 'day', ROOT / 'stage' / 'prev'
    for d in (day_dir, prev_dir):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
        shutil.copy(out / 'labels.json', d / 'labels.json')
    newest = days[-1]
    shutil.copy(out / f'kaggriculture-episodes-{newest}.npz', day_dir)
    for d in days[:-1]:
        shutil.copy(out / f'kaggriculture-episodes-{d}.npz', prev_dir)
    return day_dir, prev_dir


def common(lr):
    return ['--init-from', BASE, '--scaler', BASE / 'scaler.npz', '--split', 'episode', '--prediction-filter-length',
            '96', '--metric-horizon', 'all', '--batch-size', '64', '--lr', lr]


def smoke(day_dir, lr):
    shutil.rmtree(ROOT / 'smoke', ignore_errors=True)
    with open(ROOT / 'smoke.log', 'w') as f:
        done = subprocess.run([str(c) for c in [PY, CODE / 'train_ttm.py', '--dataset', day_dir, '--out',
                                                ROOT / 'smoke', '--max-episodes', '20', '--epochs', '1',
                                                '--window-stride', '16', *common(lr)]],
                              cwd=CODE, stdout=f, stderr=subprocess.STDOUT, env=dict(os.environ, TQDM_DISABLE='1'))
    tail = (ROOT / 'smoke.log').read_text(errors='replace')[-1500:]
    if done.returncode or not (ROOT / 'smoke' / 'best' / 'model.safetensors').exists():
        raise RuntimeError(f'smoke training failed (exit {done.returncode}): {tail}')
    log('smoke training passed')


def refit(day_dir, prev_dir, launch):
    shutil.rmtree(RUN, ignore_errors=True)
    args = ['--dataset', day_dir, '--extra-train', prev_dir, '--extra-train-decay', '2.0', '--out', RUN,
            '--eval-on-start', '--max-episodes', '100000', '--epochs', str(launch.get('epochs', 30)),
            '--patience', str(launch.get('patience', 5)), '--plateau', '--plateau-factor', '0.5',
            '--plateau-patience', '2', '--window-stride', '4', *common(launch.get('lr', '2e-5'))]
    started = time.time()
    with open(ROOT / 'train.log', 'w') as f:
        done = subprocess.run([str(c) for c in [PY, CODE / 'train_ttm.py', *args]], cwd=CODE, stdout=f,
                              stderr=subprocess.STDOUT, env=dict(os.environ, TQDM_DISABLE='1'))
    log(f'refit exit {done.returncode} after {(time.time() - started) / 60:.1f} min')
    if done.returncode or not (RUN / 'best' / 'model.safetensors').exists():
        raise RuntimeError(f'refit failed: {(ROOT / "train.log").read_text(errors="replace")[-1500:]}')


def result():
    states = sorted(RUN.glob('checkpoint-*/trainer_state.json'), key=lambda p: int(p.parent.name.split('-')[1]))
    with tarfile.open(ROOT / 'result.tar.gz.part', 'w:gz') as tar:
        tar.add(RUN / 'best', 'best')
        for path, name in ((RUN / 'scores.json', 'scores.json'), (ROOT / 'train.log', 'train.log'),
                           (states[-1] if states else None, 'trainer_state.json')):
            if path is not None and path.exists():
                tar.add(path, name)
    os.replace(ROOT / 'result.tar.gz.part', ROOT / 'result.tar.gz')
    log(f'result.tar.gz: {(ROOT / "result.tar.gz").stat().st_size / 1e6:.1f} MB')


def main():
    launch = json.loads((ROOT / 'launch.json').read_text())
    days = sorted(launch['days'])
    log(f'refit of {BASE} on {days}')
    if not (CODE / 'train_ttm.py').exists():
        CODE.mkdir(exist_ok=True)
        with tarfile.open(ROOT / 'code.tar.gz') as tar:
            tar.extractall(CODE, filter='data')
    stage('venv')
    venv()
    stage('download')
    zips = download(days)
    stage('shards')
    day_dir, prev_dir = shards(zips, days)
    stage('smoke')
    smoke(day_dir, launch.get('lr', '2e-5'))
    stage('refit')
    refit(day_dir, prev_dir, launch)
    stage('result')
    result()
    return 0


if __name__ == '__main__':
    try:
        code = main()
    except Exception:  # noqa: BLE001 -- the runner reads the traceback from the log
        traceback.print_exc()
        code = 1
    (ROOT / 'done').write_text(str(code))
    print(f'REFIT_EXIT={code}', flush=True)
    sys.exit(code)
