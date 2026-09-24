#!/usr/bin/env python3
"""Score predictor checkpoints on ladder replays, as the live oracle would (calibration data).

Shipped as `arena.py` in a payload built by calib_payload.py, so colab_run.py drives it like a
game batch: the same command line (--jobs, --root, --workers, --out, --trace-dir), one result
row per job ({'tag', 'seed' = episode id, 'a_seat': 0, 'statuses'}), ARENA_DONE at the end.

For both seats of each episode and every `stride`-th origin t >= context, the model input is
rows [t - context, t) of [scaled features | log1p targets] (the live oracle's context_block),
and the forecast covers steps t .. t+95. Per origin and product it keeps what the policy reads:
score_4 / score_24 (max log1p units over the next 4 / 24 steps) and units_24 (units summed over
24 steps), plus the truth (opponent sells over the next 4 and 24 steps). One npz per episode goes
to --trace-dir; calib_fit.py turns them into a calibration.json.

On a GPU VM (colab_run.py shape t4hm) the checkpoints run as the torch TinyTimeMixer
(tsfm_public, installed into the venv on first use; 1.7e-6 off the numpy port on a T4, ~330
forecasts/s in float32) while --workers processes rebuild the features; without a GPU they run
on the numpy port (kagg_ttm_numpy), one forecast per call, which is ~100x slower.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zipfile
from multiprocessing import Pool
from pathlib import Path

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np  # noqa: E402

ARGS = None
_NETS = {}


def nets():
    if not _NETS:
        from kagg_ttm_numpy import TTMNumpy
        for name in ARGS['models']:
            d = Path(ARGS['root']) / 'models' / name
            with np.load(d / 'scaler.npz') as sc:
                _NETS[name] = (TTMNumpy(str(d)), sc['mean'].astype(np.float32), sc['std'].astype(np.float32))
    return _NETS


def score(job):
    from extract import episode_rows
    from features import feature_names
    start = time.time()
    with zipfile.ZipFile(Path(ARGS['root']) / 'games.zip') as zf:
        doc = json.loads(zf.read(job['member']))
    rows = {0: ([], []), 1: ([], [])}
    for x, y, d in episode_rows(doc, feature_names()):
        rows[int(d[2])][0].append(x)
        rows[int(d[2])][1].append(y)
    del doc
    parts = []
    for seat, (xs, ys) in rows.items():
        X, Y = np.stack(xs), np.stack(ys).astype(np.float32)
        part = {}
        for name, (net, mean, std) in nets().items():
            ctx = net.context
            series = np.concatenate([(X - mean) / std, np.log1p(np.clip(Y, 0, None))], axis=1).astype(np.float32)
            origins = range(ctx, len(X) - 4, ARGS['stride'])
            preds = np.stack([net.predict(series[t - ctx:t]) for t in origins])   # (n, 96, 9) log1p units
            part[f'{name}_s4'] = preds[:, :4].max(1)
            part[f'{name}_s24'] = preds[:, :24].max(1)
            part[f'{name}_u24'] = np.expm1(np.clip(preds[:, :24], 0, None)).sum(1)
            part['step'] = np.array(origins)
        part['truth4'] = np.stack([Y[t:t + 4].sum(0) for t in part['step']])
        part['truth24'] = np.stack([Y[t:t + 24].sum(0) for t in part['step']])
        part['seat'] = np.full(len(part['step']), seat)
        parts.append(part)
    return finish(job, parts, start)

TORCH_INSTALL = ['torch', 'granite-tsfm', '--extra-index-url', 'https://download.pytorch.org/whl/cu128',
                 '--index-strategy', 'unsafe-best-match']


def gpu_ready():
    """torch with CUDA in this venv, installing torch + granite-tsfm with uv if needed."""
    import shutil
    import subprocess
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        pass
    if not shutil.which('nvidia-smi') or subprocess.run(['nvidia-smi'], capture_output=True).returncode:
        return False
    uv = shutil.which('uv') or '/usr/local/bin/uv'
    start = time.time()
    done = subprocess.run([uv, 'pip', 'install', '-q', '--python', sys.executable] + TORCH_INSTALL,
                          capture_output=True, text=True)
    print(f'installed torch + granite-tsfm: exit {done.returncode}, {time.time() - start:.0f} s '
          f'{done.stderr[-800:] if done.returncode else ""}', flush=True)
    import importlib
    importlib.invalidate_caches()
    import torch
    return torch.cuda.is_available()


def extract(job):
    """Feature rows of both seats: (job, {seat: (X, Y)}); CPU side of the GPU path."""
    from extract import episode_rows
    from features import feature_names
    try:
        with zipfile.ZipFile(Path(ARGS['root']) / 'games.zip') as zf:
            doc = json.loads(zf.read(job['member']))
        rows = {0: ([], []), 1: ([], [])}
        for x, y, d in episode_rows(doc, feature_names()):
            rows[int(d[2])][0].append(x)
            rows[int(d[2])][1].append(y)
        return job, {seat: (np.stack(xs), np.stack(ys).astype(np.float32)) for seat, (xs, ys) in rows.items()}, None
    except Exception as exc:  # noqa: BLE001
        return job, None, repr(exc)[:500]


def gpu_models(args, batch):
    import torch
    from tsfm_public import TinyTimeMixerForPrediction
    models = {}
    for name in args['models']:
        d = Path(args['root']) / 'models' / name
        net = TinyTimeMixerForPrediction.from_pretrained(str(d)).cuda().eval()
        with np.load(d / 'scaler.npz') as sc:
            mean, std = sc['mean'].astype(np.float32), sc['std'].astype(np.float32)
        models[name] = (net, mean, std, int(net.config.context_length))
    # check against the numpy port on real-looking blocks
    from kagg_ttm_numpy import TTMNumpy
    for name, (net, _, _, ctx) in models.items():
        x = np.random.default_rng(0).normal(size=(4, ctx, net.config.num_input_channels)).astype(np.float32)
        ref = np.stack([TTMNumpy(str(Path(args['root']) / 'models' / name)).predict(b) for b in x])
        with torch.no_grad():
            got = net(past_values=torch.from_numpy(x).cuda()).prediction_outputs.float().cpu().numpy()
        diff = float(np.abs(got - ref).max())
        print(f'model {name}: torch vs numpy max abs diff {diff:.2e}', flush=True)
        if diff > 1e-3:
            raise SystemExit(f'model {name}: torch disagrees with the numpy port ({diff})')
    return models


def gpu_score(job, seats, models, batch):
    import torch
    start = time.time()
    parts = []
    for seat, (X, Y) in seats.items():
        part = {}
        for name, (net, mean, std, ctx) in models.items():
            series = np.concatenate([(X - mean) / std, np.log1p(np.clip(Y, 0, None))], axis=1).astype(np.float32)
            origins = np.arange(ctx, len(X) - 4, ARGS['stride'])
            s = torch.from_numpy(series).cuda()
            s4, s24, u24 = [], [], []
            for lo in range(0, len(origins), batch):
                idx = torch.as_tensor(origins[lo:lo + batch], device='cuda')
                win = s[(idx[:, None] - ctx) + torch.arange(ctx, device='cuda')[None, :]]   # (b, ctx, C)
                with torch.no_grad():
                    p = net(past_values=win).prediction_outputs.float()                     # (b, 96, 9)
                s4.append(p[:, :4].amax(1).cpu().numpy())
                s24.append(p[:, :24].amax(1).cpu().numpy())
                u24.append(torch.expm1(p[:, :24].clamp(min=0)).sum(1).cpu().numpy())
            part[f'{name}_s4'], part[f'{name}_s24'], part[f'{name}_u24'] = map(np.concatenate, (s4, s24, u24))
            part['step'] = origins
        part['truth4'] = np.stack([Y[t:t + 4].sum(0) for t in part['step']])
        part['truth24'] = np.stack([Y[t:t + 24].sum(0) for t in part['step']])
        part['seat'] = np.full(len(part['step']), seat)
        parts.append(part)
    return finish(job, parts, start)


def finish(job, parts, start):
    arrays = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    arrays['episode'] = np.full(len(arrays['step']), job['seed'])
    arrays['held_out'] = np.full(len(arrays['step']), bool(job.get('held_out')))
    if ARGS['trace_dir']:
        np.savez_compressed(Path(ARGS['trace_dir']) / f"cal_{job['seed']}.npz", **arrays)
    return {'tag': job['tag'], 'seed': job['seed'], 'a_seat': job['a_seat'], 'statuses': ['DONE', 'DONE'],
            'errors': [], 'rewards': [0, 0], 'origins': int(len(arrays['step'])),
            'held_out': bool(job.get('held_out')), 'seconds': round(time.time() - start, 1)}


def error_row(job, error):
    return {'tag': job['tag'], 'seed': job['seed'], 'a_seat': job['a_seat'], 'statuses': None,
            'errors': [error], 'rewards': None}



def init(args):
    global ARGS
    ARGS = args
    sys.path[:0] = [str(Path(args['root']) / 'code')]


def safe_score(job):
    try:
        return score(job)
    except Exception as exc:  # noqa: BLE001 -- reported in the row; the runner retries it
        return error_row(job, repr(exc)[:500])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', required=True)
    ap.add_argument('--root', default='.')
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=os.cpu_count() or 1)
    ap.add_argument('--trace-dir', default=None)
    ap.add_argument('--batch', type=int, default=256, help='forecasts per GPU batch (T4: 256 = 3 GB)')
    a, _ = ap.parse_known_args()
    manifest = json.loads(Path(a.jobs).read_text())
    jobs = manifest['jobs'] if isinstance(manifest, dict) else manifest
    settings = json.loads((Path(a.root) / 'jobs.json').read_text())
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get('statuses') and not r.get('errors'):
                done.add(r['seed'])
    todo = [j for j in jobs if j['seed'] not in done]
    if a.trace_dir:
        os.makedirs(a.trace_dir, exist_ok=True)
    args = {'root': str(Path(a.root).resolve()), 'models': settings['models'], 'stride': settings['stride'],
            'trace_dir': str(Path(a.trace_dir).resolve()) if a.trace_dir else None}
    init(args)
    gpu = gpu_ready()
    print(f'calibration: {len(todo)} episodes, {a.workers} workers, stride {args["stride"]}, '
          f'models {args["models"]}, {"GPU (torch)" if gpu else "CPU (numpy)"}', flush=True)
    with Pool(a.workers, initializer=init, initargs=(args,)) as pool, open(a.out, 'a') as out:
        if gpu:
            models = gpu_models(args, a.batch)
            rows = (gpu_score(job, seats, models, a.batch) if seats else error_row(job, err)
                    for job, seats, err in pool.imap_unordered(extract, todo))
        else:
            rows = pool.imap_unordered(safe_score, todo)
        for i, row in enumerate(rows, 1):
            out.write(json.dumps(row) + '\n')
            out.flush()
            print(f"{i}/{len(todo)} episode {row['seed']} {row.get('seconds')} s {row['errors'] or ''}", flush=True)
    print('ARENA_DONE', flush=True)


if __name__ == '__main__':
    main()
