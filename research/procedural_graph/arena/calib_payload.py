#!/usr/bin/env python3
"""Build a colab_run.py payload that scores predictor checkpoints on ladder replays.

    python arena/calib_payload.py --zip replays/kaggriculture-episodes-2026-09-23.zip \
        --model old=research/procedural_graph/hazel_runtime/checkpoint \
        --model new=models/ttm_c256_h96_ft_2026-09-23 --games 400 --stride 3 --eval-id calib0923

Games: every episode in the newest model's val_episodes.json (held out from its training), then
a seeded random sample of the rest up to --games. Each job is one episode; calib_worker.py
(shipped as arena.py) writes one npz of per-origin scores and truth per episode, and
calib_fit.py fits calibration.json from them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PG = HERE.parent
REPO = PG.parents[1]
CODE = [REPO / 'research/opponent_model/extract.py', REPO / 'research/opponent_model/features.py',
        REPO / 'research/opponent_model/mechanics.py', PG / 'hazel_runtime/kagg_ttm_numpy.py']
MODEL_FILES = ('config.json', 'labels.json', 'model.safetensors', 'scaler.npz')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zip', required=True, help="a Kaggle daily replay zip")
    ap.add_argument('--model', action='append', required=True, metavar='NAME=DIR')
    ap.add_argument('--held-out', help='val_episodes.json (default: that of the last --model)')
    ap.add_argument('--games', type=int, default=400)
    ap.add_argument('--stride', type=int, default=3, help='score every Nth origin (coprime with 24)')
    ap.add_argument('--seed', type=int, default=20260924)
    ap.add_argument('--eval-id', required=True)
    args = ap.parse_args()
    models = dict(m.split('=', 1) for m in args.model)
    held_file = Path(args.held_out or Path(list(models.values())[-1]) / 'val_episodes.json')
    held = set(json.loads(held_file.read_text()))
    out = PG / 'runs' / 'arena' / args.eval_id / 'payload'
    if out.exists():
        shutil.rmtree(out)
    (out / 'code').mkdir(parents=True)
    with zipfile.ZipFile(args.zip) as zf:
        episodes = sorted(int(m[:-5]) for m in zf.namelist() if m.endswith('.json') and m[:-5].isdigit())
        chosen = [e for e in episodes if e in held]
        rest = [e for e in episodes if e not in held]
        chosen += random.Random(args.seed).sample(rest, max(0, min(len(rest), args.games - len(chosen))))
        with zipfile.ZipFile(out / 'games.zip', 'w', zipfile.ZIP_DEFLATED) as dst:
            for e in chosen:
                dst.writestr(f'{e}.json', zf.read(f'{e}.json'))
    for f in CODE:
        shutil.copy2(f, out / 'code' / f.name)
    for name, d in models.items():
        (out / 'models' / name).mkdir(parents=True)
        for f in MODEL_FILES:
            shutil.copy2(Path(d) / f, out / 'models' / name / f)
    shutil.copy2(HERE / 'calib_worker.py', out / 'arena.py')
    jobs = [{'tag': 'cal', 'seed': e, 'a_seat': 0, 'member': f'{e}.json', 'held_out': e in held} for e in chosen]
    manifest = {'evaluation_id': args.eval_id, 'kind': 'calibration', 'zip': Path(args.zip).name,
                'stride': args.stride, 'models': list(models),
                'model_sources': {n: {'dir': str(d), 'model_sha256': sha(Path(d) / 'model.safetensors')}
                                  for n, d in models.items()},
                'held_out_file': str(held_file), 'held_out_games': sum(j['held_out'] for j in jobs), 'jobs': jobs}
    (out / 'jobs.json').write_text(json.dumps(manifest, indent=1) + '\n')
    files = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*')) if p.is_file()}
    (out / 'files.json').write_text(json.dumps(files, indent=0) + '\n')
    print(json.dumps({'payload': str(out), 'jobs': len(jobs), 'held_out': manifest['held_out_games'],
                      'bytes': sum(p.stat().st_size for p in out.rglob('*') if p.is_file())}, indent=1))


if __name__ == '__main__':
    main()
