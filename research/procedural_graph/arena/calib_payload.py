#!/usr/bin/env python3
"""Build a colab_run.py payload that scores predictor checkpoints on games.

    python arena/calib_payload.py --zip replays/kaggriculture-episodes-2026-09-23.zip \
        --model old=research/procedural_graph/hazel_runtime/checkpoint \
        --model new=models/ttm_c256_h96_ft_2026-09-23 --games 400 --stride 1 --eval-id calib0923
    python arena/calib_payload.py --trace 'runs/arena/feedfix/traces/feed15@*@ours' \
        --trace 'runs/arena/model0923/traces/*@both' --model old=... --model new=... --eval-id calibown

Ladder games (--zip, a Kaggle daily replay zip): every episode in the newest model's
val_episodes.json (held out from its training), then a seeded random sample of the rest up to
--games. Our own games (--trace GLOB[@ours|theirs|both], repeatable): arena traces
(`<tag>_seed<seed>_aseat<a>.json.gz`), which the worker replays through the engine; `ours` scores
the candidate's seat (a), `theirs` the opponent's, `both` both (default). Each job is one game;
calib_worker.py (shipped as arena.py) writes one npz of per-origin scores and truth per game,
and calib_fit.py fits calibration.json from them.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import random
import re
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
    ap.add_argument('--zip', help="a Kaggle daily replay zip")
    ap.add_argument('--trace', action='append', default=[], metavar='GLOB[@ours|theirs|both]',
                    help='arena traces to replay and score (repeatable)')
    ap.add_argument('--model', action='append', required=True, metavar='NAME=DIR')
    ap.add_argument('--held-out', help='val_episodes.json (default: that of the last --model)')
    ap.add_argument('--games', type=int, default=400)
    ap.add_argument('--stride', type=int, default=3, help='score every Nth origin (coprime with 24)')
    ap.add_argument('--seed', type=int, default=20260924)
    ap.add_argument('--eval-id', required=True)
    args = ap.parse_args()
    if not args.zip and not args.trace:
        raise SystemExit('give --zip and/or --trace')
    models = dict(m.split('=', 1) for m in args.model)
    held_file = Path(args.held_out or Path(list(models.values())[-1]) / 'val_episodes.json')
    held = set(json.loads(held_file.read_text())) if args.zip else set()
    out = PG / 'runs' / 'arena' / args.eval_id / 'payload'
    if out.exists():
        shutil.rmtree(out)
    (out / 'code').mkdir(parents=True)
    jobs = []
    if args.zip:
        with zipfile.ZipFile(args.zip) as zf:
            episodes = sorted(int(m[:-5]) for m in zf.namelist() if m.endswith('.json') and m[:-5].isdigit())
            chosen = [e for e in episodes if e in held]
            rest = [e for e in episodes if e not in held]
            chosen += random.Random(args.seed).sample(rest, max(0, min(len(rest), args.games - len(chosen))))
            with zipfile.ZipFile(out / 'games.zip', 'w', zipfile.ZIP_DEFLATED) as dst:
                for e in chosen:
                    dst.writestr(f'{e}.json', zf.read(f'{e}.json'))
        jobs += [{'tag': 'cal', 'seed': e, 'a_seat': 0, 'member': f'{e}.json', 'held_out': e in held}
                 for e in chosen]
    if args.trace:
        name_re = re.compile(r'^(?P<tag>.+)_seed(?P<seed>\d+)_aseat(?P<a>[01])\.json\.gz$')
        with zipfile.ZipFile(out / 'traces.zip', 'w', zipfile.ZIP_STORED) as dst:
            for spec in args.trace:
                pattern, _, rule = spec.rpartition('@') if spec.rsplit('@', 1)[-1] in ('ours', 'theirs', 'both') \
                    else (spec, '', 'both')
                files = sorted(f for f in glob.glob(pattern) if name_re.match(Path(f).name))
                if not files:
                    raise SystemExit(f'no traces match {pattern}')
                for f in files:
                    m = name_re.match(Path(f).name)
                    a = int(m['a'])
                    member = f'{len(jobs):05d}_{Path(f).name}'
                    dst.write(f, member)
                    jobs.append({'tag': 'cal', 'seed': len(jobs) + 1, 'a_seat': 0, 'trace': member,
                                 'game_seed': int(m['seed']), 'group': m['tag'], 'held_out': False,
                                 'seats': {'ours': [a], 'theirs': [1 - a], 'both': [0, 1]}[rule]})
    for f in CODE:
        shutil.copy2(f, out / 'code' / f.name)
    for name, d in models.items():
        (out / 'models' / name).mkdir(parents=True)
        for f in MODEL_FILES:
            shutil.copy2(Path(d) / f, out / 'models' / name / f)
    shutil.copy2(HERE / 'calib_worker.py', out / 'arena.py')
    manifest = {'evaluation_id': args.eval_id, 'kind': 'calibration',
                'zip': Path(args.zip).name if args.zip else None, 'traces': args.trace,
                'stride': args.stride, 'models': list(models),
                'model_sources': {n: {'dir': str(d), 'model_sha256': sha(Path(d) / 'model.safetensors')}
                                  for n, d in models.items()},
                'held_out_file': str(held_file) if args.zip else None,
                'held_out_games': sum(j['held_out'] for j in jobs), 'jobs': jobs}
    (out / 'jobs.json').write_text(json.dumps(manifest, indent=1) + '\n')
    files = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*')) if p.is_file()}
    (out / 'files.json').write_text(json.dumps(files, indent=0) + '\n')
    groups = {}
    for j in jobs:
        groups[j.get('group', 'ladder')] = groups.get(j.get('group', 'ladder'), 0) + 1
    print(json.dumps({'payload': str(out), 'jobs': len(jobs), 'held_out': manifest['held_out_games'], 'groups': groups,
                      'bytes': sum(p.stat().st_size for p in out.rglob('*') if p.is_file())}, indent=1))


if __name__ == '__main__':
    main()
