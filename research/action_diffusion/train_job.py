"""The training job on the GPU machine: replay days -> shards -> training, overlapped.

The Kaggle notebook built by make_kaggle_kernel.py runs it on GPU T4 x2.

    python3 train_job.py --root /tmp/ad --gpus 2 --days all --start-days 4 --val-days 1 \
        --round-hours 2 --deadline-hours 11.3 --init-from best.pt -- --batch-size 64 ...

Days are taken newest first. Kaggle's published day datasets need no credentials (the API
redirects to a signed storage URL), so the machine downloads each day itself, encodes it with
data.prepare into <root>/days/<day>/ and deletes the zip.
1. The first --start-days days are encoded with every CPU; then training starts.
2. A niced background process (--extract-only, log <root>/extract.log) encodes the other days,
   --background-workers processes, newest first, while training runs.
3. Training runs in rounds of --round-hours. Before each round <root>/current/manifest.json is
   rebuilt from the finished days, so each round trains on more days. Validation stays fixed:
   the held-out episodes of the newest --val-days days (the other days' held-out episodes
   train). The first round starts from --init-from with --eval-on-start (step 0 = the
   starting weights; given several checkpoints, train.py scores each on the validation games
   and starts from the best); later rounds, and a replacement VM that got the driver's
   last.pt, resume from <root>/run/last.pt.
4. The job ends when train.py reports its schedule complete (<root>/run/complete) or train.py
   fails. Stage lines are `JOB_STAGE ...`; the last line is `JOB_EXIT=<code>`.
Rerunning the same command on the same VM skips the days already encoded, and so does a day
put into <root>/days/<day>/ beforehand: the training notebook unpacks the days its data
notebooks encoded there (make_kaggle_kernel.py --data-parts), so with --start-days covering
every day it trains on all of them from the first step.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASET = 'https://www.kaggle.com/api/v1/datasets/download/kaggle/{slug}'


def log(message):
    print(f'{time.strftime("%H:%M:%S")} {message}', flush=True)


def fetch(url, path, tries=5):
    """Download url to path through a .part file; True when the result is a readable zip."""
    part = path.with_suffix('.part')
    for attempt in range(1, tries + 1):
        code = subprocess.run(['curl', '-sSfL', '--retry', '3', '--max-time', '1800', '-o', str(part), url]).returncode
        if code == 0 and zipfile.is_zipfile(part):
            part.replace(path)
            return True
        log(f'download {path.name}: attempt {attempt} failed (curl exit {code})')
        time.sleep(10 * attempt)
    return False


def resolve_days(spec, root):
    """Newest first: a comma list of YYYY-MM-DD, last:N, or all (from the episodes index)."""
    if spec.startswith('last:') or spec == 'all':
        index = root / 'index.zip'
        if not index.exists() and not fetch(DATASET.format(slug='kaggriculture-episodes-index'), index):
            raise RuntimeError('cannot download the kaggriculture-episodes-index dataset')
        with zipfile.ZipFile(index) as z:
            rows = list(csv.DictReader(io.TextIOWrapper(z.open('manifest.csv'), encoding='utf-8')))
        days = sorted((r['date'] for r in rows), reverse=True)
        return days if spec == 'all' else days[:int(spec.split(':', 1)[1])]
    return sorted((d.strip() for d in spec.split(',') if d.strip()), reverse=True)


MIN_FREE_GB = 6.0   # encoding stops before the disk fills (training and checkpoints need room)


def train_command(gpus, arguments):
    """train.py under torchrun on `gpus` GPUs (DDP), or a plain process for one."""
    launcher = ([sys.executable, '-m', 'torch.distributed.run', '--standalone', f'--nproc_per_node={gpus}']
                if gpus > 1 else [sys.executable])
    return launcher + [str(HERE / 'train.py'), *arguments]


def extract_day(root, day, workers):
    """Encode one day into <root>/days/<day>; True when it is there (now or before)."""
    out = root / 'days' / day
    if (out / 'manifest.json').exists():
        return True
    free_gb = shutil.disk_usage(root).free / 1e9
    if free_gb < MIN_FREE_GB:
        log(f'JOB_WARN {day}: only {free_gb:.1f} GB free, not encoded')
        return False
    archive = root / 'replays' / f'{day}.zip'
    if not archive.exists() and not fetch(DATASET.format(slug=f'kaggriculture-episodes-{day}'), archive):
        log(f'JOB_WARN {day}: download failed, day skipped')
        return False
    import data
    tmp = root / 'days' / f'{day}.tmp'
    shutil.rmtree(tmp, ignore_errors=True)
    start = time.time()
    manifest = data.prepare([archive], tmp, workers=workers, progress=lambda done, total, s: (
        log(f'{day}: {done}/{total} episodes in {s:.0f}s') if done % 200 == 0 or done == total else None))
    tmp.rename(out)
    archive.unlink()
    counts = manifest['counts']
    skipped = {k[len('skipped_'):]: v for k, v in manifest['quantization'].items() if k.startswith('skipped_')}
    teams = (manifest['sources'][0].get('teams') or []) if manifest['sources'] else []
    best = f', best {teams[0]["team"]} {teams[0]["rating"]}' if teams else ''
    log(f'JOB_DAY {day}: {counts["train"]["episodes"]} train + {counts["val"]["episodes"]} val episodes, '
        f'errors {manifest["errors"]}, skipped commands {skipped}, {len(teams)} rated teams{best}, '
        f'{time.time() - start:.0f}s')
    return True


def flag_values(arguments, names):
    """The flags `names` of a train.py argument list, each with its value."""
    return [x for i, a in enumerate(arguments[:-1]) if a in names for x in (a, arguments[i + 1])]


def model_flags(arguments):
    """The flags of a train.py argument list that shape the model and its data (seat weights,
    conditions), which the self-test runs with too."""
    return flag_values(arguments, ('--rating-halving', '--weight-floor', '--margin-doubling', '--recency-halving',
                                   '--condition-dropout', '--condition-drop-all')) + \
        (['--condition'] if '--condition' in arguments else [])


def selftest(root, day, workers, init_from, gpus, weighting=()):
    """Minutes on 12 episodes before hours of work: parallel prepare, a GPU run from init_from
    (a list of checkpoints) with evaluation and checkpoints (DDP when gpus > 1), then a resume on
    the time-based schedule to completion. `weighting`: the run's model_flags."""
    log('JOB_STAGE self-test')
    archive = root / 'replays' / f'{day}.zip'
    if not archive.exists() and not fetch(DATASET.format(slug=f'kaggriculture-episodes-{day}'), archive):
        raise RuntimeError(f'self-test: cannot download {day}')
    import data
    test = root / 'selftest'
    shutil.rmtree(test, ignore_errors=True)
    manifest = data.prepare([archive], test / 'data', max_episodes_per_source=12, val_fraction=0.5, workers=workers)
    counts = manifest['counts']
    if not counts['train']['episodes'] or not counts['val']['episodes']:
        raise RuntimeError(f'self-test: prepare gave {counts}, errors {manifest["errors"]}')
    base = ['--data', str(test / 'data'), '--out', str(test / 'run'),
            '--batch-size', '16', '--workers', '1', '--val-batches', '3', '--buffer-files', '4', *weighting]
    legs = ((['--init-from', *init_from] if init_from else []) + ['--eval-on-start', '--steps', '20', '--eval-every', '10'],
            ['--resume', str(test / 'run' / 'last.pt'), '--schedule-hours', '0.01', '--eval-every', '100'])
    for leg in legs:
        code = subprocess.run(train_command(gpus, base + leg), cwd=HERE).returncode
        if code:
            raise RuntimeError(f'self-test: train.py {" ".join(leg)} exited with {code}')
    scores = json.loads((test / 'run' / 'scores.json').read_text())
    if not (test / 'run' / 'complete').exists() or not (test / 'run' / 'best.pt').exists() or scores['step'] < 20:
        raise RuntimeError(f'self-test: unexpected result {scores}')
    shutil.rmtree(test)
    (root / 'selftest.ok').write_text(json.dumps(scores) + '\n')
    log(f'JOB_STAGE self-test passed: prepare {counts}; resumed run reached step {scores["step"]}, '
        f'first-action command acc {scores["first_active_command_accuracy"]:.3f}')


def combine(root, days, val_days):
    """<root>/current/manifest.json over the encoded days: the held-out episodes of val_days
    validate, every other episode trains (the other days' held-out episodes too, unless the same
    game is a validation game). Returns (days, training files, validation files)."""
    manifests = {}
    for day in days:
        path = root / 'days' / day / 'manifest.json'
        if path.exists():
            manifests[day] = json.loads(path.read_text())
    if not manifests:
        raise RuntimeError('no day is encoded yet')
    used = list(manifests)
    held_out = {f['episode_id'] for day in val_days if day in manifests
                for f in manifests[day]['files'] if f['split'] == 'val'}
    files = []
    for day, manifest in manifests.items():
        for f in manifest['files']:
            if f['split'] == 'val' and day not in val_days:
                if f['episode_id'] in held_out:
                    continue
                f = dict(f, split='train')
            files.append(dict(f, path=f'../days/{day}/{f["path"]}'))
    header = manifests[used[0]]
    current = root / 'current'
    current.mkdir(exist_ok=True)
    combined = {k: header[k] for k in ('version', 'alignment', 'feature_dim', 'field_sizes', 'codec', 'split')}
    combined.update(files=files, days=used, val_days=list(val_days))
    tmp = current / 'manifest.json.tmp'
    tmp.write_text(json.dumps(combined))
    tmp.replace(current / 'manifest.json')
    return used, sum(f['split'] == 'train' for f in files), sum(f['split'] == 'val' for f in files)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--root', default='/content/ad')
    ap.add_argument('--days', default='all', help='comma list of YYYY-MM-DD, last:N or all')
    ap.add_argument('--start-days', type=int, default=2, help='days encoded before training starts')
    ap.add_argument('--val-days', type=int, default=1, help='newest days whose held-out episodes validate')
    ap.add_argument('--workers', type=int, default=0, help='encoding processes before training (0: every CPU)')
    ap.add_argument('--round-hours', type=float, default=2.0, help='train.py restarts this often to take new days')
    ap.add_argument('--init-from', nargs='+',
                    help='weights of the first round when there is no last.pt (several: train.py takes the best)')
    ap.add_argument('--run-dir', help='where train.py writes checkpoints and metrics (default <root>/run)')
    ap.add_argument('--gpus', type=int, default=1, help='GPUs for train.py (torchrun DDP when > 1)')
    ap.add_argument('--deadline-hours', type=float, default=0.0,
                    help='end within this many hours of the start: sets train.py --schedule-hours to the time '
                         'left after the first days are encoded (less 5%% for restarts) and caps every round')
    ap.add_argument('--margin-hours', type=float, default=0.4, help='kept free before the deadline')
    ap.add_argument('--background-workers', type=int, default=1,
                    help='encoding processes of the niced background encoder (they only take CPU training leaves)')
    ap.add_argument('--extract-only', action='store_true', help='the background encoder: --days, --workers')
    ap.add_argument('train_args', nargs=argparse.REMAINDER, help='after --: passed to train.py')
    args = ap.parse_args()
    root = Path(args.root)
    (root / 'replays').mkdir(parents=True, exist_ok=True)
    (root / 'days').mkdir(exist_ok=True)
    sys.path.insert(0, str(HERE))
    if args.extract_only:
        for day in resolve_days(args.days, root):
            extract_day(root, day, max(1, args.workers))
        log('JOB_EXTRACT_DONE')
        return 0
    started = time.monotonic()
    workers = args.workers or len(os.sched_getaffinity(0))
    run = Path(args.run_dir) if args.run_dir else root / 'run'
    run.mkdir(exist_ok=True)
    extra = [a for a in args.train_args if a != '--']
    code, extractor = 0, None
    try:
        days = resolve_days(args.days, root)
        val_days = days[:args.val_days]
        log(f'JOB_STAGE start: {len(days)} days {days[0]}..{days[-1]}; the first {args.start_days} with '
            f'{workers} workers; validation on the held-out episodes of {val_days}')
        if not (root / 'selftest.ok').exists():
            selftest(root, days[0], workers, args.init_from, args.gpus, model_flags(extra))
        for day in days[:args.start_days]:
            extract_day(root, day, workers)
        rest = days[args.start_days:]
        if rest:
            with open(root / 'extract.log', 'a') as extract_log:
                extractor = subprocess.Popen(
                    ['nice', '-n', '19', sys.executable, str(Path(__file__).resolve()), '--root', str(root),
                     '--days', ','.join(rest), '--extract-only', '--workers', str(args.background_workers)],
                    stdout=extract_log, stderr=subprocess.STDOUT, start_new_session=True)
            log(f'JOB_STAGE background encoding of {len(rest)} more days with {args.background_workers} '
                f'workers (pid {extractor.pid}, extract.log)')
        rounds, schedule = 0, []
        while not (run / 'complete').exists():
            round_hours = args.round_hours
            if args.deadline_hours:
                left = args.deadline_hours - args.margin_hours - (time.monotonic() - started) / 3600
                if left < 0.05:
                    log('JOB_STAGE deadline reached')
                    break
                round_hours = min(round_hours, left)
                if not schedule:   # the cosine spans all the training time the deadline leaves
                    schedule = ['--schedule-hours', f'{0.95 * left:.4f}']
            used, n_train, n_val = combine(root, days, val_days)
            rounds += 1
            start = ['--resume', str(run / 'last.pt')] if (run / 'last.pt').exists() else (
                (['--init-from', *args.init_from] if args.init_from else []) + ['--eval-on-start'])
            log(f'JOB_STAGE train round {rounds}: {len(used)} days ({used[-1]}..{used[0]}), '
                f'{n_train} training files, {n_val} validation files, {round_hours:.2f} h')
            command = train_command(args.gpus, ['--data', str(root / 'current'), '--out', str(run), *start,
                                                *schedule, *extra, '--max-hours', f'{round_hours:.4f}'])
            code = subprocess.run(command, cwd=HERE).returncode
            if code:
                log(f'JOB_ERROR train.py exited with {code}')
                break
        else:
            log('JOB_STAGE training schedule complete')
    except Exception as exc:   # the driver reads this log
        log(f'JOB_ERROR {type(exc).__name__}: {exc}')
        code = code or 1
    finally:
        if extractor and extractor.poll() is None:
            extractor.terminate()
    log(f'JOB_EXIT={code}')
    return code


if __name__ == '__main__':
    raise SystemExit(main())
