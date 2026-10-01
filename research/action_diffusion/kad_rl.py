#!/usr/bin/env python3
"""Expert iteration for KAD-HP-1 (policy.py) on one Colab VM with a GPU or a TPU.

    python3 kad_rl.py pack --model <weights .pt> --out <dir> [--set key=JSON ...]   # the VM's bootstrap
    python3 kad_rl.py run --root /content/kadrl [--smoke]                           # the loop, on the VM

research/procedural_graph/arena/kad_rl_colab.py makes the VM, uploads the bootstrap and runs the loop
there (AGENTS.md section 3.1 rules). The bootstrap (kadrl.tar.gz) holds this code, an arena payload
(arena.py, bundle_agent.py, kad_arena.py, the ladder opponents, the model player as agent/), the
starting model as payload/models/m0.pt, the games' pinned requirements and config.json (DEFAULTS,
changed with --set).

The model plays as `play` says (play.PolicyPlayer settings): by default a draw from every head at
temperature 0.7 with a fixed sample seed and no fit_seeds. Greedy play (every head taking its most likely
choice) was the first run's setting; on 2026-09-29 it lost $50.7k a game against the same model at 0.7
(144 paired games), and fit_seeds cost greedy play $11.5k a game (280 paired games).

The loop keeps its state in <root>/state.json, and every step can be repeated after a crash or on a
new VM that got the pulled state:
0. Baseline: the starting model plays the evaluation games (eval_opponents x eval_seeds x both seats;
   the seeds are arena/payload.py's salted validation seeds, so the first ones are the 09-28
   tournament's), plus one set per baseline_checks entry (`play` with that entry's changes), each
   compared with `play` on the same games.
Then iteration i = 1, 2, ...:
1. Training games on fresh seeds: groups of (opponent, seed, seat); in each group the current model
   plays once per entry of `temperatures` (sample seed 1000 i + j), and once greedily with
   greedy_reference. The samples are drawn at 1.0, the model's own distribution: a fine-tune on games
   drawn at 0.7 would also teach the sharper 0.7 distribution, and every iteration would move the play
   further toward greedy.
2. Selection, as each group completes: every game is scored by its margin (own minus opponent cash)
   plus farm_weight times the value of our farm (land bought, animals placed, seeds in the ground;
   kad_arena.farm_census), averaged over farm_days, so of two games with the same margin the one that
   built the bigger farm wins. The best-scoring sampled game is kept when it scores above the group's
   reference: the mean of its sampled games, or its greedy game with greedy_reference. Its seat is
   encoded (data.episode_arrays) into <root>/own as a training file labelled with the model's prompt
   (a win by its prompt team on its prompt day, at the prompt's excess margin), so the prompted policy
   moves toward it. Kept games accumulate over the iterations.
3. Fine-tune (policy_train.py; --tpu on a TPU VM) from the current model on all kept games plus
   replay_ratio times as many replay seats (winning seats of each day's top replay_top_rank teams,
   from Kaggle's public daily datasets, encoded on the VM), validated on the held-out games of val_day
   as the model always was, with --keep-conditions. A validation loss that ends above
   guard_val_ratio times its start rejects the candidate before it plays.
4. Evaluation: the candidate plays the evaluation games as `play` says (the same sample seed as the
   model it would replace) and replaces the current model when its mean margin over the same games
   (opponent, seed, seat) is higher. The farm bonus only steers the selection: acceptance is by margin.
The loop ends after `iterations`, after stop_after_rejections rejections in a row, or at a STOP file
in the root (after the current phase); a PAUSE file ends it too, with exit 75 (the runner moves the
loop to a new VM). Lines `KAD_RL <event> {json}` report progress; the last is `KAD_RL_EXIT=<code>`.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import random
import shutil
import signal
import subprocess
import sys
import tarfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EXIT_PAUSED = 75
CODE_FILES = ('data.py', 'model.py', 'policy.py', 'play.py', 'policy_train.py', 'train.py', 'kad_rl.py', 'kad_arena.py',
              'arena_main.py')
AGENT_FILES = {'play.py': 'play.py', 'model.py': 'model.py', 'policy.py': 'policy.py', 'data.py': 'data.py',
               'arena_main.py': 'main.py'}
DATASET = 'https://www.kaggle.com/api/v1/datasets/download/kaggle/kaggriculture-episodes-{day}'
VENV = Path('/content/kadrl_venv')   # the games' Python, shared by the roots of one VM

DEFAULTS = dict(
    device='cuda',            # where the model plays in games: cuda (a GPU VM) or cpu (a TPU VM)
    accelerator='gpu',        # what fine-tunes: gpu or tpu
    workers=0,                # games at once; 0: CPUs - 1 with cuda, CPUs - 2 with cpu
    eval_opponents=['tetsutani_demand_0927', 'tetsutani_demand', 'haideptry_2965_0926', 'abo_v57', 'robust_economy',
                    'pilkwang_sep', 'leoprovorov_forecast'],
    eval_seeds=20, eval_salt=20260924,   # 20: the 09-28 tournament's seeds
    train_opponents=['tetsutani_demand_0927', 'tetsutani_demand', 'haideptry_2965_0926', 'abo_v57', 'abo_v43',
                     'haideptry_shepherd'],
    train_seeds=12, train_salt=20260929,
    play=dict(temperature=0.0, sample_seed=7, fit_seeds=False),   # evaluation games (greedy best)
    temperatures=[0.7] * 3,   # 3 exploratory samples per match group at T=0.7
    greedy_reference=False,   # sample mean or baseline as reference
    farm_weight=0.0, farm_days=[10, 20],   # score by economic margin
    select_any=False,         # keep only genuine positive-advantage games
    iterations=4, stop_after_rejections=2,
    baseline_checks={},       # {tag suffix: play changes}, e.g. {"fit": {"fit_seeds": true}}
    replay_days=['2026-09-24', '2026-09-25'], val_day='2026-09-25',
    replay_ratio=2.0, replay_top_rank=30, replay_max_episodes=0,
    finetune=dict(steps=3000, lr=3e-5, warmup=100, batch_size=64, stride=1, val_stride=23, loader_workers=8,
                  buffer_files=16, max_hours=1.5),
    guard_val_ratio=1.25,
    rpc_timeout=60, startup_timeout=300,
)
SMOKE = dict(eval_opponents=['tetsutani_demand_0927'], eval_seeds=1, train_opponents=['tetsutani_demand_0927'],
             train_seeds=1, temperatures=[1.0, 1.0], select_any=True, iterations=1, replay_days=['2026-09-25'],
             replay_max_episodes=40, finetune=dict(DEFAULTS['finetune'], steps=20, warmup=5, max_hours=0.3))


class Paused(Exception):
    pass


class Stopped(Exception):
    pass


def log(message):
    print(f'{time.strftime("%H:%M:%S")} {message}', flush=True)


def event(name, **values):
    print(f'KAD_RL {name} ' + json.dumps(values, separators=(',', ':'), default=str), flush=True)


def read_jsonl(path):
    rows = []
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:   # a line still being written
                pass
    return rows


def append_jsonl(path, row):
    with open(path, 'a') as fh:
        fh.write(json.dumps(row, separators=(',', ':')) + '\n')


def salted_seeds(salt, n):
    """arena/payload.py's seeds: the first n of one salted draw are the first n of any longer one."""
    return random.Random(salt).sample(range(10_000_000, 2_000_000_000), n)


def margin(row):
    rewards = row['rewards']
    return rewards[row['a_seat']] - rewards[1 - row['a_seat']]


def finished_rows(path):
    """The games of a results file that ended without errors."""
    return [r for r in read_jsonl(path) if r.get('statuses') and not r.get('errors') and r.get('rewards')
            and None not in r['rewards']]


def farm_value(row, days, seat=None):
    """The mean value of a seat's farm (default: ours) on `days`, from the game's census
    (kad_arena.farm_census); None when the row has no census of every one of those days."""
    seat = row['a_seat'] if seat is None else seat
    farm = (row.get('farm') or {}).get(str(seat)) or {}
    values = [farm[str(d)]['value'] for d in days if str(d) in farm]
    return sum(values) / len(values) if days and len(values) == len(days) else None


def stats(rows, days=None):
    margins = [margin(r) for r in rows]
    own = [r['rewards'][r['a_seat']] for r in rows]
    out = dict(games=len(rows), wins=sum(m > 0 for m in margins), losses=sum(m < 0 for m in margins),
               ties=sum(m == 0 for m in margins), mean_margin=round(sum(margins) / max(len(rows), 1)),
               mean_cash=round(sum(own) / max(len(rows), 1)))
    if days:
        farms = [(farm_value(r, days), farm_value(r, days, 1 - r['a_seat'])) for r in rows]
        farms = [f for f in farms if None not in f]
        if farms:
            out.update(mean_farm=round(sum(f[0] for f in farms) / len(farms)),
                       opponent_farm=round(sum(f[1] for f in farms) / len(farms)))
    return out


def kill_tree(pid):
    """SIGTERM, then SIGKILL, to a process and every descendant (arena's workers start sessions of their own)."""
    def children(parent):
        found = []
        for entry in os.listdir('/proc'):
            if entry.isdigit():
                try:
                    fields = open(f'/proc/{entry}/stat').read().rsplit(')', 1)[1].split()
                except OSError:
                    continue
                if int(fields[1]) == parent:
                    found += [int(entry)] + children(int(entry))
        return found
    family = [pid] + children(pid)
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for p in family:
            try:
                os.kill(p, sig)
            except OSError:
                pass
        time.sleep(5)


class Loop:
    def __init__(self, root, smoke=False):
        self.root = Path(root)
        self.config = dict(DEFAULTS, **json.loads((self.root / 'config.json').read_text()))
        if smoke:
            self.config.update(SMOKE)
        self.smoke = smoke
        self.payload = self.root / 'payload'
        self.models = self.payload / 'models'
        self.state_path = self.root / 'state.json'
        self.state = (json.loads(self.state_path.read_text()) if self.state_path.exists() else
                      dict(iteration=0, model='m0', eval_tag=None, rejections=0, history=[]))
        self.python = VENV / 'bin' / 'python'
        sys.path.insert(0, str(self.root / 'code'))

    def save(self):
        tmp = self.state_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(self.state, indent=1) + '\n')
        tmp.replace(self.state_path)

    def check(self):
        if (self.root / 'STOP').exists():
            raise Stopped()
        if (self.root / 'PAUSE').exists():
            raise Paused()

    # --- the machine ---------------------------------------------------------------------------------------
    def setup(self):
        import torch
        c = self.config
        cuda = torch.cuda.is_available()
        event('machine', python=sys.version.split()[0], torch=torch.__version__, cpus=os.cpu_count(), cuda=cuda,
              gpu=torch.cuda.get_device_name(0) if cuda else None, tpu=os.environ.get('TPU_ACCELERATOR_TYPE'),
              free_gb=round(shutil.disk_usage(self.root).free / 1e9), workers=self.workers())
        if c['device'].startswith('cuda') and not cuda:
            raise RuntimeError('device cuda, but this VM has no GPU')
        probe = [str(self.python), '-c', 'import sys, torch, kaggle_environments; '
                 'print(sys.version.split()[0], torch.__version__, torch.cuda.is_available())']
        if not self.python.exists() or subprocess.run(probe, capture_output=True).returncode:
            self.build_venv(torch.__version__)
        done = subprocess.run(probe, capture_output=True, text=True)
        event('venv', python=str(self.python), versions=done.stdout.strip(), error=done.stderr[-800:])
        if done.returncode:
            raise RuntimeError('the games venv cannot import torch and kaggle_environments')
        if c['device'].startswith('cuda') and not done.stdout.strip().endswith('True'):
            raise RuntimeError('the games venv has no CUDA')

    def build_venv(self, system_torch):
        """Python 3.12 with the arena's pinned requirements (the environment whose games matched the Brev
        boxes) and torch: the system's version, CPU-only on a TPU VM, for the system's CUDA on a GPU VM."""
        def sh(command):
            done = subprocess.run(command, capture_output=True, text=True)
            if done.returncode:
                raise RuntimeError(f'{" ".join(command[:6])} ... failed: {(done.stdout + done.stderr)[-1500:]}')

        started = time.time()
        shutil.rmtree(VENV, ignore_errors=True)
        sh([sys.executable, '-m', 'pip', 'install', '-q', 'uv'])
        uv = [sys.executable, '-m', 'uv']
        if subprocess.run(uv + ['venv', '--python', '3.12.13', str(VENV)], capture_output=True).returncode:
            sh(uv + ['venv', '--python', '3.12', str(VENV)])
        target = ['--python', str(self.python)]
        sh(uv + ['pip', 'install', *target, '-r', str(self.root / 'requirements.txt')])
        version, _, local = system_torch.partition('+')
        flavour = 'cpu' if self.config['device'] == 'cpu' else (local if local.startswith('cu') else 'cu128')
        index = f'https://download.pytorch.org/whl/{flavour}'
        for spec in (f'torch=={version}', 'torch'):
            done = subprocess.run(uv + ['pip', 'install', *target, spec, '--index-url', index], capture_output=True,
                                  text=True)
            if done.returncode == 0:
                break
            log(f'venv: {spec} from {index} failed: {done.stderr[-400:]}')
        else:
            raise RuntimeError('torch did not install into the games venv')
        log(f'venv built in {time.time() - started:.0f} s')

    def workers(self):
        c = self.config
        return c['workers'] or max(1, (os.cpu_count() or 2) - (1 if c['device'].startswith('cuda') else 2))

    # --- games ---------------------------------------------------------------------------------------------
    def bundle(self, name, model, **settings):
        """A player bundle: the agent files and settings.json (checkpoint relative to the bundle)."""
        dst = self.payload / 'bundles' / name
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(self.payload / 'agent', dst)
        settings = dict(checkpoint=f'../../models/{model}.pt', device=self.config['device'], **settings)
        (dst / 'settings.json').write_text(json.dumps(settings) + '\n')

    def eval_seeds(self):
        return salted_seeds(self.config['eval_salt'], self.config['eval_seeds'])

    def eval_jobs(self, variant):
        return [dict(tag=f'{variant}@{opponent}', a=f'bundles/{variant}', b=f'bundles/{opponent}', seed=seed,
                     a_seat=seat, episode=False, variant=variant, opponent=opponent, role='eval',
                     group=f'{opponent}|{seed}|{seat}')
                for seed in self.eval_seeds() for opponent in self.config['eval_opponents'] for seat in (0, 1)]

    def arena(self, directory, name, jobs, episodes):
        """Start kad_arena.py on `jobs`; it appends to directory/results.jsonl and skips the jobs
        already there (so a repeated phase plays only what is missing)."""
        c = self.config
        directory.mkdir(parents=True, exist_ok=True)
        jobs_file = directory / f'jobs_{name}.json'
        jobs_file.write_text(json.dumps({'jobs': jobs}) + '\n')
        env = {k: v for k, v in os.environ.items() if k != 'KAD_EPISODE_DIR'}
        if episodes:
            env['KAD_EPISODE_DIR'] = str(directory / 'episodes')
        command = [str(self.python), str(self.payload / 'kad_arena.py'), '--jobs', str(jobs_file), '--root',
                   str(self.payload), '--workers', str(self.workers()), '--out', str(directory / 'results.jsonl'),
                   '--rpc-timeout', str(c['rpc_timeout']), '--startup-timeout', str(c['startup_timeout'])]
        log(f'arena {directory.name}/{name}: {len(jobs)} jobs, {self.workers()} at once')
        with open(directory / 'arena.log', 'a') as fh:
            return subprocess.Popen(command, cwd=self.payload, stdout=fh, stderr=subprocess.STDOUT, env=env,
                                    start_new_session=True)

    def wait(self, proc, directory, jobs, on_tick=None):
        """Until the arena exits: on_tick every 30 s, progress every 10 min; STOP or PAUSE end it."""
        keys = {(j['tag'], j['seed'], j['a_seat']) for j in jobs}
        started, reported = time.time(), 0.0
        while proc.poll() is None:
            time.sleep(30)
            if on_tick:
                on_tick()
            try:
                self.check()
            except (Paused, Stopped):
                kill_tree(proc.pid)
                proc.wait()
                if on_tick:
                    on_tick()
                raise
            if time.time() - reported >= 600:
                reported = time.time()
                done = [r for r in finished_rows(directory / 'results.jsonl') if (r['tag'], r['seed'], r['a_seat']) in keys]
                rate = len(done) / max((time.time() - started) / 3600, 1e-6)
                event('progress', phase=directory.name, done=len(done), of=len(keys), per_hour=round(rate))
        done = [r for r in finished_rows(directory / 'results.jsonl') if (r['tag'], r['seed'], r['a_seat']) in keys]
        event('arena_done', phase=directory.name, exit=proc.returncode, done=len(done), of=len(keys),
              hours=round((time.time() - started) / 3600, 2))
        if proc.returncode:
            raise RuntimeError(f'the arena exited with {proc.returncode}')
        return done

    def compare(self, new, old):
        """The candidate's evaluation games against the same games (opponent, seed, seat) of the model it
        would replace."""
        rows = [r for path in sorted(self.root.glob('it*/results.jsonl')) for r in finished_rows(path)
                if r.get('role') == 'eval']
        by = {}
        for r in rows:
            by.setdefault(r['variant'], {})[(r['opponent'], r['seed'], r['a_seat'])] = r
        a, b = by.get(new, {}), by.get(old, {})
        keys = sorted(set(a) & set(b))
        deltas = [margin(a[k]) - margin(b[k]) for k in keys]
        n = len(deltas)
        mean = sum(deltas) / n if n else 0.0
        sd = math.sqrt(sum((d - mean) ** 2 for d in deltas) / (n - 1)) if n > 1 else 0.0
        days = self.config['farm_days']
        return dict(games=n, mean_delta=round(mean), se=round(sd / math.sqrt(n)) if n else None,
                    better=sum(d > 0 for d in deltas), worse=sum(d < 0 for d in deltas),
                    new=stats([a[k] for k in keys], days), old=stats([b[k] for k in keys], days))

    # --- training data --------------------------------------------------------------------------------------
    def prompt(self, model):
        """What the model is asked to play like (its checkpoint's prompt and prompt team)."""
        import torch
        saved = torch.load(self.models / f'{model}.pt', map_location='cpu', weights_only=False)
        conditions = saved.get('conditions') or {}
        prompt = conditions.get('prompt') or {}
        return dict(day=prompt.get('day'), excess_margin=prompt.get('excess_margin') or 0.0,
                    team=conditions.get('prompt_team'))

    def select(self, it, directory, variants, greedy, prompt):
        """Decide every group whose games are all in: keep its best-scoring sampled game when it scores above
        the reference (the greedy game if there is one, else the mean of the sampled games). The score is
        the margin plus farm_weight times our farm value, or the margin alone in a group where a game has
        no farm census."""
        c = self.config
        processed = {e['group'] for e in read_jsonl(directory / 'selected.jsonl')}
        groups = {}
        for r in finished_rows(directory / 'results.jsonl'):
            if r.get('role') == 'train':
                groups.setdefault(r['group'], {})[r['variant']] = r
        samples = [v for v in variants if v != greedy]
        for group, games in groups.items():
            if group in processed or any(v not in games for v in variants):
                continue
            farms = {v: farm_value(g, c['farm_days']) for v, g in games.items()}
            bonus = c['farm_weight'] if None not in farms.values() else 0.0
            scores = {v: margin(g) + bonus * (farms[v] or 0.0) for v, g in games.items()}
            best = max(samples, key=lambda v: scores[v])
            reference = scores[greedy] if greedy else sum(scores[v] for v in samples) / len(samples)
            first = games[samples[0]]
            entry = dict(group=group, iteration=it, opponent=first['opponent'], seed=first['seed'],
                         a_seat=first['a_seat'], margins={v: margin(g) for v, g in games.items()},
                         farms={v: None if f is None else round(f) for v, f in farms.items()},
                         by_margin=max(samples, key=lambda v: margin(games[v])), selected=None)
            gain = round(scores[best] - reference)
            # Only keep games where the sample achieved genuine positive economic advantage
            if (gain > 0 and margin(games[best]) > reference) or c['select_any']:
                try:
                    entry['own'] = self.encode(it, directory, games[best], prompt, gain)
                    entry.update(selected=best, gain=gain)
                except Exception as exc:   # a broken episode costs its group, not the run
                    entry['error'] = repr(exc)[:500]
            append_jsonl(directory / 'selected.jsonl', entry)

    def encode(self, it, directory, row, prompt, gain):
        """One seat of a kept game as a training file in <root>/own, and its record."""
        import numpy as np
        import data
        with gzip.open(directory / 'episodes' / row['episode'], 'rt') as fh:
            doc = json.load(fh)
        seat = row['a_seat']
        X, A = data.episode_arrays(doc)[seat]
        own = self.root / 'own'
        own.mkdir(exist_ok=True)
        name = f"it{it}_{row['opponent']}_{row['seed']}_s{seat}"
        with open(own / f'{name}.npz.tmp', 'wb') as fh:   # a file object: numpy would append .npz to the name
            np.savez_compressed(fh, X=X, A=A, episode_id=np.asarray(name), seat=np.int64(seat))
        os.replace(own / f'{name}.npz.tmp', own / f'{name}.npz')
        rewards = row['rewards']
        real_m = margin(row)
        is_win = 1.0 if rewards[seat] > rewards[1 - seat] else 0.0
        adv_margin = min(1.0, max(-1.0, float(gain) / 10000.0))
        # Honest condition labeling with relative advantage margin for advantage-weighted regression
        append_jsonl(own / 'records.jsonl', dict(
            path=f'own/{name}.npz', episode_id=name, seat=seat, split='train', length=len(X), source=f'kadrl-it{it}',
            member=row['episode'], team=prompt['team'], cash=rewards[seat], opponent_cash=rewards[1 - seat], win=is_win,
            day=prompt['day'], team_rating=0.0, team_rank=1, margin=adv_margin, expected_margin=0.0,
            kadrl=dict(iteration=it, variant=row['variant'], opponent=row['opponent'], game_seed=row['seed'],
                       real_margin=real_m, farm=farm_value(row, self.config['farm_days']), gain=gain)))
        return name

    def purge(self, directory):
        """Training games of undecided groups whose episodes are gone (a new VM): dropped, so they replay."""
        path = directory / 'results.jsonl'
        if not path.exists():
            return
        processed = {e['group'] for e in read_jsonl(directory / 'selected.jsonl')}
        keep, dropped = [], 0
        for line in path.read_text().splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                dropped += 1
                continue
            episode = r.get('episode')
            if (r.get('role') == 'train' and r.get('group') not in processed
                    and not (isinstance(episode, str) and (directory / 'episodes' / episode).exists())):
                dropped += 1
                continue
            keep.append(line)
        if dropped:
            path.write_text(''.join(line + '\n' for line in keep))
            log(f'{directory.name}: {dropped} training games of undecided groups lost their episodes; they replay')

    def replay_days(self):
        """Encode the replay days (Kaggle's public daily datasets need no credentials) into <root>/days."""
        import data
        c = self.config
        for day in c['replay_days']:
            out = self.root / 'days' / day
            if (out / 'manifest.json').exists():
                continue
            archive = self.root / 'replays' / f'{day}.zip'
            archive.parent.mkdir(parents=True, exist_ok=True)
            started = time.time()
            if not archive.exists():
                part = archive.with_suffix('.part')
                code = subprocess.run(['curl', '-sSfL', '--retry', '5', '--max-time', '3600', '-o', str(part),
                                       DATASET.format(day=day)]).returncode
                if code:
                    raise RuntimeError(f'the replays of {day} did not download (curl exit {code})')
                part.replace(archive)
            tmp = out.with_suffix('.tmp')
            shutil.rmtree(tmp, ignore_errors=True)
            manifest = data.prepare([archive], tmp, c['replay_max_episodes'] or None,
                                    workers=max(1, (os.cpu_count() or 2) - 1))
            tmp.rename(out)
            archive.unlink()
            event('replay_day', day=day, counts=manifest['counts'], errors=manifest['errors'],
                  seconds=round(time.time() - started))

    def training_data(self, it):
        """<root>/data/manifest.json: every kept game, replay_ratio times as many replay seats (the winning
        seats of each day's top teams), and val_day's held-out games for validation."""
        c = self.config
        days = {day: json.loads((self.root / 'days' / day / 'manifest.json').read_text()) for day in c['replay_days']}
        held_out = {f['episode_id'] for f in days[c['val_day']]['files'] if f['split'] == 'val'}
        # one record per file (a group decided again after a move to a new VM writes its record again)
        own = list({r['path']: dict(r, path='../' + r['path'])
                    for r in read_jsonl(self.root / 'own' / 'records.jsonl')}.values())
        pool, val = [], []
        for day, manifest in days.items():
            for f in manifest['files']:
                f = dict(f, path=f'../days/{day}/{f["path"]}')
                if day == c['val_day'] and f['split'] == 'val':
                    val.append(f)
                elif f['episode_id'] not in held_out and f.get('win') == 1 \
                        and (f.get('team_rank') or 10 ** 6) <= c['replay_top_rank']:
                    pool.append(dict(f, split='train'))
        replay = random.Random(1000 + it).sample(pool, min(len(pool), round(c['replay_ratio'] * len(own))))
        header = days[c['val_day']]
        manifest = {k: header[k] for k in ('version', 'alignment', 'feature_dim', 'field_sizes', 'codec', 'split')}
        manifest.update(files=own + replay + val, days=list(days), val_days=[c['val_day']])
        (self.root / 'data').mkdir(exist_ok=True)
        (self.root / 'data' / 'manifest.json').write_text(json.dumps(manifest))
        return dict(own=len(own), replay=len(replay), replay_pool=len(pool), val=len(val))

    def finetune(self, it, model):
        """policy_train.py from `model` on <root>/data; the candidate's weights, or a reason to reject it."""
        import torch
        c, f = self.config, self.config['finetune']
        directory = self.root / f'it{it}'
        run = directory / 'ft'
        shutil.rmtree(run, ignore_errors=True)   # short: an interrupted fine-tune starts over
        args = ['--data', str(self.root / 'data'), '--out', str(run), '--init-from', str(self.models / f'{model}.pt'),
                '--keep-conditions', '--condition', '--steps', str(f['steps']), '--batch-size', str(f['batch_size']),
                '--lr', str(f['lr']), '--warmup', str(f['warmup']), '--stride', str(f['stride']),
                '--val-stride', str(f['val_stride']), '--eval-every', str(f['steps']), '--eval-on-start',
                '--workers', str(f['loader_workers']), '--buffer-files', str(f['buffer_files']),
                '--max-hours', str(f['max_hours']), '--save-every-min', '60']
        env = dict(os.environ)
        if c['accelerator'] == 'tpu':
            args.append('--tpu')
            env['PJRT_DEVICE'] = 'TPU'
        started = time.time()
        with open(directory / 'finetune.log', 'w') as fh:
            code = subprocess.run([sys.executable, str(self.root / 'code' / 'policy_train.py'), *args],
                                  cwd=self.root / 'code', stdout=fh, stderr=subprocess.STDOUT, env=env).returncode
        vals = [m for m in read_jsonl(run / 'metrics.jsonl') if m.get('event') == 'validation']
        keys = ('val_loss', 'first_active_command_accuracy', 'job_accuracy', 'order_accuracy', 'step')
        summary = dict(exit=code, minutes=round((time.time() - started) / 60, 1),
                       start={k: vals[0].get(k) for k in keys} if vals else None,
                       end={k: vals[-1].get(k) for k in keys} if len(vals) > 1 else None)
        (directory / 'ft_summary.json').write_text(json.dumps(summary, indent=1) + '\n')
        event('finetune', iteration=it, **summary)
        if code or not summary['end'] or not (run / 'last.pt').exists():
            return None, f'the fine-tune failed (exit {code}; see it{it}/finetune.log)'
        if summary['end']['val_loss'] > c['guard_val_ratio'] * summary['start']['val_loss']:
            return None, (f"validation loss {summary['start']['val_loss']:.3f} -> {summary['end']['val_loss']:.3f}, "
                          f"above {c['guard_val_ratio']} x its start")
        saved = torch.load(run / 'last.pt', map_location='cpu', weights_only=False)
        weights = {k: saved[k] for k in ('config', 'model', 'step', 'metrics', 'conditions', 'policy') if k in saved}
        torch.save(weights, self.models / f'c{it}.pt.tmp')
        os.replace(self.models / f'c{it}.pt.tmp', self.models / f'c{it}.pt')
        (run / 'last.pt').unlink()   # the weights above are what plays; the optimizer is not needed
        return f'c{it}', None

    # --- the loop -------------------------------------------------------------------------------------------
    def baseline(self):
        s, c = self.state, self.config
        directory = self.root / 'it0'
        model = s['model']
        self.bundle(model, model, **c['play'])
        jobs = self.eval_jobs(model)
        for suffix, changes in c['baseline_checks'].items():
            self.bundle(model + suffix, model, **dict(c['play'], **changes))
            jobs += self.eval_jobs(model + suffix)
        self.wait(self.arena(directory, 'eval', jobs, episodes=False), directory, jobs)
        record = dict(iteration=0, model=model, play=c['play'],
                      eval=stats([r for r in finished_rows(directory / 'results.jsonl') if r.get('variant') == model],
                                 c['farm_days']))
        if c['baseline_checks']:   # each check against play: a positive delta means the change plays better
            record['checks'] = {suffix: dict(changes=changes, **self.compare(model + suffix, model))
                                for suffix, changes in c['baseline_checks'].items()}
        event('baseline', **record)
        s['eval_tag'] = model
        s['history'].append(record)
        self.save()

    def variants(self, it):
        """The players of iteration it's training games, {name: bundle settings}, and the greedy one's name
        (None without greedy_reference)."""
        c = self.config
        variants = {f'it{it}s{j}': dict(c['play'], temperature=t, sample_seed=1000 * it + j)
                    for j, t in enumerate(c['temperatures'], 1)}
        greedy = None
        if c['greedy_reference']:
            greedy = f'it{it}g'
            variants[greedy] = dict(c['play'], temperature=0.0)
        return variants, greedy

    def iteration(self, it):
        s, c = self.state, self.config
        directory = self.root / f'it{it}'
        directory.mkdir(exist_ok=True)
        model = s['model']
        record = dict(iteration=it, model=model)
        # 1-2: the training games, each group decided as soon as its games are in
        variants, greedy = self.variants(it)
        if not (directory / 'games.done').exists():
            for name, settings in variants.items():
                self.bundle(name, model, **settings)
            evaluation = set(self.eval_seeds())
            seeds = [x for x in salted_seeds(c['train_salt'] + it, c['train_seeds'] + 10) if x not in evaluation]
            jobs = [dict(tag=f'{v}@{opponent}', a=f'bundles/{v}', b=f'bundles/{opponent}', seed=seed, a_seat=seat,
                         episode=True, variant=v, opponent=opponent, role='train', group=f'{opponent}|{seed}|{seat}')
                    for seed in seeds[:c['train_seeds']] for opponent in c['train_opponents'] for seat in (0, 1)
                    for v in variants]   # group by group, so groups complete (and are decided) early
            self.purge(directory)
            prompt = self.prompt(model)
            proc = self.arena(directory, 'train', jobs, episodes=True)
            self.wait(proc, directory, jobs, on_tick=lambda: self.select(it, directory, list(variants), greedy, prompt))
            self.select(it, directory, list(variants), greedy, prompt)
            (directory / 'games.done').write_text(time.strftime('%Y-%m-%d %H:%M:%S') + '\n')
        decided = read_jsonl(directory / 'selected.jsonl')
        kept = [e for e in decided if e.get('selected')]
        rows = [r for r in finished_rows(directory / 'results.jsonl') if r.get('role') == 'train']
        chosen = {(e['group'], e['selected']) for e in kept}
        days = c['farm_days']
        record.update(groups=len(decided), kept=len(kept),
                      mean_gain=round(sum(e['gain'] for e in kept) / len(kept)) if kept else None,
                      farm_changed=sum(e['selected'] != e.get('by_margin', e['selected']) for e in kept),
                      sampled=stats([r for r in rows if r['variant'] != greedy], days),
                      kept_games=stats([r for r in rows if (r['group'], r['variant']) in chosen], days))
        if greedy:
            record['greedy'] = stats([r for r in rows if r['variant'] == greedy], days)
        by_temperature = {}
        for e in kept:
            t = variants[e['selected']]['temperature']
            by_temperature[str(t)] = by_temperature.get(str(t), 0) + 1
        record['kept_by_temperature'] = by_temperature
        event('selected', **record)
        samples = [v for v in variants if v != greedy]
        if self.smoke and len(samples) > 1 and decided and \
                not any(len({e['margins'].get(v) for v in samples}) > 1 for e in decided):
            # 09-29: the graph player ignored the temperature, so every sample replayed one game and nothing was learned
            raise RuntimeError('exploration is inert: in every group all sampled games have the same margin')
        self.check()
        # 3: the fine-tune
        candidate, reason = f'c{it}', None
        if not (self.models / f'{candidate}.pt').exists():
            if not read_jsonl(self.root / 'own' / 'records.jsonl'):
                candidate, reason = None, 'no game was kept'
            else:
                self.replay_days()
                record['data'] = self.training_data(it)
                event('training_data', iteration=it, **record['data'])
                candidate, reason = self.finetune(it, model)
        if (directory / 'ft_summary.json').exists():
            record['finetune'] = json.loads((directory / 'ft_summary.json').read_text())
        # 4: the evaluation
        if candidate:
            self.check()
            self.bundle(candidate, candidate, **c['play'])
            jobs = self.eval_jobs(candidate)
            self.wait(self.arena(directory, 'eval', jobs, episodes=False), directory, jobs)
            result = self.compare(candidate, s['eval_tag'])
            record['evaluation'] = result
            if result['games'] < 0.9 * len(jobs):
                reason = f"only {result['games']} of {len(jobs)} evaluation games paired"
            elif result['mean_delta'] <= 0:
                reason = f"mean margin {result['mean_delta']:+d} a game against {s['eval_tag']}"
        record['accepted'] = candidate is not None and reason is None
        record['reason'] = reason
        if record['accepted']:
            s.update(model=candidate, eval_tag=candidate, rejections=0)
        else:
            s['rejections'] += 1
            if candidate:
                (self.models / f'{candidate}.pt').unlink(missing_ok=True)
        s['iteration'] = it
        s['history'].append(record)
        self.save()
        event('iteration', **record)
        shutil.rmtree(directory / 'episodes', ignore_errors=True)
        for name in variants:
            shutil.rmtree(self.payload / 'bundles' / name, ignore_errors=True)

    def run(self):
        s, c = self.state, self.config
        self.setup()
        if s['eval_tag'] is None:
            self.check()
            self.baseline()
        while s['iteration'] < c['iterations'] and s['rejections'] < c['stop_after_rejections']:
            self.check()
            self.iteration(s['iteration'] + 1)
        summary = dict(model=s['model'], iterations=s['iteration'], rejections=s['rejections'],
                       history=s['history'])
        (self.root / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
        event('done', model=s['model'], iterations=s['iteration'], rejections=s['rejections'])


def pack(args):
    """The VM's bootstrap: <out>/kadrl.tar.gz (see the module doc). Standard library only."""
    config = dict(DEFAULTS)
    for item in args.set or []:
        key, value = item.split('=', 1)
        if key not in DEFAULTS:
            raise SystemExit(f'unknown setting {key}')
        config[key] = json.loads(value)
    out = Path(args.out)
    stage = out / 'stage'
    shutil.rmtree(stage, ignore_errors=True)
    (stage / 'code').mkdir(parents=True)
    for name in CODE_FILES:
        shutil.copy2(HERE / name, stage / 'code' / name)
    payload = stage / 'payload'
    (payload / 'agent').mkdir(parents=True)
    juniper = REPO / 'shinka/champions/submissions/juniper_knoll'
    if juniper.is_dir():
        shutil.copytree(juniper, payload / 'agent', dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        kad_dst = payload / 'agent' / 'hazel_runtime' / 'kad'
        kad_dst.mkdir(parents=True, exist_ok=True)
        for kf in ('kad_data.py', 'kad_numpy.py', 'kad_torch.py', 'weights.npz'):
            src_f = REPO / 'research/procedural_graph/hazel_runtime/kad' / kf
            if src_f.is_file():
                shutil.copy2(src_f, kad_dst / kf)
        shutil.copy2(REPO / 'research/procedural_graph/hazel_runtime/kad_copilot.py',
                     payload / 'agent' / 'hazel_runtime' / 'kad_copilot.py')
        shutil.copy2(REPO / 'research/procedural_graph/hazel_runtime/graph_runtime.py',
                     payload / 'agent' / 'hazel_runtime' / 'graph_runtime.py')
        shutil.copy2(REPO / 'research/procedural_graph/hazel_runtime/land_plot.py',
                     payload / 'agent' / 'hazel_runtime' / 'land_plot.py')
        shutil.copy2(REPO / 'research/procedural_graph/agent_graph.py', payload / 'agent' / 'agent_graph.py')
        gp = payload / 'agent' / 'policy_graph.json'
        if gp.is_file():
            g = json.loads(gp.read_text())
            turn = g.get('turn', {})
            turn_nodes = turn.get('nodes', [])
            params = {'_KC_EVERY': 2, '_KC_FROM_STEP': 24, '_KC_TO_STEP': 671,
                      '_KC_SELL': True, '_KC_HANDS': True, '_KC_BACKEND': 'torch', '_KC_DEVICE': 'cuda'}
            kn = next((n for n in turn_nodes if n.get('id') == 'kad_copilot'), None)
            if kn is None:
                kn = {
                    'id': 'kad_copilot', 'binding': 'kad_copilot', 'enabled': True,
                    'summary': 'KAD-HP-1 (top-player replay policy) adds confident sales and idle-hand jobs',
                    'parameters': params
                }
                ids = [n['id'] for n in turn_nodes]
                before = next((x for x in ('land_plot', 'tactic', 'sanitize') if x in ids), None)
                if before:
                    idx = ids.index(before)
                    turn_nodes.insert(idx, kn)
                    prev_id = ids[idx - 1]
                    edges = turn.get('edges', [])
                    for e in edges:
                        if e.get('source') == prev_id and e.get('target') == before:
                            e['target'] = 'kad_copilot'
                            break
                    edge = {'source': 'kad_copilot', 'target': before, 'relation': 'NEXT'}
                    e_idx = next(j for j, e in enumerate(edges) if e.get('source') == prev_id and e.get('target') == 'kad_copilot')
                    edges.insert(e_idx + 1, edge)
                else:
                    turn_nodes.append(kn)
                if isinstance(g.get('nodes'), list) and not any(n.get('id') == 'kad_copilot' for n in g['nodes']):
                    g['nodes'].append({
                        'id': 'kad_copilot', 'name': 'KAD copilot',
                        'description': 'KAD-HP-1 policy', 'bindings': ['kad_copilot'],
                        'binding_semantics': 'Executable stage'
                    })
                if isinstance(g.get('edges'), list) and not any(e.get('source') == 'kad_copilot' for e in g['edges']):
                    g['edges'].append({'source': 'kad_copilot', 'target': 'sanitize',
                                      'relation': 'CONFIGURES', 'scope': 'turn'})
            else:
                kn['enabled'] = True
                kn.setdefault('parameters', {}).update(params)

            prov = g.setdefault('provenance', {})
            prov['entrypoint_sha256'] = hashlib.sha256(
                (payload / 'agent' / 'agent_graph.py').read_bytes()
            ).hexdigest()
            hashes = prov.setdefault('runtime_bundle_hashes', {})
            rt = payload / 'agent' / 'hazel_runtime'
            for rel in ('graph_runtime.py', 'kad_copilot.py', 'land_plot.py'):
                if (rt / rel).is_file():
                    hashes[rel] = hashlib.sha256((rt / rel).read_bytes()).hexdigest()
            gp.write_text(json.dumps(g, indent=2) + chr(10))
    else:
        for source, target in AGENT_FILES.items():
            shutil.copy2(HERE / source, payload / 'agent' / target)
    shutil.copy2(REPO / 'research/procedural_graph/arena/arena.py', payload / 'arena.py')
    shutil.copy2(REPO / 'shinka/evolution/pool_upgrade_bundle_agent.py', payload / 'bundle_agent.py')
    shutil.copy2(HERE / 'kad_arena.py', payload / 'kad_arena.py')
    opponents = sorted(set(config['eval_opponents']) | set(config['train_opponents']))
    for name in opponents:
        source = REPO / 'shinka/champions/ladder' / name
        if not (source / 'SOURCE.json').is_file():
            raise SystemExit(f'{name} is not a ladder agent in shinka/champions/ladder')
        shutil.copytree(source, payload / 'bundles' / name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    (payload / 'models').mkdir()
    shutil.copy2(args.model, payload / 'models' / 'm0.pt')
    shutil.copy2(REPO / 'research/procedural_graph/arena/colab_requirements.txt', stage / 'requirements.txt')
    (stage / 'config.json').write_text(json.dumps(config, indent=1) + '\n')
    with tarfile.open(out / 'kadrl.tar.gz', 'w:gz') as tar:
        tar.add(stage, arcname='.')
    shutil.rmtree(stage)
    print(json.dumps(dict(bootstrap=str(out / 'kadrl.tar.gz'), bytes=(out / 'kadrl.tar.gz').stat().st_size,
                          opponents=opponents, config=config), indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = ap.add_subparsers(dest='command', required=True)
    p = sub.add_parser('pack', help='build the bootstrap')
    p.add_argument('--model', required=True, help='the starting weights (a KAD-HP-1 best.pt)')
    p.add_argument('--out', required=True)
    p.add_argument('--set', action='append', metavar='KEY=JSON', help='a DEFAULTS setting, e.g. iterations=3')
    r = sub.add_parser('run', help='the loop, on the VM')
    r.add_argument('--root', default='/content/kadrl')
    r.add_argument('--smoke', action='store_true', help='a few games and a 20-step fine-tune (SMOKE settings)')
    args = ap.parse_args()
    if args.command == 'pack':
        pack(args)
        return 0
    code = 1
    try:
        Loop(args.root, args.smoke).run()
        code = 0
    except Paused:
        log('paused (PAUSE file): the state is saved')
        code = EXIT_PAUSED
    except Stopped:
        log('stopped (STOP file): the state is saved')
        code = 0
    except Exception as exc:
        import traceback
        log(f'KAD_RL_ERROR {type(exc).__name__}: {exc}\n{traceback.format_exc()[-3000:]}')
    print(f'KAD_RL_EXIT={code}', flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
