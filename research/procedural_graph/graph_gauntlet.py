"""Paired gauntlet for graph candidates, played with the arena harness.

A candidate plays the same jobs as the incumbent it would replace: every opponent
bundle x the run's seeds (its seat alternates with the seed index: the engine is
seat-symmetric and the agents deterministic, so one game per seed), plus a
head-to-head block against the incumbent itself. Per job, the candidate's cash
margin is compared with the incumbent's margin on the same job; the incumbent's
head-to-head baseline is its own mirror, margin 0 (measured exact). An exact sign
test over the jobs whose result changed, plus a positive mean change, decides
promotion. Games run in process-isolated agents (arena/arena.py), locally or on a
remote box over ssh; everything is written under the run directory.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import shlex
import shutil
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from arena import payload  # noqa: E402
from arena.report import sign_test  # noqa: E402
from pool_upgrade_state import atomic_json, graph_fingerprint  # noqa: E402

OPPONENTS = ('hazel', 'copper', 'orchard', 'mohui', 'mohui13')
EVOLUTION_SALT = 20260925
VALIDATION_SALT = 20260924  # arena validation rounds (arena/payload.py): never used for selection
HEAD_TO_HEAD = 'incumbent'
ENGINE_VERSION = '1.32.7'


def seed_list(n, salt=EVOLUTION_SALT, held_out_salt=VALIDATION_SALT, held_out=200):
    """Evolution seeds, disjoint from the first `held_out` arena validation seeds."""
    excluded = set(random.Random(held_out_salt).sample(range(10_000_000, 2_000_000_000), held_out))
    rng, seeds = random.Random(salt), []
    while len(seeds) < n:
        seed = rng.randrange(10_000_000, 2_000_000_000)
        if seed not in excluded and seed not in seeds:
            seeds.append(seed)
    return seeds


def _tree_digest(paths):
    h = hashlib.sha256()
    for path in sorted(paths, key=str):
        h.update(str(path).encode())
        h.update(hashlib.sha256(path.read_bytes()).digest())
    return h.hexdigest()


def job_key(job):
    return f"{job['tag']}|{job['seed']}|{job['a_seat']}"


def margins(rows, candidate):
    """{job key: candidate cash - opponent cash} for clean games; plus error records."""
    out, errors = {}, []
    for row in rows:
        rewards = row.get('rewards')
        if (row.get('errors') or not rewards or None in rewards
                or row.get('statuses') != ['DONE', 'DONE']):
            errors.append({'job': job_key(row), 'errors': row.get('errors'), 'statuses': row.get('statuses')})
            continue
        seat = row['a_seat']
        out[job_key(row)] = rewards[seat] - rewards[1 - seat]
    return out, errors


def compare(candidate, baseline, alpha=0.05):
    """Paired comparison of a candidate evaluation with its incumbent's baseline."""
    diffs = []
    for key, margin in candidate['margins'].items():
        tag = key.split('|')[0]
        if tag == HEAD_TO_HEAD:
            diffs.append((tag, margin))
        elif key in baseline['margins']:
            diffs.append((tag, margin - baseline['margins'][key]))
    wins = sum(d > 0 for _, d in diffs)
    losses = sum(d < 0 for _, d in diffs)
    n = len(diffs)
    mean = sum(d for _, d in diffs) / n if n else 0.0
    se = (math.sqrt(sum((d - mean) ** 2 for _, d in diffs) / (n - 1)) / math.sqrt(n)) if n > 1 else 0.0
    per_tag = {}
    for tag, d in diffs:
        t = per_tag.setdefault(tag, {'wins': 0, 'losses': 0, 'ties': 0, 'sum': 0.0, 'n': 0})
        t['wins' if d > 0 else 'losses' if d < 0 else 'ties'] += 1
        t['sum'] += d
        t['n'] += 1
    for t in per_tag.values():
        t['mean'] = round(t.pop('sum') / t['n'], 1)
    p = sign_test(wins, losses)
    expected = len(candidate.get('jobs') or []) or n
    valid = not candidate.get('errors') and not candidate.get('fallbacks') and n == expected
    return {'valid': valid, 'jobs': n, 'expected_jobs': expected, 'wins': wins, 'losses': losses,
            'ties': n - wins - losses, 'mean_change': round(mean, 1), 'se': round(se, 1),
            'p': p, 'alpha': alpha, 'per_opponent': per_tag,
            'promote': bool(valid and wins > losses and mean > 0 and p <= alpha)}


class LocalExecutor:
    """Plays a jobs file with arena.py on this machine."""

    def __init__(self, workers, python=None):
        self.workers = workers
        self.python = python or sys.executable

    def describe(self):
        return {'executor': 'local', 'workers': self.workers,
                'engine': metadata.version('kaggle-environments')}

    def run(self, run_dir, name, bundles):
        run_dir = Path(run_dir)
        cmd = [self.python, str(run_dir / 'arena.py'), '--jobs', str(run_dir / 'jobs' / f'{name}.json'),
               '--root', str(run_dir), '--workers', str(self.workers),
               '--out', str(run_dir / 'games' / f'{name}.jsonl')]
        with (run_dir / 'logs' / f'{name}.log').open('a') as log:
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=run_dir).returncode
        if rc != 0:
            raise RuntimeError(f'arena exited with {rc}; see logs/{name}.log')


class SSHExecutor:
    """Plays a jobs file with arena.py on a remote many-core box (e.g. Brev n2d-highcpu-80).

    Bundles and the harness are rsync'd once (unchanged files are skipped), arena runs
    detached so a dropped ssh connection does not kill a gauntlet, and results and log
    come back into the local run directory. No credentials are copied.
    """

    SETUP = ('command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh; '
             'export PATH="$HOME/.local/bin:$PATH"; '
             '[ -x "$HOME/arena/venv/bin/python" ] || uv venv --python 3.12 "$HOME/arena/venv"; '
             f'VIRTUAL_ENV="$HOME/arena/venv" uv pip install -q "kaggle-environments=={ENGINE_VERSION}"')

    def __init__(self, host, remote_dir, workers, python='~/arena/venv/bin/python', poll=30.0):
        self.host, self.remote_dir, self.workers = host, remote_dir.rstrip('/'), workers
        self.python, self.poll = python, poll
        self.ready = False

    def describe(self):
        return {'executor': 'ssh', 'host': self.host, 'workers': self.workers, 'engine': ENGINE_VERSION}

    def _ssh(self, command, check=True, timeout=600):
        return subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', self.host, command],
                              capture_output=True, text=True, check=check, timeout=timeout)

    def _rsync(self, sources, target):
        subprocess.run(['rsync', '-az', '-e', 'ssh -o BatchMode=yes', *map(str, sources),
                        f'{self.host}:{target}'], check=True, timeout=1800)

    def run(self, run_dir, name, bundles):
        run_dir, remote = Path(run_dir), self.remote_dir
        if not self.ready:
            self._ssh(f'bash -lc {shlex.quote(self.SETUP)}', timeout=1800)
            self._ssh(f'mkdir -p {remote}/bundles {remote}/jobs {remote}/games {remote}/logs')
            self.ready = True
        self._rsync([run_dir / 'arena.py', run_dir / 'bundle_agent.py'], f'{remote}/')
        self._rsync([run_dir / 'bundles' / b for b in bundles], f'{remote}/bundles/')
        self._rsync([run_dir / 'jobs' / f'{name}.json'], f'{remote}/jobs/')
        log = f'{remote}/logs/{name}.log'
        # "[j]obs": the probing shell's own command line must not match the pattern.
        alive = f'pgrep -f "[j]obs/{name}.json" >/dev/null && echo yes || true'
        attach = 'yes' in self._ssh(alive, check=False).stdout
        marker = 'ARENA_EXIT_' if attach else f'ARENA_EXIT_{time.time_ns()}'
        if not attach:
            inner = (f'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 {self.python} arena.py '
                     f'--jobs jobs/{name}.json --root . --workers {self.workers} --out games/{name}.jsonl; '
                     f'echo {marker}=$?')
            self._ssh(f'cd {remote} && nohup bash -c {shlex.quote(inner)} >> {log} 2>&1 < /dev/null &')
        # else: an earlier loop process launched this job set and the box is still playing
        # it; wait for that launch (arena output is resumable) rather than start a second.
        while True:
            time.sleep(self.poll)
            if attach and 'yes' in self._ssh(alive, check=False).stdout:
                continue
            probe = self._ssh(f'grep -o "{marker}[0-9]*=[0-9]*" {log} | tail -1', check=False, timeout=120)
            if probe.returncode == 0 and probe.stdout.strip():
                break
            if attach:
                raise RuntimeError(f'remote arena for {name} stopped without an exit marker')
        for remote_file, local in ((f'games/{name}.jsonl', run_dir / 'games' / f'{name}.jsonl'),
                                   (f'logs/{name}.log', run_dir / 'logs' / f'{name}.log')):
            subprocess.run(['scp', '-o', 'BatchMode=yes', '-q', f'{self.host}:{remote}/{remote_file}', str(local)],
                           check=True, timeout=600)
        if not probe.stdout.strip().endswith('=0'):
            raise RuntimeError(f'remote arena ended with {probe.stdout.strip()}; see logs/{name}.log')


class Gauntlet:
    def __init__(self, run_dir, executor, seeds_per_opponent=40, opponents=OPPONENTS, alpha=0.05):
        self.run_dir = Path(run_dir)
        self.executor = executor
        self.opponents = tuple(opponents)
        self.seeds = seed_list(seeds_per_opponent)
        self.alpha = alpha
        self.fingerprint = None

    def prepare(self):
        for sub in ('bundles', 'jobs', 'games', 'logs', 'scores'):
            (self.run_dir / sub).mkdir(parents=True, exist_ok=True)
        for name in self.opponents:
            if not (self.run_dir / 'bundles' / name).exists():
                payload.build_opponent(self.run_dir, name)
        shutil.copy2(HERE / 'arena' / 'arena.py', self.run_dir / 'arena.py')
        shutil.copy2(HERE.parents[1] / 'shinka/evolution/pool_upgrade_bundle_agent.py',
                     self.run_dir / 'bundle_agent.py')
        files = [p for name in self.opponents for p in (self.run_dir / 'bundles' / name).rglob('*') if p.is_file()]
        files += [p for p in (HERE / 'hazel_runtime').rglob('*')
                  if p.is_file() and '__pycache__' not in p.parts]
        files += [HERE / 'agent_graph.py', self.run_dir / 'arena.py', self.run_dir / 'bundle_agent.py']
        self.fingerprint = hashlib.sha256(json.dumps({
            'files': _tree_digest(files), 'seeds': self.seeds, 'opponents': self.opponents,
            'engine': self.executor.describe().get('engine')}, sort_keys=True).encode()).hexdigest()
        return self.fingerprint

    def bundle(self, graph):
        """Bundle directory name for a graph (content-addressed, built once)."""
        name = 'g_' + graph_fingerprint(graph)[:16]
        dst = self.run_dir / 'bundles' / name
        if not dst.exists():
            tmp = self.run_dir / 'bundles' / f'.{name}.tmp'
            shutil.rmtree(tmp, ignore_errors=True)
            payload.write_graph_bundle(tmp, graph, name)
            tmp.rename(dst)
        return name

    def jobs(self, candidate, incumbent=None):
        jobs = [{'tag': o, 'a': f'bundles/{candidate}', 'b': f'bundles/{o}', 'seed': s, 'a_seat': i % 2}
                for o in self.opponents for i, s in enumerate(self.seeds)]
        if incumbent:
            jobs += [{'tag': HEAD_TO_HEAD, 'a': f'bundles/{candidate}', 'b': f'bundles/{incumbent}',
                      'seed': s, 'a_seat': i % 2} for i, s in enumerate(self.seeds)]
        return jobs

    def _play(self, name, jobs, bundles, retries=1):
        atomic_json(self.run_dir / 'jobs' / f'{name}.json', {'evaluation_id': name, 'jobs': jobs})
        for attempt in range(retries + 1):
            self.executor.run(self.run_dir, name, bundles)
            rows = self._rows(name)
            _, errors = margins(rows, name)
            if not errors:
                break
        return rows

    def _rows(self, name):
        path = self.run_dir / 'games' / f'{name}.jsonl'
        latest = {}
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    latest[job_key(row)] = row
        return list(latest.values())

    def fallbacks(self, name, bundle):
        """Fallback lines of one graph in a job set's log (both seats share the harness stderr)."""
        log = self.run_dir / 'logs' / f'{name}.log'
        if not log.exists():
            return 0
        return sum(1 for line in log.read_text(errors='replace').splitlines()
                   if 'graph fallback' in line and f'[{bundle}]' in line)

    def evaluate(self, graph, incumbent_graph=None):
        """Play the pool jobs (+ head-to-head vs the incumbent) for a graph."""
        if self.fingerprint is None:
            raise RuntimeError('prepare() first')
        candidate = self.bundle(graph)
        incumbent = self.bundle(incumbent_graph) if incumbent_graph is not None else None
        if incumbent == candidate:
            raise ValueError('candidate and incumbent are the same graph')
        name = f'{candidate}_vs_{incumbent}' if incumbent else candidate
        jobs = self.jobs(candidate, incumbent)
        bundles = sorted({candidate, *self.opponents, *([incumbent] if incumbent else [])})
        rows = self._play(name, jobs, bundles)
        wanted = {job_key(j) for j in jobs}
        found, errors = margins([r for r in rows if job_key(r) in wanted], candidate)
        missing = sorted(wanted - set(found) - {e['job'] for e in errors})
        result = {'bundle': candidate, 'incumbent': incumbent, 'fingerprint': self.fingerprint,
                  'jobs': sorted(wanted), 'margins': found, 'errors': errors + [{'job': m, 'errors': 'missing'} for m in missing],
                  'fallbacks': self.fallbacks(name, candidate)}
        pool = {k: v for k, v in found.items() if not k.startswith(HEAD_TO_HEAD + '|')}
        if not result['errors'] and not result['fallbacks']:
            self._store(candidate, pool)
        return result

    def _store(self, bundle, pool_margins):
        atomic_json(self.run_dir / 'scores' / f'{bundle}.json',
                    {'fingerprint': self.fingerprint, 'margins': pool_margins})

    def baseline(self, graph):
        """Pool-job margins of a graph (cached per evaluation fingerprint)."""
        bundle = self.bundle(graph)
        path = self.run_dir / 'scores' / f'{bundle}.json'
        if path.exists():
            cached = json.loads(path.read_text())
            if cached.get('fingerprint') == self.fingerprint:
                return {'bundle': bundle, 'margins': cached['margins']}
        result = self.evaluate(graph)
        if result['errors'] or result['fallbacks']:
            raise RuntimeError(f'baseline evaluation of {bundle} failed: '
                               f"{len(result['errors'])} errors, {result['fallbacks']} fallbacks")
        return {'bundle': bundle, 'margins': result['margins']}

    def compare(self, candidate, baseline):
        return compare(candidate, baseline, self.alpha)

    @staticmethod
    def summary(margins_by_job):
        """Per-opponent W-L-T and mean margin of one evaluation (for prompts)."""
        out = {}
        for key, m in margins_by_job.items():
            tag = key.split('|')[0]
            t = out.setdefault(tag, {'wins': 0, 'losses': 0, 'ties': 0, 'n': 0, 'sum': 0.0})
            t['wins' if m > 0 else 'losses' if m < 0 else 'ties'] += 1
            t['n'] += 1
            t['sum'] += m
        for t in out.values():
            t['mean_margin'] = round(t.pop('sum') / t['n'], 1)
        return out
