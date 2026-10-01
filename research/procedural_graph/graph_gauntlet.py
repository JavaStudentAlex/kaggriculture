"""Paired gauntlet for graph candidates, played with the arena harness.

A candidate plays the same jobs as the incumbent it would replace: every opponent
bundle x the run's seeds (its seat alternates with the seed index: the engine is
seat-symmetric and the agents deterministic, so one game per seed), plus a
head-to-head block against the incumbent itself. Per job, the candidate's cash
margin is compared with the incumbent's margin on the same job; the incumbent's
head-to-head baseline is its own mirror, margin 0 (measured exact). A candidate is
promoted when an exact sign test over the jobs whose margin changed, plus a positive mean
change, is significant, or when one over the jobs whose result (win, tie, loss) changed is,
and never when its results got worse on balance (compare). Games run in process-isolated agents (arena/arena.py), locally or on a
remote box over ssh; everything is written under the run directory.

A staged evaluation (`stage_fraction`) plays a fixed share of the jobs first and the rest only
if those games leave the candidate a chance, so edits that change nothing, or that lose, cost
that share instead of the whole gauntlet.
"""
from __future__ import annotations

import errno
import hashlib
import json
import math
import random
import shlex
import shutil
import subprocess
import sys
import tempfile
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
STAGE_SALT = 'first stage'


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


def _bundle_digest(directory):
    """Executable bundle identity, independent of its directory and graph display name."""
    h = hashlib.sha256()
    graph = json.loads((directory / 'policy_graph.json').read_text())
    graph.pop('name', None)
    h.update(json.dumps(graph, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())
    for path in sorted(directory.rglob('*')):
        if (path.is_file() and path != directory / 'policy_graph.json'
                and '__pycache__' not in path.parts and path.suffix not in ('.pyc', '.bak')):
            h.update(str(path.relative_to(directory)).encode())
            h.update(hashlib.sha256(path.read_bytes()).digest())
    return h.hexdigest()


def job_key(job):
    return f"{job['tag']}|{job['seed']}|{job['a_seat']}"


def first_stage(key, fraction):
    """Whether a job is in the share of the jobs a staged evaluation plays first. It depends on the
    job key only, so every candidate's first stage is the same games, spread over opponents and seats."""
    return int(hashlib.sha256(f'{STAGE_SALT}|{key}'.encode()).hexdigest()[:12], 16) < fraction * 16 ** 12


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


def result_points(margin):
    """A game's result as the ladder rates it: 1 for a win, 0.5 for a tie, 0 for a loss."""
    return 1.0 if margin > 0 else 0.5 if margin == 0 else 0.0


def compare(candidate, baseline, alpha=0.05):
    """Paired comparison of a candidate evaluation with its incumbent's baseline.

    Dollars: a game is better when its margin grew (head-to-head: when the candidate won it). Results: the
    ladder rates only wins, ties and losses, so a game's result changes by the candidate's result points minus
    the incumbent's on the same job (head-to-head: minus a draw's, since a graph against itself draws). A
    valid candidate is promoted when its dollar test passes (more games better than worse, a positive mean
    change and a significant sign test) or its results improved significantly (a sign test over the games
    whose result changed), and never when its results got worse on balance (results_net < 0). On run
    ladder1's first 81 candidates the results test would have promoted 9 more and the veto none fewer
    (evolution_results/ladder_2026-09-26/flip_replay.py)."""
    diffs = []
    up = down = 0
    net = 0.0
    for key, margin in candidate['margins'].items():
        tag = key.split('|')[0]
        if tag == HEAD_TO_HEAD:
            diffs.append((tag, margin))
            change = result_points(margin) - 0.5
        elif key in baseline['margins']:
            diffs.append((tag, margin - baseline['margins'][key]))
            change = result_points(margin) - result_points(baseline['margins'][key])
        else:
            continue
        up += change > 0
        down += change < 0
        net += change
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
    results_p = sign_test(up, down)
    expected = len(candidate.get('jobs') or []) or n
    valid = not candidate.get('errors') and not candidate.get('fallbacks') and n == expected
    by_dollars = wins > losses and mean > 0 and p <= alpha
    by_results = net > 0 and results_p <= alpha
    promote = bool(valid and net >= 0 and (by_dollars or by_results))
    return {'valid': valid, 'jobs': n, 'expected_jobs': expected, 'wins': wins, 'losses': losses,
            'ties': n - wins - losses, 'mean_change': round(mean, 1), 'se': round(se, 1),
            'p': p, 'alpha': alpha, 'per_opponent': per_tag,
            'results_up': up, 'results_down': down, 'results_net': net, 'results_p': results_p,
            'promote': promote,
            'promoted_by': ('dollars and results' if by_dollars and by_results else
                            'dollars' if by_dollars else 'results') if promote else None}


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


class ColabPoolExecutor:
    """Plays a jobs file on a Colab VM pool (arena/colab_pool.py) that runs on this host.

    The request names the run directory as its root: the pool reads the bundles there and
    uploads each to a VM once. Only jobs not yet finished in games/<name>.jsonl are sent; the
    results and the VMs' arena logs (the agents' stderr, where graph fallbacks are reported)
    are appended to games/<name>.jsonl and logs/<name>.log.
    """

    def __init__(self, pool_dir, poll=15.0, timeout=8 * 3600.0):
        self.pool_dir, self.poll, self.timeout = Path(pool_dir).resolve(), poll, timeout

    def describe(self):
        return {'executor': 'colab-pool', 'pool': str(self.pool_dir), 'engine': ENGINE_VERSION}

    def run(self, run_dir, name, bundles):
        run_dir = Path(run_dir).resolve()
        jobs = json.loads((run_dir / 'jobs' / f'{name}.json').read_text())['jobs']
        out = run_dir / 'games' / f'{name}.jsonl'
        done = set()
        if out.exists():
            for line in out.read_text().splitlines():
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if row.get('statuses') and not row.get('errors'):
                    done.add(job_key(row))
        todo = [j for j in jobs if job_key(j) not in done]
        if not todo:
            return
        batch = f'{name[:80]}-{time.time_ns() % 10**12}'
        inbox, outbox = self.pool_dir / 'inbox', self.pool_dir / 'outbox'
        inbox.mkdir(parents=True, exist_ok=True)
        tmp = inbox / f'.{batch}.tmp'
        tmp.write_text(json.dumps({'name': batch, 'root': str(run_dir), 'jobs': todo}))
        tmp.rename(inbox / f'{batch}.json')
        t0 = time.time()
        while not (outbox / f'{batch}.done').exists():
            if time.time() - t0 > self.timeout:
                raise RuntimeError(f'Colab pool batch {batch} not done after {self.timeout:.0f} s')
            time.sleep(self.poll)
        summary = json.loads((outbox / f'{batch}.done').read_text())
        with out.open('a') as fh:
            fh.write((outbox / f'{batch}.jsonl').read_text())
        with (run_dir / 'logs' / f'{name}.log').open('a') as fh:
            fh.write((outbox / f'{batch}.log').read_text(errors='replace') + '\n')
        if summary['finished'] < summary['games']:
            raise RuntimeError(f'Colab pool batch {batch} incomplete: {summary}')


class Gauntlet:
    """`plan` (optional) replaces the default job set: a list of {"tag": opponent bundle, "seed",
    "a_seat", "set"} entries (e.g. every lost ladder seed against the agent that beat us there,
    from both seats, plus random seeds against the pool; ladder_seed_plan.py writes one). With a
    plan the head-to-head block plays the first `seeds_per_opponent` evolution seeds.

    A graph's pool margins are cached per job (scores/<bundle>.json). A margin stays valid while
    the fingerprint (runtime, harness, head-to-head seeds, engine version) and the digest of its
    opponent's bundle are unchanged, so a plan that grows plays only its new jobs.

    With `stage_fraction` > 0 a candidate evaluated against an incumbent's baseline plays that share
    of its jobs first (`first_stage`), and the rest only if they leave it a chance
    (`stop_after_first_stage`)."""

    INERT_CHANGES = 4  # a first stage with this many changed games or fewer stops

    def __init__(self, run_dir, executor, seeds_per_opponent=40, opponents=OPPONENTS, alpha=0.05, plan=None,
                 stage_fraction=0.0):
        self.run_dir = Path(run_dir).resolve()  # arena.py runs with cwd=run_dir
        self.executor = executor
        self.plan = [dict(e) for e in plan] if plan else None
        self.opponents = (tuple(sorted({e['tag'] for e in self.plan})) if self.plan else tuple(opponents))
        self.seeds = seed_list(seeds_per_opponent)
        self.alpha = alpha
        self.stage_fraction = stage_fraction
        self.fingerprint = None
        self.opponent_digests = {}

    def prepare(self):
        for sub in ('bundles', 'jobs', 'games', 'logs', 'scores'):
            (self.run_dir / sub).mkdir(parents=True, exist_ok=True)
        for name in self.opponents:
            if not (self.run_dir / 'bundles' / name).exists():
                payload.build_opponent(self.run_dir, name)
        shutil.copy2(HERE / 'arena' / 'arena.py', self.run_dir / 'arena.py')
        shutil.copy2(HERE.parents[1] / 'shinka/evolution/pool_upgrade_bundle_agent.py',
                     self.run_dir / 'bundle_agent.py')
        self.opponent_digests = {name: _tree_digest(p for p in (self.run_dir / 'bundles' / name).rglob('*')
                                                    if p.is_file()) for name in self.opponents}
        files = [p for p in (HERE / 'hazel_runtime').rglob('*')
                 if p.is_file() and '__pycache__' not in p.parts]
        files += [HERE / 'agent_graph.py', self.run_dir / 'arena.py', self.run_dir / 'bundle_agent.py']
        self.fingerprint = hashlib.sha256(json.dumps({
            'files': _tree_digest(files), 'seeds': self.seeds,
            'engine': self.executor.describe().get('engine')}, sort_keys=True).encode()).hexdigest()
        return self.fingerprint

    def bundle(self, graph):
        """Reuse legacy names only for matching executable contents; publish new bundles once."""
        root = self.run_dir / 'bundles'
        root.mkdir(parents=True, exist_ok=True)
        legacy = 'g_' + graph_fingerprint(graph)[:16]
        tmp = Path(tempfile.mkdtemp(prefix=f'.{legacy}.', dir=root))
        try:
            payload.write_graph_bundle(tmp, graph, legacy)
            digest = _bundle_digest(tmp)
            old = root / legacy
            if old.is_dir() and _bundle_digest(old) == digest:
                return legacy
            name = 'g_' + digest[:16]
            dst = root / name
            if dst.exists():
                if not dst.is_dir() or _bundle_digest(dst) != digest:
                    raise RuntimeError(f'bundle identity collision or damaged bundle: {dst}')
                return name
            policy = tmp / 'policy_graph.json'
            bundle_graph = json.loads(policy.read_text())
            bundle_graph['name'] = name
            policy.write_text(json.dumps(bundle_graph, indent=2) + '\n')
            try:
                tmp.rename(dst)
            except OSError as error:
                if error.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                    raise
                if not dst.is_dir() or _bundle_digest(dst) != digest:
                    raise RuntimeError(f'bundle identity collision or damaged bundle: {dst}')
            return name
        finally:
            if tmp.exists():
                shutil.rmtree(tmp)

    def jobs(self, candidate, incumbent=None):
        if self.plan:
            jobs = [{'tag': e['tag'], 'a': f'bundles/{candidate}', 'b': f"bundles/{e['tag']}", 'seed': e['seed'],
                     'a_seat': e['a_seat'], **({'set': e['set']} if 'set' in e else {})} for e in self.plan]
        else:
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

    def evaluate(self, graph, incumbent_graph=None, baseline=None):
        """Play the pool jobs (+ head-to-head vs the incumbent) for a graph. Pool jobs the graph
        already played (cached, see the class docstring) are not played again: an island champion
        offered to another island when islands mix plays only the head-to-head block.

        With a stage_fraction and the incumbent's `baseline`, the first stage's jobs are played
        first and the rest only if stop_after_first_stage() finds no reason to stop. A stopped result
        carries `stopped` and lacks the other jobs, so compare() finds it incomplete: never promoted."""
        if self.fingerprint is None:
            raise RuntimeError('prepare() first')
        candidate = self.bundle(graph)
        incumbent = self.bundle(incumbent_graph) if incumbent_graph is not None else None
        if incumbent == candidate:
            raise ValueError('candidate and incumbent are the same graph')
        name = f'{candidate}_vs_{incumbent}' if incumbent else candidate
        jobs = self.jobs(candidate, incumbent)
        cached = self._cached(candidate)
        todo = [j for j in jobs if job_key(j) not in cached]
        bundles = sorted({candidate, *self.opponents, *([incumbent] if incumbent else [])})
        stages = [todo]
        if self.stage_fraction and baseline is not None:
            first = [j for j in todo if first_stage(job_key(j), self.stage_fraction)]
            if 0 < len(first) < len(todo):
                stages = [first, todo]
        stopped = None
        for number, stage in enumerate(stages, 1):
            rows = self._play(name, stage, bundles) if stage else []
            wanted = {job_key(j) for j in stage}
            found, errors = margins([r for r in rows if job_key(r) in wanted], candidate)
            missing = sorted(wanted - set(found) - {e['job'] for e in errors})
            errors += [{'job': m, 'errors': 'missing'} for m in missing]
            found.update({job_key(j): cached[job_key(j)] for j in jobs if job_key(j) in cached})
            if errors:
                break
            if number < len(stages):
                stopped = self.stop_after_first_stage(found, baseline)
                if stopped:
                    stopped.update(played=len(stage), of=len(todo))
                    break
        result = {'bundle': candidate, 'incumbent': incumbent, 'fingerprint': self.fingerprint,
                  'jobs': sorted(job_key(j) for j in jobs), 'margins': found, 'errors': errors,
                  'fallbacks': self.fallbacks(name, candidate) if todo else 0}
        if stopped:
            result['stopped'] = stopped
        pool = {k: v for k, v in found.items() if not k.startswith(HEAD_TO_HEAD + '|')}
        if not result['errors'] and not result['fallbacks']:
            self._store(candidate, pool)
        return result

    def stop_after_first_stage(self, found, baseline):
        """Why a staged evaluation ends after its first stage, or None. A promotion needs a significant
        gain in dollars or in results over all the jobs (compare), so the rest is not played when the
        first stage changed at most INERT_CHANGES games ("inert"), or when its changed games went worse
        at least as often as better and its results did too ("not better")."""
        stats = compare({'margins': found}, baseline, self.alpha)
        if stats['wins'] + stats['losses'] <= self.INERT_CHANGES:
            reason = 'inert'
        elif stats['wins'] <= stats['losses'] and stats['results_up'] <= stats['results_down']:
            reason = 'not better'
        else:
            return None
        return {'reason': reason, 'wins': stats['wins'], 'losses': stats['losses'],
                'mean_change': stats['mean_change'], 'results_up': stats['results_up'],
                'results_down': stats['results_down']}

    def _store(self, bundle, pool_margins):
        tags = sorted({key.split('|')[0] for key in pool_margins})
        atomic_json(self.run_dir / 'scores' / f'{bundle}.json',
                    {'fingerprint': self.fingerprint, 'opponents': {t: self.opponent_digests[t] for t in tags},
                     'margins': pool_margins})

    def _cached(self, bundle):
        """Pool-job margins the bundle already has that are still valid: same fingerprint, and the
        job's opponent bundle unchanged ({} if none)."""
        path = self.run_dir / 'scores' / f'{bundle}.json'
        if path.exists():
            cached = json.loads(path.read_text())
            if cached.get('fingerprint') == self.fingerprint:
                digests = cached.get('opponents') or {}
                return {key: margin for key, margin in cached['margins'].items()
                        if digests.get(key.split('|')[0]) == self.opponent_digests.get(key.split('|')[0])}
        return {}

    def has_baseline(self, graph):
        """Whether baseline() would play nothing: every pool job of the graph is cached."""
        bundle = self.bundle(graph)
        return {job_key(j) for j in self.jobs(bundle)} <= set(self._cached(bundle))

    def baseline(self, graph):
        """Pool-job margins of a graph; only the jobs it has not played yet are played."""
        bundle = self.bundle(graph)
        cached = self._cached(bundle)
        wanted = {job_key(j) for j in self.jobs(bundle)}
        if wanted <= set(cached):
            return {'bundle': bundle, 'margins': {key: cached[key] for key in wanted}}
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
