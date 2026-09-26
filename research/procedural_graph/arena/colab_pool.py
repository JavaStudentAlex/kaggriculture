#!/usr/bin/env python3
"""A pool of Colab VMs that plays arena job batches on demand, for evolution loops.

    python3 arena/colab_pool.py --pool-dir runs/evo1/pool --run-name evo1 --vm colab2:hm --vm colab2:hm

colab_run.py plays one payload and stops its VMs. An evolution loop plays one batch after another
(one gauntlet per candidate); making VMs and building their venv for every batch would cost
minutes each time. The pool keeps its VMs between batches:

- A client (graph_gauntlet.ColabPoolExecutor) writes a request to <pool-dir>/inbox/<name>.json:
  {"name", "root": a directory holding arena.py, bundle_agent.py and bundles/, "jobs": [...]}.
  Job bundle paths (a, b) are bundles/<bundle> relative to root.
- The pool plays the jobs and writes <pool-dir>/outbox/<name>.jsonl (arena results, one line per
  game), <name>.log (the VMs' arena logs, which carry the agents' stderr) and last <name>.done
  (a summary). Requests are played one at a time, oldest first.
- A bundle is uploaded to a VM once and kept there. Bundle names must therefore be content
  addresses (graph bundles are g_<fingerprint>): a changed bundle needs a new name. The harness
  (the root's arena.py and bundle_agent.py) goes up with the first request and again whenever
  its bytes change.
- VMs are watched as in colab_run.py: a command every --poll seconds, a fresh token every 30 min
  (colab_readopt.py), results pulled every 10 min. A VM that Colab deletes is replaced (up to
  --replacements times per slot), and its unfinished games are played by the other VMs. A VM that
  served for --long-life seconds before it went away does not use up a replacement: Colab ends
  long-running VMs, and a pool that runs for days would otherwise lose its slots one by one.
- With no request for --idle-stop seconds the pool stops its VMs; the next request makes them
  again. A file <pool-dir>/STOP ends the pool after the current request. Every VM is stopped at
  the end and the accounts' session lists are checked (COLAB_VMS_LEFT=0).
Run it in tmux on cliproxyapi next to the evolution loop (AGENTS.md section 3.1).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import signal
import tarfile
import tempfile
import threading
import time
from pathlib import Path

try:
    from arena.colab_run import (WORKERS, ColabCLI, colab_link, finished, job_key, parse_endpoints)
except ImportError:  # run as a script from arena/
    from colab_run import (WORKERS, ColabCLI, colab_link, finished, job_key, parse_endpoints)

HERE = Path(__file__).resolve().parent
REMOTE = '/content/pool'
REFRESH_EVERY = 1800.0
PULL_EVERY = 600.0
LONG_LIFE = 3 * 3600.0   # a VM lost after serving this long does not count as a replacement
RETRIES = 1   # an errored game is played once more before the request reports it

SETUP = r'''
import os, subprocess, sys, tarfile
root = {root!r}
for d in ('bundles', 'batches'):
    os.makedirs(root + '/' + d, exist_ok=True)
if not os.path.exists(root + '/venv/bin/python'):
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'uv'], check=True)
    uv = [sys.executable, '-m', 'uv']
    made = subprocess.run(uv + ['venv', '--python', '3.12.13', root + '/venv'], capture_output=True, text=True)
    if made.returncode:
        subprocess.run(uv + ['venv', '--python', '3.12', root + '/venv'], check=True, capture_output=True)
    done = subprocess.run(uv + ['pip', 'install', '--python', root + '/venv/bin/python', '-r',
                                root + '_requirements.txt'], capture_output=True, text=True)
    if done.returncode:
        print('COLAB_POOL_ERROR requirements', done.stderr[-1500:])
        raise SystemExit
version = subprocess.run([root + '/venv/bin/python', '-V'], capture_output=True, text=True).stdout.strip()
print('COLAB_POOL_READY', os.cpu_count(), version)
'''

ADD_FILES = r'''
import hashlib, os, tarfile
root, archive, files = {root!r}, {archive!r}, {files!r}
with tarfile.open(archive) as t:
    t.extractall(root, filter='data')
os.remove(archive)
bad = [r for r, h in files.items()
       if hashlib.sha256(open(root + '/' + r, 'rb').read()).hexdigest() != h]
print(('COLAB_POOL_FILES_BAD ' + ' '.join(bad[:3])) if bad else 'COLAB_POOL_FILES_OK')
'''

START = r'''
import os, shutil, subprocess
root, chunk, workers = {root!r}, {chunk!r}, {workers}
d = root + '/batches/' + chunk
os.makedirs(d, exist_ok=True)
shutil.move(root + '/batches/' + chunk + '_jobs.json', d + '/jobs.json')
log = open(d + '/arena.log', 'a')
proc = subprocess.Popen([root + '/venv/bin/python', 'arena.py', '--jobs', d + '/jobs.json', '--root', '.',
                         '--workers', str(workers), '--out', d + '/results.jsonl'],
                        cwd=root, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
open(d + '/pid', 'w').write(str(proc.pid))
print('COLAB_POOL_STARTED', proc.pid)
'''

POLL = r'''
import json, os
root, chunk = {root!r}, {chunk!r}
status = {{'results': 0, 'done': False, 'alive': False, 'tail': ''}}
if chunk:
    d = root + '/batches/' + chunk
    if os.path.exists(d + '/results.jsonl'):
        status['results'] = sum(1 for _ in open(d + '/results.jsonl'))
    log = open(d + '/arena.log').read() if os.path.exists(d + '/arena.log') else ''
    status['done'] = 'ARENA_DONE' in log
    status['tail'] = log.strip().splitlines()[-1][:200] if log.strip() else ''
    try:
        status['alive'] = open('/proc/' + open(d + '/pid').read().strip() + '/stat').read().split()[2] != 'Z'
    except OSError:
        pass
print('COLAB_POOL_STATUS ' + json.dumps(status))
'''


def parse_marker(text, marker):
    for line in text.splitlines():
        if line.startswith(marker + ' ') or line == marker:
            return line[len(marker):].strip()
    return None


HARNESS = ('arena.py', 'bundle_agent.py')


def bundle_files(root, name):
    """{path relative to root: sha256} of one bundle directory (bundles/<name>/...)."""
    root = Path(root)
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((root / 'bundles' / name).rglob('*')) if p.is_file() and '__pycache__' not in p.parts}


def harness_digest(root):
    return hashlib.sha256(b''.join(hashlib.sha256((Path(root) / f).read_bytes()).digest()
                                   for f in HARNESS)).hexdigest()


class Slot:
    """One VM position of the pool; the VM in it may be replaced."""

    def __init__(self, index, command, shape, workers):
        self.index, self.command, self.shape, self.workers = index, command, shape, workers
        self.generation = 0       # replacements used since the slot was last (re)started
        self.made = 0             # VMs made for this slot so far: part of the session name
        self.session = None
        self.endpoint = None
        self.born = 0.0           # when the slot's current VM was made
        self.bundles = set()      # bundle names present on the VM
        self.harness = None       # digest of the arena.py + bundle_agent.py on the VM
        self.chunk = None         # name of the chunk it plays
        self.jobs = []            # the chunk's jobs
        self.refreshed = self.pulled = 0.0
        self.last_count = None
        self.failures = 0         # polls in a row without an answer
        self.start_failures = 0   # chunks in a row that could not be started


class ColabPool:
    def __init__(self, pool_dir, run_name, vms, poll=60.0, idle_stop=1200.0, replacements=2,
                 requirements=HERE / 'colab_requirements.txt', cli_factory=ColabCLI, log=print,
                 refresh_every=REFRESH_EVERY, pull_every=PULL_EVERY, workers=None, sleep=time.sleep,
                 long_life=LONG_LIFE):
        self.dir = Path(pool_dir).resolve()
        for sub in ('inbox', 'outbox', 'work'):
            (self.dir / sub).mkdir(parents=True, exist_ok=True)
        self.run_name, self.poll, self.idle_stop, self.replacements = run_name, poll, idle_stop, replacements
        self.long_life = long_life
        self.requirements, self.cli_factory, self.log, self.sleep = Path(requirements), cli_factory, log, sleep
        self.refresh_every, self.pull_every = refresh_every, pull_every
        self.slots = []
        for i, spec in enumerate(vms):
            command, shape = spec.split(':')
            if shape not in WORKERS:
                raise ValueError(f'unknown VM shape {shape!r}; use one of {sorted(WORKERS)}')
            self.slots.append(Slot(i, command, shape, workers or WORKERS[shape]))
        self.created = []          # (command, session) of every VM made
        self.lock = threading.Lock()
        self.stopping = False
        self.idle_since = time.time()

    # ------------------------------------------------------------------ VMs
    def session_name(self, slot):
        return f'{self.run_name}-{slot.index}-{slot.made}'

    def cli(self, slot):
        return self.cli_factory(slot.command)

    def create(self, slot, tmp):
        """Make and set up the slot's VM; True when it is ready to play."""
        slot.made += 1
        session = self.session_name(slot)
        cli = self.cli(slot)
        ok, out = cli.new(session, slot.shape in ('hm', 't4hm'), gpu='T4' if slot.shape.startswith('t4') else None)
        if not ok and session in parse_endpoints(cli.sessions()):
            ok = True
            self.log(f'[{session}] reported as not created, but it exists: using it')
        if not ok:
            cli.stop(session)
            self.log(f'[{session}] VM not created (stop sent): {out.strip()[-200:]}')
            return False
        with self.lock:
            self.created.append((slot.command, session))
        slot.session, slot.bundles, slot.chunk, slot.jobs, slot.harness = session, set(), None, [], None
        slot.born = time.time()
        slot.failures = slot.start_failures = 0
        slot.endpoint = parse_endpoints(cli.sessions()).get(session)
        if slot.endpoint:
            self.log(f'[{session}] {slot.command} link: {colab_link(slot.endpoint)}')
        try:
            ok, out = cli.upload(session, self.requirements, REMOTE + '_requirements.txt')
            if not ok:
                raise RuntimeError(f'upload of the requirements failed: {out.strip()[-300:]}')
            setup = tmp / f'{session}_setup.py'
            setup.write_text(SETUP.format(root=REMOTE))
            out = cli.exec_file(session, setup, 1200)
            ready = parse_marker(out, 'COLAB_POOL_READY')
            if ready is None:
                raise RuntimeError(f'setup failed: {out.strip()[-600:]}')
        except Exception as exc:
            self.log(f'[{session}] {exc}')
            self.stop_vm(slot)
            return False
        slot.refreshed = time.time()
        self.refresh(slot)
        self.log(f'[{session}] {slot.command} {slot.shape} ready: {slot.workers} workers ({ready})')
        return True

    def ensure(self, tmp):
        """Every slot without a VM gets one (in parallel), while it has replacements left."""
        todo = [s for s in self.slots if s.session is None and s.generation <= self.replacements]
        threads = []
        for slot in todo:
            def make(slot=slot):
                if not self.create(slot, tmp):
                    slot.generation += 1   # the next attempt is a replacement
            threads.append(threading.Thread(target=make, daemon=True))
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    def stop_vm(self, slot):
        if slot.session:
            self.cli(slot).stop(slot.session)
            self.log(f'[{slot.session}] stopped')
        slot.session = slot.endpoint = slot.chunk = slot.harness = None
        slot.jobs, slot.bundles = [], set()

    def refresh(self, slot):
        """A fresh token and a live keep-alive; 'deleted' only if two checks agree."""
        cli = self.cli(slot)
        slot.endpoint = slot.endpoint or parse_endpoints(cli.sessions()).get(slot.session)
        state = cli.refresh(slot.session, slot.endpoint) if slot.endpoint else 'error'
        if state == 'deleted':
            self.sleep(30 if self.poll else 0)
            state = cli.refresh(slot.session, slot.endpoint)
        if state != 'ok':
            self.log(f'[{slot.session}] refresh: {state}')
        return state

    def status(self, slot, tmp):
        """The slot's chunk status (also the keep-alive command); None if the VM did not answer."""
        poll = tmp / f'{slot.session}_poll.py'
        poll.write_text(POLL.format(root=REMOTE, chunk=slot.chunk))
        found = parse_marker(self.cli(slot).exec_file(slot.session, poll, 120), 'COLAB_POOL_STATUS')
        return json.loads(found) if found else None

    def keep(self, slot, tmp):
        """Refresh the token when due, then poll (the poll is also the keep-alive command). The
        chunk status, or None if the VM did not answer (then lost() decides whether it is gone)."""
        if time.time() - slot.refreshed >= self.refresh_every:
            slot.refreshed = time.time()
            self.refresh(slot)
        return self.status(slot, tmp)

    def lost(self, slot):
        """The VM stopped answering: True (and the slot emptied) if Colab deleted it."""
        slot.failures += 1
        slot.refreshed = time.time()
        if self.refresh(slot) == 'deleted' or slot.failures >= 5:
            served = time.time() - slot.born
            self.log((f'[{slot.session}] deleted by Colab' if slot.failures < 5 else
                      f'[{slot.session}] stopped answering') + f' after {served / 3600:.1f} h')
            self.stop_vm(slot)
            if served >= self.long_life:
                slot.generation = 0   # it served for hours: Colab's runtime limit, not a failing slot
            slot.generation += 1
            return True
        return False

    # ------------------------------------------------------------------ one request
    def start(self, slot, request, jobs, tmp):
        """Upload the chunk's missing bundles and jobs and start arena.py on the slot's VM."""
        cli, root = self.cli(slot), Path(request['root'])
        need = sorted({Path(j[k]).name for j in jobs for k in ('a', 'b')} - slot.bundles)
        harness = harness_digest(root)
        if need or harness != slot.harness:
            files = {}
            archive = tmp / f'{slot.session}_files.tar.gz'
            with tarfile.open(archive, 'w:gz') as tar:
                for name in need:
                    tar.add(root / 'bundles' / name, arcname=f'bundles/{name}',
                            filter=lambda ti: None if '__pycache__' in ti.name else ti)
                    files.update(bundle_files(root, name))
                if harness != slot.harness:
                    for f in HARNESS:
                        tar.add(root / f, arcname=f)
                        files[f] = hashlib.sha256((root / f).read_bytes()).hexdigest()
            remote = f'{REMOTE}_files_{slot.session}.tar.gz'
            ok, out = cli.upload(slot.session, archive, remote)
            if not ok:
                raise RuntimeError(f'upload failed: {out.strip()[-300:]}')
            script = tmp / f'{slot.session}_add.py'
            script.write_text(ADD_FILES.format(root=REMOTE, archive=remote, files=files))
            out = cli.exec_file(slot.session, script, 600)
            if parse_marker(out, 'COLAB_POOL_FILES_OK') is None:
                raise RuntimeError(f'files not installed: {out.strip()[-300:]}')
            slot.bundles |= set(need)
            slot.harness = harness
        chunk = f"{request['name']}-{slot.index}-{time.time_ns() % 10**9}"
        shard = tmp / f'{chunk}_jobs.json'
        shard.write_text(json.dumps({'jobs': jobs}))
        ok, out = cli.upload(slot.session, shard, f'{REMOTE}/batches/{chunk}_jobs.json')
        if not ok:
            raise RuntimeError(f'jobs upload failed: {out.strip()[-300:]}')
        script = tmp / f'{chunk}_start.py'
        script.write_text(START.format(root=REMOTE, chunk=chunk, workers=slot.workers))
        if parse_marker(cli.exec_file(slot.session, script, 300), 'COLAB_POOL_STARTED') is None:
            raise RuntimeError('arena did not start')
        slot.chunk, slot.jobs, slot.pulled, slot.last_count = chunk, jobs, time.time(), None
        slot.start_failures = 0
        self.log(f'[{slot.session}] {len(jobs)} games of {request["name"]}')

    def pull(self, slot, request, results, logs, tmp, final=False):
        """Merge the chunk's results so far into `results` ({job key: line}); True if downloaded."""
        cli = self.cli(slot)
        part = tmp / f'{slot.chunk}.jsonl'
        ok, _ = cli.download(slot.session, f'{REMOTE}/batches/{slot.chunk}/results.jsonl', part)
        if ok:
            for line in part.read_text().splitlines():
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                results[job_key(row)] = line
        if final:
            log = tmp / f'{slot.chunk}.log'
            if cli.download(slot.session, f'{REMOTE}/batches/{slot.chunk}/arena.log', log)[0]:
                logs.append(f'### {slot.session} {slot.chunk}\n' + log.read_text(errors='replace'))
        return ok

    def dispatch(self, request, queue, tmp):
        """Split the queued jobs over the idle VMs (in proportion to their workers) and start them."""
        idle = [s for s in self.slots if s.session and s.chunk is None]
        if not idle or not queue:
            return
        weights = [s.workers for s in idle]
        shares, start = [], 0
        for k, slot in enumerate(idle):
            n = round(len(queue) * sum(weights[:k + 1]) / sum(weights)) - start
            shares.append((slot, queue[start:start + n]))
            start += n
        queue.clear()
        errors = []

        def run(slot, jobs):
            try:
                self.start(slot, request, jobs, tmp)
            except Exception as exc:
                self.log(f'[{slot.session}] could not start: {exc}')
                errors.append((slot, jobs))

        threads = [threading.Thread(target=run, args=(s, j), daemon=True) for s, j in shares if j]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for slot, jobs in errors:
            queue.extend(jobs)
            slot.start_failures += 1
            if slot.start_failures >= 3:   # a VM that cannot start games is replaced
                self.stop_vm(slot)
                slot.generation += 1

    def play(self, request, tmp):
        """Play one request to the end; returns its results ({job key: line}) and VM logs."""
        results, logs, tries = {}, [], {}
        queue = list(request['jobs'])
        while not self.stopping:
            self.ensure(tmp)
            if not any(s.session for s in self.slots):
                self.log(f'{request["name"]}: no VM left')
                break
            self.dispatch(request, queue, tmp)
            self.sleep(self.poll)
            for slot in self.slots:
                if not slot.session:
                    continue
                jobs = slot.jobs
                status = self.keep(slot, tmp)
                if status is None:
                    if self.lost(slot):
                        queue.extend(j for j in jobs if job_key(j) not in finished_keys(results))
                    continue
                slot.failures = 0
                if slot.chunk is None:
                    continue
                if status['results'] != slot.last_count:
                    slot.last_count = status['results']
                    self.log(f'[{slot.session}] {slot.last_count}/{len(slot.jobs)} games')
                if status['done'] or not status['alive']:
                    self.pull(slot, request, results, logs, tmp, final=True)
                    done = finished_keys(results)
                    for job in slot.jobs:
                        key = job_key(job)
                        if key not in done:
                            tries[key] = tries.get(key, 0) + 1
                            if tries[key] <= RETRIES:
                                queue.append(job)
                    slot.chunk, slot.jobs = None, []
                elif time.time() - slot.pulled >= self.pull_every:
                    slot.pulled = time.time()
                    self.pull(slot, request, results, logs, tmp)
            if not queue and not any(s.chunk for s in self.slots):
                break
        return results, logs

    # ------------------------------------------------------------------ service
    def next_request(self):
        requests = sorted(self.dir.glob('inbox/*.json'), key=lambda p: p.stat().st_mtime)
        for path in requests:
            try:
                return path, json.loads(path.read_text())
            except ValueError:
                continue   # still being written
        return None, None

    def finish(self, path, request, results, logs):
        name = request['name']
        wanted = {job_key(j) for j in request['jobs']}
        lines = [line for key, line in results.items() if key in wanted]
        clean = finished_keys(results) & wanted
        out = self.dir / 'outbox'
        (out / f'{name}.jsonl').write_text(''.join(line + '\n' for line in lines))
        (out / f'{name}.log').write_text('\n'.join(logs))
        summary = {'name': name, 'games': len(wanted), 'finished': len(clean), 'errored': len(lines) - len(clean),
                   'missing': len(wanted - set(results)), 'stopped': self.stopping}
        (out / f'{name}.done').write_text(json.dumps(summary) + '\n')
        path.unlink(missing_ok=True)
        self.log(f'{name}: {summary}')

    def stop_all(self):
        for slot in self.slots:
            if slot.session:
                self.stop_vm(slot)
            slot.generation = 0

    def verify_removed(self):
        left = []
        for command in sorted({s.command for s in self.slots}):
            cli = self.cli_factory(command)
            names = {s for c, s in self.created if c == command}
            for session in sorted(names & set(parse_endpoints(cli.sessions()))):
                cli.stop(session)
            listing = cli.sessions()
            left += [f'{command}:{s}' for s in sorted(names & set(parse_endpoints(listing)))]
            left += [f'{command}:{line.split()[1]} (no local record)' for line in listing.splitlines()
                     if line.startswith('[?] ')]
        self.log(f'COLAB_VMS_LEFT={len(left)}' + (f' {left} -- stop them by hand' if left else ''))
        return left

    def serve(self, once=False):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            try:
                while not self.stopping:
                    if (self.dir / 'STOP').exists():
                        self.log('STOP file: ending the pool')
                        break
                    path, request = self.next_request()
                    if request is None:
                        if once:
                            break
                        if any(s.session for s in self.slots):
                            if time.time() - self.idle_since >= self.idle_stop:
                                self.log(f'idle for {self.idle_stop:.0f} s: stopping the VMs')
                                self.stop_all()
                            else:
                                for slot in self.slots:
                                    if slot.session and self.keep(slot, tmp) is None:
                                        self.lost(slot)
                        self.sleep(self.poll)
                        continue
                    self.log(f'{request["name"]}: {len(request["jobs"])} games')
                    results, logs = self.play(request, tmp)
                    self.finish(path, request, results, logs)
                    self.idle_since = time.time()
            finally:
                self.stop_all()
                self.verify_removed()


def finished_keys(results):
    """Keys of the games in {key: line} that finished without errors."""
    done = set()
    for key, line in results.items():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get('statuses') and not row.get('errors'):
            done.add(key)
    return done


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--pool-dir', required=True)
    ap.add_argument('--run-name', required=True, help='session name prefix (<run-name>-<slot>[-<replacement>])')
    ap.add_argument('--vm', action='append', required=True, metavar='COMMAND:SHAPE')
    ap.add_argument('--poll', type=float, default=60.0)
    ap.add_argument('--idle-stop', type=float, default=1200.0, help='stop the VMs after this long without a request')
    ap.add_argument('--replacements', type=int, default=2)
    ap.add_argument('--long-life', type=float, default=LONG_LIFE,
                    help='a VM lost after serving this many seconds does not use up a replacement')
    ap.add_argument('--workers', type=int, default=None)
    args = ap.parse_args()
    pool = ColabPool(args.pool_dir, args.run_name, args.vm, poll=args.poll, idle_stop=args.idle_stop,
                     replacements=args.replacements, workers=args.workers, long_life=args.long_life,
                     log=lambda m: print(f'{time.strftime("%H:%M:%S")} {m}', flush=True))

    def stop(signum, frame):
        pool.stopping = True
        pool.log('signal: finishing the current request, then stopping the VMs')

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    pool.serve()


if __name__ == '__main__':
    main()
