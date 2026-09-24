#!/usr/bin/env python3
"""Play an arena payload on Google Colab VMs through the Colab CLI.

    python colab_run.py --payload <payload dir> --out results.jsonl --run-name r5 \
        --vm colab2:hm --vm colab2:hm --vm colab4:std

Each `--vm COMMAND:SHAPE` starts one VM with that account's CLI wrapper (colab1..colabN,
created by colab-add-account). SHAPE is `hm` (High-RAM: 8 CPUs, 51 GB, Pro accounts only)
or `std` (2 CPUs, 12.7 GB). The payload's jobs that are not yet in --out are split across
the VMs in proportion to their workers (8 per High-RAM VM, 2 per standard VM by default).
Each VM is handled in its own thread:

1. `new` for every VM in parallel; the games are then split over the VMs that exist (a
   refused VM's share goes to the others), in proportion to their workers;
2. upload the payload (tar.gz), its shard and the pinned requirements; verify files.json;
   build a Python 3.12 venv with uv (the environment whose results matched the Brev boxes);
   start arena.py on the shard, detached;
3. poll every --poll seconds (results written, process alive, last log line);
4. download the shard's results and its per-game traces (arena.py --trace-dir, into
   <out dir>/traces/), then stop the session. Every created session is stopped,
   also after an error or Ctrl-C, and at the end every account's session list is checked
   (`COLAB_VMS_LEFT=0`; a leftover is stopped once more).

Results are appended to --out, and rerunning the same command plays only what is still
missing. `--jobs FILE` plays only the jobs listed there (same format as the payload's
jobs.json), e.g. the shards of VMs that Colab deleted, under a new --run-name. `--cleanup` only stops the sessions a run with the same --run-name and --vm
flags would create; `--attach` follows them to the end instead (for a runner that was
killed, or a `--start-only` run that left its VMs playing): wait, download, stop, merge. Run it in tmux (kagg-colab-<run>) so a closed
terminal or session does not leave VMs running.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REMOTE = '/content/arena'
WORKERS = {'hm': 8, 'std': 2}

SETUP = r'''
import hashlib, json, os, subprocess, sys, tarfile
root = {root!r}
os.makedirs(root, exist_ok=True)
with tarfile.open(root + '_payload.tar.gz') as t:
    t.extractall(root, filter='data')
files = json.load(open(root + '/payload/files.json'))
bad = [r for r, h in files.items()
       if hashlib.sha256(open(root + '/payload/' + r, 'rb').read()).hexdigest() != h]
if bad:
    print('COLAB_ARENA_ERROR payload hash mismatch', bad[:3])
    raise SystemExit
if not os.path.exists(root + '/venv/bin/python'):
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'uv'], check=True)
    uv = [sys.executable, '-m', 'uv']
    made = subprocess.run(uv + ['venv', '--python', '3.12.13', root + '/venv'], capture_output=True, text=True)
    if made.returncode:
        subprocess.run(uv + ['venv', '--python', '3.12', root + '/venv'], check=True, capture_output=True)
    done = subprocess.run(uv + ['pip', 'install', '--python', root + '/venv/bin/python', '-r',
                                root + '_requirements.txt'], capture_output=True, text=True)
    if done.returncode:
        print('COLAB_ARENA_ERROR requirements', done.stderr[-1500:])
        raise SystemExit
log = open(root + '/arena.log', 'a')
proc = subprocess.Popen([root + '/venv/bin/python', 'payload/arena.py', '--jobs', root + '_shard.json',
                         '--root', 'payload', '--workers', '{workers}', '--out', 'results.jsonl'] + {trace_args},
                        cwd=root, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
open(root + '/arena.pid', 'w').write(str(proc.pid))
version = subprocess.run([root + '/venv/bin/python', '-V'], capture_output=True, text=True).stdout.strip()
print('COLAB_ARENA_STARTED', proc.pid, os.cpu_count(), version)
'''

POLL = r'''
import json, os
root = {root!r}
path = root + '/results.jsonl'
results = sum(1 for _ in open(path)) if os.path.exists(path) else 0
log = open(root + '/arena.log').read() if os.path.exists(root + '/arena.log') else ''
alive = False
if os.path.exists(root + '/arena.pid'):
    try:
        alive = open('/proc/' + open(root + '/arena.pid').read().strip() + '/stat').read().split()[2] != 'Z'
    except OSError:
        alive = False
tail = log.strip().splitlines()[-1][:200] if log.strip() else ''
print('COLAB_ARENA_STATUS ' + json.dumps({{'results': results, 'done': 'ARENA_DONE' in log, 'alive': alive,
                                         'tail': tail}}))
'''


PACK = r'''
import os, tarfile
root = {root!r}
n = 0
with tarfile.open(root + '/traces.tgz', 'w:gz') as tar:
    if os.path.isdir(root + '/traces'):
        for name in sorted(os.listdir(root + '/traces')):
            tar.add(root + '/traces/' + name, arcname=name)
            n += 1
print('COLAB_ARENA_PACKED', n)
'''


def job_key(job):
    return job['tag'], job['seed'], job['a_seat']


def finished(path):
    """{key: line} for the games of a results file that finished without errors."""
    done = {}
    path = Path(path)
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get('statuses') and not row.get('errors'):
                done[job_key(row)] = line
    return done


def assign(jobs, capacities):
    """Split jobs across VMs in proportion to their capacities (workers)."""
    shards = [[] for _ in capacities]
    for job in jobs:
        i = min(range(len(capacities)), key=lambda k: (len(shards[k]) + 1) / capacities[k])
        shards[i].append(job)
    return shards


def parse_status(text):
    for line in text.splitlines():
        if line.startswith('COLAB_ARENA_STATUS '):
            return json.loads(line[len('COLAB_ARENA_STATUS '):])
    return None


class ColabCLI:
    """The Colab CLI calls the runner needs, through one account's wrapper command."""

    def __init__(self, command):
        self.exe = shutil.which(command) or os.path.expanduser(f'~/.local/bin/{command}')

    def _run(self, args, timeout):
        try:
            proc = subprocess.run([self.exe, *args], capture_output=True, text=True, timeout=timeout,
                                  cwd=os.path.expanduser('~'))
            return proc.returncode, proc.stdout + proc.stderr
        except subprocess.TimeoutExpired as exc:
            return 124, f'timeout after {timeout}s: {exc}'

    def new(self, session, high_mem):
        code, out = self._run(['new', '-s', session] + (['--high-mem'] if high_mem else []), 600)
        return 'Session READY' in out, out

    def upload(self, session, local, remote):
        code, out = self._run(['upload', '-s', session, str(local), remote], 900)
        return code == 0 and 'Error' not in out, out

    def exec_file(self, session, path, timeout):
        return self._run(['exec', '-s', session, '-f', str(path), '--timeout', str(timeout)], timeout + 120)[1]

    def download(self, session, remote, local):
        code, out = self._run(['download', '-s', session, remote, str(local)], 900)
        return code == 0 and Path(local).exists(), out

    def stop(self, session):
        return self._run(['stop', '-s', session], 300)

    def sessions(self):
        return self._run(['sessions'], 120)[1]


class ColabRun:
    def __init__(self, payload, out, run_name, vms, poll=60.0, requirements=HERE / 'colab_requirements.txt',
                 cli_factory=ColabCLI, workers=None, log=print, traces=True, detach=False, jobs=None):
        # absolute: the CLI runs from the home directory, so relative paths would point there
        self.payload, self.out, self.run_name = Path(payload).resolve(), Path(out).resolve(), run_name
        self.jobs = Path(jobs).resolve() if jobs else self.payload / 'jobs.json'   # the games this run plays
        self.vms = [(v.split(':')[0], v.split(':')[1]) for v in vms]
        for _, shape in self.vms:
            if shape not in WORKERS:
                raise ValueError(f'unknown VM shape {shape!r}; use hm or std')
        self.workers = [workers or WORKERS[shape] for _, shape in self.vms]
        self.poll, self.requirements, self.cli_factory, self.log = poll, Path(requirements), cli_factory, log
        self.stop_event = threading.Event()
        self.parts = self.out.parent / (self.out.name + '.parts')
        self.traces = self.out.parent / 'traces' if traces else None   # arena.py compact traces, one per game
        self.detach = detach   # start the games and leave the VMs running; --attach pulls them later
        self.lock = threading.Lock()

    def sessions(self):
        return [f'{self.run_name}-{i}' for i in range(len(self.vms))]

    def pending(self):
        jobs = json.loads(self.jobs.read_text())['jobs']
        done = finished(self.out)
        return [j for j in jobs if job_key(j) not in done], len(jobs)

    def cleanup(self):
        for (command, _), session in zip(self.vms, self.sessions()):
            self.cli_factory(command).stop(session)
            self.log(f'[{session}] stop requested')

    def attach(self):
        """Follow the sessions of a run with the same --run-name and --vm flags (e.g. after the
        runner was killed): wait for them to finish, download, stop and merge."""
        self.log(f'{self.run_name}: attaching to {", ".join(self.sessions())}')
        self.parts.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            self.join([threading.Thread(target=self.follow_vm, args=(i, Path(tmp)), daemon=True)
                       for i in range(len(self.vms))])
        return self.summary()

    def join(self, threads):
        for t in threads:
            t.start()
        try:
            while any(t.is_alive() for t in threads):
                for t in threads:
                    t.join(timeout=1.0)
        except KeyboardInterrupt:
            self.log('interrupted: stopping every VM')
            self.stop_event.set()
            for t in threads:
                t.join()

    def run(self):
        todo, total = self.pending()
        self.log(f'{self.run_name}: {len(todo)} of {total} games to play on up to {len(self.vms)} VMs')
        if not todo:
            return self.summary()
        live = self.create_all()
        if not live:
            self.log('no VM could be created')
            return self.summary()
        # games are split over the VMs that exist, in proportion to their workers (hm 8, std 2)
        shards = dict(zip(live, assign(todo, [self.workers[i] for i in live])))
        for i in live:
            self.log(f'[{self.sessions()[i]}] {self.vms[i][0]} {self.vms[i][1]}: {len(shards[i])} games')
        self.parts.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            archive = tmp / 'payload.tar.gz'
            with tarfile.open(archive, 'w:gz') as tar:
                tar.add(self.payload, arcname='payload')
            for i in live:
                if not shards[i]:   # more VMs than games
                    self.finish(self.cli_factory(self.vms[i][0]), self.sessions()[i])
            self.join([threading.Thread(target=self.run_vm, args=(i, shards[i], archive, tmp), daemon=True)
                       for i in live if shards[i]])
        if self.detach:
            running = [self.sessions()[i] for i in live if shards[i]]
            self.log(f'COLAB_DETACHED {len(todo)} games on {", ".join(running)}: the VMs keep running (and '
                     f'spending units) until the same command with --attach pulls results and traces and '
                     f'stops them')
            return 0
        return self.summary()

    def create_all(self):
        """Create every VM in parallel; the indices of those that exist."""
        live, lock = [], threading.Lock()

        def create(i):
            command, shape = self.vms[i]
            session = self.sessions()[i]
            cli = self.cli_factory(command)
            ok, out = cli.new(session, shape == 'hm')
            if ok:
                with lock:
                    live.append(i)
            else:
                # a create that reports an error can still leave a VM behind (seen 2026-09-24: 5
                # High-RAM VMs created in parallel all "failed" yet ran idle), so stop it anyway
                cli.stop(session)
                self.log(f'[{session}] {command} {shape} VM not created (stop sent): {out.strip()[-200:]}')

        self.join([threading.Thread(target=create, args=(i,), daemon=True) for i in range(len(self.vms))])
        return sorted(live)

    def run_vm(self, index, shard, archive, tmp):
        command, shape = self.vms[index]
        session = self.sessions()[index]
        cli = self.cli_factory(command)
        created, started = True, False   # create_all made the VM
        try:
            shard_file = tmp / f'{session}_shard.json'
            shard_file.write_text(json.dumps({'jobs': shard}))
            for local, remote in ((archive, REMOTE + '_payload.tar.gz'), (shard_file, REMOTE + '_shard.json'),
                                  (self.requirements, REMOTE + '_requirements.txt')):
                ok, out = cli.upload(session, local, remote)
                if not ok:
                    raise RuntimeError(f'upload of {Path(local).name} failed: {out.strip()[-300:]}')
            setup = tmp / f'{session}_setup.py'
            setup.write_text(SETUP.format(root=REMOTE, workers=self.workers[index],
                                          trace_args="['--trace-dir', 'traces']" if self.traces else '[]'))
            out = cli.exec_file(session, setup, 1200)
            marker = [l for l in out.splitlines() if l.startswith('COLAB_ARENA_STARTED')]
            if not marker:
                raise RuntimeError(f'setup failed: {out.strip()[-600:]}')
            started = True
            self.log(f'[{session}] {command} {shape}: {len(shard)} games, {self.workers[index]} workers '
                     f'({marker[0].split(maxsplit=2)[2]})')
            if self.detach:
                return
            self.watch(cli, session, len(shard), tmp)
        except Exception as exc:
            self.log(f'[{session}] error: {exc}')
        finally:
            if created and not (self.detach and started):
                self.finish(cli, session)   # a VM that failed to start is removed at once

    def follow_vm(self, index, tmp):
        """--attach: follow a session an earlier (killed) runner started, then finish it."""
        command, _ = self.vms[index]
        session = self.sessions()[index]
        cli = self.cli_factory(command)
        try:
            self.watch(cli, session, None, tmp)
        except Exception as exc:
            self.log(f'[{session}] error: {exc}')
        finally:
            self.finish(cli, session)

    def watch(self, cli, session, games, tmp):
        poll = tmp / f'{session}_poll.py'
        poll.write_text(POLL.format(root=REMOTE))
        failures, last = 0, None
        while not self.stop_event.wait(self.poll):
            status = parse_status(cli.exec_file(session, poll, 120))
            if status is None:
                failures += 1
                if failures >= 5:
                    raise RuntimeError('VM stopped answering')
                continue
            failures = 0
            if status['results'] != last:
                last = status['results']
                self.log(f'[{session}] {last}/{games or "?"} games')
            if status['done'] or not status['alive']:
                break

    def finish(self, cli, session):
        """Download the session's results and traces, then stop it and merge what it played."""
        part = self.parts / f'{session}.jsonl'
        ok, out = cli.download(session, REMOTE + '/results.jsonl', part)
        if not ok:
            self.log(f'[{session}] no results downloaded: {out.strip()[-200:]}')
        if self.traces:
            self.pull_traces(cli, session)
        cli.stop(session)
        self.log(f'[{session}] stopped')
        self.merge(part)

    def pull_traces(self, cli, session):
        pack = self.parts / f'{session}_pack.py'
        pack.write_text(PACK.format(root=REMOTE))
        packed = [l for l in cli.exec_file(session, pack, 600).splitlines() if l.startswith('COLAB_ARENA_PACKED')]
        archive = self.parts / f'{session}_traces.tgz'
        if not packed or not cli.download(session, REMOTE + '/traces.tgz', archive)[0]:
            self.log(f'[{session}] no traces downloaded')
            return
        self.traces.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive) as tar:
            tar.extractall(self.traces, filter='data')
        archive.unlink()
        self.log(f'[{session}] {packed[0].split()[1]} traces -> {self.traces}')

    def merge(self, part):
        with self.lock:
            if not Path(part).exists():
                return
            done = finished(self.out)
            new = [line for key, line in finished(part).items() if key not in done]
            with self.out.open('a') as fh:
                for line in new:
                    fh.write(line + '\n')

    def verify_removed(self):
        """After the results are pulled no session of this run may be left: list every
        account's sessions, stop any leftover once more and report what still remains. A VM
        the CLI lists as `[?]` has lost its local record (the CLI prunes records that one
        listing misses, seen 2026-09-24 after the PC woke up) and cannot be stopped by name,
        so it is reported as well."""
        left = []
        for command in sorted({c for c, _ in self.vms}):
            cli = self.cli_factory(command)
            mine = [s for s in self.sessions() if f'[{s}]' in cli.sessions()]
            for session in mine:
                cli.stop(session)
            listing = cli.sessions()
            left += [f'{command}:{s}' for s in mine if f'[{s}]' in listing]
            left += [f'{command}:{line.split()[1]} (no local record)' for line in listing.splitlines()
                     if line.startswith('[?] ')]
        self.log(f'COLAB_VMS_LEFT={len(left)}' + (f' {left} -- stop them by hand' if left else ''))
        return left

    def summary(self):
        self.verify_removed()
        todo, total = self.pending()
        per = {}
        for line in finished(self.out).values():
            row = json.loads(line)
            a, b = row['rewards'][row['a_seat']], row['rewards'][1 - row['a_seat']]
            s = per.setdefault(row['tag'], [0, 0, 0, 0.0])
            s[0] += a > b
            s[1] += a < b
            s[2] += a == b
            s[3] += a - b
        for tag, (w, l, t, m) in sorted(per.items()):
            self.log(f'COLAB_SUMMARY {tag} {w}W-{l}L-{t}T mean margin {m / max(1, w + l + t):+,.0f}')
        self.log(f'COLAB_DONE {total - len(todo)}/{total} games in {self.out}')
        return len(todo)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--payload', required=True, help='payload dir (payload.py): jobs.json, bundles/, arena.py')
    ap.add_argument('--out', required=True, help='results.jsonl; games already in it are not replayed')
    ap.add_argument('--jobs', default=None, help='play only these jobs ({"jobs": [...]}; default <payload>/jobs.json)')
    ap.add_argument('--run-name', required=True, help='session name prefix (<run-name>-<index>)')
    ap.add_argument('--vm', action='append', required=True, metavar='COMMAND:SHAPE',
                    help='one VM, e.g. colab2:hm (High-RAM) or colab4:std; repeat per VM')
    ap.add_argument('--workers', type=int, default=None, help='concurrent games per VM (default: 8 hm, 2 std)')
    ap.add_argument('--poll', type=float, default=60.0)
    ap.add_argument('--no-traces', action='store_true', help='do not keep per-game traces (<out dir>/traces/)')
    ap.add_argument('--cleanup', action='store_true', help='only stop the sessions this command would create')
    ap.add_argument('--start-only', action='store_true',
                    help='start the games and exit, leaving the VMs running; pull them later with --attach')
    ap.add_argument('--attach', action='store_true',
                    help='follow the sessions an earlier runner started, then download, stop and merge')
    args = ap.parse_args()
    runner = ColabRun(args.payload, args.out, args.run_name, args.vm, poll=args.poll, workers=args.workers,
                      traces=not args.no_traces, detach=args.start_only, jobs=args.jobs,
                      log=lambda m: print(f'{time.strftime("%H:%M:%S")} {m}', flush=True))
    if args.cleanup:
        runner.cleanup()
        return
    missing = runner.attach() if args.attach else runner.run()
    raise SystemExit(0 if missing == 0 else 3)


if __name__ == '__main__':
    main()
