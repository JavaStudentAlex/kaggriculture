#!/usr/bin/env python3
"""Train KAD-HP-1 on one Colab TPU VM through the Colab CLI (AGENTS.md section 3.1 rules).

    python3 arena/colab_train.py --account colab4 --tpu v5e1 --session kad-hp1 \
        --bootstrap runs/kad_hp1_colab/bootstrap --resume runs/kad_hp1_colab/resume \
        --out runs/kad_hp1_colab/pulled --budget-hours 24

--bootstrap holds colab_train_vm.py's inputs: code.tar.gz, launch.json and links.json (signed
download links of the encoded days; they carry their own token, so no Kaggle credential goes to
the VM, and they are never printed). --resume holds the run to continue (last.pt, best.pt,
metrics.jsonl, scores.json, config.json). The runner:
1. starts the VM (`new --tpu`; a create that reports failure can still make the VM, so the session
   list is checked before a retry) and prints its browser link;
2. uploads the bootstrap, the run files and colab_train_vm.py, and starts the latter detached;
3. polls every --poll s. The command also keeps the VM alive: Colab deletes a VM about an hour
   after the last command. It refreshes the VM's access token every 30 min (colab_readopt.py),
   pulls scores.json every --pull-every s, and the checkpoints (last.pt, best.pt) plus gzipped
   copies of metrics.jsonl and the VM's log every --checkpoint-every s, into --out;
4. ends when the job exits or --budget-hours after the VM started (the job's own --deadline-hours
   should end it first): pulls everything, stops the VM, checks the account's session list
   (`COLAB_VMS_LEFT=0`) and deletes the session's CLI history.
If Colab deletes the VM, the runner ends with `COLAB_TRAIN_VM_LOST`. The checkpoints in --out are at
most --checkpoint-every old; resume with fresh links (they expire) and --resume <out>.
`--attach` follows a VM that a killed runner left running (no create, upload or start).
Run it in tmux (kagg-colab-<run>) on cliproxyapi, with the kagg-colab HOME and PATH.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from colab_run import ColabCLI, colab_link, parse_endpoints  # noqa: E402

REMOTE = '/content/kad'
RUN_FILES = ('last.pt', 'best.pt', 'metrics.jsonl', 'scores.json', 'config.json')
BOOTSTRAP = ('code.tar.gz', 'launch.json', 'links.json')
REFRESH_EVERY = 1800.0

START = r'''
import os, shutil, subprocess, sys
root, run_files = {root!r}, {run_files!r}
os.makedirs(root + '/run', exist_ok=True)
for name in {staged!r}:   # uploaded as <root>_<name>
    shutil.move(root + '_' + name, root + ('/run/' if name in run_files else '/') + name)
log = open(root + '/launch.log', 'a')
proc = subprocess.Popen([sys.executable, root + '/colab_train_vm.py'], cwd=root, stdout=log, stderr=subprocess.STDOUT,
                        start_new_session=True)
open(root + '/launch.pid', 'w').write(str(proc.pid))
print('KAD_STARTED', proc.pid)
'''

POLL = r'''
import json, os, re
root = {root!r}
path = root + '/launch.log'
text = ''
if os.path.exists(path):
    with open(path, 'rb') as f:
        f.seek(max(0, os.path.getsize(path) - 400000))
        text = f.read().decode(errors='replace')
lines = text.splitlines()
alive = False
try:
    alive = open('/proc/' + open(root + '/launch.pid').read().strip() + '/stat').read().split()[2] != 'Z'
except OSError:
    pass
marks = [l for l in lines if any(m in l for m in ('KAD_DAYS', 'JOB_STAGE', 'JOB_ERROR', 'JOB_WARN', 'JOB_EXIT',
                                                   'Traceback', 'Error'))]
train = [l for l in lines if l.startswith('{{') and '"event": "train"' in l]
val = [l for l in lines if l.startswith('{{') and '"event": "validation"' in l]
done = re.findall(r'KAD_COLAB_EXIT=(-?\d+)', text)
def stat(name):
    p = root + '/run/' + name
    return [os.path.getsize(p), int(os.path.getmtime(p))] if os.path.exists(p) else None
print('KAD_STATUS ' + json.dumps({{'alive': alive, 'exit': int(done[-1]) if done else None,
    'mark': marks[-1][:400] if marks else '', 'train': train[-1][:700] if train else '',
    'val': val[-1][:900] if val else '', 'tail': lines[-1][:400] if lines else '',
    'last': stat('last.pt'), 'best': stat('best.pt'), 'complete': os.path.exists(root + '/run/complete')}}))
'''

PACK = r'''
import gzip, os, shutil
root = {root!r}
os.makedirs(root + '/pull', exist_ok=True)
for src, name in ((root + '/run/metrics.jsonl', 'metrics.jsonl.gz'), (root + '/launch.log', 'launch.log.gz')):
    if os.path.exists(src):
        with open(src, 'rb') as a, gzip.open(root + '/pull/' + name + '.tmp', 'wb') as b:
            shutil.copyfileobj(a, b)
        os.replace(root + '/pull/' + name + '.tmp', root + '/pull/' + name)
print('KAD_PACKED')
'''


def log(message):
    print(f'{time.strftime("%Y-%m-%d %H:%M:%S")} {message}', flush=True)


class Trainer:
    def __init__(self, args):
        self.args = args
        self.cli = ColabCLI(args.account)
        self.session = args.session
        self.out = Path(args.out).resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.endpoint = None
        self.created = False
        self.pulled = {}   # remote run file -> [size, mtime] last pulled

    def script(self, name, template, **values):
        path = Path(self.tmp) / name
        path.write_text(template.format(root=REMOTE, **values))
        return path

    def exec(self, name, template, timeout=180, **values):
        return self.cli.exec_file(self.session, self.script(name, template, **values), timeout)

    def usage(self):
        out = self.cli._run(['usage'], 120)[1]
        return ' | '.join(line.strip() for line in out.splitlines() if 'balance' in line or 'rate' in line)

    def create(self):
        """The VM's endpoint, or None. A reported failure can still have made the VM: look first."""
        for attempt in range(1, 4):
            ok, out = self.cli._run(['new', '-s', self.session, '--tpu', self.args.tpu], 900)
            endpoint = parse_endpoints(self.cli.sessions()).get(self.session)
            if endpoint:
                return endpoint
            log(f'create attempt {attempt} failed: {out.strip().splitlines()[-1][:300] if out.strip() else ""}')
            time.sleep(60 * attempt)
        return None

    def upload_all(self):
        # absolute: the CLI runs from the home directory, so relative paths would point there
        bootstrap, resume = Path(self.args.bootstrap).resolve(), Path(self.args.resume).resolve()
        files = [(bootstrap / name, name) for name in BOOTSTRAP]
        files += [(resume / name, name) for name in RUN_FILES if (resume / name).exists()]
        files.append((HERE / 'colab_train_vm.py', 'colab_train_vm.py'))
        for local, name in files:
            started = time.time()
            ok, out = self.cli.upload(self.session, local, f'{REMOTE}_{name}')
            if not ok:
                raise RuntimeError(f'upload of {name} failed: {out[-300:]}')
            log(f'uploaded {name} ({local.stat().st_size / 1e6:.1f} MB, {time.time() - started:.0f} s)')
        return [name for _, name in files]

    def poll(self):
        out = self.exec('poll.py', POLL, 180)
        for line in out.splitlines():
            if line.startswith('KAD_STATUS '):
                return json.loads(line[len('KAD_STATUS '):])
        return None

    def pull(self, remote, name, expected=None):
        """Download REMOTE/remote to out/name through a temporary file; True when it arrived whole."""
        tmp = self.out / (name + '.part')
        ok, out = self.cli.download(self.session, f'{REMOTE}/{remote}', tmp)
        if ok and (expected is None or tmp.stat().st_size == expected):
            tmp.replace(self.out / name)
            return True
        tmp.unlink(missing_ok=True)
        log(f'pull of {remote} failed: {out.strip()[-200:]}')
        return False

    def pull_checkpoints(self, status, final=False):
        for name in ('last.pt', 'best.pt'):
            stat = status.get(name.split('.')[0]) if status else None
            if stat and stat != self.pulled.get(name):
                if self.pull(f'run/{name}', name, expected=stat[0]):
                    self.pulled[name] = stat
                    log(f'pulled {name} ({stat[0] / 1e6:.0f} MB)')
        if 'KAD_PACKED' in self.exec('pack.py', PACK, 600):
            for name in ('metrics.jsonl.gz', 'launch.log.gz'):
                self.pull(f'pull/{name}', name)
        for name in ('scores.json', 'config.json') + (('complete',) if final else ()):
            self.pull(f'run/{name}', name)

    def run(self):
        args = self.args
        with tempfile.TemporaryDirectory() as self.tmp:
            started_file = self.out / 'vm_started'
            if args.attach:
                self.endpoint = parse_endpoints(self.cli.sessions()).get(self.session)
                if not self.endpoint:
                    log(f'COLAB_TRAIN_ERROR no session {self.session} on {args.account} to attach to')
                    return 1
                self.created = True
                vm_started = float(started_file.read_text()) if started_file.exists() else time.time()
            else:
                log(f'{args.account} before: {self.usage()}')
                self.endpoint = self.create()
                if not self.endpoint:
                    log(f'COLAB_TRAIN_ERROR no {args.tpu} VM on {args.account}')
                    return 1
                self.created = True
                vm_started = time.time()
                started_file.write_text(str(vm_started))
                log(f'[{self.session}] {args.account} link: {colab_link(self.endpoint)}')
                log(f'{args.account} with the VM: {self.usage()}')
                staged = self.upload_all()
                out = self.exec('start.py', START, 300, staged=staged, run_files=list(RUN_FILES))
                if 'KAD_STARTED' not in out:
                    log(f'COLAB_TRAIN_ERROR start failed: {out[-500:]}')
                    return 1
                log('started colab_train_vm.py on the VM')
            return self.watch(vm_started)

    def watch(self, vm_started):
        args = self.args
        last_refresh = last_pull = last_checkpoint = last_train_log = last_usage = time.time()
        failures, seen_mark, seen_val, status = 0, None, None, None
        while True:
            now = time.time()
            new = self.poll()
            if new is None:
                failures += 1
                state = self.cli.refresh(self.session, self.endpoint)
                if state == 'deleted':
                    log('COLAB_TRAIN_VM_LOST Colab deleted the VM; the checkpoints in --out are the latest')
                    self.created = False
                    return 3
                if failures >= 10:
                    log('COLAB_TRAIN_ERROR ten polls in a row failed')
                    return 4
                time.sleep(args.poll)
                continue
            failures, status = 0, new
            if status['mark'] != seen_mark:
                seen_mark = status['mark']
                log(f'VM: {seen_mark}')
            if status['val'] != seen_val and status['val']:
                seen_val = status['val']
                log(f'VM validation: {seen_val}')
            if status['train'] and now - last_train_log >= 600:
                last_train_log = now
                log(f'VM train: {status["train"]}')
            if status['exit'] is not None or not status['alive']:
                log(f'job ended: exit {status["exit"]}, alive {status["alive"]}, complete {status["complete"]}; '
                    f'last line: {status["tail"]}')
                self.pull_checkpoints(status, final=True)
                return 0 if status['exit'] == 0 else 5
            if now - vm_started >= args.budget_hours * 3600:
                log(f'COLAB_TRAIN_BUDGET {args.budget_hours} h reached: pulling the checkpoints and stopping the VM')
                self.pull_checkpoints(status, final=True)
                return 6
            if now - last_refresh >= REFRESH_EVERY:
                last_refresh = now
                log(f'token refresh: {self.cli.refresh(self.session, self.endpoint)}')
            if now - last_usage >= 3600:
                last_usage = now
                log(f'{args.account}: {self.usage()}')
            if now - last_pull >= args.pull_every:
                last_pull = now
                self.pull('run/scores.json', 'scores.json')
            if now - last_checkpoint >= args.checkpoint_every:
                last_checkpoint = now
                self.pull_checkpoints(status)
            time.sleep(args.poll)

    def left(self):
        """(sessions on the account, the listing); the CLI's own notices ([colab] ...) are not sessions."""
        listing = self.cli.sessions()
        return len([name for name in parse_endpoints(listing) if name != 'colab']) + listing.count('[?]'), listing

    def finish(self):
        if self.created:
            log(f'stopping {self.session}: {self.cli.stop(self.session)[1].strip()[-200:]}')
        left, listing = self.left()
        if left and self.created:
            self.cli.stop(self.session)
            left, listing = self.left()
        log(f'{self.args.account} sessions after: {listing.strip()[-300:]}')
        print(f'COLAB_VMS_LEFT={left}', flush=True)
        if not left:
            Path(f'~/.config/colab-cli/history/{self.session}.jsonl').expanduser().unlink(missing_ok=True)
        log(f'{self.args.account} after: {self.usage()}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--account', default='colab4', help='the colab<N> wrapper of the account')
    ap.add_argument('--tpu', default='v5e1', help='v5e1 or v6e1')
    ap.add_argument('--session', default='kad-hp1')
    ap.add_argument('--bootstrap', help='dir with code.tar.gz, launch.json, links.json')
    ap.add_argument('--resume', help='dir with the run to continue (last.pt, best.pt, metrics.jsonl, ...)')
    ap.add_argument('--out', required=True, help='where checkpoints, scores and logs are pulled to')
    ap.add_argument('--budget-hours', type=float, default=24.0, help='stop the VM this long after it started')
    ap.add_argument('--poll', type=float, default=60.0)
    ap.add_argument('--pull-every', type=float, default=600.0)
    ap.add_argument('--checkpoint-every', type=float, default=3600.0)
    ap.add_argument('--attach', action='store_true', help='follow a VM a killed runner left running')
    args = ap.parse_args()
    if not args.attach and not (args.bootstrap and args.resume):
        ap.error('--bootstrap and --resume are required unless --attach')
    trainer = Trainer(args)
    code = 1
    try:
        code = trainer.run()
    except KeyboardInterrupt:
        log('interrupted')
        code = 130
    except Exception as exc:
        log(f'COLAB_TRAIN_ERROR {type(exc).__name__}: {str(exc)[:500]}')
    finally:
        trainer.finish()
    return code


if __name__ == '__main__':
    raise SystemExit(main())
