#!/usr/bin/env python3
"""Refit the opponent model (TTM) on one Colab GPU VM through the Colab CLI (AGENTS.md section 3.1 rules), because the
Kaggle GPU quota that ops/kaggle/train_5days.py uses was spent until 10-03 (2026-09-29).

    python3 arena/colab_refit.py --account colab2 --gpu L4 --session refit-0928 \
        --bootstrap runs/refit_0928/bootstrap --out runs/refit_0928/pulled

--bootstrap holds code.tar.gz (research/opponent_model's scripts), launch.json ({"days": [...5 days...], "lr": "2e-5"})
and base/ (the checkpoint to refit: model.safetensors, config.json, scaler.npz, labels.json). The runner:
1. starts the VM (`new --gpu`; a create that reports failure can still make the VM, so the session list is checked
   before a retry) and prints its browser link;
2. uploads the bootstrap and colab_refit_vm.py, and starts the latter detached (its stages: venv, download, shards,
   smoke, refit, result);
3. polls every --poll s, which also keeps the VM alive (Colab deletes a VM about an hour after the last command),
   refreshes its access token every 30 min (colab_readopt.py) and logs each new stage and evaluation;
4. when the job ends (or after --budget-hours), downloads result.tar.gz and the logs into --out, stops the VM, checks
   the account's session list (`COLAB_VMS_LEFT=0`) and deletes the session's CLI history.
Run it in tmux (kagg-colab-<run>) on cliproxyapi, with the kagg-colab HOME and PATH. Ends with COLAB_REFIT_EXIT=<code>.
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

REMOTE = '/content/refit'
BASE_FILES = ('model.safetensors', 'config.json', 'scaler.npz', 'labels.json')
REFRESH_EVERY = 1800.0

START = r'''
import os, shutil, subprocess, sys
root = {root!r}
os.makedirs(root + '/base', exist_ok=True)
for name in {staged!r}:   # uploaded as <root>_<name>
    dest = root + '/base/' + name[5:] if name.startswith('base_') else root + '/' + name
    shutil.move(root + '_' + name, dest)
log = open(root + '/launch.log', 'a')
proc = subprocess.Popen([sys.executable, root + '/colab_refit_vm.py'], cwd=root, stdout=log, stderr=subprocess.STDOUT,
                        start_new_session=True)
open(root + '/launch.pid', 'w').write(str(proc.pid))
print('REFIT_STARTED', proc.pid)
'''

POLL = r'''
import json, os, re
root = {root!r}
def tail(path, n=300000):
    if not os.path.exists(path):
        return ''
    with open(path, 'rb') as f:
        f.seek(max(0, os.path.getsize(path) - n))
        return f.read().decode(errors='replace')
text = tail(root + '/launch.log')
lines = text.splitlines()
train = tail(root + '/train.log').replace('\r', '\n').splitlines()
alive = False
try:
    alive = open('/proc/' + open(root + '/launch.pid').read().strip() + '/stat').read().split()[2] != 'Z'
except OSError:
    pass
stages = [l for l in lines if 'REFIT_STAGE' in l]
evals = [l for l in train if "'eval_auc_any_sell'" in l or '"eval_auc_any_sell"' in l]
steps = [l for l in train if "'loss'" in l and "'epoch'" in l]
done = re.findall(r'REFIT_EXIT=(-?\d+)', text)
errors = [l for l in lines if 'Error' in l or 'Traceback' in l]
print('REFIT_STATUS ' + json.dumps({{'alive': alive, 'exit': int(done[-1]) if done else None,
    'stage': stages[-1][:200] if stages else '', 'evals': [e[:600] for e in evals], 'step': steps[-1][:300] if steps else '',
    'tail': lines[-1][:400] if lines else '', 'error': errors[-1][:600] if errors else '',
    'result': os.path.exists(root + '/result.tar.gz')}}))
'''


def log(message):
    print(f'{time.strftime("%Y-%m-%d %H:%M:%S")} {message}', flush=True)


class Refit:
    def __init__(self, args):
        self.args = args
        self.cli = ColabCLI(args.account)
        self.session = args.session
        self.out = Path(args.out).resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.endpoint = None
        self.created = False

    def exec(self, name, template, timeout=180, **values):
        path = Path(self.tmp) / name
        path.write_text(template.format(root=REMOTE, **values))
        return self.cli.exec_file(self.session, path, timeout)

    def usage(self):
        out = self.cli._run(['usage'], 120)[1]
        return ' | '.join(line.strip() for line in out.splitlines() if 'balance' in line or 'rate' in line)

    def create(self):
        """The VM's endpoint, or None. A reported failure can still have made the VM: look first."""
        for attempt in range(1, 4):
            ok, out = self.cli.new(self.session, False, self.args.gpu)
            endpoint = parse_endpoints(self.cli.sessions()).get(self.session)
            if endpoint:
                return endpoint
            log(f'create attempt {attempt} failed: {out.strip().splitlines()[-1][:300] if out.strip() else ""}')
            time.sleep(60 * attempt)
        return None

    def upload_all(self):
        # absolute: the CLI runs from the home directory, so relative paths would point there
        bootstrap = Path(self.args.bootstrap).resolve()
        files = [(bootstrap / 'code.tar.gz', 'code.tar.gz'), (bootstrap / 'launch.json', 'launch.json'),
                 (HERE / 'colab_refit_vm.py', 'colab_refit_vm.py')]
        files += [(bootstrap / 'base' / name, f'base_{name}') for name in BASE_FILES]
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
            if line.startswith('REFIT_STATUS '):
                return json.loads(line[len('REFIT_STATUS '):])
        return None

    def pull(self, remote, name):
        tmp = self.out / (name + '.part')
        ok, out = self.cli.download(self.session, f'{REMOTE}/{remote}', tmp)
        if ok:
            tmp.replace(self.out / name)
            log(f'pulled {name} ({(self.out / name).stat().st_size / 1e6:.1f} MB)')
            return True
        tmp.unlink(missing_ok=True)
        log(f'pull of {remote} failed: {out.strip()[-200:]}')
        return False

    def pull_all(self):
        for remote in ('result.tar.gz', 'launch.log', 'train.log', 'smoke.log'):
            self.pull(remote, remote)

    def run(self):
        args = self.args
        with tempfile.TemporaryDirectory() as self.tmp:
            if args.attach:
                self.endpoint = parse_endpoints(self.cli.sessions()).get(self.session)
                if not self.endpoint:
                    log(f'COLAB_REFIT_ERROR no session {self.session} on {args.account} to attach to')
                    return 1
                self.created = True
            else:
                log(f'{args.account} before: {self.usage()}')
                self.endpoint = self.create()
                if not self.endpoint:
                    log(f'COLAB_REFIT_ERROR no {args.gpu} VM on {args.account}')
                    return 1
                self.created = True
                log(f'[{self.session}] {args.account} link: {colab_link(self.endpoint)}')
                log(f'{args.account} with the VM: {self.usage()}')
                staged = self.upload_all()
                out = self.exec('start.py', START, 300, staged=staged)
                if 'REFIT_STARTED' not in out:
                    log(f'COLAB_REFIT_ERROR start failed: {out[-500:]}')
                    return 1
                log('started colab_refit_vm.py on the VM')
            return self.watch()

    def watch(self):
        args = self.args
        started = last_refresh = last_usage = time.time()
        failures, seen_stage, seen_evals, seen_error, last_step_log = 0, None, 0, '', 0.0
        while True:
            now = time.time()
            status = self.poll()
            if status is None:
                failures += 1
                state = self.cli.refresh(self.session, self.endpoint)
                if state == 'deleted':
                    log('COLAB_REFIT_VM_LOST Colab deleted the VM')
                    self.created = False
                    return 3
                if failures >= 10:
                    log('COLAB_REFIT_ERROR ten polls in a row failed')
                    return 4
                time.sleep(args.poll)
                continue
            failures = 0
            if status['stage'] != seen_stage:
                seen_stage = status['stage']
                log(f'VM: {seen_stage}')
            for line in status['evals'][seen_evals:]:
                log(f'VM eval: {line}')
            seen_evals = len(status['evals'])
            if status['error'] and status['error'] != seen_error:
                seen_error = status['error']
                log(f'VM error line: {seen_error}')
            if status['step'] and now - last_step_log >= 900:
                last_step_log = now
                log(f'VM train: {status["step"]}')
            if status['exit'] is not None or not status['alive']:
                log(f'job ended: exit {status["exit"]}, alive {status["alive"]}, result {status["result"]}; '
                    f'last line: {status["tail"]}')
                self.pull_all()
                return 0 if status['exit'] == 0 and status['result'] else 5
            if now - started >= args.budget_hours * 3600:
                log(f'COLAB_REFIT_BUDGET {args.budget_hours} h reached: pulling what exists and stopping the VM')
                self.pull_all()
                return 6
            if now - last_refresh >= REFRESH_EVERY:
                last_refresh = now
                log(f'token refresh: {self.cli.refresh(self.session, self.endpoint)}')
            if now - last_usage >= 3600:
                last_usage = now
                log(f'{args.account}: {self.usage()}')
            time.sleep(args.poll)

    def left(self):
        """(sessions on the account, the listing); the CLI's own notices ([colab] ...) are not sessions."""
        listing = self.cli.sessions()
        return len([name for name in parse_endpoints(listing) if name != 'colab']) + listing.count('[?]'), listing

    def finish(self):
        """Stop our VM. COLAB_VMS_LEFT counts only this session: other runs (the evolution pool) share the account."""
        if self.created:
            log(f'stopping {self.session}: {self.cli.stop(self.session)[1].strip()[-200:]}')
        listing = self.cli.sessions()
        mine = 1 if self.session in parse_endpoints(listing) else 0
        if mine and self.created:
            self.cli.stop(self.session)
            listing = self.cli.sessions()
            mine = 1 if self.session in parse_endpoints(listing) else 0
        log(f'{self.args.account} sessions after: {listing.strip()[-400:]}')
        print(f'COLAB_VMS_LEFT={mine}', flush=True)
        if not mine:
            Path(f'~/.config/colab-cli/history/{self.session}.jsonl').expanduser().unlink(missing_ok=True)
        log(f'{self.args.account} after: {self.usage()}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--account', default='colab2', help='the colab<N> wrapper of the account')
    ap.add_argument('--gpu', default='L4', help='T4, L4, G4 or A100')
    ap.add_argument('--session', required=True)
    ap.add_argument('--bootstrap', help='dir with code.tar.gz, launch.json and base/')
    ap.add_argument('--out', required=True, help='where result.tar.gz and the logs are pulled to')
    ap.add_argument('--budget-hours', type=float, default=4.0, help='stop the VM this long after the runner started')
    ap.add_argument('--poll', type=float, default=60.0)
    ap.add_argument('--attach', action='store_true', help='follow a VM a killed runner left running')
    args = ap.parse_args()
    if not args.attach and not args.bootstrap:
        ap.error('--bootstrap is required unless --attach')
    refit = Refit(args)
    code = 1
    try:
        code = refit.run()
    except KeyboardInterrupt:
        log('interrupted')
        code = 130
    except Exception as exc:  # noqa: BLE001
        log(f'COLAB_REFIT_ERROR {type(exc).__name__}: {str(exc)[:500]}')
    finally:
        refit.finish()
    return code


if __name__ == '__main__':
    raise SystemExit(main())
