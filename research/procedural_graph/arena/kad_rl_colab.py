#!/usr/bin/env python3
"""Run KAD-HP-1's expert iteration (research/action_diffusion/kad_rl.py) on a Colab VM through the Colab CLI,
under AGENTS.md section 3.1's rules.

    python3 arena/kad_rl_colab.py --account colab4 --vm gpu:L4 --session kadrl \
        --bootstrap runs/kadrl/kadrl.tar.gz --out runs/kadrl/pulled [--smoke-first | --smoke-only]

--bootstrap is `kad_rl.py pack`'s kadrl.tar.gz. --vm is gpu:<T4|L4|G4|A100|H100> (the model plays on the
GPU, which also fine-tunes) or tpu:<v5e1|v6e1> (the model plays on the CPUs, the TPU fine-tunes); the
bootstrap's config must say the same (device/accelerator). The runner:
1. makes the VM (a create that reports failure can still make one, so the session list decides), logs
   its browser link, uploads the bootstrap and starts `kad_rl.py run` detached in /content/kadrl
   (--smoke-first: first a --smoke run in /content/kadrl_smoke, and the real one only if it exits 0);
2. polls every --poll s (the command also keeps the VM alive: Colab deletes a VM about an hour after the
   last command), logs the loop's KAD_RL events, refreshes the VM's token every 30 min
   (colab_readopt.py), and every --pull-every s pulls the loop's state (the VM packs state.json, the
   results, the kept games and the logs) into --out/state and its models into --out/models (only the
   current model and the newest candidate stay there);
3. moves the loop to a new VM, uploading the pulled state and models, when the VM reaches
   --rotate-hours (a PAUSE file: the loop stops at its next check and exits 75; Colab ends VMs at 12 h)
   or when Colab deletes it (then from the last pull; up to --replacements times);
4. ends when the loop exits, at --max-hours or when the account's balance falls below --min-units (a
   STOP file: the loop stops at its next check): pulls everything, stops the VM, checks the account's
   session list (`COLAB_VMS_LEFT=0`) and deletes the sessions' CLI history.
A STOP or PAUSE file put in --out goes to the loop at the next poll (`touch <out>/STOP` ends the run
cleanly; PAUSE moves the loop to a new VM, with a newly packed --bootstrap's settings).
`--attach` follows a VM that a killed runner left running (session named by --session). Run it in tmux
(kagg-colab-<run>) on cliproxyapi, with the kagg-colab HOME and PATH.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from colab_run import ColabCLI, colab_link, parse_endpoints  # noqa: E402

ROOT = '/content/kadrl'
SMOKE_ROOT = '/content/kadrl_smoke'
BOOT = '/content/kadrl_bootstrap.tar.gz'
STATE = '/content/kadrl_state.tgz'
EXIT_PAUSED = 75
REFRESH_EVERY = 1800.0

START = r'''
import os, subprocess, sys, tarfile, time
root, boot, state, smoke = {root!r}, {boot!r}, {state!r}, {smoke!r}
os.makedirs(root, exist_ok=True)
if not os.path.exists(root + '/.bootstrap_ok'):   # a marker, not config.json: a retried start may meet a partial one
    with tarfile.open(boot) as t:
        t.extractall(root, filter='data')
    open(root + '/.bootstrap_ok', 'w').write('ok\n')
if state and os.path.exists(state):
    with tarfile.open(state) as t:
        t.extractall(root, filter='data')
    os.remove(state)
for flag in ('PAUSE', 'STOP'):
    if os.path.exists(root + '/' + flag):
        os.remove(root + '/' + flag)
if os.path.exists(root + '/rl.log'):   # a fresh log, so an old exit line is not read as this run's
    os.replace(root + '/rl.log', root + '/rl.log.' + time.strftime('%Y%m%d%H%M%S'))
log = open(root + '/rl.log', 'a')
command = [sys.executable, root + '/code/kad_rl.py', 'run', '--root', root] + (['--smoke'] if smoke else [])
proc = subprocess.Popen(command, cwd=root + '/code', stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
open(root + '/rl.pid', 'w').write(str(proc.pid))
print('KAD_RL_STARTED', proc.pid)
'''

POLL = r'''
import json, os, re
root = {root!r}
text = ''
path = root + '/rl.log'
if os.path.exists(path):
    with open(path, 'rb') as f:
        f.seek(max(0, os.path.getsize(path) - 400000))
        text = f.read().decode(errors='replace')
lines = text.splitlines()
alive = False
try:
    alive = open('/proc/' + open(root + '/rl.pid').read().strip() + '/stat').read().split()[2] != 'Z'
except OSError:
    pass
done = re.findall(r'KAD_RL_EXIT=(-?\d+)', text)
errors = [l for l in lines if 'KAD_RL_ERROR' in l or 'Error' in l or 'Traceback' in l]
models = {{}}
if os.path.isdir(root + '/payload/models'):
    for name in os.listdir(root + '/payload/models'):
        if name.endswith('.pt'):
            models[name] = os.path.getsize(root + '/payload/models/' + name)
state = {{}}
try:
    state = json.load(open(root + '/state.json'))
except Exception:
    pass
print('KAD_RL_STATUS ' + json.dumps({{'alive': alive, 'exit': int(done[-1]) if done else None,
    'events': [l[:2000] for l in lines if l.startswith('KAD_RL ')][-40:], 'error': errors[-1][:1500] if errors else '',
    'tail': lines[-1][:300] if lines else '', 'iteration': state.get('iteration'), 'model': state.get('model'),
    'rejections': state.get('rejections'), 'models': models}}))
'''

PACK = r'''
import os, tarfile
root = {root!r}
with tarfile.open(root + '_pull.tgz.tmp', 'w:gz') as t:
    for name in ('state.json', 'summary.json', 'rl.log', 'own'):
        if os.path.exists(root + '/' + name):
            t.add(root + '/' + name, arcname=name)
    for d in sorted(os.listdir(root)):
        if d.startswith('it') and os.path.isdir(root + '/' + d):
            for f in ('results.jsonl', 'selected.jsonl', 'games.done', 'ft_summary.json', 'finetune.log',
                      'ft/metrics.jsonl', 'arena.log'):
                if os.path.exists(root + '/' + d + '/' + f):
                    t.add(root + '/' + d + '/' + f, arcname=d + '/' + f)
os.replace(root + '_pull.tgz.tmp', root + '_pull.tgz')
print('KAD_RL_PACKED', os.path.getsize(root + '_pull.tgz'))
'''

TOUCH = r'''
open({root!r} + '/' + {flag!r}, 'w').write('runner\n')
print('KAD_RL_FLAG', {flag!r})
'''


def log(message):
    print(f'{time.strftime("%Y-%m-%d %H:%M:%S")} {message}', flush=True)


class Runner:
    def __init__(self, args):
        self.args = args
        self.cli = ColabCLI(args.account)
        self.kind, _, self.variant = args.vm.partition(':')
        if self.kind not in ('gpu', 'tpu') or not self.variant:
            raise SystemExit('--vm is gpu:<type> or tpu:<type>')
        self.out = Path(args.out).resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / 'models').mkdir(exist_ok=True)
        self.session = self.endpoint = None
        self.sessions = []   # every session this runner made (its CLI history is deleted at the end)
        self.started = time.time()
        self.seen = set()
        self.tmp = None

    # --- Colab calls ------------------------------------------------------------------------------------
    def exec(self, name, template, timeout=180, **values):
        path = Path(self.tmp) / name
        path.write_text(template.format(**values))
        return self.cli.exec_file(self.session, path, timeout)

    def usage(self):
        out = self.cli._run(['usage'], 120)[1]
        return ' | '.join(line.strip() for line in out.splitlines() if 'balance' in line or 'rate' in line)

    def balance(self):
        for part in self.usage().split('|'):
            if 'balance' in part:
                try:
                    return float(part.split(':')[1].split()[0])
                except (IndexError, ValueError):
                    return None
        return None

    def create(self, session):
        flag = ['--gpu', self.variant] if self.kind == 'gpu' else ['--tpu', self.variant]
        for attempt in range(1, 4):
            ok, out = self.cli._run(['new', '-s', session, *flag], 900)
            endpoint = parse_endpoints(self.cli.sessions()).get(session)
            if endpoint:
                return endpoint
            log(f'create attempt {attempt} failed: {out.strip().splitlines()[-1][:300] if out.strip() else ""}')
            time.sleep(60 * attempt)
        return None

    def start_vm(self, number):
        self.session = self.args.session if number == 0 else f'{self.args.session}-{number}'
        log(f'{self.args.account} before: {self.usage()}')
        self.endpoint = self.create(self.session)
        if not self.endpoint:
            log(f'KAD_RL_RUNNER_ERROR no {self.args.vm} VM on {self.args.account}')
            return False
        self.sessions.append(self.session)
        self.vm_started = time.time()
        log(f'[{self.session}] {self.args.account} link: {colab_link(self.endpoint)}')
        log(f'{self.args.account} with the VM: {self.usage()}')
        ok, out = self.cli.upload(self.session, Path(self.args.bootstrap).resolve(), BOOT)
        if not ok:
            raise RuntimeError(f'bootstrap upload failed: {out[-300:]}')
        log(f'uploaded the bootstrap ({Path(self.args.bootstrap).stat().st_size / 1e6:.0f} MB)')
        return True

    def upload_state(self):
        """The pulled state and models, for the loop to resume on this VM; False when there is none."""
        state = self.out / 'state'
        if not (state / 'state.json').exists():
            return False
        path = Path(self.tmp) / 'state.tgz'
        with tarfile.open(path, 'w:gz') as tar:
            for item in sorted(state.iterdir()):
                if not item.name.startswith('rl.log'):
                    tar.add(item, arcname=item.name)
            for model in sorted((self.out / 'models').glob('*.pt')):
                tar.add(model, arcname=f'payload/models/{model.name}')
        ok, out = self.cli.upload(self.session, path, STATE)
        if not ok:
            raise RuntimeError(f'state upload failed: {out[-300:]}')
        log(f'uploaded the pulled state ({path.stat().st_size / 1e6:.0f} MB)')
        return True

    def launch(self, root, smoke=False, state=False):
        for attempt in range(1, 4):
            out = self.exec('start.py', START, 300, root=root, boot=BOOT, state=STATE if state else '', smoke=smoke)
            if 'KAD_RL_STARTED' in out:
                break
            # an exec can hang after the loop started (09-30 00:03 UTC the runner stopped the VM on such a hang):
            # look before starting a second loop
            status = self.poll(root)
            if status and status['alive']:
                log(f'start attempt {attempt} got no reply ({out.strip()[-200:]}), but the loop is running')
                break
            log(f'start attempt {attempt} failed: {out.strip()[-300:]}')
            if attempt == 3:
                raise RuntimeError(f'start failed: {out[-800:]}')
            time.sleep(30)
        log(f'started kad_rl.py run in {root}' + (' (--smoke)' if smoke else '') + (' from the pulled state' if state else ''))

    def poll(self, root):
        out = self.exec('poll.py', POLL, 180, root=root)
        for line in out.splitlines():
            if line.startswith('KAD_RL_STATUS '):
                return json.loads(line[len('KAD_RL_STATUS '):])
        return None

    def flag(self, root, name):
        out = self.exec('flag.py', TOUCH, 120, root=root, flag=name)
        log(f'{name} file on the VM: {"ok" if "KAD_RL_FLAG" in out else out[-300:]}')

    def pull(self, root, status, smoke=False):
        """The loop's state into out/state (out/smoke) and its models into out/models."""
        out = self.exec('pack.py', PACK, 900, root=root)
        if 'KAD_RL_PACKED' not in out:
            log(f'pack failed: {out[-300:]}')
            return False
        part = self.out / 'pull.tgz.part'
        ok, text = self.cli.download(self.session, root + '_pull.tgz', part)
        if not ok:
            log(f'state download failed: {text[-300:]}')
            part.unlink(missing_ok=True)
            return False
        dest = self.out / ('smoke' if smoke else 'state')
        fresh = dest.with_name(dest.name + '.new')
        shutil.rmtree(fresh, ignore_errors=True)
        fresh.mkdir()
        with tarfile.open(part) as tar:
            tar.extractall(fresh, filter='data')
        shutil.rmtree(dest, ignore_errors=True)
        fresh.rename(dest)
        part.unlink()
        if smoke or not status:
            return True
        wanted = {name: size for name, size in (status.get('models') or {}).items() if name != 'm0.pt'}
        keep = {f'{status.get("model")}.pt'}
        candidates = sorted((n for n in wanted if n.startswith('c')), key=lambda n: int(n[1:-3]) if n[1:-3].isdigit() else -1)
        if candidates:
            keep.add(candidates[-1])
        for name in sorted(keep & set(wanted)):
            local = self.out / 'models' / name
            if local.exists() and local.stat().st_size == wanted[name]:
                continue
            tmp = local.with_suffix('.part')
            ok, text = self.cli.download(self.session, f'{root}/payload/models/{name}', tmp)
            if ok and tmp.stat().st_size == wanted[name]:
                tmp.replace(local)
                log(f'pulled model {name} ({wanted[name] / 1e6:.0f} MB)')
            else:
                tmp.unlink(missing_ok=True)
                log(f'model {name} download failed: {text[-200:]}')
        for local in (self.out / 'models').glob('*.pt'):
            if local.name not in keep:
                local.unlink()
                log(f'removed the local copy of {local.name} (neither the current model nor the newest candidate)')
        return True

    # --- the watch --------------------------------------------------------------------------------------
    def watch(self, root, smoke=False):
        """Until the loop on this VM exits: ('exit', code), ('lost', None) or ('stopped', code)."""
        args = self.args
        last_pull = last_refresh = last_usage = time.time()
        failures, status, flagged = 0, None, None
        while True:
            new = self.poll(root)
            now = time.time()
            if new is None:
                failures += 1
                state = self.cli.refresh(self.session, self.endpoint)
                if state == 'deleted':
                    log('KAD_RL_VM_LOST Colab deleted the VM')
                    return 'lost', None
                if failures >= 10:
                    log('KAD_RL_VM_LOST ten polls in a row failed')
                    return 'lost', None
                time.sleep(args.poll)
                continue
            failures, status = 0, new
            for line in status['events']:
                if line not in self.seen:
                    self.seen.add(line)
                    log(f'VM: {line}')
            if status['exit'] is not None or not status['alive']:
                log(f'loop ended: exit {status["exit"]}, alive {status["alive"]}; last line: {status["tail"]}')
                if status['exit'] not in (0, EXIT_PAUSED) and status['error']:
                    log(f'VM error: {status["error"]}')
                self.pull(root, status, smoke)
                return 'exit', status['exit']
            if now - last_refresh >= REFRESH_EVERY:
                last_refresh = now
                log(f'token refresh: {self.cli.refresh(self.session, self.endpoint)}')
            if now - last_pull >= args.pull_every:
                last_pull = now
                self.pull(root, status, smoke)
            if now - last_usage >= 1800:
                last_usage = now
                balance = self.balance()
                log(f'{args.account}: {self.usage()}')
                if balance is not None and balance < args.min_units and flagged != 'STOP':
                    log(f'KAD_RL_BUDGET the balance {balance} is below --min-units {args.min_units}: stopping the loop')
                    self.flag(root, 'STOP')
                    flagged = 'STOP'
            if now - self.started >= args.max_hours * 3600 and flagged != 'STOP':
                log(f'KAD_RL_DEADLINE --max-hours {args.max_hours} reached: stopping the loop')
                self.flag(root, 'STOP')
                flagged = 'STOP'
            if not smoke and now - self.vm_started >= args.rotate_hours * 3600 and flagged is None:
                log(f'the VM is {args.rotate_hours} h old: pausing the loop to move it to a new VM')
                self.flag(root, 'PAUSE')
                flagged = 'PAUSE'
            for name in ('STOP', 'PAUSE'):   # a flag file in --out, e.g. `touch <out>/STOP` to end the run
                local = self.out / name
                if local.exists() and flagged != 'STOP':
                    log(f'{local} found: {name} for the loop')
                    self.flag(root, name)
                    flagged = name
                    local.unlink()
            time.sleep(args.poll)

    def stop_vm(self):
        if self.session:
            log(f'stopping {self.session}: {self.cli.stop(self.session)[1].strip()[-200:]}')
            self.session = None

    def run(self):
        args = self.args
        with tempfile.TemporaryDirectory() as self.tmp:
            number, lost = 0, 0
            if args.attach:
                self.session = args.session
                self.endpoint = parse_endpoints(self.cli.sessions()).get(self.session)
                if not self.endpoint:
                    log(f'KAD_RL_RUNNER_ERROR no session {self.session} to attach to')
                    return 1
                self.sessions.append(self.session)
                self.vm_started = time.time()
            elif not self.start_vm(number):
                return 1
            if (args.smoke_first or args.smoke_only) and not args.attach:
                self.launch(SMOKE_ROOT, smoke=True)
                outcome, code = self.watch(SMOKE_ROOT, smoke=True)
                log(f'KAD_RL_SMOKE {outcome} exit {code}')
                if outcome != 'exit' or code != 0 or args.smoke_only:
                    return 0 if outcome == 'exit' and code == 0 else 2
            if not args.attach:
                self.launch(ROOT, state=self.upload_state())
            while True:
                outcome, code = self.watch(ROOT)
                if outcome == 'exit' and code == EXIT_PAUSED:
                    log('moving the loop to a new VM')
                elif outcome == 'lost':
                    lost += 1
                    self.session = None   # Colab deleted it
                    if lost > args.replacements:
                        log(f'KAD_RL_RUNNER_ERROR {lost} VMs lost; the pulled state is in {self.out}')
                        return 3
                    log(f'replacing the lost VM ({lost} of {args.replacements}) from the last pull')
                else:
                    log(f'KAD_RL_RUNNER_DONE loop exit {code}; results in {self.out}')
                    return 0 if code == 0 else 4
                self.stop_vm()
                if time.time() - self.started >= args.max_hours * 3600:
                    log('KAD_RL_DEADLINE reached between VMs: not starting another')
                    return 0
                balance = self.balance()
                if balance is not None and balance < args.min_units:
                    log(f'KAD_RL_BUDGET the balance {balance} is below --min-units: not starting another VM')
                    return 0
                number += 1
                if not self.start_vm(number):
                    return 1
                self.launch(ROOT, state=self.upload_state())

    def finish(self):
        self.stop_vm()
        listing = self.cli.sessions()
        left = len([n for n in parse_endpoints(listing) if n != 'colab']) + listing.count('[?]')
        if left:
            for name in self.sessions:
                self.cli.stop(name)
            listing = self.cli.sessions()
            left = len([n for n in parse_endpoints(listing) if n != 'colab']) + listing.count('[?]')
        log(f'{self.args.account} sessions after: {listing.strip()[-300:]}')
        print(f'COLAB_VMS_LEFT={left}', flush=True)
        if not left:
            for name in self.sessions:
                Path(f'~/.config/colab-cli/history/{name}.jsonl').expanduser().unlink(missing_ok=True)
        log(f'{self.args.account} after: {self.usage()}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--account', default='colab4', help='the colab<N> wrapper of the account')
    ap.add_argument('--vm', default='gpu:L4', help='gpu:<T4|L4|G4|A100|H100> or tpu:<v5e1|v6e1>')
    ap.add_argument('--session', default='kadrl')
    ap.add_argument('--bootstrap', help="kad_rl.py pack's kadrl.tar.gz")
    ap.add_argument('--out', required=True, help='where the state, models and logs are pulled to')
    ap.add_argument('--smoke-first', action='store_true', help='a --smoke loop first; the real one only if it passes')
    ap.add_argument('--smoke-only', action='store_true', help='only the --smoke loop, then stop the VM')
    ap.add_argument('--poll', type=float, default=60.0)
    ap.add_argument('--pull-every', type=float, default=600.0)
    ap.add_argument('--rotate-hours', type=float, default=10.0, help='move the loop to a new VM at this VM age')
    ap.add_argument('--replacements', type=int, default=4, help='VMs that Colab deleted, replaced at most this often')
    ap.add_argument('--max-hours', type=float, default=48.0, help='stop the loop this long after the runner started')
    ap.add_argument('--min-units', type=float, default=15.0, help="stop the loop below this balance of the account")
    ap.add_argument('--attach', action='store_true', help='follow the loop of a VM a killed runner left running')
    args = ap.parse_args()
    if not args.attach and not args.bootstrap:
        ap.error('--bootstrap is required unless --attach')
    runner = Runner(args)
    code = 1
    try:
        code = runner.run()
    except KeyboardInterrupt:
        log('interrupted')
        code = 130
    except Exception as exc:
        log(f'KAD_RL_RUNNER_ERROR {type(exc).__name__}: {str(exc)[:800]}')
    finally:
        runner.finish()
    print(f'KAD_RL_RUNNER_EXIT={code}', flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
