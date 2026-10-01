#!/usr/bin/env python3
"""Calibrate a refit predictor for our agent, end to end (AGENTS.md section 13).

    python3 research/procedural_graph/arena/calibrate.py --refit models/ttm_c256_h96_ft_2026-09-25

1. Payload: the committed game set (calibration/game_set: 600 arena games of feed15 with the
   reference predictor), the reference (hazel_runtime/checkpoint = ttm_c256_h96_ft_2026-09-13,
   the predictor the agent's thresholds were tuned with) and the refit.
2. On cliproxyapi (always on), colab_run.py runs in tmux `kagg-colab-<eval-id>` on one Colab T4
   High-RAM VM (colab2:t4hm). The VM replays every game, scores both models on every turn, and
   is stopped at the end. It takes about 40 min and a little under 1 compute unit.
3. Waits for COLAB_RUN_EXIT, checks COLAB_VMS_LEFT=0, pulls the results and the per-game npz
   files, and deletes the finished sessions' CLI history on the host.
4. Fits calibration/<refit>/own_games/: calibration.json (all games), calibration_pool.json and
   calibration_mirror.json (subsets), fit_report*.txt, jobs.json and a README. It prints the
   factors and each product's AUC for both models.

Rerunning with the same --eval-id resumes: a finished remote run is only pulled, a running one is
waited for. Run it in tmux (it waits ~40 min); a sleeping PC only pauses the waiting, since the
runner itself lives on cliproxyapi.

`--host local` runs every step on the machine itself (no ssh, plain copies): on cliproxyapi, from a repo copy that
has calibration/game_set, so the payload build and the fit do not run on the laptop (2026-09-29). The copy also
needs research/opponent_model/{extract,features,mechanics}.py: calib_payload.py bundles them for the VM.
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PG = HERE.parent
REPO = PG.parents[1]
REFERENCE = PG / 'hazel_runtime' / 'checkpoint'
GAME_SET = PG / 'calibration' / 'game_set'
GROUPS = {'': 'calibration.json', '^feed15(@|$)': 'calibration_pool.json', 'feed15new': 'calibration_mirror.json'}


def log(msg):
    print(time.strftime('%H:%M:%S'), msg, flush=True)


def ssh(host, command, check=True):
    if host == 'local':
        return subprocess.run(['bash', '-c', command], capture_output=True, text=True, check=check, timeout=600)
    return subprocess.run(['ssh', '-o', 'BatchMode=yes', host, command], capture_output=True, text=True,
                          check=check, timeout=600)


def at(host, path):
    """An rsync location on `host` ('local': the path itself)."""
    return str(path) if host == 'local' else f'{host}:{path}'


def remote_state(host, remote, eid):
    out = ssh(host, f'grep -a "COLAB_RUN_EXIT" {remote}/runs/{eid}/colab.log 2>/dev/null; '
                    f'tmux has-session -t kagg-colab-{eid} 2>/dev/null && echo RUNNING', check=False).stdout
    if 'COLAB_RUN_EXIT' in out:
        return 'finished'
    return 'running' if 'RUNNING' in out else 'absent'


def launch(args, eid):
    payload = PG / 'runs' / 'arena' / eid / 'payload'
    subprocess.run([sys.executable, str(HERE / 'calib_payload.py'), '--game-set', str(args.game_set),
                    '--model', f'old={args.reference}', '--model', f'new={args.refit}',
                    '--stride', '1', '--eval-id', eid], check=True, stdout=subprocess.DEVNULL)
    ssh(args.host, f'mkdir -p {args.remote}/runs/{eid}')
    subprocess.run(['rsync', '-a', str(payload), at(args.host, f'{args.remote}/runs/{eid}/')], check=True)
    vms = ' '.join(f'--vm {v}' for v in args.vm)
    inner = (f'cd {args.remote} && export HOME={args.remote}/home PATH={args.remote}/home/.local/bin:$PATH && '
             f'python3 arena/colab_run.py --payload runs/{eid}/payload --out runs/{eid}/results.jsonl '
             f'--run-name {eid} {vms} > runs/{eid}/colab.log 2>&1; '
             f'echo COLAB_RUN_EXIT=$? >> runs/{eid}/colab.log')
    ssh(args.host, f'tmux new -d -s kagg-colab-{eid} {shlex.quote(inner)}')
    log(f'started kagg-colab-{eid} on {args.host} ({len(args.vm)} VM)')


def wait(args, eid):
    seen, last = set(), 0.0
    while True:
        text = ssh(args.host, f'cat {args.remote}/runs/{eid}/colab.log 2>/dev/null', check=False).stdout
        for line in text.splitlines():
            if 'link:' in line and line not in seen:
                seen.add(line)
                log('VM ' + line.split(' ', 1)[1])
        if 'COLAB_RUN_EXIT' in text:
            return text
        progress = [ln for ln in text.splitlines() if re.search(r'\d+/\d+ games$', ln)]
        if progress and time.time() - last > 300:
            log(progress[-1].split(' ', 1)[1])
            last = time.time()
        time.sleep(60)


def auc_rows(report):
    """{(metric, product): (base, ref auc, refit auc)} from a calib_fit.py report."""
    rows, metric = {}, None
    for line in report.splitlines():
        if line.startswith('== '):
            metric = line.split()[1]
        m = re.match(r'  (\w+)\s+base\s+([\d.]+)%\s+AUC \w+ (\S+) \w+ (\S+)', line)
        if m and metric:
            rows[(metric, m[1])] = (float(m[2]), m[3], m[4])
    return rows


def fit(args, eid):
    run = PG / 'runs' / 'arena' / eid
    out = Path(args.out) if args.out else PG / 'calibration' / Path(args.refit).name / 'own_games'
    out.mkdir(parents=True, exist_ok=True)
    reports = {}
    for group, name in GROUPS.items():
        cmd = [sys.executable, str(HERE / 'calib_fit.py'), str(run / 'traces'), '--manifest',
               str(run / 'payload' / 'jobs.json'), '--out', str(out / name)] + (['--group', group] if group else [])
        text = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
        report = out / name.replace('calibration', 'fit_report').replace('.json', '.txt')
        report.write_text(text)
        reports[name] = text
    (out / 'jobs.json').write_text((run / 'payload' / 'jobs.json').read_text())
    factors = {n: json.loads((out / n).read_text())['factors'] for n in GROUPS.values()}
    ref = json.loads((out / 'calibration.json').read_text()).get('reference') or {}
    ref_label = (f"{ref.get('dir', args.reference)}, model sha256 {ref.get('model_sha256', '')[:12]}…"
                 if isinstance(ref, dict) else str(ref))
    aucs = auc_rows(reports['calibration.json'])
    products = list(factors['calibration.json']['score_4'])
    lines = [f'# Calibration of {Path(args.refit).name} for our agent (own games)', '',
             f'Made by `arena/calibrate.py` on {time.strftime("%Y-%m-%d")}: the committed game set '
             f'(`calibration/game_set`, 600 games), reference `{ref_label}` '
             f'(the predictor the thresholds were tuned with), every turn, one '
             f'Colab T4. `calibration.json` fits all games; `_pool` only feed15 vs Mohui and the champions, '
             f'`_mirror` only feed15 vs feed15. Reports: `fit_report*.txt`.', '',
             '| product | base rate | score_4 AUC reference | score_4 AUC refit | factors score_4 / score_24 / units_24 |',
             '|---|---|---|---|---|']
    for p in products:
        base, ra, na = aucs.get(('score_4', p), (0.0, 'nan', 'nan'))
        f = factors['calibration.json']
        lines.append(f'| {p} | {base:.1f} % | {ra} | {na} | '
                     f'{f["score_4"][p]:.3f} / {f["score_24"][p]:.3f} / {f["units_24"][p]:.3f} |')
    lines += ['', 'AUC is unchanged by any calibration: where the refit ranks worse than the reference, '
              'the calibrated agent still predicts our opponents worse. Check with the rematch '
              '(AGENTS.md section 13) before the refit plays.', '']
    (out / 'README.md').write_text('\n'.join(lines))
    print('\n'.join(lines))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--refit', required=True, help='the refit checkpoint dir, e.g. models/ttm_c256_h96_ft_2026-09-25')
    ap.add_argument('--reference', default=str(REFERENCE), help='the predictor the thresholds were tuned with')
    ap.add_argument('--game-set', default=str(GAME_SET))
    ap.add_argument('--eval-id', help='default: calib-<refit dir name>')
    ap.add_argument('--vm', action='append', default=None, help='default: colab2:t4hm (one T4 High-RAM VM)')
    ap.add_argument('--out', help='output dir (default: calibration/<refit dir name>/own_games)')
    ap.add_argument('--host', default='cliproxyapi', help="the runner's host, or 'local' when run on it")
    ap.add_argument('--remote', default='/home/alex/kagg-colab')
    args = ap.parse_args()
    args.vm = args.vm or ['colab2:t4hm']
    args.refit, args.reference = str(Path(args.refit).resolve()), str(Path(args.reference).resolve())
    eid = args.eval_id or f'calib-{Path(args.refit).name}'
    state = remote_state(args.host, args.remote, eid)
    if state == 'absent':
        launch(args, eid)
    else:
        log(f'remote run {eid} is {state}: resuming')
    text = wait(args, eid)
    code = re.findall(r'COLAB_RUN_EXIT=(\d+)', text)[-1]
    left = re.findall(r'COLAB_VMS_LEFT=(.*)', text)
    log(f'runner exit {code}; VMs left: {left[-1] if left else "?"}')
    run = PG / 'runs' / 'arena' / eid
    run.mkdir(parents=True, exist_ok=True)
    subprocess.run(['rsync', '-a', at(args.host, f'{args.remote}/runs/{eid}/results.jsonl'),
                    at(args.host, f'{args.remote}/runs/{eid}/traces'), at(args.host, f'{args.remote}/runs/{eid}/colab.log'),
                    str(run) + '/'], check=True)
    ssh(args.host, f'rm -f {args.remote}/home/.config/colab-cli/history/{eid}-*.jsonl', check=False)
    rows = [json.loads(line) for line in (run / 'results.jsonl').read_text().splitlines() if line.strip()]
    bad = [r for r in rows if r.get('errors') or not r.get('statuses')]
    jobs = len(json.loads((run / 'payload' / 'jobs.json').read_text())['jobs'])
    log(f'{len(rows)} of {jobs} games scored, {len(bad)} failed')
    if code != '0' or not left or left[-1].strip() != '0' or len(rows) - len(bad) < jobs:
        raise SystemExit('the run did not finish cleanly: read the log above, stop any VM left '
                         '(colab<N> stop -s <session>) and rerun this command to resume')
    out = fit(args, eid)
    shown = out.relative_to(REPO) if out.is_relative_to(REPO) else out
    log(f'wrote {shown}: commit it (git add {shown})')


if __name__ == '__main__':
    main()
