"""Where did the KAD copilot help, and where did it hurt? (2026-09-29; cliproxyapi, after kad_experiment.py)

    python where_kad_helped.py --run ~/kagg-evo/kadexp/run --variant sell [--top 12] [--workers 3]

Takes the variant's games from <run>/kad_experiment.json (per game: Juniper's cached margin and the variant's), replays
the --top games it improved most and the --top it worsened most, each in its own process with the variant's bundle and
the same opponent, seed and seat as in the pool (the replay must reproduce the pool's margin to the dollar), with
KAD_COPILOT_LOG set, so every intervention is logged. Prints and writes <run>/where_<variant>.json:
- per game: margin change, interventions by kind (sale of a product with units, hand job) and by game day;
- over the improved and the worsened games: interventions by day and by kind, side by side.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

GAME_PY = r'''
import json, os, sys
sys.path.insert(0, sys.argv[1])
from kaggle_environments import make
from pool_upgrade_bundle_agent import BundleAgent
ours, theirs, seed, seat = sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
os.chdir('/')
a = BundleAgent(ours, entrypoint='main.py', timeout=600.0)
b = BundleAgent(theirs, entrypoint='main.py', timeout=600.0)
a.start(); b.start()
try:
    agents = [None, None]
    agents[seat] = lambda obs, config: a(obs, config)
    agents[1 - seat] = lambda obs, config: b(obs, config)
    env = make('kaggriculture', configuration={'seed': seed}, debug=False)
    env.run(agents)
finally:
    a.close(); b.close()
r = [s.reward for s in env.steps[-1]]
print('GAME_RESULT ' + json.dumps({'rewards': r, 'margin': r[seat] - r[1 - seat],
                                   'statuses': [s.status for s in env.steps[-1]]}), flush=True)
'''


def play(py, script, helper_dir, run, bundle, key, workdir):
    tag, seed, seat = key.rsplit('|', 2)
    log = Path(workdir) / (key.replace('|', '_') + '.jsonl')
    log.unlink(missing_ok=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith('KAGG_') and k != 'PYTHONPATH'}
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', KAD_COPILOT_LOG=str(log))
    done = subprocess.run([py, script, helper_dir, str(run / 'bundles' / bundle), str(run / 'bundles' / tag), seed, seat],
                          cwd='/', env=env, text=True, capture_output=True)
    line = next((l for l in done.stdout.splitlines() if l.startswith('GAME_RESULT ')), None)
    result = json.loads(line[len('GAME_RESULT '):]) if line else {'error': done.stderr[-600:]}
    rows = [json.loads(l) for l in log.read_text().splitlines()] if log.exists() else []
    return key, result, rows


def summarize(rows):
    kinds, days, units = Counter(), Counter(), Counter()
    for r in rows:
        if r['kind'] == 'sell':
            kind = f"sell {r['item']}"
            units[r['item']] += r['qty']
        elif r['kind'] == 'hand':
            kind = f"hand {r['op']} {r.get('crop') or r.get('animal') or ''}".strip()
        else:
            kind = r['kind']
        kinds[kind] += 1
        days[r['step'] // 24] += 1
    return kinds, days, units


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--variant', required=True)
    ap.add_argument('--top', type=int, default=12)
    ap.add_argument('--workers', type=int, default=3)
    args = ap.parse_args()
    exp = json.loads((args.run / 'kad_experiment.json').read_text())
    bundle = exp['bundles'][args.variant]
    games = {k: v for k, v in exp['games'][args.variant].items() if v[0] is not None}
    ranked = sorted(games, key=lambda k: games[k][1] - games[k][0])
    worst = [k for k in ranked[:args.top] if games[k][1] < games[k][0]]
    best = [k for k in ranked[::-1][:args.top] if games[k][1] > games[k][0]]
    helper = Path(__file__).resolve().parents[3] / 'shinka' / 'evolution' / 'pool_upgrade_bundle_agent.py'
    results = {}
    with tempfile.TemporaryDirectory(prefix='kad_where_') as tmp:
        script = Path(tmp) / 'game.py'
        script.write_text(GAME_PY)
        shutil.copy(helper, tmp)
        with ThreadPoolExecutor(args.workers) as pool:
            futures = [pool.submit(play, sys.executable, str(script), tmp, args.run.resolve(), bundle, k, tmp)
                       for k in best + worst]
            for f in futures:
                key, result, rows = f.result()
                results[key] = (result, rows)
    report = {'variant': args.variant, 'bundle': bundle, 'games': {}}
    sides = {'improved': best, 'worsened': worst}
    agg = {side: (Counter(), Counter(), Counter()) for side in sides}
    fidelity = 0
    for side, keys in sides.items():
        print(f'\n== {side} games ({len(keys)}): change = variant margin - Juniper margin')
        for k in keys:
            result, rows = results[k]
            same = result.get('margin') == games[k][1]
            fidelity += same
            kinds, days, units = summarize(rows)
            for total, part in zip(agg[side], (kinds, days, units)):
                total.update(part)
            change = games[k][1] - games[k][0]
            print(f'{k:42s} {change:+8.0f}  replay {"==" if same else "!="} pool  {len(rows):3d} interventions  '
                  f'{dict(kinds.most_common(4))}  days {sorted(days)[:1]}-{sorted(days)[-1:]}')
            report['games'][k] = {'side': side, 'change': change, 'replay_equal': same, 'kinds': dict(kinds),
                                  'days': dict(days), 'units': dict(units), 'result': result}
    print(f'\nreplays equal to the pool: {fidelity} of {len(best) + len(worst)}')
    for side in sides:
        kinds, days, units = agg[side]
        print(f'\n{side}: interventions by kind {dict(kinds.most_common(10))}')
        print(f'{side}: by day {dict(sorted(days.items()))}')
        print(f'{side}: units sold by KAD {dict(units.most_common())}')
    report['aggregate'] = {side: {'kinds': dict(a[0]), 'days': dict(a[1]), 'units': dict(a[2])} for side, a in agg.items()}
    (args.run / f'where_{args.variant}.json').write_text(json.dumps(report, indent=1) + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
