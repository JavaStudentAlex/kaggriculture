"""Does the KAD copilot help Juniper Knoll? Paired games through the land run's Colab pool (2026-09-29; cliproxyapi).

    python kad_experiment.py --run ~/kagg-evo/kadexp/run --pool ~/kagg-evo/pool \
        --seed-graph <land seed_graph.json> --plan <land plan.json> --baseline <land run scores/g_d46eaa6322231c85.json>

The seed graph is Juniper Knoll's (the land run's seed; its margins on the plan's games are cached in --baseline).
Variants add the kad_copilot stage and nothing else; each plays the land gauntlet's first stage (the same 258 games
every candidate plays first) against the same opponents, seeds and seats, so every game pairs with the baseline:
  control  the seed graph rebuilt under this runtime (no KAD node): must equal the baseline, game for game
  sell     KAD's confident SELLs appended after the engine's orders, every 2nd turn from turn 8 to 719
  hands    idle hands do KAD's job where they stand, every 2nd turn from turn 8 to 719
Writes <run>/kad_experiment.json (per game: baseline and variant margins) and prints the paired comparison.
"""
import argparse
import json
import math
import sys
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
PG = HERE.parent
sys.path.insert(0, str(PG))
import graph_edits  # noqa: E402
import graph_gauntlet as G  # noqa: E402

WINDOW = {'_KC_EVERY': 2, '_KC_FROM_STEP': 8, '_KC_TO_STEP': 719}
VARIANTS = {
    'sell': {**WINDOW, '_KC_SELL': True, '_KC_HANDS': False},
    'hands': {**WINDOW, '_KC_SELL': False, '_KC_HANDS': True},
}
CONTROL_GAMES = 24


def sign_p(up, down):
    n = up + down
    if n == 0:
        return 1.0
    k = min(up, down)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def points(m):
    return 1.0 if m > 0 else 0.5 if m == 0 else 0.0


def compare(name, base, found):
    keys = sorted(k for k in found if k in base)
    d = [found[k] - base[k] for k in keys]
    better, worse = sum(x > 0 for x in d), sum(x < 0 for x in d)
    up = sum(points(found[k]) > points(base[k]) for k in keys)
    down = sum(points(found[k]) < points(base[k]) for k in keys)
    mean = sum(d) / len(d) if d else 0.0
    wins_b = sum(base[k] > 0 for k in keys)
    wins_v = sum(found[k] > 0 for k in keys)
    print(f'{name}: {len(keys)} games, {better} better / {worse} worse (p={sign_p(better, worse):.3g}), '
          f'mean {mean:+.0f} $/game, results +{up}/-{down} (p={sign_p(up, down):.3g}), wins {wins_b} -> {wins_v}',
          flush=True)
    by = {}
    for k in keys:
        tag = k.split('|')[0]
        tag = 'replay' if tag.startswith('replay_') else tag
        by.setdefault(tag, []).append(found[k] - base[k])
    for tag, ds in sorted(by.items(), key=lambda kv: sum(kv[1])):
        print(f'    {tag:28s} {len(ds):3d} games  {sum(x > 0 for x in ds):3d} better {sum(x < 0 for x in ds):3d} worse  '
              f'mean {sum(ds) / len(ds):+8.0f}', flush=True)
    return {'games': len(keys), 'better': better, 'worse': worse, 'mean': mean, 'results_up': up,
            'results_down': down}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--pool', type=Path, required=True)
    ap.add_argument('--seed-graph', type=Path, required=True)
    ap.add_argument('--plan', type=Path, required=True)
    ap.add_argument('--baseline', type=Path, required=True)
    ap.add_argument('--fraction', type=float, default=0.3)
    ap.add_argument('--only', default=None, help='comma list of control,sell,hands')
    args = ap.parse_args()
    seed = json.loads(args.seed_graph.read_text())
    plan = json.loads(args.plan.read_text())
    base = json.loads(args.baseline.read_text())['margins']
    gauntlet = G.Gauntlet(args.run, G.ColabPoolExecutor(args.pool), plan=plan, stage_fraction=args.fraction)
    gauntlet.prepare()
    graphs = {'control': seed}
    for name, params in VARIANTS.items():
        graphs[name] = graph_edits.apply_edit(seed, {'channels': {'kad_copilot': True}, 'parameters': params})
    wanted = args.only.split(',') if args.only else list(graphs)
    bundles = {name: gauntlet.bundle(graphs[name]) for name in wanted}
    for name, bundle in bundles.items():
        (args.run / f'{name}_graph.json').write_text(json.dumps(graphs[name], indent=2) + '\n')
        print(f'{name}: bundle {bundle}', flush=True)
    results, threads = {}, []

    def play(name):
        bundle = bundles[name]
        jobs = [j for j in gauntlet.jobs(bundle) if G.first_stage(G.job_key(j), args.fraction)]
        if name == 'control':
            jobs = jobs[::max(1, len(jobs) // CONTROL_GAMES)][:CONTROL_GAMES]
        rows = gauntlet._play(f'kad_{name}_{bundle}', jobs, sorted({bundle, *{j['tag'] for j in jobs}}))
        found, errors = G.margins(rows, bundle)
        results[name] = {'found': found, 'errors': errors, 'jobs': len(jobs)}
        print(f'{name}: {len(found)} of {len(jobs)} games played, {len(errors)} errors', flush=True)

    for name in wanted:
        t = threading.Thread(target=play, args=(name,), daemon=True)
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    summary = {}
    for name in wanted:
        found = results.get(name, {}).get('found', {})
        if name == 'control':
            same = sum(found[k] == base.get(k) for k in found)
            print(f'control: {same} of {len(found)} games equal the baseline '
                  f'({"INERT" if found and same == len(found) else "NOT INERT"})', flush=True)
            summary[name] = {'games': len(found), 'equal': same}
        else:
            summary[name] = compare(name, base, found)
    out = {'summary': summary, 'bundles': bundles,
           'games': {name: {k: [base.get(k), v] for k, v in results.get(name, {}).get('found', {}).items()}
                     for name in wanted}}
    (args.run / 'kad_experiment.json').write_text(json.dumps(out, indent=1) + '\n')
    print('KAD_EXPERIMENT_DONE', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
