#!/usr/bin/env python3
"""Paired validation of ladder-engine graphs on the Colab VM pool (run it where the pool runs).

    python ladder_validate.py --run_dir ~/kagg-evo/runs/ladder1 --pool_dir ~/kagg-evo/pool \
        --name validate1 --graph seed=seed --graph opening=island:Island-Opening \
        --graph combo=merge:Island-Opening+Island-Endgame --seeds 20 \
        --lost evolution_results/ladder_2026-09-26/new_losses.json

    # a backtest against the ladder: every graph against the rivals' recorded moves of our games
    python ladder_validate.py ... --name backtest1 --graph champion=island:Island-Opening --graph plain=seed \
        --seeds 0 --replays ~/kagg-evo/repo/shinka/champions/replay_opponents

Every graph plays the same jobs: --seeds arena validation seeds (salt 20260924, the seeds
arena/payload.py uses; evolution never selects on them) against every opponent bundle of the run,
the graph's seat alternating, plus every --lost entry ({"seed", "tag", ...}) from both seats
against its opponent, plus every --replays bundle (replay_<episode>, make_replay_opponents.py)
once, from the seat we played on the ladder: set replay:W, replay:L or replay:T by the ladder result.
Against a replay the graph that played the game reproduces its ladder margin, so a replay backtest
shows what another graph would have scored against the same rival moves. Graphs: `seed` (the run's seed graph), `island:<name>` (that island's
champion in the run's checkpoint), `merge:<A>+<B>+...` (island A's champion plus the non-default
settings of the other islands' champions, later ones winning a conflict), or a graph JSON file.

Bundles are built in the run directory (content-addressed: the evolution's own are reused) and the
jobs go to the pool as one request, like a gauntlet batch (results in <run_dir>/games/<name>.jsonl;
rerunning plays only missing games). The report (printed, and <run_dir>/validation/<name>.json)
gives each graph's W-L-T and mean margin per opponent and set, and every graph paired against the
first one on identical games (changed games W-L, mean change, exact sign test).
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import graph_edits  # noqa: E402
from arena import payload  # noqa: E402
from graph_gauntlet import VALIDATION_SALT, ColabPoolExecutor, Gauntlet, job_key, margins, sign_test  # noqa: E402

MERGE_KEYS = ('parameters', 'stages', 'engine_parameters', 'channels', 'experimental')


def validation_seeds(n):
    """The first n arena validation seeds (arena/payload.py --seeds n)."""
    return random.Random(VALIDATION_SALT).sample(range(10_000_000, 2_000_000_000), n)


def merged(base, others, seed, constants):
    """`base` with what each graph in `others` changed relative to `seed` (later ones win)."""
    start = graph_edits.settings(seed, constants)
    graph = base
    for other in others:
        s = graph_edits.settings(other, constants)
        edit = {}
        for key in MERGE_KEYS:
            changed = {name: value for name, value in (s.get(key) or {}).items()
                       if (start.get(key) or {}).get(name) != value}
            if changed:
                edit[key] = changed
        if edit:
            graph = graph_edits.apply_edit(graph, edit, constants)
    return graph


def resolve(spec, state, constants):
    islands = {i['name']: i['graph'] for i in state['islands']}
    if spec == 'seed':
        return state['seed_graph']
    if spec.startswith('island:'):
        return islands[spec[7:]]
    if spec.startswith('merge:'):
        names = spec[6:].split('+')
        return merged(islands[names[0]], [islands[n] for n in names[1:]], state['seed_graph'], constants)
    return json.loads(Path(spec).read_text())


def summarize(rows, labels):
    """Per graph: results by opponent and set; per graph after the first: paired changes."""
    by_label = {label: {} for label in labels}
    meta = {}
    for row in rows:
        label, opponent = row['tag'].split('@', 1)
        if label not in by_label:
            continue
        found, _ = margins([row], None)
        for key, margin in found.items():
            game = f"{opponent}|{row['seed']}|{row['a_seat']}"
            by_label[label][game] = margin
            meta[game] = row.get('set', 'validation')
    report = {'graphs': {}, 'paired': {}}
    for label, games in by_label.items():
        per = collections.defaultdict(lambda: {'wins': 0, 'losses': 0, 'ties': 0, 'sum': 0.0, 'n': 0})
        for game, margin in games.items():
            for key in (game.split('|')[0], f"set:{meta[game]}", 'all'):
                t = per[key]
                t['wins' if margin > 0 else 'losses' if margin < 0 else 'ties'] += 1
                t['sum'] += margin
                t['n'] += 1
        report['graphs'][label] = {k: {'wins': t['wins'], 'losses': t['losses'], 'ties': t['ties'], 'games': t['n'],
                                       'mean_margin': round(t['sum'] / t['n'], 1)} for k, t in sorted(per.items())}
    first = labels[0]
    for label in labels[1:]:
        per = collections.defaultdict(lambda: {'better': 0, 'worse': 0, 'same': 0, 'sum': 0.0, 'n': 0})
        for game, margin in by_label[label].items():
            if game not in by_label[first]:
                continue
            d = margin - by_label[first][game]
            for key in (game.split('|')[0], f"set:{meta[game]}", 'all'):
                t = per[key]
                t['better' if d > 0 else 'worse' if d < 0 else 'same'] += 1
                t['sum'] += d
                t['n'] += 1
        report['paired'][f'{label} vs {first}'] = {
            k: {'better': t['better'], 'worse': t['worse'], 'same': t['same'], 'mean_change': round(t['sum'] / t['n'], 1),
                'p': sign_test(t['better'], t['worse'])} for k, t in sorted(per.items())}
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run_dir', type=Path, required=True)
    ap.add_argument('--pool_dir', type=Path, required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--graph', action='append', required=True, metavar='LABEL=SPEC')
    ap.add_argument('--opponents', default='', help='comma list (default: every opponent bundle of the run)')
    ap.add_argument('--seeds', type=int, default=20, help='validation seeds per opponent')
    ap.add_argument('--lost', type=Path, default=None, help='JSON list of {"seed", "tag", ...}: played from both seats')
    ap.add_argument('--replays', type=Path, default=None,
                    help='directory of replay_<episode> bundles: each played once, from our ladder seat')
    ap.add_argument('--dry_run', action='store_true', help='build the bundles and the jobs, submit nothing')
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    state = json.loads((run_dir / 'checkpoint.json').read_text())
    constants = graph_edits.catalog()
    gauntlet = Gauntlet(run_dir, ColabPoolExecutor(args.pool_dir))
    labels, bundles = [], {}
    for item in args.graph:
        label, spec = item.split('=', 1)
        if '@' in label or label in bundles:
            raise SystemExit(f'bad or repeated label {label!r}')
        labels.append(label)
        bundles[label] = gauntlet.bundle(resolve(spec, state, constants))
        print(f'{label}: {spec} -> bundles/{bundles[label]}', flush=True)
    opponents = [o for o in args.opponents.split(',') if o] or sorted(
        p.name for p in (run_dir / 'bundles').iterdir() if p.is_dir() and not p.name.startswith(('g_', '.', 'replay_')))
    lost = json.loads(args.lost.read_text()) if args.lost else []
    replays = []
    for src in sorted(args.replays.glob('replay_*/SOURCE.json')) if args.replays else []:
        meta = json.loads(src.read_text())
        if not (run_dir / 'bundles' / meta['name']).is_dir():
            payload.build_opponent(run_dir, meta['name'])
        margin = meta.get('recorded_margin_for_us') or 0
        replays.append((meta, 'W' if margin > 0 else 'L' if margin < 0 else 'T'))
    jobs = []
    for label in labels:
        for opponent in opponents:
            jobs += [{'tag': f'{label}@{opponent}', 'a': f'bundles/{bundles[label]}', 'b': f'bundles/{opponent}',
                      'seed': s, 'a_seat': i % 2, 'set': 'validation'} for i, s in enumerate(validation_seeds(args.seeds))]
        for entry in lost:
            jobs += [{'tag': f"{label}@{entry['tag']}", 'a': f'bundles/{bundles[label]}', 'b': f"bundles/{entry['tag']}",
                      'seed': int(entry['seed']), 'a_seat': seat, 'set': 'lost'} for seat in (0, 1)]
        jobs += [{'tag': f"{label}@{meta['name']}", 'a': f'bundles/{bundles[label]}', 'b': f"bundles/{meta['name']}",
                  'seed': int(meta['seed']), 'a_seat': int(meta['our_seat']), 'set': f'replay:{result}'}
                 for meta, result in replays]
    keys = [job_key(j) for j in jobs]
    if len(set(keys)) != len(keys):
        raise SystemExit('duplicate jobs (a lost seed equal to a validation seed against the same opponent?)')
    missing = sorted({j['b'] for j in jobs if not (run_dir / j['b']).is_dir()})
    if missing:
        raise SystemExit(f'opponent bundles missing in the run directory: {missing}')
    print(f'{len(jobs)} jobs: {len(labels)} graphs x ({len(opponents)} opponents x {args.seeds} validation seeds '
          f'+ {len(lost)} lost seeds x 2 seats + {len(replays)} replays)', flush=True)
    if args.dry_run:
        return
    (run_dir / 'jobs' / f'{args.name}.json').write_text(json.dumps({'evaluation_id': args.name, 'jobs': jobs}))
    gauntlet.executor.run(run_dir, args.name, sorted(set(bundles.values()) | set(opponents)))
    rows = gauntlet._rows(args.name)
    wanted = set(keys)
    report = summarize([r for r in rows if job_key(r) in wanted], labels)
    report.update(name=args.name, graphs_spec=dict(zip(labels, args.graph)), bundles=bundles,
                  jobs=len(jobs), played=sum(1 for r in rows if job_key(r) in wanted))
    out = run_dir / 'validation' / f'{args.name}.json'
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=1) + '\n')
    for label, per in report['graphs'].items():
        a = per['all']
        print(f"{label}: {a['wins']}W-{a['losses']}L-{a['ties']}T, mean margin ${a['mean_margin']:+,.0f} | " + '; '.join(
            f"{k} {t['wins']}-{t['losses']}-{t['ties']} ${t['mean_margin']:+,.0f}" for k, t in per.items()
            if k != 'all' and not k.startswith('replay_')))
    for pair, per in report['paired'].items():
        a = per['all']
        print(f"{pair}: {a['better']} better, {a['worse']} worse, {a['same']} same, mean change ${a['mean_change']:+,.0f}, "
              f"p={a['p']:.2g} | " + '; '.join(f"{k} {t['better']}-{t['worse']} ${t['mean_change']:+,.0f}"
                                                for k, t in per.items() if k != 'all' and not k.startswith('replay_')))
    print(f'report: {out}')


if __name__ == '__main__':
    main()
