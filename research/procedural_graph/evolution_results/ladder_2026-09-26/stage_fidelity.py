"""Play a packaged submission through the Kaggle loader on gauntlet games where one of its stages changes the
margin, and require the arena's cached margin to the dollar (mirror_fidelity.py generalised, 2026-09-28; run on
cliproxyapi).

    python stage_fidelity.py ARCHIVE --with BUNDLE --without BUNDLE --opponent TAG [--opponent TAG ...] [--n 2]

BUNDLE ids are the run's arena bundles (runs/ladder1/scores/<id>.json): --with is the graph the archive packages,
--without the same graph without the stage (e.g. the champion before the rival emulator). For each --opponent, the
first --n plan jobs (alternating seats) whose cached margins differ between the two are played; the packaged agent
must reproduce --with's margin exactly, so the stage acts in the archive as it did in the arena.
"""
import argparse
import json
import os
import sys
import tarfile
import tempfile
from pathlib import Path

REPO = Path('/home/alex/kagg-evo/repo')
RUN = Path('/home/alex/kagg-evo/runs/ladder1')
sys.path.insert(0, str(REPO / 'shinka/evolution'))
from kaggle_environments import make  # noqa: E402
from pool_upgrade_bundle_agent import BundleAgent  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('archive', type=Path)
    ap.add_argument('--with', dest='with_stage', required=True)
    ap.add_argument('--without', required=True)
    ap.add_argument('--opponent', action='append', required=True)
    ap.add_argument('--n', type=int, default=2)
    args = ap.parse_args()
    with_stage = json.loads((RUN / f'scores/{args.with_stage}.json').read_text())['margins']
    without = json.loads((RUN / f'scores/{args.without}.json').read_text())['margins']
    plan = json.loads((REPO / 'research/procedural_graph/evolution_results/ladder_2026-09-26/plan.json').read_text())
    jobs = []
    for tag in args.opponent:
        seats, picked = [], 0
        for e in plan:
            k = f"{e['tag']}|{e['seed']}|{e['a_seat']}"
            if e['tag'] != tag or k not in with_stage or k not in without or with_stage[k] == without[k]:
                continue
            if picked and e['a_seat'] in seats and len(set(seats)) < 2:
                continue    # take the other seat next
            jobs.append((k, e))
            seats.append(e['a_seat'])
            picked += 1
            if picked == args.n:
                break
    ok = bool(jobs)
    with tempfile.TemporaryDirectory(prefix='kagg_stage_fidelity_') as tmp:
        x = Path(tmp) / 'x'
        x.mkdir()
        with tarfile.open(args.archive.resolve()) as tar:
            tar.extractall(x, filter='data')
        os.chdir('/')
        for k, e in jobs:
            opp = BundleAgent(RUN / 'bundles' / e['tag'], entrypoint='main.py', timeout=600.0)
            opp.start()
            agents = [None, None]
            agents[e['a_seat']] = str(x / 'main.py')
            agents[1 - e['a_seat']] = lambda obs, config, _o=opp: _o(obs, config)
            env = make('kaggriculture', configuration={'seed': e['seed']}, debug=False)
            env.run(agents)
            opp.close()
            r = [s.reward for s in env.steps[-1]]
            margin = r[e['a_seat']] - r[1 - e['a_seat']]
            same = margin == with_stage[k]
            ok = ok and same
            print(json.dumps({'job': k, 'rewards': r, 'margin': margin, 'arena_with': with_stage[k],
                              'arena_without': without[k], 'identical': same}), flush=True)
    print('STAGE FIDELITY', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
