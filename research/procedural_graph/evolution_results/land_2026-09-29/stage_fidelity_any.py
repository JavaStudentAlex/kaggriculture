"""stage_fidelity.py for any run: score files and plan are arguments, and the games play in parallel processes
(2026-09-29, for Juniper Knoll; run on cliproxyapi).

    python stage_fidelity_any.py ARCHIVE --with_scores FILE --without_scores FILE --run DIR --plan FILE \
        --opponent TAG [--opponent TAG ...] [--n 2] [--workers 4]

--with_scores holds the arena's cached margins of the graph the archive packages, --without_scores those of a graph
without the change under test; opponents come from DIR/bundles. For each --opponent the first --n plan jobs
(alternating seats) whose two cached margins differ are played through the Kaggle loader, and the packaged agent must
reproduce the --with_scores margin to the dollar, with both agents DONE.

Every game runs in its own `python -I` process with an empty HOME and one BLAS thread, as make_graph_submission.py's
validation and fidelity games do and as Kaggle starts every episode in a fresh process. The first version played the
games in a reused process pool without those settings: the engine's module-level game state carried over into the next
game of a process (four games ended at the $3,000 starting cash) and three more games differed.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path('/home/alex/kagg-evo/repo')

GAME_PY = '''
import json, os, sys, tarfile, tempfile
sys.path.insert(0, sys.argv[4])
from kaggle_environments import make
from pool_upgrade_bundle_agent import BundleAgent
archive, run, job = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
with tempfile.TemporaryDirectory(prefix="kagg_stage_fidelity_") as tmp:
    x = os.path.join(tmp, "x")
    os.mkdir(x)
    with tarfile.open(archive) as tar:
        tar.extractall(x, filter="data")
    os.chdir("/")
    opp = BundleAgent(os.path.join(run, "bundles", job["tag"]), entrypoint="main.py", timeout=600.0)
    opp.start()
    try:
        agents = [None, None]
        agents[job["a_seat"]] = os.path.join(x, "main.py")
        agents[1 - job["a_seat"]] = lambda obs, config, _o=opp: _o(obs, config)
        env = make("kaggriculture", configuration={"seed": job["seed"]}, debug=False)
        env.run(agents)
    finally:
        opp.close()
    last = env.steps[-1]
    rewards = [s.reward for s in last]
    print("GAME_RESULT " + json.dumps({"rewards": rewards, "margin": rewards[job["a_seat"]] - rewards[1 - job["a_seat"]],
                                       "statuses": [s.status for s in last]}), flush=True)
'''


def play(py, script, archive, run, key, job, agent_dir):
    with tempfile.TemporaryDirectory(prefix='kagg_stage_home_') as home:
        env = {k: v for k, v in os.environ.items() if not k.startswith('KAGG_') and k != 'PYTHONPATH'}
        env.update({'HOME': home, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
        proc = subprocess.run([py, '-I', script, archive, run, json.dumps(job), agent_dir],
                              cwd='/', env=env, text=True, capture_output=True)
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith('GAME_RESULT ')]
    if proc.returncode or not lines:
        return key, None, None, [f'exit {proc.returncode}: {proc.stderr[-400:]}']
    out = json.loads(lines[-1][len('GAME_RESULT '):])
    return key, out['rewards'], out['margin'], out['statuses']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('archive', type=Path)
    ap.add_argument('--with_scores', type=Path, required=True)
    ap.add_argument('--without_scores', type=Path, required=True)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--plan', type=Path, required=True)
    ap.add_argument('--opponent', action='append', required=True)
    ap.add_argument('--n', type=int, default=2)
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    with_stage = json.loads(args.with_scores.read_text())['margins']
    without = json.loads(args.without_scores.read_text())['margins']
    plan = json.loads(args.plan.read_text())
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
    with tempfile.TemporaryDirectory(prefix='kagg_stage_script_') as tmp:
        script = Path(tmp) / 'game.py'
        script.write_text(GAME_PY)
        # only the bundle-agent helper next to the script, as the builder's fidelity game has it
        shutil.copy(REPO / 'shinka/evolution/pool_upgrade_bundle_agent.py', tmp)
        with ThreadPoolExecutor(args.workers) as pool:
            futures = [pool.submit(play, sys.executable, str(script), str(args.archive.resolve()), str(args.run.resolve()),
                                   k, e, tmp) for k, e in jobs]
            for f in futures:
                k, rewards, margin, statuses = f.result()
                same = margin == with_stage[k] and statuses == ['DONE', 'DONE']
                ok = ok and same
                print(json.dumps({'job': k, 'rewards': rewards, 'statuses': statuses, 'margin': margin,
                                  'arena_with': with_stage[k], 'arena_without': without[k], 'identical': same}),
                      flush=True)
    print(f'STAGE FIDELITY {"PASS" if ok else "FAIL"} ({len(jobs)} games)')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
