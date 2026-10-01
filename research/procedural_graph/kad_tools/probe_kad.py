"""Play the start of one real game with a KAD graph and print what the copilot did (2026-09-29; cliproxyapi).

    python probe_kad.py GRAPH.json [--steps 80] [--seed 7] [--opponent starter]

Builds the graph's bundle (arena/payload.write_graph_bundle), plays --steps turns in a fresh process through
kaggle_environments against --opponent, and prints the copilot's report (calls, sales, hand jobs, failures), its latest
advice (what a tactic sees as info['kad']) and the mean time of a turn with and without a KAD call.
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from arena import payload  # noqa: E402

PLAY = r'''
import json, os, sys, time
sys.path.insert(0, sys.argv[1])
os.chdir(sys.argv[1])
import main as agent_graph   # the bundle ships agent_graph.py as main.py
from kaggle_environments import make
engine = agent_graph.get_engine()
times = {'kad': [], 'plain': []}
def agent(obs, config):
    t = time.perf_counter()
    out = agent_graph.agent(obs, config)
    k = engine.kad
    step = int(obs['step'])
    kad_turn = k is not None and k.p['_KC_FROM_STEP'] <= step <= k.p['_KC_TO_STEP'] and step % k.p['_KC_EVERY'] == 0
    times['kad' if kad_turn else 'plain'].append((time.perf_counter() - t) * 1000)
    return out
env = make('kaggriculture', configuration={'seed': int(sys.argv[3]), 'episodeSteps': int(sys.argv[2])})
env.run([agent, sys.argv[4]])
k = engine.kad
mean = lambda xs: round(sum(xs) / len(xs), 1) if xs else None
print('PROBE ' + json.dumps({'report': k.report if k else None, 'last': k.last if k else None,
                             'ms_kad_turn': mean(times['kad']), 'ms_other_turn': mean(times['plain']),
                             'statuses': [s.status for s in env.steps[-1]], 'rewards': [s.reward for s in env.steps[-1]]}))
'''


def main():
    graph = json.loads(Path(sys.argv[1]).read_text())
    args = dict(zip(sys.argv[2::2], sys.argv[3::2]))
    with tempfile.TemporaryDirectory(prefix='kad_probe_') as tmp:
        bundle = Path(tmp) / 'bundle'
        payload.write_graph_bundle(bundle, graph, 'kad_probe')
        env = dict(os.environ, OMP_NUM_THREADS='1', KAD_COPILOT_LOG=str(Path(tmp) / 'kad.jsonl'))
        done = subprocess.run([sys.executable, '-c', PLAY, str(bundle), args.get('--steps', '80'),
                               args.get('--seed', '7'), args.get('--opponent', 'starter')],
                              env=env, capture_output=True, text=True)
        line = next((l for l in done.stdout.splitlines() if l.startswith('PROBE ')), None)
        print(line or f'probe failed (exit {done.returncode}): {done.stderr[-2000:]}')
        log = Path(tmp) / 'kad.jsonl'
        if log.exists():
            rows = [json.loads(l) for l in log.read_text().splitlines()]
            print(f'{len(rows)} interventions; first ones: {rows[:6]}')
    return 0 if line else 1


if __name__ == '__main__':
    sys.exit(main())
