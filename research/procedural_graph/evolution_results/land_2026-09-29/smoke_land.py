"""Full games on cliproxyapi: the oracle-run seed graph (guard off for speed) with and without land-plot variants,
against the public 0927 engine, both seats, same seeds. Prints money, plot report and SE use per game."""
import concurrent.futures as cf, copy, json, os, subprocess, sys, tempfile
from pathlib import Path
PG = Path(sys.argv[1]); os.chdir(PG); sys.path.insert(0, str(PG))
import graph_edits
seeds = [int(s) for s in sys.argv[2].split(',')]
base = json.loads((PG / 'evolution_results/oracle_2026-09-29/seed_graph.json').read_text())
base['require_oracle'] = False
base = graph_edits.apply_edit(base, {'channels': {'oracle_guard': False}})
VARIANTS = {'base': None,
            'goose8w2': {'_LP_USE': 'GOOSE', '_LP_TILES': 8, '_LP_WORKERS': 2},
            'goose8w1': {'_LP_USE': 'GOOSE', '_LP_TILES': 8, '_LP_WORKERS': 1},
            'goose12w3': {'_LP_USE': 'GOOSE', '_LP_TILES': 12, '_LP_WORKERS': 3},
            'goose8w2d10': {'_LP_USE': 'GOOSE', '_LP_TILES': 8, '_LP_WORKERS': 2, '_LP_DAY': 10, '_LP_MIN_MONEY': 500},
            'goose8w2h2': {'_LP_USE': 'GOOSE', '_LP_TILES': 8, '_LP_WORKERS': 2, '_LP_HIRE_HOUR': 2}}
only = sys.argv[3].split(',') if len(sys.argv) > 3 else list(VARIANTS)
GAME = r'''
import json, os, sys
from pathlib import Path
os.environ['KAGG_GRAPH_PATH'] = sys.argv[1]
seed, seat = int(sys.argv[2]), int(sys.argv[3])
sys.path.insert(0, os.getcwd())
import agent_graph
engine = agent_graph.get_engine()
from hazel_runtime import engines
opp = engines.Engine(Path('hazel_runtime/engines/tetsutani_demand_0927'))
from kaggle_environments import make
env = make('kaggriculture', configuration={'seed': seed}, debug=False)
agents = [agent_graph.agent, opp.agent] if seat == 0 else [opp.agent, agent_graph.agent]
env.run(agents)
last = env.steps[-1]
farm = last[0]['observation']['farms'][seat]
se = [farm['tiles'][y][x] for y in range(5, 10) for x in range(5, 10)]
plot = getattr(engine, 'plot', None)
print(json.dumps({'ours': last[seat]['reward'], 'theirs': last[1 - seat]['reward'],
                  'statuses': [s['status'] for s in last], 'fallbacks': engine.fallback_count, 'error': engine.last_error,
                  'report': plot.report if plot else None, 'quadrants': farm['unlocked_quadrants'],
                  'geese_se': sum(1 for t in se if isinstance(t, dict) and t.get('animal') == 'GOOSE'),
                  'plants_se': sum(1 for t in se if isinstance(t, dict) and t.get('kind') == 'PLANT')}))
'''
tmp = Path(tempfile.mkdtemp())
paths = {}
for name in only:
    g = base if VARIANTS[name] is None else graph_edits.apply_edit(base, {'channels': {'land_plot': True}, 'parameters': VARIANTS[name]})
    paths[name] = tmp / f'{name}.json'
    paths[name].write_text(json.dumps(graph_edits.repinned(g)))
def play(job):
    name, seed, seat = job
    env = dict(os.environ, KAGG_ORACLE_BACKEND='numpy', CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1')
    p = subprocess.run([sys.executable, '-c', GAME, str(paths[name]), str(seed), str(seat)], cwd=PG, env=env,
                       capture_output=True, text=True, timeout=1800)
    try:
        return job, json.loads(p.stdout.strip().splitlines()[-1])
    except Exception:
        return job, {'crash': (p.stderr or p.stdout)[-800:]}
jobs = [(n, s, seat) for s in seeds for seat in (0,) for n in only]
res = {}
with cf.ThreadPoolExecutor(int(os.environ.get('WORKERS', '7'))) as ex:
    for job, r in ex.map(play, jobs):
        res[job] = r
        print(job, json.dumps(r)[:900], flush=True)
print('\nSUMMARY (ours - theirs, and ours vs base on the same seed/seat):')
for n in only:
    diffs, gains = [], []
    for s in seeds:
        for seat in (0,):
            r, b = res.get((n, s, seat), {}), res.get(('base', s, seat), {})
            if 'ours' in r:
                diffs.append(r['ours'] - r['theirs'])
                if 'ours' in b:
                    gains.append(r['ours'] - b['ours'])
    w = sum(d > 0 for d in diffs)
    print(f'  {n:10s} games {len(diffs)} wins {w} mean margin {sum(diffs)/max(1,len(diffs)):9.0f} '
          f'mean gain vs base {sum(gains)/max(1,len(gains)):8.0f}  gains {[round(g) for g in gains]}')
