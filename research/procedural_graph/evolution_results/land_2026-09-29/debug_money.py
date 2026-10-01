"""Seed 404 vs the public 0927 engine: base and goose8w1, per day at hour 23: both players' money, our shed total,
prices of wheat/egg/fertilizer/milk, and the plot's spend estimate. Shows where the plot's games part."""
import json, os, subprocess, sys
from pathlib import Path
PG = Path(sys.argv[1]); os.chdir(PG); sys.path.insert(0, str(PG))
import graph_edits
base = json.loads((PG / 'evolution_results/oracle_2026-09-29/seed_graph.json').read_text())
base['require_oracle'] = False
base = graph_edits.apply_edit(base, {'channels': {'oracle_guard': False}})
plot = graph_edits.apply_edit(base, {'channels': {'land_plot': True},
                                     'parameters': {'_LP_USE': 'GOOSE', '_LP_TILES': 8, '_LP_WORKERS': 1}})
GAME = r'''
import json, os, sys
from pathlib import Path
os.environ['KAGG_GRAPH_PATH'] = sys.argv[1]
sys.path.insert(0, os.getcwd())
import agent_graph
engine = agent_graph.get_engine()
from hazel_runtime import engines
opp = engines.Engine(Path('hazel_runtime/engines/tetsutani_demand_0927'))
from kaggle_environments import make
env = make('kaggriculture', configuration={'seed': int(sys.argv[2])}, debug=False)
env.run([agent_graph.agent, opp.agent])
rows = []
for t in range(10 * 24 + 23, 720, 24):
    o0 = env.steps[t][0]['observation']
    p = o0['market']['prices']
    shed = o0['private']['shed']
    rows.append([t // 24, round(o0['farms'][0]['money']), round(o0['farms'][1]['money']), sum(shed.values()),
                 p['WHEAT'], p['EGG'], p['FERTILIZER'], p['MILK'], p['STRAWBERRY'], len(o0['farms'][0]['hands'])])
print(json.dumps(rows))
'''
out = {}
for name, g in (('base', base), ('goose', plot)):
    path = Path(f'/tmp/dbg_{name}.json'); path.write_text(json.dumps(graph_edits.repinned(g)))
    r = subprocess.run([sys.executable, '-c', GAME, str(path), sys.argv[2]], cwd=PG, capture_output=True, text=True,
                       env=dict(os.environ, KAGG_ORACLE_BACKEND='numpy'))
    out[name] = json.loads(r.stdout.strip().splitlines()[-1])
print('day | base: ours theirs shed | goose: ours theirs shed | prices base (wheat egg fert milk straw) | goose prices | hands b/g')
for b, g in zip(out['base'], out['goose']):
    print(f"{b[0]:3d} | {b[1]:7d} {b[2]:7d} {b[3]:3d} | {g[1]:7d} {g[2]:7d} {g[3]:3d} | {b[4:9]} | {g[4:9]} | {b[9]}/{g[9]}")
