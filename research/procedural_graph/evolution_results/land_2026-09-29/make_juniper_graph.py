"""Write the graph packaged as Juniper Knoll (2026-09-29): the land run's seed with `require_oracle` off.

    python make_juniper_graph.py OUT.json

In the arena `require_oracle` makes a game without working forecasts fail, so it cannot count. On Kaggle that would
end the game in an error, so the package plays with it off: a failed predictor then leaves the engine's own orders,
as every earlier oracle submission did. Nothing else changes.
"""
import json
import sys
from pathlib import Path

SEED = Path(__file__).resolve().parent / 'seed_graph.json'
if not SEED.exists():   # on cliproxyapi the script runs from ~/kagg-evo/juniper_knoll
    SEED = Path('/home/alex/kagg-evo/land-20260929/repo/research/procedural_graph/evolution_results/'
                'land_2026-09-29/seed_graph.json')

graph = json.loads(SEED.read_text())
assert graph.get('require_oracle') is True, graph.get('require_oracle')
graph['require_oracle'] = False
Path(sys.argv[1]).write_text(json.dumps(graph, indent=2) + '\n')
nodes = {node['id']: node for node in graph['turn']['nodes']}
engine = nodes['backbone']['engine_parameters']
guard = nodes['oracle_guard']['parameters']
print(f'{sys.argv[1]}: from {SEED}; require_oracle True -> False')
print('engine parameters:', json.dumps(engine, sort_keys=True))
print('guard parameters:', json.dumps(guard, sort_keys=True))
