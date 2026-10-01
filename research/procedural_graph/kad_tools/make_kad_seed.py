"""Write the KAD run's seed graph (2026-09-29): Juniper Knoll's graph (the land run's seed: Aspen Vale + _S809_LOOK 4 +
_CA_MARGIN -22 + _SR_MARGIN 14 + the oracle guard) with the kad_copilot stage on, the predictor required
(require_oracle) and KAD required (require_kad: the evolution cannot switch it off, only learn to use it).

    python make_kad_seed.py LAND_SEED_GRAPH OUT.json [KC_NAME=VALUE ...]

KAD's settings default to kad_copilot.PARAMETERS (every 4th turn from turn 24 to 671: not the opening day, never the
engine's liquidation on days 28-29); name=value pairs (JSON values) override them.
"""
import copy
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import graph_edits  # noqa: E402
import highcpu_island_evolution as evo  # noqa: E402


def main():
    seed = json.loads(Path(sys.argv[1]).read_text())
    params = {}
    for pair in sys.argv[3:]:
        name, value = pair.split('=', 1)
        params[name] = json.loads(value)
    graph = graph_edits.apply_edit(copy.deepcopy(seed), {'channels': {'kad_copilot': True},
                                                         'parameters': params or {'_KC_EVERY': 4}})
    graph['require_oracle'] = True
    graph['require_kad'] = True
    evo.check_required_oracle(graph)
    evo.check_required_kad(graph)
    Path(sys.argv[2]).write_text(json.dumps(graph, indent=2) + '\n')
    node = next(n for n in graph['turn']['nodes'] if n['id'] == 'kad_copilot')
    print(f'{sys.argv[2]}: kad_copilot {json.dumps(node.get("parameters"))}; require_oracle, require_kad')
    return 0


if __name__ == '__main__':
    sys.exit(main())
