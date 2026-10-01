"""Write the KAD copilot graphs (2026-09-29; run on cliproxyapi in the experiment's code copy) and check each with
graph_edits.validate_graph (a 30-turn game, then later steps on its last observation).

    python make_kad_graphs.py SEED_GRAPH OUT_DIR

timing   Juniper Knoll's graph (require_oracle off, as packaged) + KAD every turn from turn 8 to 719, both levers: the
         worst case for Kaggle's time limits (make_graph_submission.py --validate measures it)
sell     the seed graph + KAD's confident sales, every 2nd turn from turn 8 to 719
hands    the seed graph + idle hands doing KAD's job where they stand, every 2nd turn from turn 8 to 719
"""
import copy
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import graph_edits  # noqa: E402

WINDOW = {'_KC_EVERY': 2, '_KC_FROM_STEP': 8, '_KC_TO_STEP': 719}
EDITS = {
    'timing': {'_KC_EVERY': 1, '_KC_FROM_STEP': 8, '_KC_TO_STEP': 719, '_KC_SELL': True, '_KC_HANDS': True},
    'sell': {**WINDOW, '_KC_SELL': True, '_KC_HANDS': False},
    'hands': {**WINDOW, '_KC_SELL': False, '_KC_HANDS': True},
}


def main():
    seed = json.loads(Path(sys.argv[1]).read_text())
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    ok = True
    for name, params in EDITS.items():
        graph = graph_edits.apply_edit(copy.deepcopy(seed), {'channels': {'kad_copilot': True}, 'parameters': params})
        if name == 'timing':
            graph['require_oracle'] = False
        path = out / f'kad_{name}_graph.json'
        path.write_text(json.dumps(graph, indent=2) + '\n')
        try:
            graph_edits.validate_graph(path)
            print(f'{name}: {path} valid', flush=True)
        except Exception as exc:  # noqa: BLE001
            ok = False
            print(f'{name}: INVALID {type(exc).__name__}: {str(exc)[:800]}', flush=True)
    print('KAD_GRAPHS ' + ('OK' if ok else 'FAIL'), flush=True)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
