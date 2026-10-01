"""The ladder1 island champions against their seeds and against Aspen Vale's graph (cliproxyapi, read-only): which
graphs are worth a validation on Aspen Vale's games.

    python champions_diff.py [checkpoint.json]
"""
import json
import sys
from pathlib import Path

PG = Path('/home/alex/kagg-evo/repo/research/procedural_graph')
sys.path.insert(0, str(PG))
import graph_edits  # noqa: E402

checkpoint = json.load(open(sys.argv[1] if len(sys.argv) > 1 else '/home/alex/kagg-evo/runs/ladder1/checkpoint.json'))
aspen = json.load(open(PG / 'evolution_results/ladder_2026-09-26/next_counter_emulator_graph.json'))
constants = graph_edits.catalog()
for island in checkpoint['islands']:
    seed = island.get('seed') or checkpoint['seed_graph']
    graph = island['graph']
    engine = graph_edits.engine_name(graph)
    print(f"== {island['name']} (engine {engine}); promotions in history: {len(island.get('history') or [])}")
    for line in graph_edits.diff(seed, graph, constants):
        print('   vs seed :', line[:150])
    if engine == graph_edits.engine_name(aspen):
        lines = graph_edits.diff(aspen, graph, constants)
        print('   vs Aspen Vale:', 'identical' if not lines else '')
        for line in lines:
            print('   vs Aspen:', line[:150])
