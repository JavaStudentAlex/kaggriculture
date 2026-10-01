#!/bin/bash
# The KAD run (from 2026-09-29): Juniper Knoll's graph with the predictor and the KAD copilot required, eight islands
# that learn to use them (--islands kad), on the land run's 763-game plan and the shared Colab pool (tmux
# kagg-colab-evo). Run in tmux kagg-evo-kad on cliproxyapi; log ~/kagg-evo/kad-20260929/run.log, marker
# KAD_EVOLUTION_EXIT=. Does not touch the pool's STOP file (the pool outlives the run).
set -uo pipefail
BASE=/home/alex/kagg-evo/kad-20260929
PG=$BASE/repo/research/procedural_graph
PY=/home/alex/kagg-evo/venv/bin/python
cd "$PG"
OMP_NUM_THREADS=1 "$PY" highcpu_island_evolution.py \
    --executor colab-pool --pool_dir /home/alex/kagg-evo/pool \
    --run_dir "$BASE/run" --iterations "${ITERATIONS:-60}" --islands kad \
    --seed_graph "$BASE/seed_graph.json" --plan evolution_results/land_2026-09-29/plan.json \
    --knowledge evolution_knowledge_ladder.md --ideas evolution_ideas_ladder.md \
    --queue "$BASE/queue.json" \
    --seeds_per_opponent 20 --supervisor_interval 8 --mix_interval 8 \
    --stage_fraction 0.3 --prefetch
echo "KAD_EVOLUTION_EXIT=$?"
