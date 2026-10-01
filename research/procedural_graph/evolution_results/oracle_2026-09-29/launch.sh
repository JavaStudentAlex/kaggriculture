#!/bin/bash
# Run in its own tmux session on cliproxyapi. Shares the existing pool; does not own its lifetime.
set -euo pipefail
BASE=/home/alex/kagg-evo/oracle-20260929
PG=$BASE/repo/research/procedural_graph
PY=/home/alex/kagg-evo/venv/bin/python
trap 'code=$?; echo "ORACLE_EVOLUTION_EXIT=$code"' EXIT
cd "$PG"
"$PY" highcpu_island_evolution.py \
    --executor colab-pool --pool_dir /home/alex/kagg-evo/pool \
    --run_dir "$BASE/run" --iterations "${ITERATIONS:-40}" --islands oracle --require_oracle \
    --seed_graph evolution_results/oracle_2026-09-29/seed_graph.json \
    --plan evolution_results/oracle_2026-09-29/plan.json \
    --knowledge evolution_knowledge_ladder.md --ideas evolution_ideas_ladder.md \
    --queue evolution_results/oracle_2026-09-29/queue.json \
    --seeds_per_opponent 20 --supervisor_interval 6 --mix_interval 6 \
    --stage_fraction 0.3 --prefetch
