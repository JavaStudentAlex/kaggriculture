#!/bin/bash
# Run in tmux kagg-evo-land on cliproxyapi. Shares the existing pool (tmux kagg-colab-evo); does not own its lifetime.
# First the runtime check that carries the seed's 763 cached games (inert_land.py), then the loop.
set -uo pipefail
BASE=/home/alex/kagg-evo/land-20260929
PG=$BASE/repo/research/procedural_graph
PY=/home/alex/kagg-evo/venv/bin/python
R=evolution_results/land_2026-09-29
cd "$PG"
if [ ! -f "$BASE/run/checkpoint.json" ]; then
    "$PY" $R/inert_land.py "$BASE/run" /home/alex/kagg-evo/oracle-20260929/run $R/seed_graph.json $R/plan.json \
        --old_bundle g_2e6e1664c98cc2a9 --check 24 2>&1 | tee -a "$BASE/inert.log"
    grep -q '^INERT_LAND_EXIT=0' "$BASE/inert.log" || { echo "LAND_EVOLUTION_EXIT=inert_check_failed"; exit 1; }
fi
"$PY" highcpu_island_evolution.py \
    --executor colab-pool --pool_dir /home/alex/kagg-evo/pool \
    --run_dir "$BASE/run" --iterations "${ITERATIONS:-40}" --islands land --require_oracle \
    --seed_graph $R/seed_graph.json --plan $R/plan.json \
    --knowledge evolution_knowledge_ladder.md --ideas evolution_ideas_ladder.md \
    --queue $R/queue.json \
    --seeds_per_opponent 20 --supervisor_interval 6 --mix_interval 6 \
    --stage_fraction 0.3 --prefetch
echo "LAND_EVOLUTION_EXIT=$?"
