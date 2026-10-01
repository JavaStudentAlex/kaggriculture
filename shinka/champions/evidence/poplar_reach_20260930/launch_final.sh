#!/bin/bash
# Final mutation run (2026-09-30 evening): full graphs seeded from candidate E, KAD/Oracle islands, fresh LLMs only
# (no qwen, deepseek, fugu), gauntlet = flippable losses of our last five submissions + close wins + Ice and Fire.
# tmux kagg-evo-final on cliproxyapi; log ~/kagg-evo/final-0930/run.log; marker FINAL_EVOLUTION_EXIT=. Leaves the pool's STOP file alone.
set -uo pipefail
BASE=/home/alex/kagg-evo/final-0930
PG=/home/alex/kagg-evo/lit0930e/repo/research/procedural_graph
PY=/home/alex/kagg-evo/venv/bin/python
cd "$PG"
OMP_NUM_THREADS=1 "$PY" highcpu_island_evolution.py \
    --executor colab-pool --pool_dir /home/alex/kagg-evo/pool \
    --run_dir "$BASE/run" --iterations "${ITERATIONS:-8}" --islands kad \
    --seed_graph "$BASE/seed_graph_full.json" --plan "$BASE/plan_full.json" \
    --knowledge evolution_knowledge_final.md --ideas evolution_ideas_ladder.md \
    --queue "$BASE/queue_full.json" \
    --models gpt-6.1-sol,claude-opus-5.5,claude-sonnet-5.5,gemini-3.1-pro-preview,gemini-3.8-flash-high,gemini-3.8-flash,gpt-6-astra,gpt-6-luna \
    --seeds_per_opponent 5 --supervisor_interval 8 --mix_interval 4 \
    --stage_fraction 0.3 --prefetch
echo "FINAL_EVOLUTION_EXIT=$?"
