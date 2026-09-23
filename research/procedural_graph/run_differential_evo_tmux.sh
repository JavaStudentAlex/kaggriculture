#!/usr/bin/env bash
# Launch the current growing-champion procedural-graph evolution pipeline.
# Run this script inside a tmux session named `island-evo`.
set -euo pipefail

export HOME=/home/alex
export KAGGLE_CONFIG_DIR=/home/alex/.kaggle
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd /home/alex/kaggriculture/research/procedural_graph
exec /home/alex/kaggriculture/.venv/bin/python -u kaggle_island_evolution.py \
  --iterations 200 \
  --supervisor_interval 8 \
  "$@"
