#!/usr/bin/env bash
set -e
cd /home/alex/kaggriculture/research/procedural_graph
echo "Starting Distributed Kaggle CPU Evolution in tmux at $(date)..."
/home/alex/kaggriculture/.venv/bin/python -u kaggle_island_evolution.py --iterations 200 --supervisor_interval 8 2>&1 | tee -a differential_evo.log
echo "Kaggle Distributed Evolution exited at $(date)."
