#!/bin/bash
# BEFORE_START of the restart after iteration 55 of run ladder1 (2026-09-27):
#   1. the staged gauntlet and proposals ahead (code staged in ~/kagg-evo/dev) replace the loop's copy;
#   2. the plan grows by Birch Hollow's 28 lost games: both seats against the pool bundle that plays like the
#      rival (match_ladder_games.py evidence, else the opening's bundle, else tetsutani_demand), and their replays.
# The plan is not in the gauntlet fingerprint: cached margins stay valid and only the new jobs are played.
set -euo pipefail
R=/home/alex/kagg-evo/repo
PG=$R/research/procedural_graph
PY=/home/alex/kagg-evo/venv/bin/python
BH=/home/alex/kagg-evo/bh
rsync -a --exclude __pycache__ --exclude evolution_results/ladder_2026-09-26/plan.json \
    /home/alex/kagg-evo/dev/research/procedural_graph/ $PG/
echo "code synced"
cat $BH/match/bhL/*.jsonl > $BH/match_bhL_all.jsonl
cp $PG/evolution_results/ladder_2026-09-26/plan.json /home/alex/kagg-evo/runs/ladder1/plan_before_bh.json
cd $PG
$PY ladder_seed_plan.py --extend evolution_results/ladder_2026-09-26/plan.json --with-ties \
    --losses $BH/index_bhL.json=$BH/replays_bhL --evidence $BH/match_bhL_all.jsonl --fallback tetsutani_demand \
    --replay-opponents $R/shinka/champions/replay_opponents \
    --out evolution_results/ladder_2026-09-26/plan.json > $BH/plan_extend.log 2>&1
echo "plan: $($PY -c "import json; print(len(json.load(open('evolution_results/ladder_2026-09-26/plan.json'))))") jobs"
echo "BEFORE_START done"
