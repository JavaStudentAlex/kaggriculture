#!/bin/bash
# BEFORE_START of the ladder1 restart that adds Aspen Vale's losses (2026-09-28; before_restart_cr.sh of
# ../cedar_ridge_20260928 for this submission): the plan grows by its 31 lost games, both seats against the pool bundle
# that plays like the rival (match_ladder_games.py evidence, leo_pi now in the pool; else the opening's bundle, else
# tetsutani_demand), and their replays. The plan is not in the gauntlet fingerprint: cached margins stay valid and
# only the new jobs are played.
set -euo pipefail
R=/home/alex/kagg-evo/repo
PG=$R/research/procedural_graph
PY=/home/alex/kagg-evo/venv/bin/python
AV=/home/alex/kagg-evo/aspen_vale
cat $AV/match/avL/*.jsonl > $AV/match_avL_all.jsonl
cp $PG/evolution_results/ladder_2026-09-26/plan.json /home/alex/kagg-evo/runs/ladder1/plan_before_av.json
cd $PG
$PY ladder_seed_plan.py --extend evolution_results/ladder_2026-09-26/plan.json --with-ties \
    --losses $AV/index_avL.json=$AV/replays_avL --evidence $AV/match_avL_all.jsonl --fallback tetsutani_demand \
    --replay-opponents $R/shinka/champions/replay_opponents \
    --out evolution_results/ladder_2026-09-26/plan.json > $AV/plan_extend.log 2>&1
echo "plan: $($PY -c "import json; print(len(json.load(open('evolution_results/ladder_2026-09-26/plan.json'))))") jobs"
echo "BEFORE_START done"
