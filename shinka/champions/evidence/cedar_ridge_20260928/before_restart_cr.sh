#!/bin/bash
# BEFORE_START of the ladder1 restart that adds Cedar Ridge's losses (2026-09-28): the plan grows by its 57 lost games
# against rivals rated 1,900+ or unlisted, both seats against the pool bundle that plays like the rival
# (match_ladder_games.py evidence, else the opening's bundle, else tetsutani_demand), and their replays. The plan is
# not in the gauntlet fingerprint: cached margins stay valid and only the new jobs are played.
set -euo pipefail
R=/home/alex/kagg-evo/repo
PG=$R/research/procedural_graph
PY=/home/alex/kagg-evo/venv/bin/python
CR=/home/alex/kagg-evo/cedar_ridge
cat $CR/match/crL/*.jsonl > $CR/match_crL_all.jsonl
cp $PG/evolution_results/ladder_2026-09-26/plan.json /home/alex/kagg-evo/runs/ladder1/plan_before_cr.json
cd $PG
$PY ladder_seed_plan.py --extend evolution_results/ladder_2026-09-26/plan.json --with-ties \
    --losses $CR/index_crL.json=$CR/replays_crL --evidence $CR/match_crL_all.jsonl --fallback tetsutani_demand \
    --replay-opponents $R/shinka/champions/replay_opponents \
    --out evolution_results/ladder_2026-09-26/plan.json > $CR/plan_extend.log 2>&1
echo "plan: $($PY -c "import json; print(len(json.load(open('evolution_results/ladder_2026-09-26/plan.json'))))") jobs"
echo "BEFORE_START done"
