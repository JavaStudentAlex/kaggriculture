#!/bin/bash
# BEFORE_START of the 2026-09-27 restart of run ladder1 (restart_ladder1.sh runs it after killing the loop).
#   0. the Island-Market champion rebuilt with the staged runtime (renamed only, ~/kagg-evo/inert_test.sh) must
#      replay its 20 games against the 09-27 engine exactly, or nothing changes and the loop stays stopped;
#   1. the staged code (~/kagg-evo/dev, a copy of the repo with the rival_counter stage, the public engine's
#      09-27 version, the per-island seeds) replaces the loop's copy;
#   2. the plan grows by Alder Ford's 29 newer lost and tied games (both seats against the bundle that plays
#      like the rival, and their replays) and 10 random seeds against the 09-27 engine; earlier lost games
#      that a pool bundle now matches longer move to it;
#   3. the cached margins carry over to the new fingerprint (graphs without the stage play as before);
#   4. two islands evolve on the 09-27 engine;
#   5. the queue: the mirror counters and the guard on the 09-27 engine.
set -euo pipefail
R=/home/alex/kagg-evo/repo
D=/home/alex/kagg-evo/dev
RUN=/home/alex/kagg-evo/runs/ladder1
PY=/home/alex/kagg-evo/venv/bin/python
PG=$R/research/procedural_graph
OLD_FP=41075e62041c89f805a056143d230f0fc96b660fa3aa48268f497ef8f565b668

for _ in $(seq 240); do grep -q '^INERT_TEST_EXIT=' /home/alex/kagg-evo/inert_test.log && break; sleep 15; done
grep -q '^INERT_TEST_EXIT=0' /home/alex/kagg-evo/inert_test.log
$PY $D/research/procedural_graph/evolution_results/ladder_2026-09-26/same_games.py $RUN \
    vs_engine0927:market inert_test:market_rt --expect 20

rsync -a --exclude __pycache__ --exclude evolution_results/ladder_2026-09-26/plan.json \
    $D/research/procedural_graph/ $PG/
rsync -a --exclude __pycache__ $D/shinka/champions/ladder/ $R/shinka/champions/ladder/
rsync -a $D/shinka/champions/replay_opponents/ $R/shinka/champions/replay_opponents/
echo "code synced"

cat /home/alex/kagg-evo/rival/match/*/*.jsonl > /home/alex/kagg-evo/rival/ladder_match_all.jsonl
cp $PG/evolution_results/ladder_2026-09-26/plan.json $RUN/plan_before_0927.json
cd $PG
$PY ladder_seed_plan.py --extend evolution_results/ladder_2026-09-26/plan.json --with-ties \
    --losses /home/alex/kagg-evo/alder_ford2/index_lt.json=/home/alex/kagg-evo/alder_ford2/replays \
    --evidence /home/alex/kagg-evo/rival/ladder_match_all.jsonl --fallback tetsutani_demand \
    --replay-opponents $R/shinka/champions/replay_opponents \
    --random-opponents tetsutani_demand_0927 --random-seeds 10 \
    --out evolution_results/ladder_2026-09-26/plan.json | head -5
echo "plan: $(python3 -c "import json; print(len(json.load(open('evolution_results/ladder_2026-09-26/plan.json'))))") jobs"

$PY evolution_results/ladder_2026-09-26/carry_fingerprint.py $RUN evolution_results/ladder_2026-09-26/plan.json \
    --old $OLD_FP --apply

$PY evolution_results/ladder_2026-09-26/add_islands.py $RUN evolution_results/ladder_2026-09-27/seed_graph_engine0927.json \
    --island "Island-Next-Market=The public engine's 09-27 version (tetsutani_demand_0927) as the backbone: its sale timing against the rival (the race layers _RACE_* and V9_RACE*, the library-based predictor of the rival's sales _V92_P_* and _V92_Q_*, the numbered sale layers _S720_* to _S1009_*, order-slot priority _OR2_*, quote reordering _R37_*) and our layers on top of it: the oracle guard (_OG_*) and rival counters." \
    --island "Island-Next-Production=The public engine's 09-27 version (tetsutani_demand_0927) as the backbone: its production (the hybrid opening _ALT_* and _R42_*, routes by shops _R108_*, _R110_*, _V92_TABLE, the herd _HD2_* and _CS_*, crops _CA_*, V9_CARROT_*, V9_FERT_*, feed and fertilizer economics _R85_* and _R88_*, the cash reserve _CXD_*, shed and endgame _SR_*, _CH_*, V9_COURIER_*, _Y_*) and our rival counters." \
    --apply

echo "BEFORE_START done"
