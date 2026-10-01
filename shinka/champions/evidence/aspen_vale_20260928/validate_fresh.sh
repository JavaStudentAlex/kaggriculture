#!/bin/bash
# Fresh-seed test of Aspen Vale with the oracle on, before any submission (user: "lets test it first"; cliproxyapi):
# plain Aspen Vale and the oracle-guard variant, always on (reference predictor), on the arena validation seeds against every
# opponent bundle of the run, plus head-to-head against Aspen Vale's own arena bundle (g_16bb4af65a704f00); 10 seeds per
# opponent, seats alternating. Waits until validation round 2 has been submitted, so the pool plays the smaller rounds
# 2 and 3 first. Report: runs/ladder1/validation/fresh_av.json (every graph paired with aspen).
set -uo pipefail
AV=/home/alex/kagg-evo/aspen_vale
PG=/home/alex/kagg-evo/repo/research/procedural_graph
PY=/home/alex/kagg-evo/venv/bin/python
RUN=/home/alex/kagg-evo/runs/ladder1
until [ -e $AV/backtest_av_round1.log ]; do sleep 60; done
sleep 120
OPP=$(ls -d $RUN/bundles/*/ | xargs -n1 basename | grep -v -e '^g_' -e '^replay_' -e '^\.' | tr '\n' ',')g_16bb4af65a704f00
echo "$(date -u +%T) opponents: $OPP"
G="--graph aspen=evolution_results/ladder_2026-09-26/next_counter_emulator_graph.json --graph aspen_guard=$AV/aspen_guard_graph.json"
cd $PG
$PY ladder_validate.py --run_dir $RUN --pool_dir /home/alex/kagg-evo/pool --name fresh_av $G --opponents "$OPP" --seeds 10 \
    > $AV/fresh_av.log 2>&1
echo FRESH_AV_EXIT=$? >> $AV/fresh_av.log
