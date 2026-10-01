#!/bin/bash
# Second round of validate_av.sh (run on cliproxyapi): once backtest_av and lost_av have finished, the same two
# requests again for aspen and three new graphs, each Aspen Vale with one change:
#   aspen_t7000, aspen_t6000  the tomato investment's revenue gate lowered (_CXTB_MIN_REVENUE 9,000 -> 7,000 / 6,000;
#                             counter T-B in the 09-27 engine, main.py:6700-6800);
#   aspen_guard               the predictor's oracle guard with the settings the old engine's islands evolved (MILK,
#                             WOOL, STRAWBERRY; score 0.5, batch 4, keep 2, price ratio 0.75), as iteration 55 tried it;
#   (aspen_guardx, the guard switched off while the emulator identifies an exact public engine, was dropped: the user
#    wants the oracle on in every game)
# The game cache is keyed by run name and label, so aspen's games are reused and only the new graphs' games are
# played; the reports pair them with aspen.
set -uo pipefail
AV=/home/alex/kagg-evo/aspen_vale
PG=/home/alex/kagg-evo/repo/research/procedural_graph
PY=/home/alex/kagg-evo/venv/bin/python
until grep -q '^BACKTEST_AV_EXIT=' $AV/backtest_av.log 2>/dev/null && grep -q '^LOST_AV_EXIT=' $AV/lost_av.log 2>/dev/null; do
    sleep 60
done
cp $AV/backtest_av.log $AV/backtest_av_round1.log
cp $AV/lost_av.log $AV/lost_av_round1.log
# only file graphs: an island: label resolves against the checkpoint of the moment, and the cache is keyed by label
G="--graph aspen=evolution_results/ladder_2026-09-26/next_counter_emulator_graph.json --graph aspen_t7000=$AV/aspen_t7000_graph.json --graph aspen_t6000=$AV/aspen_t6000_graph.json --graph aspen_guard=$AV/aspen_guard_graph.json"
cd $PG
{ $PY ladder_validate.py --run_dir /home/alex/kagg-evo/runs/ladder1 --pool_dir /home/alex/kagg-evo/pool --name backtest_av $G \
    --seeds 0 --replays $AV/replay_bundles > $AV/backtest_av.log 2>&1; echo BACKTEST_AV_EXIT=$? >> $AV/backtest_av.log; } &
{ $PY ladder_validate.py --run_dir /home/alex/kagg-evo/runs/ladder1 --pool_dir /home/alex/kagg-evo/pool --name lost_av $G \
    --seeds 0 --lost $AV/av_losses.json > $AV/lost_av.log 2>&1; echo LOST_AV_EXIT=$? >> $AV/lost_av.log; } &
wait
echo VALIDATE_AV2_DONE
