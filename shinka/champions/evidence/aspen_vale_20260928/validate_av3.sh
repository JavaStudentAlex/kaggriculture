#!/bin/bash
# Third round of validate_av.sh (run on cliproxyapi): the oracle-guard variant of round 2 (aspen_guard, always on;
# the user wants the oracle on in every game) with the newest predictor instead of the reference one. It is built from ~/kagg-evo/pred26, a copy
# of the evolution's repo whose hazel_runtime/checkpoint is models/ttm_c256_h96_ft_2026-09-26 plus its calibration
# for our agent (arena/calibrate.py, AGENTS.md section 13), so the running evolution's runtime and cache are untouched.
# Waits until round 1 has finished and the calibration is in place. Its own run names (round 2 may be writing
# backtest_av and lost_av at the same time), started from copies of round 1's game files, so aspen's games are reused
# (its play does not run the predictor) and the reports pair the new labels with aspen.
set -uo pipefail
AV=/home/alex/kagg-evo/aspen_vale
PG=/home/alex/kagg-evo/pred26/research/procedural_graph
PY=/home/alex/kagg-evo/venv/bin/python
GAMES=/home/alex/kagg-evo/runs/ladder1/games
until [ -s $PG/hazel_runtime/checkpoint/calibration.json ] && grep -q '^BACKTEST_AV_EXIT=' $AV/backtest_av_round1.log 2>/dev/null \
        && grep -q '^LOST_AV_EXIT=' $AV/lost_av_round1.log 2>/dev/null; do
    sleep 60
done
echo "$(date -u +%T) predictor: $(sha256sum $PG/hazel_runtime/checkpoint/model.safetensors | cut -c1-16), calibration: $(sha256sum $PG/hazel_runtime/checkpoint/calibration.json | cut -c1-16)"
[ -e $GAMES/backtest_av3.jsonl ] || cp $GAMES/backtest_av.jsonl $GAMES/backtest_av3.jsonl
[ -e $GAMES/lost_av3.jsonl ] || cp $GAMES/lost_av.jsonl $GAMES/lost_av3.jsonl
G="--graph aspen=evolution_results/ladder_2026-09-26/next_counter_emulator_graph.json --graph aspen_guard26=$AV/aspen_guard_graph.json"
cd $PG
# build the bundles once before the two requests run in parallel (two builders of one new bundle collide)
$PY ladder_validate.py --run_dir /home/alex/kagg-evo/runs/ladder1 --pool_dir /home/alex/kagg-evo/pool --name backtest_av3 $G \
    --seeds 0 --replays $AV/replay_bundles --dry_run > $AV/backtest_av3_dry.log 2>&1
{ $PY ladder_validate.py --run_dir /home/alex/kagg-evo/runs/ladder1 --pool_dir /home/alex/kagg-evo/pool --name backtest_av3 $G \
    --seeds 0 --replays $AV/replay_bundles > $AV/backtest_av3.log 2>&1; echo BACKTEST_AV3_EXIT=$? >> $AV/backtest_av3.log; } &
{ $PY ladder_validate.py --run_dir /home/alex/kagg-evo/runs/ladder1 --pool_dir /home/alex/kagg-evo/pool --name lost_av3 $G \
    --seeds 0 --lost $AV/av_losses.json > $AV/lost_av3.log 2>&1; echo LOST_AV3_EXIT=$? >> $AV/lost_av3.log; } &
wait
echo VALIDATE_AV3_DONE
