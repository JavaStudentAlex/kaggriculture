#!/bin/bash
# Unattended start of the KAD run (2026-09-29, user: "automate, then exit"). tmux kagg-kad-chain on cliproxyapi, log
# ~/kagg-evo/kad-20260929/chain.log, final marker KAD_CHAIN_EXIT=<code>.
# 1. waits for the 09-28 refit (tmux kagg-colab-refit-0928) and unpacks its best checkpoint into models/;
# 2. calibrates it for our agent on one colab2 T4 (arena/calibrate.py --host local: everything on this host);
# 3. installs refit + calibration.json as the run's predictor (hazel_runtime/checkpoint of the run's code copy; the
#    09-13 reference stays in checkpoint_0913). Without a usable refit or calibration it keeps the 09-13 predictor;
# 4. waits for the KAD lever games (tmux kagg-kadexp) and keeps the levers that did not lose (chain_helpers.py levers);
# 5. writes the seed (make_kad_seed.py: Juniper Knoll + KAD every 2nd turn from turn 24 to 671, predictor and KAD
#    required), validates it with a 30-turn game, and starts launch_kad_run.sh in tmux kagg-evo-kad.
set -uo pipefail
B=/home/alex/kagg-evo/kad-20260929
PG=$B/repo/research/procedural_graph
PY=/home/alex/kagg-evo/venv/bin/python
T=$PG/kad_tools
REFIT=/home/alex/kagg-colab/runs/refit_0928
NAME=ttm_c256_h96_ft_2026-09-28
MODEL=$B/models/$NAME
CAL=$B/calibration/$NAME/own_games
EXP=/home/alex/kagg-evo/kadexp
LAND_SEED=/home/alex/kagg-evo/land-20260929/repo/research/procedural_graph/evolution_results/land_2026-09-29/seed_graph.json
export OMP_NUM_THREADS=1
log() { echo "$(date -u '+%m-%d %H:%M:%S') $*"; }
PREDICTOR=ttm_c256_h96_ft_2026-09-13

log "1. waiting for the refit runner (tmux kagg-colab-refit-0928)"
while tmux has-session -t kagg-colab-refit-0928 2>/dev/null; do sleep 60; done
tail -4 $REFIT/runner.log
if [ -f $REFIT/pulled/result.tar.gz ]; then
    rm -rf $MODEL /tmp/kad_chain_refit && mkdir -p $MODEL /tmp/kad_chain_refit
    tar -xzf $REFIT/pulled/result.tar.gz -C /tmp/kad_chain_refit
    cp /tmp/kad_chain_refit/best/* $MODEL/ && cp /tmp/kad_chain_refit/scores.json /tmp/kad_chain_refit/trainer_state.json $MODEL/ 2>/dev/null
    read E0 BEST EPOCH <<< "$($PY $T/chain_helpers.py refit $MODEL/trainer_state.json)"
    log "refit: held-out AUC epoch 0 (the 09-26 base) $E0, best $BEST at epoch $EPOCH; files: $(ls $MODEL | tr '\n' ' ')"
    for f in model.safetensors config.json scaler.npz labels.json; do [ -f $MODEL/$f ] || { log "refit lacks $f"; BEST=nan; }; done
else
    log "no refit result (result.tar.gz missing)"; BEST=nan
fi

if [ "$BEST" != "nan" ]; then
    log "2. calibrating $NAME (one colab2 T4, ~40 min)"
    cd $PG
    $PY arena/calibrate.py --host local --remote /home/alex/kagg-colab --refit $MODEL --reference $B/checkpoint_0913 \
        --eval-id calib-$NAME --out $CAL
    CAL_EXIT=$?
    log "calibration exit $CAL_EXIT (colab2 T4 High-RAM)"
    if [ $CAL_EXIT -ne 0 ]; then   # a High-RAM T4 can be refused (Service Unavailable): a standard T4, 2 workers
        log "retrying the calibration on a standard colab2 T4 (slower, ~2 h)"
        $PY arena/calibrate.py --host local --remote /home/alex/kagg-colab --refit $MODEL --reference $B/checkpoint_0913 \
            --eval-id calib-$NAME-t4 --vm colab2:t4 --out $CAL
        CAL_EXIT=$?
        log "calibration exit $CAL_EXIT (colab2 T4)"
    fi
    if [ $CAL_EXIT -eq 0 ] && [ -f $CAL/calibration.json ]; then
        log "3. installing $NAME + calibration.json as the run's predictor"
        cp $MODEL/model.safetensors $MODEL/config.json $MODEL/scaler.npz $MODEL/labels.json $PG/hazel_runtime/checkpoint/
        cp $CAL/calibration.json $PG/hazel_runtime/checkpoint/calibration.json
        PREDICTOR=$NAME
    else
        log "3. calibration failed: keeping the 09-13 predictor (Juniper Knoll's)"
    fi
else
    log "2-3. no usable refit: keeping the 09-13 predictor (Juniper Knoll's)"
fi

log "4. waiting for the KAD lever games (tmux kagg-kadexp; at most 3 h)"
for i in $(seq 1 180); do grep -q "KAD_EXPERIMENT_EXIT" $EXP/experiment.log && break; sleep 60; done
grep -A 30 "^sell:\|^hands:" $EXP/experiment.log | grep -v "^--" | head -40
if [ -f $EXP/run/kad_experiment.json ]; then
    LEVERS=$($PY $T/chain_helpers.py levers $EXP/run/kad_experiment.json)
else
    LEVERS="_KC_SELL=true _KC_HANDS=true"; log "no experiment result: both levers on"
fi
log "levers: $LEVERS"

log "5. seed: Juniper Knoll + KAD ($LEVERS, every 2nd turn, turns 24-671) + predictor $PREDICTOR"
cd $PG
$PY $T/make_kad_seed.py $LAND_SEED $B/seed_graph.json _KC_EVERY=2 _KC_FROM_STEP=24 _KC_TO_STEP=671 $LEVERS || {
    log "seed failed"; echo "KAD_CHAIN_EXIT=5"; exit 5; }
if [ "$PREDICTOR" = "$NAME" ]; then
    $PY $T/chain_helpers.py provenance $B/seed_graph.json $PG/hazel_runtime/checkpoint $NAME
fi
$PY -c "import sys; sys.path.insert(0, '.'); import graph_edits; graph_edits.validate_graph('$B/seed_graph.json'); print('seed valid')" || {
    log "seed validation failed"; echo "KAD_CHAIN_EXIT=6"; exit 6; }
ls /home/alex/kagg-evo/pool/STOP 2>/dev/null && { log "pool STOP file present: not starting"; echo "KAD_CHAIN_EXIT=7"; exit 7; }
tmux has-session -t kagg-evo-kad 2>/dev/null && { log "kagg-evo-kad already exists"; echo "KAD_CHAIN_EXIT=8"; exit 8; }
tmux new -d -s kagg-evo-kad "bash $T/launch_kad_run.sh >> $B/run.log 2>&1"
sleep 30
log "started tmux kagg-evo-kad; run.log:"
tail -5 $B/run.log
echo "KAD_CHAIN_EXIT=0"
