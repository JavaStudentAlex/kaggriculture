#!/usr/bin/env bash
# Base-model training: warm start from an existing checkpoint (BASE) on every day
# in the shard directory, 96-step (4 game days) supervision, context CONTEXT turns
# (256 = 8 patches of 32, so the in-game oracle forecasts from day 10 h16), 10 % of
# episodes held out (best/val_episodes.json). BASE's scaler is reused, never refit;
# when only the context changes, just the patcher (0.6 % of the weights) is
# re-initialised. Window stride 17 is coprime with the 24-turn day, so training
# origins cover every hour the live oracle runs at.
#   BASE=<model dir>       required: weights + scaler.npz + labels.json (e.g. the newest models/*)
#   CONTEXT=256 PATCH=32 STRIDE=17 GPUS=3 LR=1e-4 PATIENCE=6 EPOCHS=40
#   DATASET=/results/kagg/datasets/shards   staged by ops/copy.sh / ops/extract.sh
#   RESUME=<checkpoint-N dir or run dir>    resumes the HF Trainer state
#   TAG=ttm_c<CONTEXT>_h96                  run name -> /results/kagg/runs/<TAG>, mirrored to
#                                           research/opponent_model/runs/<TAG> every 10 min
# Launch: tmux new -d -s kagg-train 'BASE=models/<dir> bash research/opponent_model/ops/train.sh 2>&1 | tee /results/kagg/logs/train.log'
# Output: runs/<TAG>/{best,scores.json,scaler.npz,eval.json,eval.txt}; best/ carries config.json,
# scaler.npz, labels.json and val_episodes.json = everything the oracle and evaluate.py need.
# Promote best/ to models/<name>/ and chain daily refits with BASE=<that> ops/finetune.sh.
set -u
CONTEXT=${CONTEXT:-256}; PATCH=${PATCH:-32}; STRIDE=${STRIDE:-17}; GPUS=${GPUS:-3}; LR=${LR:-1e-4}
PATIENCE=${PATIENCE:-6}; EPOCHS=${EPOCHS:-40}; PORT=${PORT:-29576}; RESUME=${RESUME:-}
K=/results/kagg; H=/home/jovyan/kaggriculture; SRC=$H/research/opponent_model
DATASET=${DATASET:-$K/datasets/shards}; TAG=${TAG:-ttm_c${CONTEXT}_h96}; RUN=$K/runs/$TAG; DEST=$SRC/runs/$TAG
log(){ echo "[$(date -u '+%H:%M:%S')] $*"; }
[ -n "${BASE:-}" ] && [ -s "$BASE/scaler.npz" ] || { echo "BASE=<model dir with scaler.npz> is required"; echo TRAIN_EXIT=1; exit 1; }
[ -s "$DATASET/labels.json" ] || { echo "no shards in $DATASET (run ops/copy.sh or ops/extract.sh first)"; echo TRAIN_EXIT=1; exit 1; }
log "dataset $DATASET: $(ls "$DATASET"/*.npz | wc -l) shards, $(cat "$DATASET/labels.json")"
mkdir -p "$DEST" "$K/logs" "$SRC/logs"; cd "$H"
( while sleep 600; do
    for d in "$RUN"/checkpoint-* "$RUN"/best; do [ -d "$d" ] && rsync -a "$d" "$DEST/" 2>/dev/null; done
    for d in "$DEST"/checkpoint-*; do [ -d "$d" ] && [ ! -d "$RUN/$(basename "$d")" ] && rm -rf "$d"; done
  done ) &
SYNC=$!
if [ "$GPUS" -gt 1 ]; then LAUNCH="$K/venv-cuda/bin/torchrun --nproc_per_node=$GPUS --master_port=$PORT"
else LAUNCH="env CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} $K/venv-cuda/bin/python"; fi
log "TRAIN $TAG: context $CONTEXT / patch $PATCH, 96-step supervision, warm start from $BASE, $GPUS GPU(s), lr $LR, stride $STRIDE${RESUME:+, resuming from $RESUME}"
$LAUNCH "$SRC/train_ttm.py" \
  --dataset "$DATASET" --out "$RUN" --init-from "$BASE" --scaler "$BASE/scaler.npz" --split episode \
  --context-length "$CONTEXT" --patch-length "$PATCH" \
  --prediction-filter-length 96 --metric-horizon all --eval-on-start \
  --max-episodes 100000 --epochs "$EPOCHS" --patience "$PATIENCE" \
  --plateau --plateau-factor 0.5 --plateau-patience 2 \
  --batch-size 64 --window-stride "$STRIDE" --lr "$LR" ${RESUME:+--resume-from "$RESUME"}
RC=$?; log "train exit=$RC"
kill $SYNC 2>/dev/null
rm -rf "$RUN"/checkpoint-* "$DEST"/checkpoint-*
log "canonical evaluation (the held-out episodes, hour-balanced grid)"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} "$K/venv-cuda/bin/python" "$SRC/evaluate.py" --checkpoint "$RUN/best" --dataset "$DATASET" \
  --stride 1 --include-leaky --out "$RUN/eval.json" 2>&1 | grep -vE 'HTTP|Warning|warn' | tee "$RUN/eval.txt"
log "copying best checkpoint + scores home"
rsync -ah "$RUN/best" "$RUN/scores.json" "$RUN/scaler.npz" "$RUN/eval.json" "$RUN/eval.txt" "$DEST/" 2>/dev/null
cp "$K"/logs/train*.log "$SRC/logs/" 2>/dev/null
echo TRAIN_EXIT=$RC; log "TRAIN_DONE"
