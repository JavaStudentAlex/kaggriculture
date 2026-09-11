#!/usr/bin/env bash
# Daily fine-tune of the opponent sell model on Kaggle's own daily datasets
# (kaggle/kaggriculture-episodes-<date>, ~660 top episodes, published ~00:10 UTC
# the next day). Base = models/ttm_v3_h96 with its input scaler (never refit).
#   DAY=2026-09-10       newest training day (default: yesterday UTC)
#   PREV_DAYS=4          days before DAY added to training in full
#   VAL_DAY=2026-09-11   validate on this whole day (forward in time) instead of
#                        a 10% episode split of DAY; DAY is then trained on in
#                        full. Waits until Kaggle publishes it.
#   GPUS=3 LR=6e-5       torchrun on GPUS devices; LR scales with the batch
#                        (2e-5 at 1 GPU x 64, linear rule -> 6e-5 at 3 GPUs)
#   BASE=<model dir>     weights + scaler.npz to start from (default v3 best;
#                        point it at yesterday's runs/ft_*/best to chain refits)
#   DECAY=2 STRIDE=4     recency weighting: the newest day is windowed at
#                        STRIDE, each day further back at STRIDE*DECAY^age, so
#                        with DECAY=2 every day back contributes half as many
#                        windows (newest day ~50% of training with 4 prev days)
# Epoch 0 in the log is the un-tuned v3 scored on the same val windows.
# dataset_v2 stays frozen at 42 shards (v3's training set, EVALUATION.md
# depends on it); new daily shards go to $K/dataset_daily, mirrored home.
set -u
DAY=${DAY:-$(date -u -d yesterday +%F)}; PREV_DAYS=${PREV_DAYS:-0}; VAL_DAY=${VAL_DAY:-}
GPUS=${GPUS:-1}; LR=${LR:-2e-5}; PORT=${PORT:-29581}; DECAY=${DECAY:-1}; STRIDE=${STRIDE:-8}
K=/results/kagg; H=/home/jovyan/kaggriculture; SRC=$H/research/opponent_model
BASE=${BASE:-$H/models/ttm_v3_h96}; PY=$K/venv-cuda/bin/python; KG=$K/venv-cuda/bin/kaggle
TAG=ft_$DAY$([ "$PREV_DAYS" -gt 0 ] && echo "_p$PREV_DAYS")${VAL_DAY:+_val$VAL_DAY}$([ "$GPUS" -gt 1 ] && echo "_g$GPUS")$([ "$DECAY" != 1 ] && echo "_d$DECAY")$([ "$STRIDE" != 8 ] && echo "_s$STRIDE")
RUN=$K/runs/$TAG; DEST=$SRC/runs/$TAG
log(){ echo "[$(date -u '+%H:%M:%S')] $*"; }
die(){ log "$*"; echo FT_EXIT=1; exit 1; }
[ -s "$BASE/scaler.npz" ] || { echo "no scaler.npz in $BASE"; echo FT_EXIT=1; exit 1; }
cd "$H"; mkdir -p "$K/dataset_daily" "$K/replays_daily_in" "$K/replays" "$DEST"

shard(){ for s in "$K/dataset_daily" "$K/dataset_v2"; do
           [ -s "$s/kaggriculture-episodes-$1.npz" ] && { echo "$s/kaggriculture-episodes-$1.npz"; return 0; }; done; return 1; }
ensure_day(){  # make sure the day's shard exists: reuse, else download Kaggle's dataset (waiting for it) and extract
  local d=$1 zip=$K/replays/kaggriculture-episodes-$1.zip
  shard "$d" >/dev/null && return 0
  [ -s "$zip" ] || { [ -s "$H/replays/kaggriculture-episodes-$d.zip" ] && cp "$H/replays/kaggriculture-episodes-$d.zip" "$zip"; }
  if [ ! -s "$zip" ]; then
    until "$KG" datasets files "kaggle/kaggriculture-episodes-$d" 2>/dev/null | grep -q '\.json'; do
      log "kaggle/kaggriculture-episodes-$d is not published yet; checking again in 5 min"; sleep 300
    done
    log "downloading kaggle/kaggriculture-episodes-$d"
    "$KG" datasets download "kaggle/kaggriculture-episodes-$d" -p "$K/replays" || return 1
    [ -s "$zip" ] || return 1
    ( cp "$zip" "$H/replays/" ) &   # canonical copy on ceph, slow, in the background
  fi
  ln -sfn "$zip" "$K/replays_daily_in/"
  log "extracting $d"
  "$PY" "$SRC/extract_parallel.py" --replays "$K/replays_daily_in" --out "$K/dataset_daily" --workers 16
  shard "$d" >/dev/null
}

for i in $(seq 0 "$PREV_DAYS"); do d=$(date -u -d "$DAY - $i day" +%F); ensure_day "$d" || die "no data for $d"; done
[ -z "$VAL_DAY" ] || ensure_day "$VAL_DAY" || die "no data for $VAL_DAY"

D=$K/dataset_$TAG; rm -rf "$D"; mkdir -p "$D/day" "$D/prev" "$D/val"
ln -s "$(shard "$DAY")" "$D/day/"
EXTRA=""; VALARG=""
for i in $(seq 1 "$PREV_DAYS"); do ln -s "$(shard "$(date -u -d "$DAY - $i day" +%F)")" "$D/prev/"; done
[ "$PREV_DAYS" -gt 0 ] && EXTRA="--extra-train $D/prev --extra-train-decay $DECAY"
[ -n "$VAL_DAY" ] && { ln -s "$(shard "$VAL_DAY")" "$D/val/"; VALARG="--val-dataset $D/val"; }
log "days in train: $(ls "$D/day" "$D/prev" | grep npz | sed 's/kaggriculture-episodes-//; s/.npz//' | sort | tr '\n' ' ')"
log "validation: ${VAL_DAY:+whole day $VAL_DAY}${VAL_DAY:-10% episode split of $DAY}"

if [ "$GPUS" -gt 1 ]; then LAUNCH="$K/venv-cuda/bin/torchrun --nproc_per_node=$GPUS --master_port=$PORT"
else LAUNCH="env CUDA_VISIBLE_DEVICES=0 $PY"; fi
log "FINETUNE $TAG: init from $BASE (+ its scaler), $GPUS GPU(s), lr $LR, stride $STRIDE, decay $DECAY"
$LAUNCH "$SRC/train_ttm.py" \
  --dataset "$D/day" $EXTRA $VALARG --out "$RUN" --init-from "$BASE" --scaler "$BASE/scaler.npz" \
  --split episode --prediction-filter-length 96 --metric-horizon all --eval-on-start \
  --max-episodes 100000 --epochs 30 --patience 5 \
  --plateau --plateau-factor 0.5 --plateau-patience 2 \
  --batch-size 64 --window-stride "$STRIDE" --lr "$LR"
RC=$?; log "finetune exit=$RC"
rm -rf "$RUN"/checkpoint-*
rsync -ah "$RUN/best" "$RUN/scores.json" "$RUN/scaler.npz" "$DEST/" 2>/dev/null
rsync -ah "$K/dataset_daily/" "$SRC/dataset_daily/"
rsync -ah "$K/logs/" "$SRC/logs/"
echo FT_EXIT=$RC; log "FT_DONE"
