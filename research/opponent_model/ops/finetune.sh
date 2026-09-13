#!/usr/bin/env bash
# Daily fine-tune of the opponent sell model on Kaggle's own daily datasets
# (kaggle/kaggriculture-episodes-<date>, ~660 top episodes, published ~00:10 UTC
# the next day): the newest day PLUS the 4 days before it, recency-weighted (each
# day back contributes half as many windows, the newest day ~48 % of every epoch),
# validated on 10 % of the newest day's episodes. Chains from BASE (required) with
# BASE's input scaler (never refit). Weighting toward the newest day was the biggest
# lever when the recipe was tuned (+0.03 AUC / +0.05 AP over equal weighting).
#   BASE=models/<newest promoted dir> DAY=2026-09-12 bash research/opponent_model/ops/finetune.sh
#   BASE=<model dir>     weights + scaler.npz + labels.json to start from: the newest
#                        promoted models/* (or a runs/*/best to chain unpromoted refits).
#                        Its config.json sets the context length; its labels.json must
#                        say next_action (the shards' rule) -- anything else is refused.
#   DAY=2026-09-12       newest training day (default: yesterday UTC)
#   PREV_DAYS=4          days before DAY added to training in full
#   DECAY=2 STRIDE=4     recency weighting: the newest day is windowed at STRIDE, each
#                        day further back at STRIDE*DECAY^age
#   GPUS=3 LR=6e-5       torchrun on GPUS devices; LR scales with the batch
#                        (2e-5 at 1 GPU x 64, linear rule -> 6e-5 at 3 GPUs)
#   VAL_DAY=2026-09-13   forward-in-time check instead of the 10 % split: validate on
#                        this whole day (waits until Kaggle publishes it), DAY trained in full
#   PORT=29581           torchrun master port; change it when two runs overlap
# Epoch 0 in the log is the un-tuned BASE scored on the same val windows -- the number
# the refit has to beat. Shards: one directory for every day, /results/kagg/datasets/shards
# (SSD), mirrored to ~/kaggriculture/datasets/shards; a missing day is downloaded and
# extracted here. Launch in tmux, tee to /results/kagg/logs/finetune_<DAY>.log.
set -u
DAY=${DAY:-$(date -u -d yesterday +%F)}; PREV_DAYS=${PREV_DAYS:-4}; VAL_DAY=${VAL_DAY:-}
GPUS=${GPUS:-3}; LR=${LR:-6e-5}; PORT=${PORT:-29581}; DECAY=${DECAY:-2}; STRIDE=${STRIDE:-4}
K=/results/kagg; H=/home/jovyan/kaggriculture; SRC=$H/research/opponent_model
SHARDS=$K/datasets/shards; PY=$K/venv-cuda/bin/python; KG=$K/venv-cuda/bin/kaggle
log(){ echo "[$(date -u '+%H:%M:%S')] $*"; }
die(){ log "$*"; echo FT_EXIT=1; exit 1; }
[ -n "${BASE:-}" ] || die "BASE=<model dir> is required (chain from the newest models/*)"
[ -s "$BASE/scaler.npz" ] || die "no scaler.npz in $BASE"
ALIGN=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("alignment", "legacy"))' "$BASE/labels.json" 2>/dev/null || echo legacy)
[ "$ALIGN" = next_action ] || die "$BASE was trained on '$ALIGN' labels; only next_action checkpoints can be fine-tuned on these shards"
TAG=ft_$DAY$([ "$PREV_DAYS" -gt 0 ] && echo "_p$PREV_DAYS")${VAL_DAY:+_val$VAL_DAY}$([ "$GPUS" -gt 1 ] && echo "_g$GPUS")$([ "$DECAY" != 1 ] && echo "_d$DECAY")$([ "$STRIDE" != 8 ] && echo "_s$STRIDE")
RUN=$K/runs/$TAG; DEST=$SRC/runs/$TAG
cd "$H"; mkdir -p "$SHARDS" "$K/replays_daily_in" "$K/replays" "$DEST" "$SRC/logs"

shard(){ [ -s "$SHARDS/kaggriculture-episodes-$1.npz" ] && echo "$SHARDS/kaggriculture-episodes-$1.npz"; }
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
  "$PY" "$SRC/extract_parallel.py" --replays "$K/replays_daily_in" --out "$SHARDS" --workers 16 --alignment next_action
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
rm -rf "$RUN"/checkpoint-* "$D"
rsync -ah "$RUN/best" "$RUN/scores.json" "$RUN/scaler.npz" "$DEST/" 2>/dev/null
rsync -ah "$SHARDS/" "$H/datasets/shards/"
cp "$K"/logs/finetune*.log "$SRC/logs/" 2>/dev/null
echo FT_EXIT=$RC; log "FT_DONE"
