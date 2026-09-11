#!/usr/bin/env bash
# Base-model training from scratch-ish: v3 = 96-step (4 game days) supervision
# on dataset_v2, warm-started from the v2 checkpoint (24-step; kept outside git
# at research/opponent_model/runs/ttm_v2/best on ceph, staged by ops/copy.sh).
# This is how models/ttm_v3_h96 was produced on 2026-09-11 (best epoch 19,
# early-stopped at 27). Checkpoints are synced to ceph every 10 min.
# RESUME=<checkpoint-N dir or run dir> resumes the HF Trainer state.
set -u
RESUME=${RESUME:-}
K=/results/kagg; H=/home/jovyan/kaggriculture
SRC=$H/research/opponent_model; RUN=$K/runs/ttm_v3_h96; DEST=$SRC/runs/ttm_v3_h96
log(){ echo "[$(date -u '+%H:%M:%S')] $*"; }
until grep -q ENV_EXIT=0 "$K/logs/env.log" 2>/dev/null && grep -q COPY_EXIT=0 "$K/logs/copy.log" 2>/dev/null; do sleep 20; done
mkdir -p "$DEST"; cd "$H"
( while sleep 600; do
    for d in "$RUN"/checkpoint-* "$RUN"/best; do [ -d "$d" ] && rsync -a "$d" "$DEST/" 2>/dev/null; done
    for d in "$DEST"/checkpoint-*; do [ -d "$d" ] && [ ! -d "$RUN/$(basename "$d")" ] && rm -rf "$d"; done
  done ) &
SYNC=$!
log "TRAIN v3: 96-step supervision, warm start from v2 best, 3 GPUs${RESUME:+, resuming from $RESUME}"
"$K/venv-cuda/bin/torchrun" --nproc_per_node=3 --master_port=29574 "$SRC/train_ttm.py" \
  --dataset "$K/dataset_v2" --out "$RUN" --init-from "$K/runs/ttm_v2/best" --split series \
  --prediction-filter-length 96 --metric-horizon all \
  --max-episodes 100000 --epochs 200 --patience 8 \
  --plateau --plateau-factor 0.5 --plateau-patience 2 \
  --batch-size 64 --window-stride 8 --lr 1e-4 ${RESUME:+--resume-from "$RESUME"}
log "v3 train exit=$?"
kill $SYNC 2>/dev/null
log "canonical evaluation (episode-disjoint, hour-balanced grid)"
CUDA_VISIBLE_DEVICES=0 "$K/venv-cuda/bin/python" "$SRC/evaluate.py" --checkpoint "$RUN/best" --dataset "$K/dataset_v2" \
  --split series --max-episodes 100000 --stride 1 --include-leaky --out "$RUN/eval.json" 2>&1 | grep -vE 'HTTP|Warning|warn' | tee "$RUN/eval.txt"
log "copying best checkpoint + scores home"
rsync -ah "$RUN/best" "$RUN/scores.json" "$RUN/scaler.npz" "$RUN/eval.json" "$RUN/eval.txt" "$DEST/" 2>/dev/null
rsync -ah "$K/logs/" "$SRC/logs/"
log "V3_DONE"
