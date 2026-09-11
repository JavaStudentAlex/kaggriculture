#!/usr/bin/env bash
# Stage the training inputs on the SSD: raw daily replay zips, the extracted
# datasets (dataset_v2 = v3's frozen training set, dataset_daily = newer days)
# and the models. Ceph is read once here; training then never touches it.
set -u
K=/results/kagg; H=/home/jovyan/kaggriculture
log(){ echo "[$(date -u '+%H:%M:%S')] $*"; }
mkdir -p "$K/replays" "$K/dataset_v2" "$K/dataset_daily" "$K/runs/ttm_v2" "$K/models"
log "replay zips (Kaggle daily datasets, ~0.5 GB/day)"
rsync -ah --stats "$H"/replays/*.zip "$K/replays/" || { echo COPY_EXIT=1; exit 1; }
log "dataset_v2 (42 shards, 2 GB) + dataset_daily"
rsync -ah --stats "$H/research/opponent_model/dataset_v2/" "$K/dataset_v2/" || { echo COPY_EXIT=1; exit 1; }
rsync -ah "$H/research/opponent_model/dataset_daily/" "$K/dataset_daily/" 2>/dev/null || true
log "models (+ the v2 checkpoint for train_v3.sh, if it is still on ceph)"
rsync -ah "$H/models/" "$K/models/" || { echo COPY_EXIT=1; exit 1; }
[ -d "$H/research/opponent_model/runs/ttm_v2/best" ] && rsync -ah "$H/research/opponent_model/runs/ttm_v2/best" "$K/runs/ttm_v2/"
log "done: $(ls $K/replays/*.zip | wc -l) zips, $(ls $K/dataset_v2/*.npz | wc -l)+$(ls $K/dataset_daily/*.npz 2>/dev/null | wc -l) shards"; echo COPY_EXIT=0
