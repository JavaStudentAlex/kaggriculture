#!/usr/bin/env bash
# Stage the training inputs on the SSD after a pod restart: raw replay zips, the
# shard directory and the run dirs (checkpoints for RESUME). Ceph is read once
# here; training then never touches it. Models (4 MB each) are read from ceph.
set -u
K=/results/kagg; H=/home/jovyan/kaggriculture
log(){ echo "[$(date -u '+%H:%M:%S')] $*"; }
mkdir -p "$K/replays" "$K/datasets/shards" "$K/runs" "$K/logs"
log "replay zips (Kaggle daily datasets, ~0.5 GB/day)"
rsync -ah --stats "$H"/replays/*.zip "$K/replays/" || { echo COPY_EXIT=1; exit 1; }
log "shards (datasets/shards, ~50 MB/day)"
rsync -ah --stats "$H/datasets/shards/" "$K/datasets/shards/" || { echo COPY_EXIT=1; exit 1; }
log "run dirs (checkpoints for RESUME)"
rsync -ah "$H/research/opponent_model/runs/" "$K/runs/" 2>/dev/null || true
log "done: $(ls $K/replays/*.zip | wc -l) zips, $(ls $K/datasets/shards/*.npz | wc -l) shards"; echo COPY_EXIT=0
