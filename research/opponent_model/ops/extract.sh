#!/usr/bin/env bash
# Extract every staged replay day that has no shard yet into the shard directory
# (one worker per zip, ~10 min per day on the pod), then mirror the whole directory
# home (labels.json included). Resumable and idempotent: run it after each daily
# download, or once to rebuild everything after a pod restart.
#   DAYS_DIR=/results/kagg/replays         zips to extract (default: all staged days)
#   OUT=/results/kagg/datasets/shards      shard dir on the SSD (labels.json records the rule)
#   MIRROR=~/kaggriculture/datasets/shards ceph copy
set -u
K=/results/kagg; H=/home/jovyan/kaggriculture; SRC=$H/research/opponent_model
DAYS_DIR=${DAYS_DIR:-$K/replays}; OUT=${OUT:-$K/datasets/shards}; MIRROR=${MIRROR:-$H/datasets/shards}; WORKERS=${WORKERS:-44}
log(){ echo "[$(date -u '+%H:%M:%S')] $*"; }
cd "$H"; mkdir -p "$OUT" "$MIRROR"
log "EXTRACT $(ls "$DAYS_DIR"/*.zip | wc -l) zips from $DAYS_DIR -> $OUT (next_action), $WORKERS workers"
"$K/venv-cuda/bin/python" "$SRC/extract_parallel.py" --replays "$DAYS_DIR" --out "$OUT" --workers "$WORKERS" --alignment next_action
RC=$?
log "extract exit=$RC; shards: $(ls "$OUT"/*.npz 2>/dev/null | wc -l)"
rsync -ah "$OUT/" "$MIRROR/"
echo EXTRACT_EXIT=$RC; log "EXTRACT_DONE"
