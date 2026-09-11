#!/usr/bin/env bash
# Fine-tune the base model on ONE day of Kaggle's daily replays (the newest
# published day by default), validating on 10% of that day's episodes.
# Measured on 2026-09-10: v3 AUC 0.8115 / AP 0.1396 -> 0.8326 / 0.1605.
#   DAY=2026-09-10 bash finetune_1day.sh      (DAY defaults to yesterday UTC)
# Runs from the repo root; start it in tmux and tee to /results/kagg/logs/.
set -u
cd "$(dirname "$0")/../../.." || exit 1
DAY="${DAY:-$(date -u -d yesterday +%F)}" PREV_DAYS=0 GPUS=3 LR=6e-5 STRIDE=4 \
  bash research/opponent_model/ops/finetune.sh
