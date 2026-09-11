#!/usr/bin/env bash
# The production recipe: fine-tune on the newest day PLUS the 4 days before it,
# recency-weighted (each day back contributes half as many windows, the newest
# day ~48% of every epoch), validating on 10% of the newest day's episodes.
# Measured on 2026-09-10: v3 AUC 0.8115 / AP 0.1396 -> 0.8655 / 0.2124
# (equal weighting of the same 5 days only reached 0.8374 / 0.1633).
#   DAY=2026-09-10 bash finetune_4days.sh     (DAY defaults to yesterday UTC)
#   BASE=<model dir> to chain from a previous fine-tune instead of v3.
set -u
cd "$(dirname "$0")/../../.." || exit 1
DAY="${DAY:-$(date -u -d yesterday +%F)}" PREV_DAYS=4 DECAY=2 STRIDE=4 GPUS=3 LR=6e-5 \
  bash research/opponent_model/ops/finetune.sh
