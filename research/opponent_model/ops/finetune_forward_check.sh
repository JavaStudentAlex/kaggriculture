#!/usr/bin/env bash
# Forward-in-time check of the 4-day recipe: train on DAY and the 4 days before
# it (all episodes, recency-weighted) and validate on the WHOLE next day, which
# Kaggle publishes ~00:10 UTC the day after. Waits for it if not out yet.
# Epoch 0 in the log = the base model on that day; the refit is validated if a
# later epoch beats it. Use PORT=29582 if another fine-tune is running.
#   DAY=2026-09-10 bash finetune_forward_check.sh   (validates on 2026-09-11)
set -u
cd "$(dirname "$0")/../../.." || exit 1
DAY="${DAY:-$(date -u -d '2 days ago' +%F)}"
DAY="$DAY" VAL_DAY="$(date -u -d "$DAY + 1 day" +%F)" PREV_DAYS=4 DECAY=2 STRIDE=4 GPUS=3 LR=6e-5 PORT="${PORT:-29582}" \
  bash research/opponent_model/ops/finetune.sh
