#!/usr/bin/env bash
# Push the 5-day TTM opponent model fine-tuning job to Kaggle GPU cluster
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Pushing Kaggle kernel: sunshinethroughfog/kaggriculture-ttm-5day-finetune..."
kaggle kernels push -p "$SCRIPT_DIR"

echo "Checking initial status..."
sleep 3
kaggle kernels status sunshinethroughfog/kaggriculture-ttm-5day-finetune
