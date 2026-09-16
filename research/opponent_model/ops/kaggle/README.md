# Kaggle Dual-GPU Opponent Model Fine-Tuning

This directory contains the automation to fine-tune the IBM Granite TinyTimeMixer (TTM) opponent action model on Kaggle's free GPU compute (Dual Tesla T4 GPUs) using the 5 most recent tournament days.

## Components
- `train_5days.py`: Complete pipeline executed inside the Kaggle GPU container:
  1. Installs `granite-tsfm` and `kaggle-environments`.
  2. Locates the base model checkpoint (`models/ttm_c256_h96_ft_2026-09-13`).
  3. Patches precision for Turing GPUs (`fp16` on sm_75 Tesla T4).
  4. Extracts 5 days of mounted episode JSONs into `.npz` feature shards in parallel across 4 CPU workers.
  5. Stages recency-weighted dataset split (`DECAY=2.0, STRIDE=4`).
  6. Launches distributed training across 2× Tesla T4 GPUs via `torchrun --nproc_per_node=2`.
  7. Exports best checkpoint (`model.safetensors`, `scaler.npz`, `scores.json`) to `/kaggle/working/`.
- `kernel-metadata.json`: Kaggle API configuration for private, dual-GPU, internet-enabled job execution.
- `push_job.sh`: Convenience script to push and trigger the job via `kaggle kernels push`.

## Usage
To push a new fine-tuning run:
```bash
bash research/opponent_model/ops/kaggle/push_job.sh
```

To pull the output checkpoint when complete:
```bash
kaggle kernels output sunshinethroughfog/kaggriculture-ttm-5day-finetune -p models/ttm_c256_h96_ft_<date>/
```
