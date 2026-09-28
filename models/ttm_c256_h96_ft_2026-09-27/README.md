# ttm_c256_h96_ft_2026-09-27 — opponent sell model, context 256, refit through 2026-09-27

Daily refit of the 256-context base (`ttm_c256_h96`, initialized from `ttm_c256_h96_ft_2026-09-26`)
on the newest Kaggle tournament days (2026-09-23 through 2026-09-27, recency-weighted).
Trained on Kaggle's dual NVIDIA Tesla T4 GPUs via Distributed Data Parallel (`torchrun --nproc_per_node=2`).

## Execution & Convergence
- **Training Duration:** 162.6 minutes (~2.71 hours)
- **Batch Size:** 64 per GPU (effective 128)
- **Learning Rate:** 4e-5 (scaled for 2 GPUs) with ReduceLROnPlateau (`factor=0.5, patience=2`)
- **Early Stopping:** Patience 5 epochs, best epoch 10 of 30 (stopped at epoch 15)
- **Epoch 0 baseline:** AUC 0.8529 / AP 0.3733 -> Best: AUC 0.8543 / AP 0.3727

## Evaluation Metrics (Held-out validation on 2026-09-27)
- **Windows Evaluated (N):** 852,480 (all horizons pooled)
- **AUC (Any Sell):** 0.8543
- **Average Precision (AP Any Sell):** 0.3727 (Baseline rate: 0.0780 -> **4.78x lift**)
- **MAE (when sold):** 4.13 (vs 4.71 all-zero baseline)

### Per-Product Detection AUC
- Product 0: 0.8050
- Product 1: 0.9134
- Product 2: 0.9392
- Product 3: 0.7611
- Product 4: 0.8765
- Product 5: 0.8772
- Product 6: 0.7631
- Product 7: 0.8407
- Product 8: 0.8221
