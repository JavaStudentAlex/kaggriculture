# ttm_c256_h96_ft_2026-09-19 — opponent sell model, context 256, refit through 2026-09-19

Daily refit of the 256-context base (`ttm_c256_h96`, initialized from `ttm_c256_h96_ft_2026-09-18`)
on the newest Kaggle tournament days (2026-09-14 through 2026-09-19, recency-weighted).
Trained on Kaggle's dual NVIDIA Tesla T4 GPUs via Distributed Data Parallel (`torchrun --nproc_per_node=2`).

## Execution & Convergence
- **Training Duration:** 284.3 minutes (~4.74 hours)
- **Batch Size:** 64 per GPU (effective 128)
- **Learning Rate:** 4e-5 (scaled for 2 GPUs) with ReduceLROnPlateau (`factor=0.5, patience=2`)
- **Early Stopping:** Patience 5 epochs, best epoch 19 of 30 (stopped at epoch 24)
- **Epoch 0 baseline:** AUC 0.8420 / AP 0.3451 -> Best: AUC 0.8467 / AP 0.3557

## Evaluation Metrics (Held-out validation on 2026-09-19)
- **Windows Evaluated (N):** 909,312 (all horizons pooled)
- **AUC (Any Sell):** 0.8467
- **Average Precision (AP Any Sell):** 0.3557 (Baseline rate: 0.0657 -> **5.41x lift**)
- **MAE (when sold):** 4.13 (vs 4.71 all-zero baseline)

### Per-Product Detection AUC
- Product 0: 0.8026
- Product 1: 0.8685
- Product 2: 0.9033
- Product 3: 0.7782
- Product 4: 0.8813
- Product 5: 0.8572
- Product 6: 0.7885
- Product 7: 0.8097
- Product 8: 0.8157
