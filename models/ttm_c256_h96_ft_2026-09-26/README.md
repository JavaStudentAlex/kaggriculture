# ttm_c256_h96_ft_2026-09-26 — opponent sell model, context 256, refit through 2026-09-26

Daily refit of the 256-context base (`ttm_c256_h96`, initialized from `ttm_c256_h96_ft_2026-09-25`)
on the newest Kaggle tournament days (2026-09-22 through 2026-09-26, recency-weighted).
Trained on Kaggle's dual NVIDIA Tesla T4 GPUs via Distributed Data Parallel (`torchrun --nproc_per_node=2`).

## Execution & Convergence
- **Training Duration:** 97.1 minutes (~1.62 hours)
- **Batch Size:** 64 per GPU (effective 128)
- **Learning Rate:** 4e-5 (scaled for 2 GPUs) with ReduceLROnPlateau (`factor=0.5, patience=2`)
- **Early Stopping:** Patience 5 epochs, best epoch 5 of 30 (stopped at epoch 10)
- **Epoch 0 baseline:** AUC 0.8531 / AP 0.3950 -> Best: AUC 0.8553 / AP 0.3999

## Evaluation Metrics (Held-out validation on 2026-09-26)
- **Windows Evaluated (N):** 838,272 (all horizons pooled)
- **AUC (Any Sell):** 0.8553
- **Average Precision (AP Any Sell):** 0.3999 (Baseline rate: 0.0861 -> **4.64x lift**)
- **MAE (when sold):** 3.92 (vs 4.54 all-zero baseline)

### Per-Product Detection AUC
- Product 0: 0.8020
- Product 1: 0.8977
- Product 2: 0.9487
- Product 3: 0.7599
- Product 4: 0.8900
- Product 5: 0.8788
- Product 6: 0.7681
- Product 7: 0.8083
- Product 8: 0.8313
