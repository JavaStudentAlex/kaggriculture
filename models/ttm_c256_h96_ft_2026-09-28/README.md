# ttm_c256_h96_ft_2026-09-28 — opponent sell model, context 256, refit through 2026-09-28

Daily refit of the 256-context base (`ttm_c256_h96`, initialized from `ttm_c256_h96_ft_2026-09-27`)
on the newest Kaggle tournament days (2026-09-24 through 2026-09-28, recency-weighted).
Trained on Kaggle's dual NVIDIA Tesla T4 GPUs via Distributed Data Parallel (`torchrun --nproc_per_node=2`).

## Execution & Convergence
- **Training Duration:** 79.6 minutes (~1.33 hours)
- **Batch Size:** 64 per GPU (effective 128)
- **Learning Rate:** 4e-5 (scaled for 2 GPUs) with ReduceLROnPlateau (`factor=0.5, patience=2`)
- **Early Stopping:** Patience 5 epochs, best epoch 2 of 30 (stopped at epoch 7)
- **Epoch 0 baseline:** AUC 0.8559 / AP 0.3856 -> Best: AUC 0.8557 / AP 0.3852

## Evaluation Metrics (Held-out validation on 2026-09-28)
- **Windows Evaluated (N):** 838,272 (all horizons pooled)
- **AUC (Any Sell):** 0.8557
- **Average Precision (AP Any Sell):** 0.3852 (Baseline rate: 0.0822 -> **4.69x lift**)
- **MAE (when sold):** 4.08 (vs 4.67 all-zero baseline)

### Per-Product Detection AUC
- Product 0: 0.8151
- Product 1: 0.9113
- Product 2: 0.9305
- Product 3: 0.7573
- Product 4: 0.9125
- Product 5: 0.8851
- Product 6: 0.7440
- Product 7: 0.8330
- Product 8: 0.8328
