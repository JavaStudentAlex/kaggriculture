# ttm_c256_h96_ft_2026-09-23 — opponent sell model, context 256, refit through 2026-09-23

Daily refit of the 256-context base (`ttm_c256_h96`, initialized from `ttm_c256_h96_ft_2026-09-22`)
on the newest Kaggle tournament days (2026-09-19 through 2026-09-23, recency-weighted).
Trained on Kaggle's dual NVIDIA Tesla T4 GPUs via Distributed Data Parallel (`torchrun --nproc_per_node=2`).

## Execution & Convergence
- **Training Duration:** 194.2 minutes (~3.24 hours)
- **Batch Size:** 64 per GPU (effective 128)
- **Learning Rate:** 4e-5 (scaled for 2 GPUs) with ReduceLROnPlateau (`factor=0.5, patience=2`)
- **Early Stopping:** Patience 5 epochs, best epoch 13 of 30 (stopped at epoch 18)
- **Epoch 0 baseline:** AUC 0.8534 / AP 0.3907 -> Best: AUC 0.8554 / AP 0.3976

## Evaluation Metrics (Held-out validation on 2026-09-23)
- **Windows Evaluated (N):** 852,480 (all horizons pooled)
- **AUC (Any Sell):** 0.8554
- **Average Precision (AP Any Sell):** 0.3976 (Baseline rate: 0.0718 -> **5.54x lift**)
- **MAE (when sold):** 4.26 (vs 4.94 all-zero baseline)

### Per-Product Detection AUC
- Product 0: 0.8063
- Product 1: 0.8695
- Product 2: 0.9150
- Product 3: 0.7842
- Product 4: 0.8953
- Product 5: 0.8720
- Product 6: 0.7796
- Product 7: 0.8361
- Product 8: 0.8552
