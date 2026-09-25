# ttm_c256_h96_ft_2026-09-24 — opponent sell model, context 256, refit through 2026-09-24

Daily refit of the 256-context base (`ttm_c256_h96`, initialized from `ttm_c256_h96_ft_2026-09-23`)
on the newest Kaggle tournament days (2026-09-20 through 2026-09-24, recency-weighted).
Trained on Kaggle's dual NVIDIA Tesla T4 GPUs via Distributed Data Parallel (`torchrun --nproc_per_node=2`).

## Execution & Convergence
- **Training Duration:** 127.0 minutes (~2.12 hours)
- **Batch Size:** 64 per GPU (effective 128)
- **Learning Rate:** 4e-5 (scaled for 2 GPUs) with ReduceLROnPlateau (`factor=0.5, patience=2`)
- **Early Stopping:** Patience 5 epochs, best epoch 7 of 30 (stopped at epoch 12)
- **Epoch 0 baseline:** AUC 0.8523 / AP 0.4054 -> Best: AUC 0.8535 / AP 0.4106

## Evaluation Metrics (Held-out validation on 2026-09-24)
- **Windows Evaluated (N):** 838,272 (all horizons pooled)
- **AUC (Any Sell):** 0.8535
- **Average Precision (AP Any Sell):** 0.4106 (Baseline rate: 0.0789 -> **5.21x lift**)
- **MAE (when sold):** 3.99 (vs 4.65 all-zero baseline)

### Per-Product Detection AUC
- Product 0: 0.8025
- Product 1: 0.8887
- Product 2: 0.9056
- Product 3: 0.7801
- Product 4: 0.8734
- Product 5: 0.8659
- Product 6: 0.7808
- Product 7: 0.8176
- Product 8: 0.8537
