# ttm_c256_h96_ft_2026-09-18 — opponent sell model, context 256, refit through 2026-09-18

Daily refit of the 256-context base (`ttm_c256_h96`, initialized from `ttm_c256_h96_ft_2026-09-17`)
on the newest Kaggle tournament days (2026-09-13 through 2026-09-18, recency-weighted).
Trained on Kaggle's dual NVIDIA Tesla T4 GPUs via Distributed Data Parallel (`torchrun --nproc_per_node=2`).

## Execution & Convergence
- **Training Duration:** 254.2 minutes (~4.24 hours)
- **Batch Size:** 64 per GPU (effective 128)
- **Learning Rate:** 4e-5 (scaled for 2 GPUs) with ReduceLROnPlateau (`factor=0.5, patience=2`)
- **Early Stopping:** Patience 5 epochs, best epoch 19 of 30 (stopped at epoch 24)
- **Epoch 0 baseline:** AUC 0.8355 / AP 0.3316 -> Best: AUC 0.8408 / AP 0.3426

## Evaluation Metrics (Held-out validation on 2026-09-18)
- **Windows Evaluated (N):** 909,312 (all horizons pooled)
- **AUC (Any Sell):** 0.8408
- **Average Precision (AP Any Sell):** 0.3426 (Baseline rate: 0.0672 -> **5.10x lift**)
- **MAE (when sold):** 4.09 (vs 4.66 all-zero baseline)

### Per-Product Detection AUC
- Product 0: 0.7951
- Product 1: 0.8542
- Product 2: 0.8909
- Product 3: 0.7477
- Product 4: 0.8249
- Product 5: 0.8542
- Product 6: 0.7749
- Product 7: 0.8226
- Product 8: 0.8198
