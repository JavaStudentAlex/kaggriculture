# ttm_c256_h96_ft_2026-09-22 — opponent sell model, context 256, refit through 2026-09-22

Daily refit of the 256-context base (`ttm_c256_h96`, initialized from `ttm_c256_h96_ft_2026-09-21`)
on the newest Kaggle tournament days (2026-09-18 through 2026-09-22, recency-weighted).
Trained on Kaggle's dual NVIDIA Tesla T4 GPUs via Distributed Data Parallel (`torchrun --nproc_per_node=2`).

## Execution & Convergence
- **Training Duration:** 247.4 minutes (~4.12 hours)
- **Batch Size:** 64 per GPU (effective 128)
- **Learning Rate:** 4e-5 (scaled for 2 GPUs) with ReduceLROnPlateau (`factor=0.5, patience=2`)
- **Early Stopping:** Patience 5 epochs, best epoch 18 of 30 (stopped at epoch 23)
- **Epoch 0 baseline:** AUC 0.8493 / AP 0.3619 -> Best: AUC 0.8533 / AP 0.3713

## Evaluation Metrics (Held-out validation on 2026-09-22)
- **Windows Evaluated (N):** 880,896 (all horizons pooled)
- **AUC (Any Sell):** 0.8533
- **Average Precision (AP Any Sell):** 0.3713 (Baseline rate: 0.0656 -> **5.66x lift**)
- **MAE (when sold):** 4.71 (vs 5.39 all-zero baseline)

### Per-Product Detection AUC
- Product 0: 0.7962
- Product 1: 0.8939
- Product 2: 0.9027
- Product 3: 0.7935
- Product 4: 0.8849
- Product 5: 0.8717
- Product 6: 0.7809
- Product 7: 0.8181
- Product 8: 0.8435
