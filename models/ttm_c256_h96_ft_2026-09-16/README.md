# ttm_c256_h96_ft_2026-09-16 — opponent sell model, context 256, refit through 2026-09-16

Daily refit of the 256-context base (`ttm_c256_h96`, initialized from `ttm_c256_h96_ft_2026-09-15`)
on the newest Kaggle tournament days (2026-09-11 through 2026-09-16, recency-weighted).
Trained on Kaggle's dual NVIDIA Tesla T4 GPUs via Distributed Data Parallel (`torchrun --nproc_per_node=2`).

## Execution & Convergence
- **Training Duration:** 171.2 minutes (~2.85 hours)
- **Batch Size:** 64 per GPU (effective 128)
- **Learning Rate:** 4e-5 (scaled for 2 GPUs) with ReduceLROnPlateau (`factor=0.5, patience=2`)
- **Early Stopping:** Patience 5 epochs, best epoch 10 of 15
- **Epoch 0 baseline:** AUC 0.8515 / AP 0.3389 -> Best: AUC 0.8535 / AP 0.3501

## Evaluation Metrics (Held-out validation on 2026-09-16)
- **Windows Evaluated (N):** 895,104 (all horizons pooled)
- **AUC (Any Sell):** 0.8535
- **Average Precision (AP Any Sell):** 0.3501 (Baseline rate: 0.0578 -> **6.06x lift**)
- **MAE (when sold):** 4.74 (vs 5.34 all-zero baseline)

### Per-Product Detection AUC
- Product 0: 0.8310
- Product 1: 0.8728
- Product 2: 0.8992
- Product 3: 0.7807
- Product 4: 0.8205
- Product 5: 0.8828
- Product 6: 0.7886
- Product 7: 0.8222
- Product 8: 0.8370
