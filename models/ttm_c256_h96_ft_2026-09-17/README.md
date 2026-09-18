# ttm_c256_h96_ft_2026-09-17 — opponent sell model, context 256, refit through 2026-09-17

Daily refit of the 256-context base (`ttm_c256_h96`, initialized from `ttm_c256_h96_ft_2026-09-16`)
on the newest Kaggle tournament days (2026-09-12 through 2026-09-17, recency-weighted).
Trained on Kaggle's dual NVIDIA Tesla T4 GPUs via Distributed Data Parallel (`torchrun --nproc_per_node=2`).

## Execution & Convergence
- **Training Duration:** 225.4 minutes (~3.76 hours)
- **Batch Size:** 64 per GPU (effective 128)
- **Learning Rate:** 4e-5 (scaled for 2 GPUs) with ReduceLROnPlateau (`factor=0.5, patience=2`)
- **Early Stopping:** Patience 5 epochs, best epoch 15 of 20
- **Epoch 0 baseline:** AUC 0.8476 / AP 0.3600 -> Best: AUC 0.8480 / AP 0.3617

## Evaluation Metrics (Held-out validation on 2026-09-17)
- **Windows Evaluated (N):** 923,520 (all horizons pooled)
- **AUC (Any Sell):** 0.8480
- **Average Precision (AP Any Sell):** 0.3617 (Baseline rate: 0.0644 -> **5.62x lift**)
- **MAE (when sold):** 4.22 (vs 4.82 all-zero baseline)

### Per-Product Detection AUC
- Product 0: 0.7956
- Product 1: 0.8797
- Product 2: 0.8967
- Product 3: 0.7640
- Product 4: 0.8204
- Product 5: 0.8785
- Product 6: 0.7811
- Product 7: 0.8178
- Product 8: 0.8251
