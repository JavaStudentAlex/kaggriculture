# ttm_c256_h96_ft_2026-09-21 — opponent sell model, context 256, refit through 2026-09-21

Daily refit of the 256-context base (`ttm_c256_h96`, initialized from `ttm_c256_h96_ft_2026-09-20`)
on the newest Kaggle tournament days (2026-09-16 through 2026-09-21, recency-weighted).
Trained on Kaggle's dual NVIDIA Tesla T4 GPUs via Distributed Data Parallel (`torchrun --nproc_per_node=2`).

## Execution & Convergence
- **Training Duration:** 168.9 minutes (~2.81 hours)
- **Batch Size:** 64 per GPU (effective 128)
- **Learning Rate:** 4e-5 (scaled for 2 GPUs) with ReduceLROnPlateau (`factor=0.5, patience=2`)
- **Early Stopping:** Patience 5 epochs, best epoch 9 of 30 (stopped at epoch 14)
- **Epoch 0 baseline:** AUC 0.8490 / AP 0.3551 -> Best: AUC 0.8535 / AP 0.3626

## Evaluation Metrics (Held-out validation on 2026-09-21)
- **Windows Evaluated (N):** 895,104 (all horizons pooled)
- **AUC (Any Sell):** 0.8535
- **Average Precision (AP Any Sell):** 0.3626 (Baseline rate: 0.0657 -> **5.52x lift**)
- **MAE (when sold):** 4.81 (vs 5.15 all-zero baseline)

### Per-Product Detection AUC
- Product 0: 0.7839
- Product 1: 0.8895
- Product 2: 0.9114
- Product 3: 0.7803
- Product 4: 0.8931
- Product 5: 0.8996
- Product 6: 0.7813
- Product 7: 0.8275
- Product 8: 0.8090
