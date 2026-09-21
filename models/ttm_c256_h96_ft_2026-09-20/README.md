# ttm_c256_h96_ft_2026-09-20 — opponent sell model, context 256, refit through 2026-09-20

Daily refit of the 256-context base (`ttm_c256_h96`, initialized from `ttm_c256_h96_ft_2026-09-19`)
on the newest Kaggle tournament days (2026-09-15 through 2026-09-20, recency-weighted).
Trained on Kaggle's dual NVIDIA Tesla T4 GPUs via Distributed Data Parallel (`torchrun --nproc_per_node=2`).

## Execution & Convergence
- **Training Duration:** 151.8 minutes (~2.53 hours)
- **Batch Size:** 64 per GPU (effective 128)
- **Learning Rate:** 4e-5 (scaled for 2 GPUs) with ReduceLROnPlateau (`factor=0.5, patience=2`)
- **Early Stopping:** Patience 5 epochs, best epoch 9 of 30 (stopped at epoch 14)
- **Epoch 0 baseline:** AUC 0.8433 / AP 0.3451 -> Best: AUC 0.8495 / AP 0.3549

## Evaluation Metrics (Held-out validation on 2026-09-20)
- **Windows Evaluated (N):** 895,104 (all horizons pooled)
- **AUC (Any Sell):** 0.8495
- **Average Precision (AP Any Sell):** 0.3549 (Baseline rate: 0.0654 -> **5.43x lift**)
- **MAE (when sold):** 4.26 (vs 4.84 all-zero baseline)

### Per-Product Detection AUC
- Product 0: 0.7812
- Product 1: 0.8760
- Product 2: 0.9119
- Product 3: 0.7657
- Product 4: 0.8586
- Product 5: 0.8814
- Product 6: 0.7721
- Product 7: 0.8182
- Product 8: 0.8358
