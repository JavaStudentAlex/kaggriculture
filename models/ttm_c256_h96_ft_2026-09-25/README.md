# ttm_c256_h96_ft_2026-09-25 — opponent sell model, context 256, refit through 2026-09-25

Daily refit of the 256-context base (`ttm_c256_h96`, initialized from `ttm_c256_h96_ft_2026-09-24`)
on the newest Kaggle tournament days (2026-09-21 through 2026-09-25, recency-weighted).
Trained on Kaggle's dual NVIDIA Tesla T4 GPUs via Distributed Data Parallel (`torchrun --nproc_per_node=2`).

## Execution & Convergence
- **Training Duration:** 68.2 minutes (~1.14 hours)
- **Batch Size:** 64 per GPU (effective 128)
- **Learning Rate:** 4e-5 (scaled for 2 GPUs) with ReduceLROnPlateau (`factor=0.5, patience=2`)
- **Early Stopping:** Patience 5 epochs, best epoch 2 of 30 (stopped at epoch 7)
- **Epoch 0 baseline:** AUC 0.8519 / AP 0.4082 -> Best: AUC 0.8537 / AP 0.4139

## Evaluation Metrics (Held-out validation on 2026-09-25)
- **Windows Evaluated (N):** 809,856 (all horizons pooled)
- **AUC (Any Sell):** 0.8537
- **Average Precision (AP Any Sell):** 0.4139 (Baseline rate: 0.0855 -> **4.84x lift**)
- **MAE (when sold):** 3.98 (vs 4.65 all-zero baseline)

### Per-Product Detection AUC
- Product 0: 0.7966
- Product 1: 0.8807
- Product 2: 0.9032
- Product 3: 0.7715
- Product 4: 0.8522
- Product 5: 0.8619
- Product 6: 0.7776
- Product 7: 0.8318
- Product 8: 0.8672
