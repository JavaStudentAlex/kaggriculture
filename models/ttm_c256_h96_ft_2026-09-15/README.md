# ttm_c256_h96_ft_2026-09-15 — opponent sell model, context 256, refit through 2026-09-15

Daily refit of the 256-context base (`ttm_c256_h96`, initialized from `ttm_c256_h96_ft_2026-09-13`)
on the newest Kaggle tournament days (2026-09-11 through 2026-09-15, 5 days recency-weighted).
Trained on Kaggle's dual NVIDIA Tesla T4 GPUs via Distributed Data Parallel (`torchrun --nproc_per_node=2`).

## Execution & Convergence
- **Training Duration:** 136.0 minutes (~2.26 hours)
- **Batch Size:** 64 per GPU (effective 128)
- **Learning Rate:** 4e-5 (scaled for 2 GPUs) with ReduceLROnPlateau (`factor=0.5, patience=2`)
- **Early Stopping:** Patience 5 epochs

## Evaluation Metrics (Held-out validation on 2026-09-15)
- **Windows Evaluated (N):** 923,520 (all horizons pooled)
- **AUC (Any Sell):** 0.8490
- **Average Precision (AP Any Sell):** 0.3455 (Baseline rate: 0.0622 -> **5.55x lift**)
- **MAE (when sold):** 4.32 (vs 4.89 all-zero baseline)

### Per-Product Detection AUC
- Product 0: 0.7747
- Product 1: 0.8965
- Product 2: 0.8965
- Product 3: 0.7701
- Product 4: 0.7876
- Product 5: 0.8621
- Product 6: 0.7868
- Product 7: 0.8253
- Product 8: 0.8237
