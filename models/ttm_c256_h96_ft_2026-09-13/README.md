# ttm_c256_h96_ft_2026-09-13 — opponent sell model, context 256, refit through 2026-09-13

The daily refit of the 256-context base (`ttm_c256_h96`, held-out AUC 0.8668 on 44 days
07-30..09-11; in git history, commit c5b8684) on the newest Kaggle day plus the four
before it, by `research/opponent_model/ops/finetune.sh` with its defaults:
`DAY=2026-09-13 BASE=models/ttm_c256_h96` → train days 09-09, 09-10, 09-11, 09-12,
**09-13**, recency-weighted (`STRIDE=4 DECAY=2`: the newest day is 49 % of the 222,570
training windows — 108,376 / 60,260 / 30,222 / 15,768 / 7,944 from newest to oldest),
3×A100, batch 64/GPU, lr 6e-5 with reduce-on-plateau (×0.5, patience 2; halved twice),
early-stop patience 5, ≤ 30 epochs. The base's `scaler.npz` is reused unchanged (never
refit); `labels.json`: next_action. Ran 2026-09-14 06:28–07:04 UTC (36 min) and
**stopped early at epoch 21: best = epoch 16**, five flat epochs after it.

Validation = 10 % of 09-13's episodes (`val_episodes.json`, 65 episodes / 130 series,
never trained on). Epoch 0 is the un-tuned base on the same windows.

| eval (65 held-out 09-13 episodes) | AUC | AP |
|---|---|---|
| base `ttm_c256_h96`, trainer val, all 96 steps pooled (epoch 0) | 0.859 | 0.357 |
| **this checkpoint**, trainer val, all 96 steps pooled (`scores.json`) | **0.878** | **0.381** |
| canonical (`eval.json`: 46,800 windows, stride 1, hour-balanced), pooled | 0.878 | 0.381 (lift 7.5×) |
| canonical day 1 / 2 / 3 / 4 | 0.891 / 0.886 / 0.876 / 0.859 | 0.407 / 0.397 / 0.376 / 0.347 |
| canonical t+1 | 0.885 | 0.425 (lift 9.2×; clock baseline 0.141) |
| canonical t+1 per product (AUC / AP) | WHEAT 0.83 / 0.37, CARROT 0.87 / 0.34, TOMATO 0.94 / 0.50, STRAWBERRY 0.87 / 0.44, MELON 0.87 / 0.36, EGG 0.91 / 0.56, MILK 0.83 / 0.39, WOOL 0.86 / 0.32, FERTILIZER 0.84 / 0.50 | |

Held-out curve: 0.859 (0) → 0.872 (1) → 0.875 (3) → 0.877 (7) → 0.878 (16) → flat.
No seat leak (clean AP = un-cleaned AP 0.425 at t+1). Top-decile volume recall 0.729.
The gain over the base (+0.019 AUC / +0.024 AP on the newest day) is the ladder drift
the daily refits exist for; the numbers are on one day's 65 games, so per-product APs
rest on a few hundred positives.

**Magnitudes are for ranking only** (as for the base): when a sell happens the predicted
units are shrunk ~7× (6.1 actual vs 0.86 predicted, pooled); day-1..4 volume totals are
under-predicted 2.5–3×. Use the scores, not the units.

Load with `TinyTimeMixerForPrediction.from_pretrained("models/ttm_c256_h96_ft_2026-09-13")`
plus `scaler.npz` (input mean/std, `(x - mean) / std`, targets as `log1p` channels
141–149; `past_values` of shape `(B, 256, 150)`). `labels.json` tells the oracle which
accounting to rebuild (next_action). The next daily refit chains from this directory:
`DAY=<next day> BASE=models/ttm_c256_h96_ft_2026-09-13 bash research/opponent_model/ops/finetune.sh`.
