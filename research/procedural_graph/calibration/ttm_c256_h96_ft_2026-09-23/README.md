# Calibration of ttm_c256_h96_ft_2026-09-23 to the 09-13 predictor (2026-09-24)

The graph's oracle thresholds were tuned while it played with `ttm_c256_h96_ft_2026-09-13`
(Hazel Weir's predictor, model sha256 3c94bfc5…). These thresholds are the front-run score 0.30,
the batch steps 0.45 and 0.60, `score_24` 0.45, `units_24` 2.0, and the cadence bypass 0.25, 0.40
and 0.50. The 09-23 refit's raw scores run higher, so with the same thresholds it fires more
often. In a head-to-head test, feed15 with the uncalibrated 09-23 model lost to feed15 with the
09-13 model 24W-174L, by −$263 a game. `calibration.json` scales the 09-23 model's `score_4`,
`score_24` and `units_24` per product, so each threshold fires as often as it did with the 09-13
model. The oracle (`kagg_oracle.forecast()`) applies it when the file sits in the checkpoint
directory.

Data: 400 games of `kaggle/kaggriculture-episodes-2026-09-23`. These are all 60 games that the
09-23 refit held out (its `val_episodes.json`) plus 340 others, which it trained on. Both seats
were scored at every turn from 256 to 714, 367,200 origins per model. The run used one Colab T4
High-RAM VM on account 2 (`calib0923`, 2026-09-24), about 45 minutes, and the torch forecasts
matched the numpy port to 1.7e-6. The episode list and model hashes are in `jobs.json`.

| file | what |
|---|---|
| `calibration.json` | factors fitted on all 400 games (the one to use) |
| `calibration_heldout.json` | the same fit on the 60 held-out games only, as a check on in-sample bias |
| `fit_report.txt`, `fit_report_heldout.txt` | per product and threshold: firing rate, precision and AUC of both models |

Findings:
- **Ranking.** For `score_4`, the 09-23 model ranks better or the same for every product: AUC 0.806
  → 0.813 for wheat, 0.906 → 0.927 for tomato, 0.859 → 0.903 for melon, 0.874 → 0.892 for egg.
  Strawberry, milk and wool are flat. On `score_24` it is slightly worse for milk (0.709 → 0.690)
  and fertilizer (0.741 → 0.722).
- **Scale.** Most factors are between 0.74 and 0.97: the new model's scores are inflated. The
  exceptions above 1 are milk (1.08-1.11), carrot `score_24` (1.11) and fertilizer `units_24`
  (1.03). The strongest shrinks are tomato (0.82 / 0.85 / 0.74 for `score_4` / `score_24` /
  `units_24`) and melon (0.77 / 0.85 / 0.71); the policy never front-runs melon anyway.
- **At the old firing rates** the 09-23 model is as precise or more precise, for example tomato
  at 0.60: 0.82 vs 0.78.
- **In-sample bias is small.** The held-out-only factors match the all-400 ones to within 0.04.
  The exception is melon, whose base rate is only 4.5 %: 0.73 vs 0.77 on `score_4` and 0.58 vs
  0.71 on `units_24`.

Next: a rematch of feed15 with 09-23 + `calibration.json` against feed15 with 09-13, on the same
200 seeds as the uncalibrated test.
