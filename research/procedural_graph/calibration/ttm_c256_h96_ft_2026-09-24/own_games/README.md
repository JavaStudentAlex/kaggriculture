# Calibration of ttm_c256_h96_ft_2026-09-24 for our agent (own games)

Made by `arena/calibrate.py` on 2026-09-25: the committed game set (`calibration/game_set`, 600 games), reference `hazel_runtime/checkpoint, model sha256 3c94bfc5d5c8…` (the predictor the thresholds were tuned with), every turn, one Colab T4. `calibration.json` fits all games; `_pool` only feed15 vs Mohui and the champions, `_mirror` only feed15 vs feed15. Reports: `fit_report*.txt`.

| product | base rate | score_4 AUC reference | score_4 AUC refit | factors score_4 / score_24 / units_24 |
|---|---|---|---|---|
| WHEAT | 27.9 % | 0.682 | 0.636 | 0.810 / 0.658 / 1.487 |
| CARROT | 3.7 % | 0.939 | 0.960 | 1.802 / 1.124 / 0.811 |
| TOMATO | 0.0 % | nan | nan | 1.000 / 1.000 / 1.000 |
| STRAWBERRY | 21.7 % | 0.805 | 0.777 | 0.994 / 1.013 / 1.236 |
| MELON | 0.9 % | 0.960 | 0.962 | 1.483 / 1.990 / 0.976 |
| EGG | 0.0 % | nan | nan | 1.000 / 1.000 / 1.000 |
| MILK | 28.4 % | 0.633 | 0.644 | 1.123 / 1.101 / 1.105 |
| WOOL | 15.5 % | 0.693 | 0.692 | 0.878 / 0.946 / 0.847 |
| FERTILIZER | 28.8 % | 0.786 | 0.766 | 1.192 / 1.295 / 1.145 |

AUC is unchanged by any calibration: where the refit ranks worse than the reference, the calibrated agent still predicts our opponents worse. Check with the rematch (AGENTS.md section 13) before the refit plays.
