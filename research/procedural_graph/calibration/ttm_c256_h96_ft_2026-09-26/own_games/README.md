# Calibration of ttm_c256_h96_ft_2026-09-26 for our agent (own games)

Made by `arena/calibrate.py` on 2026-09-29: the committed game set (`calibration/game_set`, 600 games), reference `/home/alex/PycharmProjects/kaggriculture/research/procedural_graph/hazel_runtime/checkpoint, model sha256 3c94bfc5d5c8…` (the predictor the thresholds were tuned with), every turn, one Colab T4. `calibration.json` fits all games; `_pool` only feed15 vs Mohui and the champions, `_mirror` only feed15 vs feed15. Reports: `fit_report*.txt`.

| product | base rate | score_4 AUC reference | score_4 AUC refit | factors score_4 / score_24 / units_24 |
|---|---|---|---|---|
| WHEAT | 27.9 % | 0.682 | 0.625 | 0.763 / 0.723 / 1.033 |
| CARROT | 3.7 % | 0.939 | 0.962 | 1.384 / 1.032 / 0.731 |
| TOMATO | 0.0 % | nan | nan | 1.000 / 1.000 / 1.000 |
| STRAWBERRY | 21.7 % | 0.805 | 0.760 | 1.011 / 1.067 / 1.210 |
| MELON | 0.9 % | 0.960 | 0.906 | 1.604 / 2.000 / 0.930 |
| EGG | 0.0 % | nan | nan | 1.000 / 1.000 / 1.000 |
| MILK | 28.4 % | 0.633 | 0.642 | 1.136 / 1.137 / 1.105 |
| WOOL | 15.5 % | 0.693 | 0.681 | 0.861 / 0.956 / 0.864 |
| FERTILIZER | 28.8 % | 0.786 | 0.754 | 1.244 / 1.448 / 1.148 |

AUC is unchanged by any calibration: where the refit ranks worse than the reference, the calibrated agent still predicts our opponents worse. Check with the rematch (AGENTS.md section 13) before the refit plays.
