# Calibration of ttm_c256_h96_ft_2026-09-24 for the ladder (2026-09-25)

`calibration.json` is fitted on top players' games: 378 of 400 games from
`kaggle/kaggriculture-episodes-2026-09-24`. The 400 were all 59 games that the 09-24 refit held
out plus a seeded sample of the rest. The per-turn files of the last 22 games were lost when
the runner's final pack command on the VM hung. Both models were scored on every turn from 256
to 714, 347,004 origins, on one Colab T4 VM (`caliblad0924`). The reference is
`ttm_c256_h96_ft_2026-09-13` (`hazel_runtime/checkpoint`), the predictor the agent's thresholds
were tuned with. The feed15 submission for the ladder uses this file. `own_games/` holds the
calibration on our arena games, for arena play.

| product | base rate | score_4 AUC 09-13 → 09-24 | score_4 factor: all (held-out only) | own games | 09-23 ladder |
|---|---|---|---|---|---|
| WHEAT | 36.3 % | 0.794 → 0.801 | 0.957 (0.961) | 0.810 | 0.948 |
| CARROT | 14.5 % | 0.895 → 0.908 | 0.907 (0.905) | 1.802 | 0.942 |
| TOMATO | 12.3 % | 0.899 → 0.919 | 0.809 (0.803) | 1.000 | 0.816 |
| STRAWBERRY | 38.8 % | 0.803 → 0.806 | 0.863 (0.865) | 0.994 | 0.890 |
| MELON | 4.2 % | 0.856 → 0.900 | 0.756 (0.776) | 1.483 | 0.775 |
| EGG | 21.0 % | 0.863 → 0.884 | 0.911 (0.922) | 1.000 | 0.911 |
| MILK | 36.3 % | 0.742 → 0.746 | 1.109 (1.115) | 1.123 | 1.077 |
| WOOL | 30.5 % | 0.841 → 0.841 | 0.970 (0.974) | 0.878 | 0.963 |
| FERTILIZER | 24.6 % | 0.861 → 0.872 | 0.884 (0.895) | 1.192 | 0.886 |

On top players' games 09-24 ranks every product better than 09-13, or equally (wool). On our
Mohui-family arena games it ranks wheat, strawberry and fertilizer worse (`own_games/README.md`).
The factors for the ladder and for our games differ a lot, so each is only valid for its own
kind of opponent. The held-out-only fit agrees with the all-games fit to within about 0.02
(melon 0.76 vs 0.78). Reports: `fit_report*.txt`; payload and model hashes: `jobs.json`.
