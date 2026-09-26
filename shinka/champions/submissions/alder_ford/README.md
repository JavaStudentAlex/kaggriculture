# Alder Ford

Submission bundle built 2026-09-26T14:39:15Z by `research/procedural_graph/make_graph_submission.py`.

- graph: `research/procedural_graph/evolution_results/ladder_2026-09-26/opening_champion_graph.json` (sha256 `2f4da012e222f1d5…`), runtime `hazel_merged_v1`
- predictor: `research/procedural_graph/hazel_runtime/checkpoint` (model sha256 `3c94bfc5d5c8df7c…`, numpy backend)
- no calibration
- archive: `../AlderFord.tar.gz` (4,583,541 bytes, sha256 `b514b6a573274141…`)
- validation: PASS; fidelity: PASS

ladder1 Island-Opening champion (arena bundle g_560d76b781114338, promoted in iteration 11): the tetsutani_demand engine (public notebook tetsutani/demand-preserving-turn-sale-timing, Apache-2.0) + oracle_guard on MILK/WOOL/STRAWBERRY (score 0.5, batch 4, keep 2, price ratio 0.75) + engine _CXD_BUDGET 1050; predictor: the committed reference ttm_c256_h96_ft_2026-09-13, no calibration (as in the arena). validate1, 260 games on arena validation seeds: 217-35-8 against the 12-agent ladder pool; +53 USD a game over the plain engine (92 better, 49 worse, p=0.0004); 14-0-8 head-to-head against tetsutani_demand.

See `MANIFEST.json`. Re-validate with `HOME=<empty> python -I validate.py <extracted dir>`. Procedure: `.agents/skills/kaggle-simulation-competitions/references/submission-promotion.md`.
