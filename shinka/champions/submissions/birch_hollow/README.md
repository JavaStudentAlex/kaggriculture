# Birch Hollow

Submission bundle built 2026-09-26T23:26:38Z by `research/procedural_graph/make_graph_submission.py`.

- graph: `research/procedural_graph/evolution_results/ladder_2026-09-26/alder_ford_look4_graph.json` (sha256 `865f88a1e00493fc…`), runtime `hazel_merged_v1`
- predictor: `research/procedural_graph/hazel_runtime/checkpoint` (model sha256 `3c94bfc5d5c8df7c…`, numpy backend)
- no calibration
- archive: `../BirchHollow.tar.gz` (4,583,560 bytes, sha256 `3582a0595d8d3674…`)
- validation: PASS; fidelity: PASS

Alder Ford (submission 56582140, ladder1 Island-Opening champion) with one change: engine _ADV_LOOK 3 -> 4 (the ready-stock layer sells milk/wool/eggs/strawberries/melons/carrots/tomatoes up to 4 turns before the route tape instead of 3, i.e. one turn ahead of rivals running the public engine). Arena bundle g_303b063b37c58a0c. Replay backtest on Alder Ford's 72 recorded ladder games (rivals rated >= 1900): 44W-27L-1T instead of 36W-28L-8T, +60 USD a game (49 better, 15 worse, p=2e-5), all 36 wins kept. Fresh arena games (validate1, 300): +32 USD a game (158 better, 111 worse, p=0.005); vs tetsutani_demand 12-0-8 -> 19-1-0. Evidence: shinka/champions/evidence/alder_ford_20260927/.

See `MANIFEST.json`. Re-validate with `HOME=<empty> python -I validate.py <extracted dir>`. Procedure: `.agents/skills/kaggle-simulation-competitions/references/submission-promotion.md`.
