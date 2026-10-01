# Aspen Vale

Submission bundle built 2026-09-28T08:56:37Z by `research/procedural_graph/make_graph_submission.py`.

- graph: `research/procedural_graph/evolution_results/ladder_2026-09-26/next_counter_emulator_graph.json` (sha256 `97da7708b8a4a2b0…`), runtime `hazel_merged_v1`
- predictor: `research/procedural_graph/hazel_runtime/checkpoint` (model sha256 `3c94bfc5d5c8df7c…`, numpy backend)
- no calibration
- archive: `../AspenVale.tar.gz` (5,255,247 bytes, sha256 `139d095caccd5af0…`)
- validation: PASS; fidelity: PASS

ladder1 Island-Next-Counter champion after iteration 68 (arena bundle g_16bb4af65a704f00) = the public engine's 09-27 version (tetsutani_demand_0927) + engine _SR_MARGIN 12 + the rival_emulator stage (both tetsutani versions, _EM_LOCK 12, _EM_RACE on); no predictor. Candidate for Cedar Ridge's losses (shinka/champions/evidence/cedar_ridge_20260928).

See `MANIFEST.json`. Re-validate with `HOME=<empty> python -I validate.py <extracted dir>`. Procedure: `.agents/skills/kaggle-simulation-competitions/references/submission-promotion.md`.
