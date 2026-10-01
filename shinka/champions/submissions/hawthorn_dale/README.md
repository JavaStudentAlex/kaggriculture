# Hawthorn Dale

Submission bundle built 2026-09-30T08:07:25Z by `research/procedural_graph/make_graph_submission.py`.

- graph: `/home/alex/kagg-evo/hawthorn_dale/hawthorn_dale_graph.json` (sha256 `3752d1341b52b328…`), runtime `hazel_merged_v1`
- predictor: `/home/alex/kagg-evo/kad-20260929/models/ttm_c256_h96_ft_2026-09-28` (model sha256 `878699cdbc19d799…`, numpy backend)
- calibration `/home/alex/kagg-evo/kad-20260929/calibration/ttm_c256_h96_ft_2026-09-28/own_games/calibration.json` (sha256 `6ac918acf0fd0a58…`)
- archive: `../HawthornDale.tar.gz` (43,518,656 bytes, sha256 `3f1f897f7fbc80a1…`)
- validation: PASS; fidelity: PASS

KAD run (~/kagg-evo/kad-20260929, from 20:59 UTC 09-29) best graph after iteration 7: the Island-KAD-Hands champion (iteration 2, gemini-3.8-flash; arena bundle g_a59fcff157ff0fa5) = the run's seed (Juniper Knoll's graph + predictor ttm_c256_h96_ft_2026-09-28 with its own-games calibration + KAD-HP-1 advice every 2nd turn, levers off) + _KC_HANDS on, _KC_JOBS [WATER], _KC_JOB_P 0.6, _KC_FROM_STEP 144; require_oracle and require_kad off in the package. Gauntlet vs the seed: 15 of 783 games changed, all better, +$2 a game, results +1/-0. Seed vs Juniper Knoll on 763 cached games: 693 identical, 22 better / 48 worse, -$3.0 a game, results 0/0.

See `MANIFEST.json`. Re-validate with `HOME=<empty> python -I validate.py <extracted dir>`. Procedure: `.agents/skills/kaggle-simulation-competitions/references/submission-promotion.md`.
