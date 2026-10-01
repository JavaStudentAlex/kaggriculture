# Poplar Reach

Submission bundle built 2026-09-30T16:42:27Z by `research/procedural_graph/make_graph_submission.py`.

- graph: `/home/alex/kagg-evo/candidate_b/graph.json` (sha256 `a29a750dfc29a9db…`), runtime `hazel_merged_v1`
- predictor: `/home/alex/kagg-evo/kad-20260929/models/ttm_c256_h96_ft_2026-09-28` (model sha256 `878699cdbc19d799…`, numpy backend)
- calibration `/home/alex/kagg-evo/kad-20260929/calibration/ttm_c256_h96_ft_2026-09-28/own_games/calibration.json` (sha256 `6ac918acf0fd0a58…`)
- archive: `../PoplarReach.tar.gz` (44,832,484 bytes, sha256 `1eedf71d29f0f8e0…`)
- validation: PASS; fidelity: PASS

Elm Crossing (56712000) with a wider rival-emulator repair: after the 12 original market lists, up to 120 more single edits of the predicted list (drops, blanks, moves, quantity changes, swaps) per repair, capped at 160 a step and 5000 a game, off below 25 s of overage bank (runtime copy lit0930d, rival_emulator sha 30fc9a72); graph identical to Elm Crossing

See `MANIFEST.json`. Re-validate with `HOME=<empty> python -I validate.py <extracted dir>`. Procedure: `.agents/skills/kaggle-simulation-competitions/references/submission-promotion.md`.
