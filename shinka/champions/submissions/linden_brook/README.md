# Linden Brook

Submission bundle built 2026-09-25T07:57:49Z by `research/procedural_graph/make_graph_submission.py`.

- graph: `research/procedural_graph/evolution_results/feed_fix_2026-09-24/feed15_graph.json` (sha256 `92d48eed5fbee447…`), runtime `hazel_merged_v1`
- predictor: `models/ttm_c256_h96_ft_2026-09-24` (model sha256 `7dcb717531d04962…`, numpy backend)
- calibration `research/procedural_graph/calibration/ttm_c256_h96_ft_2026-09-24/calibration.json` (sha256 `5a3b75984fa8d194…`)
- archive: `../LindenBrook.tar.gz` (4,397,009 bytes, sha256 `5045e622afb84fec…`)
- validation: PASS; fidelity: PASS

feed15 graph (evolution_results/feed_fix_2026-09-24/feed15_graph.json: 15/10 opening, _FEED_RESERVE_LOOKAHEAD, front-run batch 4 / price ratio 0.7) with the predictor ttm_c256_h96_ft_2026-09-24 and its ladder calibration (calibration/ttm_c256_h96_ft_2026-09-24/calibration.json, fitted on 378 top-player games of 2026-09-24 to the thresholds tuned with ttm_c256_h96_ft_2026-09-13). Arena: feed15 with 09-13 went 122-77 vs plain Mohui and 40-0 vs each champion; the new predictor is not arena-validated (it ranks our Mohui-family opponents worse, top players better); submitted to test it on the ladder.

See `MANIFEST.json`. Re-validate with `HOME=<empty> python -I validate.py <extracted dir>`. Procedure: `.agents/skills/kaggle-simulation-competitions/references/submission-promotion.md`.
