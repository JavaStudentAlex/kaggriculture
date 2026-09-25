# Rowan Glen

Submission bundle built 2026-09-25T22:32:45Z by `research/procedural_graph/make_graph_submission.py`.

- graph: `research/procedural_graph/evolution_results/feed_fix_2026-09-24/feed15_graph.json` (sha256 `92d48eed5fbee447…`), runtime `hazel_merged_v1`
- predictor: `models/ttm_c256_h96_ft_2026-09-24` (model sha256 `7dcb717531d04962…`, numpy backend)
- calibration `research/procedural_graph/calibration/ttm_c256_h96_ft_2026-09-24/calibration.json` (sha256 `5a3b75984fa8d194…`)
- backbone patch: `research/procedural_graph/arena/yarn_second_fix.py` (`hazel_runtime/mohui_v66/candidate_v66_meta_closed_loop.py` sha256 `185e7bfa8864e061…`)
- archive: `../RowanGlen.tar.gz` (4,397,542 bytes, sha256 `1df9cdb225d04d14…`)
- validation: PASS; fidelity: PASS

Linden Brook (56545702) + the yarn-second route fix (research/procedural_graph/arena/yarn_second_fix.py): feed15 graph, predictor ttm_c256_h96_ft_2026-09-24 and its ladder calibration, and the v66 backbone takes bakery_yarn at step 144 when the second town shop is a YARN_STORE after a shop other than YARN_STORE/PET_CAFE with no route active. Why: Linden Brook lost all 9 of its ladder games with a day-6 yarn store (shinka/champions/evidence/linden_brook_loss_audit_20260925). Arena vs Linden Brook on those 59 ladder seeds, both seats, each twice (run yarnfix, 236 games, Colab): 16-0 in the 8 seeds where the rule fires (+21,772 on average), identical play elsewhere.

See `MANIFEST.json`. Re-validate with `HOME=<empty> python -I validate.py <extracted dir>`. Procedure: `.agents/skills/kaggle-simulation-competitions/references/submission-promotion.md`.
