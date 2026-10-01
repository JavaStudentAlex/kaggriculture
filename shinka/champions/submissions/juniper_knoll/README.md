# Juniper Knoll

Submission bundle built 2026-09-29T14:19:33Z by `research/procedural_graph/make_graph_submission.py`.

- graph: `/home/alex/kagg-evo/juniper_knoll/juniper_knoll_graph.json` (sha256 `5273b061cb68b729…`), runtime `hazel_merged_v1`
- predictor: `research/procedural_graph/hazel_runtime/checkpoint` (model sha256 `3c94bfc5d5c8df7c…`, numpy backend)
- no calibration
- archive: `../JuniperKnoll.tar.gz` (5,255,364 bytes, sha256 `4cc19f98da22390b…`)
- validation: PASS; fidelity: PASS

Land run seed = predictor-required run seed (arena bundles g_2e6e1664c98cc2a9, land runtime g_d46eaa6322231c85): ladder1 Island-Next-Counter champion after iteration 92 (g_99508e7525447d95 = the public engine's 09-27 version + _SR_MARGIN 12 + the rival emulator + _S809_LOOK 4 + _CA_MARGIN -22) + _SR_MARGIN 14 + the oracle guard for every rival family (MILK/WOOL/STRAWBERRY, score 0.5, strong 0.6, batch 4, keep 2, price ratio 0.75, steps 256-696); predictor ttm_c256_h96_ft_2026-09-13 on numpy, no calibration; require_oracle off in the package. Against Aspen Vale on 727 cached development games: +$54 a game, 400 better / 191 worse, results +18/-5, wins 581 -> 595 (research/procedural_graph/evolution_results/land_2026-09-29/README.md).

See `MANIFEST.json`. Re-validate with `HOME=<empty> python -I validate.py <extracted dir>`. Procedure: `.agents/skills/kaggle-simulation-competitions/references/submission-promotion.md`.
