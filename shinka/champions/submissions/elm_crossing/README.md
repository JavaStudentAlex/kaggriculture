# Elm Crossing

Submission bundle built 2026-09-30T14:02:19Z by `research/procedural_graph/make_graph_submission.py`.

- graph: `/home/alex/kagg-evo/candidate_b/graph.json` (sha256 `f326dabbe19f5174…`), runtime `hazel_merged_v1`
- predictor: `/home/alex/kagg-evo/kad-20260929/models/ttm_c256_h96_ft_2026-09-28` (model sha256 `878699cdbc19d799…`, numpy backend)
- calibration `/home/alex/kagg-evo/kad-20260929/calibration/ttm_c256_h96_ft_2026-09-28/own_games/calibration.json` (sha256 `6ac918acf0fd0a58…`)
- archive: `../ElmCrossing.tar.gz` (44,831,364 bytes, sha256 `04f12e444c417dd4…`)
- validation: PASS; fidelity: PASS

Laurel Field (56704713) + rival emulator: _EM_ENGINES adds haodou_ledger_0928 and leo_pi (each played one of today's lost or tied ladder games move for move) and _EM_REPAIR 300 (new: a hypothesis whose farm actions were right but whose market list was not is kept with the market list that reproduces the observation; on today's 34 lost games tracking rose from 9,218 to 12,073 predicted steps, 98% exact). Built from ~/kagg-evo/lit0930/repo (patched rival_emulator.py and graph_runtime.py).

See `MANIFEST.json`. Re-validate with `HOME=<empty> python -I validate.py <extracted dir>`. Procedure: `.agents/skills/kaggle-simulation-competitions/references/submission-promotion.md`.
