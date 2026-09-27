# Cedar Ridge

Submission bundle built 2026-09-27T21:28:29Z by `research/procedural_graph/make_graph_submission.py`.

- graph: `research/procedural_graph/evolution_results/ladder_2026-09-26/opening_mirror_counter_graph.json` (sha256 `48db96f87cbad4a9…`), runtime `hazel_merged_v1`
- predictor: `research/procedural_graph/hazel_runtime/checkpoint` (model sha256 `3c94bfc5d5c8df7c…`, numpy backend)
- no calibration
- archive: `../CedarRidge.tar.gz` (4,589,069 bytes, sha256 `aece1a2b8f9cd070…`)
- validation: PASS; fidelity: PASS
- mirror counter: PASS. The extracted archive, played through the Kaggle loader against the tetsutani_demand pool
  bundle on seed 1324216955 from both seats, won by $1,555 each time, exactly the arena's cached margin for bundle
  g_900a45ee4b3107a7; the same graph without the counter loses that game by $51 (`~/kagg-evo/mirror_fidelity.py`
  on cliproxyapi)

ladder1 Island-Opening champion after iteration 58 (arena bundle g_900a45ee4b3107a7) = Birch Hollow (submission 56603928) + engine _OR2_SLOT_MARGIN 12 -> 0 (promoted by the island mixing after iteration 36) + the rival_counter stage with the mirror counter (iteration 53: against rivals that still play exactly as we do at step 93, _EV_H/_DP_H/_MP_H 8 -> 12 and _ADV_LOOK 4). Gauntlet (463 pool games): 348 won; vs tetsutani_demand 122-6 (+704 USD a game), the counter changes no game against other opponents; vs tetsutani_demand_0927 1-23. Replay backtest on Birch Hollow's 63 recorded ladder games (rivals rated >= 1900 or unlisted): 41W-22L instead of 35W-28L, +297 USD a game (41 better, 10 worse, p=1.5e-5); 6 of the 28 losses won (all against mirrors), none of the 35 wins lost. Evidence: shinka/champions/evidence/birch_hollow_20260927/.

Kaggle submission 56619997, submitted 21:43 UTC 09-27 from this PC, COMPLETE by 21:50 (start rating 600).

See `MANIFEST.json`. Re-validate with `HOME=<empty> python -I validate.py <extracted dir>`. Procedure: `.agents/skills/kaggle-simulation-competitions/references/submission-promotion.md`.
