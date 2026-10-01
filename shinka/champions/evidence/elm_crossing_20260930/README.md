# Elm Crossing: the rival emulator keeps up with modified copies (2026-09-30)

**Elm Crossing** (Kaggle 56712000, submitted 15:41 UTC 2026-09-30) is Laurel Field (56704713, `../laurel_field_20260930/`)
with two changes to the `rival_emulator` stage:
- `_EM_ENGINES` also runs `haodou_ledger_0928` and `leo_pi` (the ladder-pool bundles of the same names) besides
  `tetsutani_demand` and `tetsutani_demand_0927`;
- `_EM_REPAIR` 300, a new parameter (default 0 = the old behaviour): see below.

Package: `../../submissions/ElmCrossing.tar.gz` (sha256 `04f12e44…`), staging dir `../../submissions/elm_crossing/`.
Built on cliproxyapi from the copy `~/kagg-evo/lit0930/repo` (the KAD run's code + the two engines under
`hazel_runtime/engines/` + the patched `rival_emulator.py` and `graph_runtime.py`) by `~/kagg-evo/candidate_b/build_elm.sh`;
validation PASS (8 games; step 0 takes 12-14 s against ~9.4 s for Laurel Field, because four engines load; mean
145-155 ms a step), fidelity seed 101 identical, and `stage_fidelity_any.py` reproduced 8 arena games where the
emulator acts (Ledger 09-28, leo_pi, 2965 09-26, V57) to the dollar.

## What the ladder showed

On 09-30 Hawthorn Dale and Laurel Field lost most games against rivals rated 1,700-1,900 by less than $600. The rivals'
recorded moves of 34 of those lost or tied games were matched against every public bundle (`match_ladder_games.py`;
work dir `~/kagg-evo/today0930` on cliproxyapi):
- exact (all 719 moves): tetsutani_demand_0927 (zby_4240, -$304), haodou_ledger_0928 (koala_bear, -$188), leo_pi
  (Rifqi Haikal, tie). The emulator ran only the first of these three engines.
- 400-577 identical moves: seven games, all the 0927 family.
- 48-264 identical moves, most at exactly 150: the largest group. At the first difference their **farm actions are
  identical** to the engine's; only the market list differs: SELL WOOL 6 instead of 4 at step 150, one extra
  BUY_PRODUCT FERTILIZER, no SELL WHEAT 1 at step 80, a HIRE after a wheat purchase instead of before it.
- 7 games: other openings (private agents); the four biggest losses.

The emulator dropped such a rival at its first difference, and with it the race (our market slots arranged against the
rival's known orders), which is what turned the 0927 mirror games from 8-2-14 into 23-1-0 on 09-28.

## The repair

In `RivalEmulator._verify`, when **no** hypothesis reproduces the observed step as predicted, each failed hypothesis is
tried with up to `REPAIR_SIMULATIONS` (12) other market lists, with its farm actions kept: each product's order
changed by the units the observed market inventory differs from the failed simulation (an order added or dropped when
needed), one HIRE more or less when the rival's hands differ, then single edits of the predicted list (an order moved
to another slot, an order dropped). A list that reproduces the observation (the environment's own interpreter) keeps the
hypothesis alive with the rebuilt private state; at most `_EM_REPAIR` repairs a game. A repaired hypothesis gives no
engine identity (`identity()`), so per-engine counters never apply to a modified copy. Hypotheses that still match
exactly are never repaired alongside, because the emulator predicts only when all living hypotheses agree.

`em_harness.py` (next to this file) drives the emulator over our own recorded observations: on the 34 lost games the
steps with a prediction went from 9,218 to 12,073, 98% of them exactly the rival's recorded action; for example the
-$279 game from step 48 to 529, the -$73, -$79 and -$232 games to the end.

## Results

| check | games | Laurel Field (or its fast graph) | Elm Crossing (or its fast graph) |
|---|---|---|---|
| backtest, fast graphs, 179 replayed ladder rivals (today's 106 + Juniper Knoll's 73) | 179 | 101W-78L | 108W-71L; 62 better / 5 worse, +$57 a game, p=1.4e-13 |
| the same, Juniper Knoll's 73 only (not used to build the repair) | 73 | 33W-40L | 37W-36L; 27 better / 2 worse, +$65 |
| backtest, full graphs, Juniper Knoll's 73 | 73 | 35W-38L | 38W-35L; 27 better / 2 worse, +$66, p=1.6e-6 |
| arena, full graphs, 20 validation seeds x 12 pool opponents | 240 | 216W-24L | 222W-18L; 69 better / 52 worse, +$37 (Ledger 09-28 13-7 -> 17-3, leo_pi 17-3 -> 19-1) |
| four engines without the repair, fast backtest | 179 | 101W-78L | 102W-77L (+$4) |

Against engines of other families (V55, V57, Shepherd, Forecast) the repair changed many games by small amounts, more
of them for the worse (mean still +$18-21): a hypothesis that matched only the first steps can be repaired. The next
candidate repairs only hypotheses that had locked (`_EM_LOCK` steps as predicted), copy `~/kagg-evo/lit0930c`.

## Candidate C (repair only after lock): not submitted

Same graph on the `lit0930c` runtime (`rival_emulator.py` sha `92f49b17`: the repair also needs `h.matched >= self.lock`).
Fast graphs, against the wheat-8 base:

| check | Elm Crossing's repair | C |
|---|---|---|
| backtest, 179 replayed ladder rivals | 108W-71L; 62 better / 5 worse, +$57 | 108W-71L; 59 better / 5 worse, +$54 |
| 200 fresh games (salt 20260930, 10 opponents x 10 seeds x 2 seats; base 175W-25L) | 179W-21L; 60 better / 38 worse, +$42, p=0.033 | 179W-21L; 30 better / 30 worse, +$20 |

The guard removed the repair's gains against the 2965 family (09-26 +$113 with 14 better / 0 worse, 2965 +$51 10/0,
V57 +$55): those rivals leave the public engine's moves before 12 steps, and the repair then kept a hypothesis that
predicted them. Shepherd (2 better / 18 worse, -$10) and Forecast (4/12, -$4) came out the same in both versions, so
the lock guard does not explain them, and no result changed against either (18-2 and 20-0 in all three graphs).
Package `MapleBend.tar.gz` (sha256 `223c8a79…`) was built in `~/kagg-evo/lit0930c` and not submitted.
