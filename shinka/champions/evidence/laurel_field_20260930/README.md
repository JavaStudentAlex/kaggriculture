# Laurel Field and the 2026-09-30 screens

**Laurel Field** (Kaggle 56704713, submitted 10:25 UTC 2026-09-30, COMPLETE 10:32, started at 600) is Hawthorn Dale
(56702363: Juniper Knoll's graph + the 09-28 predictor refit with its own-games calibration + KAD-HP-1 advice every
2nd turn, hands take KAD's WATER jobs from turn 144 at p >= 0.6) with one engine constant changed:
`_R51_INPUT_CROPS` WHEAT (2, 4, 6) -> (2, 4, 8). Package: `../../submissions/LaurelField.tar.gz` (sha256 `96f70008…`),
staging dir `../../submissions/laurel_field/`; built on cliproxyapi by `~/kagg-evo/laurel_field/build_laurel.sh` from the
KAD run's code copy (validation PASS on 8 games, fidelity seed 101 identical).

## Why that constant

The KAD run (`~/kagg-evo/kad-20260929`, iteration 8, Island-Herd) promoted it: 768 of 783 gauntlet games changed,
+$92 a game, p=3.6e-10, results +45/-28. The engine's input planner (`_r68_joint_plans` in tetsutani_demand_0927) buys
fertilizer for wheat and carrot tiles when value >= 1.5 x cost + 50 and keeps $3,000 in cash; the game caps wheat at
6 units, so a cap of 8 only raises the planner's value of extra watering and fertilizer, and more plans pass its test.

## Checks (all on Colab, run from cliproxyapi)

| check | games | result |
|---|---|---|
| fast screen, fresh seeds (salt 20260930), guard and KAD off: without the change vs with | 200 paired | 78 better / 122 worse, -$75 a game, p=0.0023 |
| full graph, arena validation seeds x 12 pool opponents: Laurel vs Hawthorn | 240 paired | 146 better / 94 worse, +$79 a game, p=0.00095; wins 216 vs 217 |
| backtest against the rivals' recorded moves of Juniper Knoll's 73 newest ladder games (rivals >= 1,900 or unlisted) | 73 | Hawthorn 34W-39L, Laurel 35W-38L (6 recorded losses won, 6 recorded wins lost), +$56 a game (44 better / 26 worse) |

## What else was tried (fast graphs, 200 paired fresh games each, against the wheat-8 base)

| change | better / worse | mean |
|---|---|---|
| wheat cap 10 | 0 / 0 (identical) | $0 |
| carrot cap 4 -> 6 | 34 / 152 | -$108 |
| `_R51_INPUT_MAX_WORKERS` 3 | 0 / 0 | $0 |
| `V9_FERT_FIRST_DAY` 14 -> 10 / 12 | 102 / 98, 28 / 28 | +$15, -$20 |
| `V9_FERT_AGES` (1,) -> (1, 2) | 78 / 116 | -$2 |
| `_CA_MARGIN` -22 -> -30 | 66 / 82 | +$6 |

Cash reserves: engine copy `tetsutani_demand_0927r` (`~/kagg-evo/lit0930/repo`, made by `make_engine_r.py` here; seven literals as `_XR_*` constants, identical play at the defaults: 200 + 73 games the same).
Tomato-plot money 12,000 -> 9,000, crop-worker reserve 3,000 -> 1,500, sheep reserves 3,000/1,000 -> 1,500/500 and the
day-11 check 10,000 -> 7,000 changed none of 200 games (never binding). The input planner's reserve 3,000 -> 1,500
with ROI 1.5 -> 1.25: 114 better / 80 worse but -$39 a game; in the ladder backtest 34W-39L against 33W-40L (+$7).
Not used.

## Where the gap is

Juniper Knoll's 106 newest games: 60W-46L; against rivals rated 1,900-2,000 14-13, against 2,000 and up 1-14; 25 of
the 46 losses by less than $500. In its four biggest recent losses (-$12.6k to -$17k, rivals 2,013-2,232) the rival
has on day 10 a third quadrant already planted with 18-28 wheat and 3-9 geese; we have two quadrants, 4-5 wheat and
0-2 geese, and a little more cash. Our route tapes buy the third quadrant on day 11 in every game, and no engine
constant moves it; the land-plot stage that bought it earlier lost (-$948 to -$2,515 per changed game) because it
works against the tape's own plan. That, not a constant, is what separates us from the 2,100+ band.

Stock left at the end is not a leak: the step-718 action drops the workers' bags into the shed and sells them.
