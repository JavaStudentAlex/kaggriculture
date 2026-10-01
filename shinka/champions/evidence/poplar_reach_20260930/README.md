# Poplar Reach: a wider search when the rival emulator repairs (2026-09-30)

**Poplar Reach** (Kaggle 56715579, submitted 18:15 UTC 2026-09-30) is Elm Crossing (56712000,
`../elm_crossing_20260930/`) with one file changed: `hazel_runtime/rival_emulator.py` (sha256 `30fc9a72…`, copy
`rival_emulator_d.py` here). The graph is byte-identical to Elm Crossing's apart from its name. Package
`../../submissions/PoplarReach.tar.gz` (sha256 `1eedf71d…`), staging dir `../../submissions/poplar_reach/`; built on
cliproxyapi from the copy `~/kagg-evo/lit0930d/repo` by `~/kagg-evo/candidate_b/build_d.sh`. Validation PASS (8 games,
step 0 12-14 s as in Elm Crossing, mean ~150 ms a step), fidelity seed 101 identical; the package differs from Elm
Crossing's only in that file, the name and the provenance hash.

## Why

Elm Crossing's repair keeps a hypothesis alive when the rival left the engine only in its market orders, but it tried
12 market lists. `em_diag.py` found where the emulator still lost the rivals of the 34 games lost on 09-30 morning:
21 of 29 times on the market list, and `em_debug.py` showed that in every case checked **the rival's true market list
with the engine's farm actions reproduced the observation from our rebuilt state**, so the state was right and the
search too narrow. The differences: an order moved one to five slots (the buy before the hires, two sells swapped),
a quantity (sell 6 wool, not 3; buy 2 fertilizer, not 4, where the inventory rule changed the sell of the same
product instead), an order dropped or kept at 0.

## The change

After the 12 original lists, up to 120 more single edits of the predicted list per repair (`WIDE_SIMULATIONS`), most
common first: the inventory adjustment applied to the buy order, an order dropped / blanked / set to 0, moved one or two
slots, a trade's quantity doubled, +-1, +-2, halved or set to the rival's whole stock, moved further, two swapped. A list
that repaired one hypothesis is tried first on the others. Caps: 160 wide simulations a step, 5,000 a game, none below
25 s of overage bank. One simulation takes ~1 ms on cliproxyapi.

## Checks (fast graphs: predictor guard and KAD off; all games on Colab from cliproxyapi)

| check | Elm Crossing's repair | Poplar Reach's |
|---|---|---|
| emulator harness, 34 lost games of 09-30: steps with a prediction (exact) | 12,073 (11,858) | 14,102 (13,798); 13 games longer, none shorter |
| emulator time a step over those games: p99 / p99.9 / max | 34 / 184 / 581 ms | 42 / 210 / 582 ms |
| backtest, 179 replayed ladder rivals | 108W-71L | **112W-67L**; 25 better / 4 worse, +$29, p=0.0001 |
| **the 311 lost or tied games of our last five submissions** (Aspen Vale 180, Juniper Knoll 77, Hawthorn Dale 31, Laurel Field 23), rivals replayed | 85 won (Laurel Field's graph: 70) | **92 won**; 62 better / 4 worse, +$34; 8 won that Elm lost, 1 the reverse |
| 200 fresh games against 10 public engines | 179W-21L | 179W-21L; 10 better / 2 worse, +$1 |
| 40 games against Ice and Fire 09-30 (below) | 24-16, +$85 | 25-15, +$331 |

## Candidate E (not submitted; the seed of the final mutation run)

The notebooks run on 09-30 afternoon (pulled read-only, agents recovered statically): Harvest Ledger 15:56 is
byte-identical to `haodou_ledger_0928` (0927 + a mirror detector); **Ice and Fire 15:31 is `leo_pi` with `_OR2_SN_K = 1`**,
which tracks the rival's stock from market movements and sells ahead the units its route tape plans for the next 24
steps while the rival holds MILK/STRAWBERRY/WOOL/MELON/EGG. E = Poplar Reach + engine `leo_icefire_0930` in
`_EM_ENGINES` (runtime copy `~/kagg-evo/lit0930e`, package SorrelMeadow.tar.gz there, validation PASS, step 0 up to 17 s):
- against Ice and Fire, 40 games: **32-8, +$537** (Poplar Reach 25-15, Elm Crossing 24-16, Laurel Field's graph 21-19);
- but on history slightly behind Poplar Reach: 179 replays 111-68 (2 better / 9 worse, -$17), the 311 lost games 89
  won (3 better / 29 worse, -$16): a fifth hypothesis alive means more turns on which the hypotheses disagree and the
  emulator predicts nothing.
Rivals submitted on 09-30 (49 of their games matched): 0927 exact 8, Harvest Ledger family 20 (1 exact), leo_pi 3,
Ice and Fire family 2, other openings 13; Elm Crossing's three losses: a Harvest Ledger copy (-$403, 340 moves equal),
an Ice and Fire copy (-$675, 361), another opening (-$7,807). Poplar Reach went in as the better agent on history; E
seeds the final mutation run so that the user's last submission can hedge towards Ice and Fire.

## Final mutation run (from 18:18 UTC)

`launch_final.sh` / `setup_final_full.py` here (cliproxyapi `~/kagg-evo/final-0930`, tmux `kagg-evo-final`): full graphs
(predictor + KAD required), seed E, KAD/Oracle islands, fresh LLMs only (no qwen, deepseek or fugu, the user's
instruction), gauntlet = the 243 lost/tied games of the five submissions within $2,000 + 72 close wins + 20 games against
Ice and Fire; queued: `_OR2_SN_K` 1, the guard at score 0.4 on five products for rivals of class `other_opening` only
(where the emulator is blind), the guard on EGG and MELON for all.

**Iteration 1 promoted `_OR2_SN_K` 0 -> 1** (candidate F = E + front-running; package SycamoreBend.tar.gz, sha256
`103bf00d…`, validation PASS, step 0 up to 18.5 s): 344 of 345 games changed, 257 better / 87 worse, +$164, p=2e-15,
results +63/-15. By group: Aspen Vale's losses +30 net wins, Juniper's +6, Hawthorn's +2, Laurel's +3, but the close
wins -5, and Ice and Fire 12/8 (+$91). Independent full-graph checks against Elm Crossing (cached games):

| check | F vs Elm Crossing |
|---|---|
| 240 arena games, 12 public engines (fresh0930) | 107 better / 132 worse, -$20; 212W-28L vs 222W-18L |
| Juniper Knoll's 73 replays (bt_jk2) | 49 better / 23 worse, +$121, p=0.003; 41-32 vs 38-35 |
| head-to-head, our submitted tarballs (h2h_f) | 70W-10L: Poplar Reach 17-3, Elm 18-2, Laurel 18-2, Juniper 17-3 |

Front-running beats the 0927 family (Ledger 09-28 14-6 +$111, leo_pi 13-7 +$100, 0927 14-6 +$96) and our own agents,
and loses to other families (V55 5-15, V57 6-14, Shepherd 8-12, Forecast 8-12, Robust Economy 4-16, Pioneers 7-13).
Candidate **F2** = F with `counters: {other_opening: {_OR2_SN_K: 0}}` (off against rivals whose opening differs from
ours before step 92) went into the same three checks at ~21:10 UTC.

**Loss census** (`census.py`, the 311 recorded lost/tied games of the five submissions, rival minus us at days
5-25): losses under $500 (167) show no farm difference at all (same land, crops, animals, money within $100): sale
timing. Losses of $500-2,000 (76): same farm, the money gap opens after day 20 (+$270 d20, +$550 d25). Losses over
$2,000 (68): the rival invests earlier (2.4 vs 2.0 quadrants at day 10, +1.8 geese, +6 wheat, -$711 cash at day 10) and
grows tomatoes and carrots instead of strawberries in the second half (+4.8 tomatoes, -4.6 strawberries at day 20);
+$4,000 by day 25. Iteration 3 plays the matching engine lever, the tomato gate `_CXTB_MIN_REVENUE` 9000 -> 7000.

**Tomatoes** (`tomato_census.py`): in the losses over $2,000 the rival grew tomatoes in 46% of the games (we: 10%),
from a median day 16 (a quarter by day 11; we: day 18) and 5.8 tiles at day 20 against our 1.0; in the close and
mid losses 18% of rivals did (we 18% / 12%, both from day 18), and in our 126 won games 13% (we 13%). Iteration 3
PROMOTED the tomato gate 7000 on E (43 of 345 games changed, 25 better / 18 worse, +$99, results +10/-2, p=0.039).
Candidate **F2t** = F2 + that gate (package UmberField.tar.gz) went into the 240-game fresh arena check (main pool,
plus a second 2-VM pool `~/kagg-evo/pool2` as a backup) at ~22:00.

The gate is the engine's day-18 decision (step 432) to buy the last quadrant (4,000) and plant ten tomato tiles for a
harvest on days 26-29, taken when we hold 12,000 and the projected tomato revenue clears `_CXTB_MIN_REVENUE`; the
projection subtracts the rival's standing tomato tiles, so the gate opens least where the rival already grows them. It
does not reproduce the rivals' day 11-16 tomato plans, which live in the route tapes. Candidate **Et** = E + the gate
(`candEt.json`, byte-for-byte in play the champion that iteration 3 promoted on Island-KAD-Early) was built as the
fallback in place of E: package VetchHill.tar.gz (sha256 `3aec30e8…`), validation PASS, fidelity identical (its timing
numbers were taken while three builds shared cliproxyapi's 8 cores, so they are not representative).

## Outcome: the last slot was not used

Kaggle closed submissions at 23:59 UTC (`submissions_disabled`); the final pair is **Poplar Reach + Elm Crossing**
(ratings at 04:55 UTC 10-01: 1648.9 and 1761.1). Every check below finished before the deadline (F2's fresh games 22:19,
F2t's 22:59, F2's head-to-head 23:25, its replays 23:38), but the watcher on the PC hung while the PC slept, so the
results were not reported in time and no go-ahead was given.

Full graphs, 240 fresh arena games (12 public engines x 20 validation seeds, both seats), paired by game:

| graph | W-L | 0927 family | other engines | mean margin | paired |
|---|---|---|---|---|---|
| Laurel Field | 216-24 | 67-13 | 149-11 | +$892 | |
| Elm Crossing | 222-18 | 73-7 | 149-11 | +$929 | |
| F | 212-28 | 72-8 | 140-20 | +$908 | vs Elm 107 better / 132 worse, -$20 |
| **F2** | 219-21 | 72-8 | 147-13 | **+$945** | vs Elm 75 / 52, +$16, p=0.05 |
| F2t | 210-30 | 69-11 | 141-19 | +$899 | vs F2 **0 better / 12 worse**, -$45 (pool2's copy: 0 / 17) |

F2 also: Juniper Knoll's 73 replays 41W-32L against Elm's 38-35 (39 better / 12 worse, +$134, p=0.0002); head-to-head
against our submitted tarballs 70W-10L (Poplar Reach 17-3 +$190, Elm 18-2 +$223, Laurel 18-2 +$267, Juniper 17-3 +$569).
The tomato gate changed only games it lost on fresh seeds, so F2t would not have gone in. **F2 was the pick**, but its
package TamarackRise.tar.gz (sha256 `116a366c…`) failed the fidelity check: seed 101 gave $181,445 through the Kaggle
loader and $124,872 through the arena harness, where F, F2t and Et gave $181,445 both ways. It ran while three builds
shared the machine; a rerun would have been needed (or Vetch Hill as the fallback).

Final mutation run, all eight iterations (gauntlet of 345 games; results = won/lost games flipped):

| it | island | edit | changed games | results | verdict |
|---|---|---|---|---|---|
| 1 | KAD-Sell | `_OR2_SN_K` 0 -> 1 | 257 better / 87 worse, +$164 | +63/-15 | promoted (= F) |
| 2 | KAD-Hands | `_KC_FROM_STEP` 144 -> 96 | 1 / 0 | +1/-0 | stopped after 100 games |
| 3 | KAD-Early | `_CXTB_MIN_REVENUE` 9000 -> 7000 | 25 / 18, +$99 | +10/-2 | promoted (= Et) |
| 4 | KAD-Oracle | guard on five products vs `other_opening` | 12 / 5, +$1 | +3/-0 | rejected |
| mix | all | donor KAD-Sell: `_OR2_SN_K` 1 | 257 / 87 | +63/-15 (KAD-Early +60/-19) | promoted on all seven |
| 5 | KAD-Tactics | `_OR2_SN_ITEMS` + WHEAT/CARROT/TOMATO | 5 / 95, -$3 | +1/-62 | stopped after 100 games |
| 6 | Oracle | `_OG_ITEMS` + EGG/MELON | 27 / 20, -$3 | +1/-1 | rejected |
| 7 | Tactics | `_OR2_SN_H` 24 -> 48 (on F) | 199 / 94, +$76 | +24/-4 | promoted, not checked on fresh seeds |
| 8 | Herd | counter vs mirror `_S738_LOOK` 7 | 114 / 61, +$35 | +12/-4 | promoted (by dollars); the run ended here, 05:04 UTC 10-01 |

All VMs were stopped afterwards (`COLAB_VMS_LEFT=0` for both pools, no live session on colab2 or colab4) and the
session history files were deleted.
