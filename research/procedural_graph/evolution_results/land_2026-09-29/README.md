# Land evolution: the fourth quadrant as an evolvable lever (2026-09-29)

**Why.** Aspen Vale's large ladder losses (09-29 recheck) were under-investment: on day 10 our farm always has 2
plots, 5 wheat, 20 strawberries and no geese. The top teams (DSM, 83% wins, and three others; tape mining of
09-22..09-28) own 3 quadrants by day 9 and 43% own 4 by day 12. Our engine's 41 route tapes buy NE at step 150
(day 6) and SW at step 265 (day 11) in every game, never SE, and no engine parameter moves those purchases. The
user asked for the evolution (and the KAD model) to get the chance to vary the land it buys. The KAD RL already does:
its kept games buy land on days 2-15, a 4th quadrant in 45 of 131.

**The lever.** `hazel_runtime/land_plot.py`, the optional `land_plot` turn stage (channel `land_plot`, `_LP_*`
parameters, `graph_edits` catalog and prompt section LAND PLOT, tests `test_land_plot.py`). From `_LP_DAY` it buys SE
(SW first when the tape has not bought it), hires `_LP_WORKERS` hands a day after the engine's hires, farms
`_LP_TILES` SE tiles with `_LP_USE` (WHEAT, CARROT, TOMATO, STRAWBERRY, MELON or GOOSE), buys its own inputs and
sells its produce. Its orders are appended after the engine's (the engine's slots, deliberate empty ones included,
keep their indices).

**Test games on cliproxyapi** (`smoke_land*.py`, 30-day games, guard off for speed):
- Against the public 0927 engine every goose variant lost $5-24k of margin on 5 seeds. That engine races a rival it
  recognises as its clone (same quadrants, 90-95% of occupied tiles alike: `_r37_similarity`, `_race_clone`); the
  fourth quadrant ends the recognition, the rival leaves the race and earns $11-15k more (seed 404: milk $19-45
  against $40-99 in the base game from day 21). Hence `_LP_MIRROR` 0.9: no land while the rival's farm is that
  alike. haideptry_2965, leoprovorov_forecast and robust_economy also look alike (the plot stayed off, 0 change);
  against abo_v57_open13 it bought and lost (goose8w2 -$7.8k a game, day-10 goose -$22.7k).
- The mechanics work (land bought day 12, 8 geese placed by day 14-15, eggs sold, fertilizer used by the engine);
  whether any setting pays is for the gauntlet to show.

**Run.** Code snapshot `~/kagg-evo/land-20260929/repo`, run dir `~/kagg-evo/land-20260929/run`, tmux `kagg-evo-land`
(`launch.sh`, log `~/kagg-evo/land-20260929/run.log`, marker `LAND_EVOLUTION_EXIT=`), pool tmux `kagg-colab-evo`
(restarted 13:54 UTC as `evo290929`, 5 colab2 High-RAM VMs). Seed = the predictor-required run's seed (Aspen line +
`_CA_MARGIN` -22 + `_SR_MARGIN` 14 + guard, `require_oracle`), plan = its 763 games. Islands (`--islands land`):
Land-Geese, Land-Crops, Land-Timing, Oracle, Tactics, Herd; mixing and supervisor every 6 iterations; staged
gauntlet 30%; budget 40 iterations. `queue.json`: goose 8/2 hands day 12, wheat 12 day 12, goose day 9 (SW+SE early),
guard + egg/carrot, goose 12/3 hands, tomato 10.

**The seed against Aspen Vale** (checked 14:10 UTC 09-29 on cached margins, no new games: `vs_aspen.py`,
`aspen_losses.py`, run in `~/kagg-evo`). On the 727 development games both played, the seed (Aspen Vale +
`_S809_LOOK` 4 + `_CA_MARGIN` -22 + `_SR_MARGIN` 14 + the guard) is better in 400 and worse in 191, +$54 a game
(p=5e-18), results +18/-5 (p=0.011), wins 581 → 595. It wins 5 of Aspen Vale's 31 recorded ladder losses (24 better,
2 worse) and keeps 34 of its 36 recorded wins (2 lost by $15 and $195). Without the predictor (ladder1's Next-Counter
champion) it is +$43 a game, results +14/-5. Every old-engine island of ladder1 is worse than Aspen Vale (-$76 to
-$113 a game). The loop chose these changes on these games; only the guard alone had a fresh-seed check (180 games,
157W-13L-10T → 165W-10L-5T). ladder1's and the predictor run's margins compare game by game: the runtimes differ only in
the `require_oracle` check.

**Cache carry.** The seed's bundle is new under the new runtime (`g_d46eaa6322231c85`, was `g_2e6e1664c98cc2a9`),
fingerprint `560451fde0e8`. The gauntlet's opponent digests include absolute paths, so copied bundles in a new run
dir look changed (`inert_land.py` therefore checked 0 games and wrongly passed; the loop started replaying the 763
games). The interrupted replay gave the real check: **259 of 259 games equal the cached margins**; then `fix_carry.py`
re-keyed the 763 margins to the new run's digests (222 of 222 opponent bundles byte-identical).
