# Aspen Vale's first 84 ladder games (2026-09-28)

Aspen Vale (Kaggle 56643792, submitted 14:31 UTC 09-28) is the ladder1 Island-Next-Counter champion: the public engine's
09-27 version (`tetsutani_demand_0927`) with `_SR_MARGIN` 12 and the rival emulator, no predictor (bundle
g_16bb4af65a704f00). By 20:16 UTC it had played 84 games: 53W-31L-0T, rating 2,206.5 (team rank 621 of 10,138 at 20:22;
rank 100 is 2,589). Cedar Ridge, the other submission that still plays, was at 1,912.9.

| games | record | rivals' median rating |
|---|---|---|
| 1-10 (14:38-15:14 UTC) | 10-0 | 1,042 |
| 11-20 | 9-1 | 1,819 |
| 21-30 | 7-3 | 2,101 |
| 31-50 | 13-7 | 2,191-2,202 |
| 51-84 (17:30-20:16) | 14-20 | 2,238-2,364 |

The same pattern as every submission so far: it wins almost everything on the way up and is about even against the
2,200-2,400 band. 19 of the 31 losses and 29 of the 36 wins against strong rivals were decided by less than $600.

## Files

- `fetch_new_games.py` (Cedar Ridge's), `index.json`: the 84 games (read-only API); `index_avL.json` the 31 losses,
  `index_avW.json` the 36 wins against rivals rated 1,900+ or no longer listed. The 67 replays are in
  `replays/ours/aspen_vale/` (not in git); on cliproxyapi the losses' stay in `~/kagg-evo/aspen_vale/replays_avL` for
  `ladder_seed_plan.py`. Each game is a replay opponent `replay_<episode>` in the repo's `replay_opponents` there.
- `recover_0928b.py`: the notebooks that ran after the 09-28 08:04 UTC pull, pulled again at 20:26 UTC and recovered
  statically on cliproxyapi into `~/kagg-evo/aspen_vale/cands4/`. tetsutani's 17:41 run repackages the engine (base85
  instead of base64) but its files equal `tetsutani_demand_0927`'s byte for byte; Dvorkin's v31 and leoprovorov's Ice and
  Fire are unchanged; new are flexonafft's "C95 top-meta replay" agent (`flexon_multiroute_0928`, reported at 2,837 by
  its author), Harvest Ledger's 12:01 version (`haodou_ledger_0928b`) and leoprovorov's God's mode
  (`leo_gods_mode_0928b`). None of the three reproduced a lost game for 100 moves.
- `run_batch_av.sh`: on cliproxyapi, the rival's per-step public state (`rival_features.py`), the replay opponents and
  `match_ladder_games.py` for 44 public bundles; `summarize_av.py`: the rival class (MirrorTracker) and best match of
  each game (`classes_avL.json`, `best_match_avL.txt`); `divergence_av.py` (`divergence_avL.txt`): where a rival that
  ran a public engine left it, and how.
- `flows_av.py` (`flows_avL.json`, `flows_avW.json`): every recorded step re-run through the environment's interpreter,
  logging each unit the market commits with its price, each hire and land purchase, for both seats. The simulated money
  equals the recorded money at every step of all 67 games, so the split below is exact. `story_av.py`
  (`stories_avL.json`): the money gap by day and both farms at days 10 and 20 (its sales column is an early estimate;
  use `flows_av.py`).

## Who beat Aspen Vale (31 losses)

By class: mirror 17, wheat92_seller 6, other_opening 8. By the public agent that reproduced the rival's moves longest:

| rival | losses | median margin |
|---|---|---|
| the public engine's 09-27 version (our own base), 145-497 moves | 13 | -$184 |
| the old public engine, 151-408 moves | 5 | -$708 |
| leoprovorov's `leo_pi` exactly (all 719 moves) | 1 | -$114 |
| Harvest Ledger 09-28 (673 moves), 2965 (290), tt_metav4 (159) | 3 | |
| no public agent | 9 | -$920 |

The rivals that ran our engine and then left it almost all left it in the market list, not on the farm
(`divergence_avL.txt`): a purchase moved ahead of the hires (Yunho Hwang, Aniket Sharma, Malyshev Danil), one more unit
sold (wool 6 against 5, 14 against 9; eggs 3; fertilizer 2), wheat bought to trade (Mario 38, Haalandspring 60), a
fertilizer round trip (satoooh). The market runs both lists slot by slot, one unit at a time at a shared price, so the
slot of an order decides who gets the better price. These look like agents tuned against the public engine, which is
what we play. The rival emulator follows such a rival until its first deviation and then drops the engine.

## Where the money went (exact)

Ours minus the rival's, by product (sales net of the product bought back) and by cost:

| category | 31 losses | per loss | 36 wins | per win |
|---|---|---|---|---|
| tomato | -$32,237 | -$1,040 | +$55,485 | +$1,541 |
| egg (geese) | -$16,085 | -$519 | -$14,169 | -$394 |
| carrot | -$7,619 | -$246 | -$9,355 | -$260 |
| strawberry | -$2,900 | -$94 | -$8,335 | -$232 |
| wheat | -$2,681 | -$86 | -$2,939 | -$82 |
| wool | -$2,294 | -$74 | +$46,476 | +$1,291 |
| fertilizer | -$1,820 | -$59 | -$1,599 | -$44 |
| melon | +$1,935 | +$62 | +$4,608 | +$128 |
| milk | +$9,115 | +$294 | -$6,909 | -$192 |
| land | +$4,000 | +$129 | -$16,000 | -$444 |
| hires | +$4,618 | +$149 | -$13,591 | -$378 |
| seeds, animals | +$2,550 | +$82 | -$1,500 | -$42 |

Tomatoes and land separate the wins from the losses: where we bought the fourth quadrant and grew tomatoes we won,
where the rival did and we did not we lost. Tomatoes pay from day 26 (planted around days 16-20, $80-$330 a unit
depending on the town's pizza shops and farmers' markets):
- Ichika (-$8,778): the same farm as ours to day 20 except 18 tomato plants against our 10: +$13.9k in tomatoes for them;
- Kaggriculture Agent (-$3,668; we were $5,278 ahead on day 25): it bought the fourth quadrant and grew 10 tomatoes where
  we stayed at three quadrants with none: -$11.0k in tomatoes, $4k saved on land;
- leave you (-$3,904): 11 tomato and 7 carrot plants where we grew wheat: -$6.2k in tomatoes.
The largest loss, PrasadChopade213 (-$9,776, no public agent), is a stronger economy from day 10: a third quadrant and
21 wheat by day 10, 7 geese against our 3 (-$8.0k in eggs), more sheep.

This matches the plan mining of 2,990 top games (`research/procedural_graph/benchmark/results/plan_mining_2026-09-26`):
the top winners keep twice our geese and grow tomatoes and carrots where our tapes grow strawberries. In the close losses
the rival's edge is small and scattered: carrots where we grew wheat (UCAS_fish_wang, ShionMatsuoka, vn1111, xxxx0314),
wheat trading (Mario, Ignat, ekkkcz: exactly the margin), a larger wool or milk sale.
