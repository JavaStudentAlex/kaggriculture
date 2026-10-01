# Aspen Vale's ladder games after the 09-28 review (2026-09-29)

Aspen Vale (Kaggle 56643792: the public engine's 09-27 version `tetsutani_demand_0927`, `_SR_MARGIN` 12, the rival
emulator, no predictor) played 44 more games between 20:24 UTC 09-28 and 06:34 UTC 09-29: **19W-25L**. Its rating
went from 2,206.5 to **2,175.4** (09:00 UTC 09-29); all 128 games so far are 72W-56L. Cedar Ridge, the other
submission that still plays, fell from 1,912.9 to 1,812.1. The first 84 games are in `../aspen_vale_20260928/`.

| games | time (UTC) | record | rivals' median rating |
|---|---|---|---|
| 1-20 | 09-28 14:38-15:42 | 19-1 | 1,180 |
| 21-40 | 15:46-16:50 | 13-7 | 2,152 |
| 41-60 | 16:50-18:01 | 11-9 | 2,162 |
| 61-80 | 18:05-19:25 | 9-11 | 2,190 |
| 81-100 | 19:25-22:34 | 7-13 | 2,216 |
| 101-120 | 22:38-02:26 | 10-10 | 2,193 |
| 121-128 | 09-29 02:50-06:34 | 3-5 | 2,155 |

All 44 new games were against rivals rated 1,900+ or no longer listed. 17 of the 19 wins were by less than $600
(median +$81); 12 of the 25 losses were by less than $600, and 8 by $5,000-$24,448.

## Who beat Aspen Vale (25 new losses)

Rival class (rival_counter's MirrorTracker on the rival's public state) and the public bundle that reproduced its
recorded moves longest (`match_ladder_games.py` over 46 public bundles; `summarize_avL2.log`):

| rival | losses | median margin |
|---|---|---|
| no public agent, 10 of them with another opening than ours (`other_opening`) | 12 | -$7,045 |
| our own base engine, `tetsutani_demand_0927`, 150-576 moves | 6 | -$260 |
| Harvest Ledger 09-28 (`haodou_ledger_0928`), 447-673 moves | 3 | -$367 |
| the old public engine (`tetsutani_demand`), 288-299 moves | 2 | -$782 |
| leoprovorov's `leo_pi`, 221-457 moves | 2 | -$90 |

No rival played a public engine for all 719 moves: the 13 copies all left it, late, and lost us close races. The
rival emulator follows a rival only until its first deviation. Every loss by more than $5,000 but one (Carson
Rodrigues, an old-engine copy: -$5,242) is to an agent no public notebook reproduces.

The 19 wins (`summarize_avW2.log`): 17 mirror class, 2 other openings; 6 rivals ran our base engine exactly (all 719
moves, median +$155) and 3 ran `leo_pi` exactly (+$40), 7 were copies of our base engine that left it (+$81), and 3
matched no public agent (+$446). Together, over the 44 games:

| rival | record |
|---|---|
| a public engine played exactly (719 moves) | 9-0 |
| a copy that left a public engine after 150-670 moves | 7-13 (our base engine's copies 7-6; Harvest Ledger 0-3, the old engine 0-2, `leo_pi` 0-2) |
| no public agent | 3-12 |

## Where the money went (exact)

`flows_av.py` re-runs every recorded step through the environment's interpreter; the simulated money equals the
recorded money at every step of all 44 games. Ours minus the rival's, by product (sales net of the same product bought
back) and by cost (positive = we spent less), with the 09-28 sets for comparison (`money_split.py`, `money_split.txt`):

| category | 25 new losses | per loss | 19 new wins | per win | 31 losses 09-28 | per loss |
|---|---|---|---|---|---|---|
| wheat | -$51,101 | -$2,044 | +$2,177 | +$115 | -$2,681 | -$86 |
| egg (geese) | -$28,784 | -$1,151 | +$287 | +$15 | -$16,085 | -$519 |
| strawberry | -$19,397 | -$776 | +$1,438 | +$76 | -$2,900 | -$94 |
| tomato | -$19,191 | -$768 | $0 | $0 | -$32,237 | -$1,040 |
| carrot | -$18,519 | -$741 | -$1,114 | -$59 | -$7,619 | -$246 |
| wool | -$13,461 | -$538 | +$1,039 | +$55 | -$2,294 | -$74 |
| fertilizer | +$154 | +$6 | +$408 | +$21 | -$1,820 | -$59 |
| land | +$4,000 | +$160 | $0 | $0 | +$4,000 | +$129 |
| milk | +$8,415 | +$337 | +$278 | +$15 | +$9,115 | +$294 |
| hires | +$9,198 | +$368 | $0 | $0 | +$4,618 | +$149 |
| seeds, animals | +$9,680 | +$387 | +$150 | +$8 | +$2,550 | +$82 |
| melon | +$20,450 | +$818 | $0 | $0 | +$1,935 | +$62 |

The new wins are mirror races: no difference in tomatoes, land or hires, and every product within about $100 a game.

## How the big losses happen: we keep cash, they invest it

`story_av.py` (`stories_avL2.json`, `story_avL2.log`) gives the money gap by day and both farms at days 10 and 20
(`farms_av2.py`). Our farm on day 10 is the same in every game: two quadrants, 5 wheat, 20 strawberries, 12 melons,
6 cows (9 in two games), 4-7 sheep, no geese (2 in one game). The rivals that beat us by thousands:

| rival | our cash lead day 10 | day 20 | rival's farm on day 10 (ours: 2 quadrants, 5 wheat, 0 geese) |
|---|---|---|---|
| juliencst (2,393), -$24,448 | +$3,523 | -$17,654 | 3 quadrants, 25 wheat, 2 geese, 12 sheep, 8 cows |
| cununn, -$16,386 | +$3,443 | -$4,616 | 3 quadrants, 19 wheat, 2 geese, 11 sheep, 3 tomatoes |
| Breaking1800 (2,351), -$8,477 | +$3,132 | +$715 | 3 quadrants, 20 wheat, 12 sheep, 4 tomatoes, 1 goose |
| Zhu Liang (2,562), -$8,096 | +$3,049 | +$1,745 | 2 quadrants, 9 wheat; 38 wheat and 6 carrots by day 20 |
| GPTatoes (2,341), -$7,557 | +$811 | -$12,786 | 3 quadrants, 23 wheat, 6 geese, 11 cows |
| Closed Loop (2,168), -$7,045 | +$2,152 | -$1,363 | 3 quadrants, 10 wheat, 3 geese, 11 cows, 1 tomato |

In 6 of these 6 we had more cash on day 10 and lost it by the end: the rivals had bought the third quadrant, four to
five times our wheat, geese and early tomatoes. We bought land too, later: in juliencst's game we owned 4 quadrants by
day 20 against their 3 ($7,000 against $3,000) and planted them with strawberries and sheep. So more land alone is not
the difference; buying it before day 10 and putting wheat, geese and tomatoes on it is. HireMe (-$12,565) had our farm
on day 10 and bought the fourth quadrant for 43 strawberries against our 33 by day 20.

## Seats

Aspen Vale's seat split is seat 0 40-23, seat 1 32-33 over all 128 games (12-9 and 7-16 in the new 44). This is
probably not a mechanism: the environment's market quotes both seats the same price for every unit of a slot ("Both
players see the same pre-commit inventory for this unit", `kaggriculture.py` 612), seat 0 wins 51.3% of the 2,990
top games of the plan mining, and Alder Ford and Birch Hollow did better in seat 1.

## Files

- `index.json`: all 128 games (`fetch_new_games.py` of ../aspen_vale_20260928, read-only API); `index_avL2.json` the
  25 new losses, `index_avW2.json` the 19 new wins. Replays: `replays/ours/aspen_vale/` on the PC (not in git); the
  copies on cliproxyapi were deleted after the run (disk).
- `run_av2.sh` (cliproxyapi, tmux kagg-av2): flows, then `run_batch_av.sh` of ../aspen_vale_20260928 per batch
  (rival features, replay opponents `replay_<episode>` in `~/kagg-evo/repo/shinka/champions/replay_opponents`, the
  46-bundle match), then `summarize_av.py`.
- `flows_avL2.json`, `flows_avW2.json` (+ logs), `money_split.py` and `money_split.txt`, `stories_avL2.json` and
  `story_avL2.log`, `farms_av2.py` and `farms_avL2.txt`, `summarize_av{L2,W2}.log`, `classes_av{L2,W2}.json`,
  `best_match_av{L2,W2}.txt`.

## What it means for the next agent

- The engine's production plan is below the band it now meets. The plan mining of 2,990 top games (09-22..09-26,
  `research/procedural_graph/benchmark/results/plan_mining_2026-09-26/report_tetsutani_demand_0927.md`) says the same
  at scale: by day 29 top players have bought 3.6 plots, about 20 geese, 23 cows and 288 hires; the 09-27 engine's
  route tapes buy 2 plots, 0-5 geese, 6-9 cows and about 252 hires. Winners and losers there buy alike, so this is the
  standard of the 2,900+ band, not what separates its players. Early land is in the engine's hard-coded route tapes
  (no parameter); the one land lever parameters reach is the tomato-plot branch (`_CXTB`, one plot and 10 tomato
  seeds, gated by `_CXTB_MIN_REVENUE`), queued for the predictor run's Tomatoes island.
- Close races against copies that leave a public engine are the other half. The emulator wins every exact copy;
  keeping a hypothesis after small market deviations (REVIEW_20260929.md of ../aspen_vale_20260928, item 5) and
  sale timing against such copies (the predictor run's Oracle island) are the levers.
- The 44 games are not in the predictor run's plan (763 games: ladder1's 727 plus 36 older Aspen wins). The replay
  opponents `replay_<episode>` for all 44 exist in `~/kagg-evo/repo/shinka/champions/replay_opponents`.
