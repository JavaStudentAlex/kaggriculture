# Ladder-engine evolution, run ladder1 (2026-09-26)

The graph runs tetsutani's public "Demand-Preserving" agent (`tetsutani_demand`, the strongest agent
of the ladder pool, `shinka/champions/ladder/`) as its production engine; the edit-based SIFT island
loop (`highcpu_island_evolution.py --islands ladder`) tunes the engine's constants and switches our
layers on top of it. It runs on cliproxyapi (`~/kagg-evo`, tmux `kagg-evo-ladder1`), and the games
are played by a persistent Colab VM pool (`arena/colab_pool.py`, 5 High-RAM VMs on colab2, tmux
`kagg-colab-evo`).

## Files

- `seed_graph.json`: the seed. It plays the engine as-is: the farmer, hands and market channels pass
  its action through and the oracle guard is off.
- `plan.json`: the gauntlet's 210 pool jobs (`ladder_seed_plan.py`).
  - Every lost ladder seed of Rowan Glen and Linden Brook, played from both seats against the bundle
    that plays like the rival who beat us there. The bundle comes from move-for-move evidence, or
    else from the rival's step-0 opening.
  - 10 random seeds against each pool member.
  - The loop adds 20 head-to-head games against the island champion.
- `engine_activity.json`: which of the engine's 76 layer functions changed its actions in 48 plan
  games, and which engine parameters each layer reads (`engine_activity.py`; summary in
  `evolution_knowledge_ladder.md`).
- `new_losses.json`: Rowan Glen's and Linden Brook's 10 ladder losses after the plan was built (6 and
  4 games, 06:28-09:56 UTC). Each carries the opponent bundle assigned by opening; 7 of them fall back
  to haideptry_2965. These seeds are held out from the evolution and used in validation only.
- `upgrade_scores.py`: the one-off conversion of the run's cached pool margins to the gauntlet's
  per-game cache (19:35 UTC restart).
- `opening_champion_graph.json`: Alder Ford (the Opening champion, submitted 09-26).
- `alder_ford_look4_graph.json`: Alder Ford with engine `_ADV_LOOK` 4 (arena bundle g_303b063b37c58a0c),
  packaged as "Birch Hollow" (`shinka/champions/submissions/birch_hollow/`, Kaggle submission 56603928, 09-27).
- `alder_ford_look4_lead12_graph.json`: the same plus the lead-sell horizons `_EV_H`, `_DP_H`, `_MP_H` 12
  (bundle g_e5e21e76b673be23), backtest2's best.

## Loop

Each iteration works on one of 6 islands in turn: Herd, Crops, Market, Oracle, Opening, Endgame.

1. SIFT stage 1: a UCB1 bandit over the proxy's models picks 3 of them, and each proposes one edit.
2. SIFT stage 2: a pairwise judge (gpt-6-astra) and a Bradley-Terry ranking pick one winner.
3. The winner plays the 230-game paired gauntlet. It is promoted if an exact sign test over the
   changed games gives p <= 0.05 with a positive mean.

A meta-supervisor writes guidance every 6 iterations.

**Island mixing** (user's requirement; on since the iteration-19 boundary, 09-26 ~14:30 UTC). The islands
evolve separately, and an edit whose settings were already played on any island is refused as a repeat,
so the models could not pass a champion to another island: in iteration 15, two models proposed the
promoted guard for the Market island and both were refused. Every 12 iterations (`--mix_interval 12`; the
first mixing ran at the restart, after iteration 18) the island whose champion gains most over the seed
is the donor. Every other island is offered its champion plus the donor's changes, keeping its own value
where it changed the same setting. The same gauntlet decides. A candidate whose pool games were already
played (an existing champion) plays only the 20 head-to-head games.

**Restarts.**
- 10:49 UTC, iteration-12 boundary: loop fixes (below); the budget went from 24 to 72 iterations.
- 12:35 UTC, iteration-15 boundary: 200 iterations, at the user's request. An iteration takes 12-35
  minutes (35 when both graphs run the predictor), so 200 iterations take about 3-4.5 days and 100-140
  compute units; the account had 180.8 units at 14:00 UTC.
- ~14:30 UTC, iteration-19 boundary: island mixing, and the Colab pool restarted with new code. A VM
  that served 3 hours before Colab ended it no longer uses up one of its slot's 2 replacements, so the
  pool survives Colab's runtime limits over several days.
- ~19:35 UTC, iteration-29 boundary: the plan grew to 258 pool games (next section), and the gauntlet's
  cache became per game, so the champions play only the new games (`upgrade_scores.py` carried the 28
  cached graphs over). The ideas and knowledge files now steer the models away from small guard tweaks
  and towards the ready-stock advancing layer (`_ADV_*`).
- ~21:10 UTC, iteration-31 boundary: replay opponents and the edit queue (next section).

Fixes in the 10:49 loop:
- The knowledge file is re-read every iteration.
- Later models see the edits proposed earlier in the same iteration. Before, three models proposed
  the same `_FX_FLOW_MIN` edit and two of them were refused as duplicates.
- A stop that arrives during a gauntlet keeps that iteration's result.

Iterations 1-11 saw a stale engine description: the knowledge file still named haideptry_2965,
while the controls table said tetsutani_demand. The corrected file adds the layer-activity map.

The pool fingerprint was unchanged by the restart (checked), so every cached baseline was reused.

## Alder Ford's losses in the plan (19:35 UTC)

Alder Ford, the Opening champion on the ladder, reached 2,078 after 70 games (46W-22L-2T); most of
its losses at 2,000-2,250 were by a few hundred dollars. Its 22 lost and 2 tied seeds were added from
both seats (48 games), each against the pool agent that reproduces the rival longest
(`shinka/champions/evidence/alder_ford_20260926/`): tetsutani_demand 28 games, haideptry_2965_0926 10,
leoprovorov_forecast 8, tetsutani_shape_shop 2. Two new pool agents came out of the matching:
- haideptry_2965_0926: the 2965 notebook's version of 09-26 13:54 UTC, which plays one rival exactly.
- tetsutani_shape_shop: tetsutani's earlier "Shape the Shop" notebook. It plays three of the 13-wheat
  rivals who beat Rowan Glen and Linden Brook exactly, and two more for 106-623 moves. Those five
  games (10 plan entries) moved to it from abo_v57_open13, which matches none of them past step 0.

## Replay opponents and queued edits (iteration 31)

The loss analysis (`shinka/champions/evidence/alder_ford_20260926/README.md`) showed that the stand-in
agents are weaker than the rivals who beat Alder Ford: on 21 of its 24 lost or tied seeds the champion beat
the stand-in 13 times. So the plan now also plays each of those 24 games against a replay of the rival's
recorded moves (`shinka/champions/replay_opponents/`): 282 pool games plus 20 head-to-head. Against all
24 replays the champion reproduced the ladder margins to the dollar.

`--queue evolution_queue_ladder.json` plays a listed edit as an island's next candidate. Queued first, from
the rivals' most common change (cows where the engine buys sheep or geese):
- Herd (iteration 31): the HERD2 layer may turn the tape's goose purchase on days 8-15 only into cows,
  whenever their expected value is at least the geese's (`_HD2_OPTIONS` COW, `_HD2_RATIO` 1.0,
  `_HD2_MIN_GAIN` 0, `_HD2_FUTURE` 1.0).
- Crops (iteration 32): the same, choosing cows or sheep by expected value. Withdrawn before it played:
  the backtest (below) showed the cows-only edit losing $337 a game against the recorded rivals.

A sheep purchase cannot become cows by a parameter: the engine's tape harvests sheep every third day.

## Backtest against the recorded rivals (22:15 UTC)

Alder Ford's 72 games against rivals rated 1,900 or more (or unlisted) were played again with one change
each, against the rivals' recorded moves (`ladder_validate.py --replays`; table in
`shinka/champions/evidence/alder_ford_20260926/README.md`). Alder Ford reproduced all 72 ladder results.
`_ADV_LOOK` 3 -> 4 gained $60 a game (49 better, 15 worse, p = 2e-5): the 36-28-8 record would have been
44-27-1, with gains against every rival family. The gauntlet's stand-ins had rejected the same change in
iteration 24. The plain engine would have scored $16 a game less than Alder Ford; guard variants changed
little; cows-only HERD2 lost $337 a game.

On fresh arena games (validate1, 300 per graph) `_ADV_LOOK` 4 gained $32 a game (158 better, 111 worse,
p = 0.005), and against the public engine 12-0-8 became 19-1-0. Backtest 2 (23:39 UTC, all 82 recorded games,
`~/kagg-evo/backtest2.sh`): `_ADV_LOOK` 4 plus lead sells 12 turns ahead (`_EV_H`, `_DP_H`, `_MP_H` 8 -> 12)
went 56W-26L-0T against Alder Ford's 40W-31L-11T, +$365 a game; `_ADV_LOOK` 5 and 6 added nothing over 4.
On validate1's 300 fresh games (2026-09-27 00:09 UTC) the lead-sell variant gained only $37 a game over
`_ADV_LOOK` 4 alone (145 better, 155 worse; 269W-31L against 267W-33L): +$345 against the public engine and
+$318 on the held-out lost seeds, -$60 to -$122 against pilkwang_sep, Shepherd, Forecast, V55 and V57. Queued for
Island-Market (iteration 51), whose champion already has `_ADV_LOOK` 4; the cows-only edit left the queue
after its second rejection (iteration 37). Tables and caveats:
`shinka/champions/evidence/alder_ford_20260927/README.md`.

## Results so far

| it | island | model | edit | changed games | mean | p | verdict |
|---|---|---|---|---|---|---|---|
| 1 | Herd | gpt-6-luna | `_CS_MIN_GAIN` 600 → -5000, `_CS_RATIO` 1.3 → 0 | 2-0 | $0 | 0.5 | rejected (dormant layer) |
| 2 | Crops | gemini-3.8-flash | `V9_CARROT_LAST_DAY` 23 → 25, `V9_CARROT_RATIO` 2.0 → 1.5 | 3-24 | -$38 | 5e-5 | rejected (woke the carrot planner) |
| 3 | Market | claude-sonnet-5 | `_FX_FLOW_MIN` 999 → 20 | 2-0 | $0 | 0.5 | rejected |
| 4 | Oracle | gpt-6-sol | oracle guard on; MILK/WOOL/STRAWBERRY; `_OG_SCORE` 0.45 | 90-66 | +$29 | 0.065 | rejected |
| 5 | Opening | gemini-3.8-flash | `_CXD_BUDGET` 800 → 1050 | 40-22 | +$2 | 0.03 | **promoted** |
| 6 | Endgame | claude-opus-5.5 | `_SR_MARGIN` 8 → 12 (night shed guard) | 81-38 | +$9 | 1e-4 | **promoted** |
| 7 | Herd | gpt-6-sol | `V9_HERD_*` limits | 2-1 | $0 | 1 | rejected (load-time table) |
| 8 | Crops | gpt-6-astra | `V9_FERT_FIRST_DAY` 14 → 16 | 33-44 | +$24 | 0.25 | rejected |
| 9 | Market | gpt-6-luna | `_SETTINGS.terminal_liquidation` on | 1-0 | $0 | 1 | rejected |
| 10 | Oracle | claude-opus-5.5 | oracle guard on; MILK/WOOL/STRAWBERRY; score 0.5, batch 4, keep 2, price ratio 0.75 | 76-47 | +$31 | 0.011 | **promoted** |
| 11 | Opening | claude-opus-5.5 | the same guard on the Opening champion | 73-50 | +$31 | 0.047 | **promoted** |
| 12 | Endgame | (restarted loop) | `_CH_SHED` 100 → 90 | 6-27 | -$9 | 3e-4 | rejected |
| 13 | Herd | | `_HD2_MIN_GAIN` 600 → 300 | 2-1 | $0 | 1 | rejected (herd layer acts in 6 of 48 games) |
| 14 | Crops | | `_CA_BUFFER` 8 → 12 | 22-60 | -$3 | 3e-5 | rejected |
| 15 | Market | gemini-3.8-flash | `_ADV_FRONT` false → true | 51-177 | +$7 | 2e-15 | rejected (more games worse) |
| 16 | Oracle | claude-opus-5.5 | `_OG_TO_STEP` 696 → 714 | 2-0 | $0 | 0.5 | rejected (inert) |
| 17 | Opening | claude-opus-5.5 | `_SR_MARGIN` 8 → 12 on the guard + `_CXD_BUDGET` line | 75-42 | -$22 | 0.003 | rejected (mean < 0) |
| 18 | Endgame | claude-opus-5.5 | the promoted guard on the `_SR_MARGIN` 12 line | 66-55 | -$2 | 0.36 | rejected |
| mix | Herd, Crops, Market | (mixing) | the Opening champion (guard + `_CXD_BUDGET` 1050) | 92-65 | +$31 | 0.038 | **promoted** on all three |
| mix | Oracle / Endgame | (mixing) | the same changes | 39-23 / 82-73 | $0 / -$2 | 0.056 / 0.52 | rejected |
| 19 | Herd | claude-opus-5.5 | `V9_HERD_MAX_EGG_SHOPS_COW` 0 → 1 | 1-1 | $0 | 1 | rejected |
| 20 | Crops | gemini-3.8-flash | `_CA_FEED_DAYS` 1 → 2 | 24-69 | -$100 | 3e-6 | rejected |
| 21 | Market | gpt-6-astra | `_OG_STRONG_SCORE` 0.6 → 0.7 | 12-10 | +$1 | 0.83 | rejected |
| 22 | Oracle | claude-opus-5.5 | `_OG_PRICE_RATIO` 0.75 → 0.8 | 4-6 | +$1 | 0.75 | rejected |
| 23 | Opening | gemini-3.8-flash | `_BD_CAP` 64 → 40 | 6-6 | -$1 | 1 | rejected |
| 24 | Endgame | gpt-6-luna | `_ADV_LOOK` 3 → 4 | 104-93 | +$13 | 0.48 | rejected (+$226 head-to-head, +$124 against tetsutani, about -$95 against V55, V57, Forecast) |
| mix | Oracle / Endgame | (mixing) | nothing new: the donor Herd had the Opening champion's changes | | | | rejected again from the cache |
| 25 | Herd | gemini-3.1-pro-preview | `V9_HERD_MIN_MILK_SHOPS` 3 → 2 | 1-0 | $0 | 1 | rejected |
| 26 | Crops | gemini-3.8-flash | `_CA_BUFFER` 8 → 6 | 4-3 | $0 | 1 | rejected |
| 27 | Market | gpt-6-astra | `_ADV_FRONT` false → true | 52-176 | +$4 | 2e-15 | rejected (more games worse) |
| 28 | Oracle | gpt-6-sol | `_OG_TO_STEP` 696 → 712 | 2-1 | $0 | 1 | rejected |
| 29 | Opening | gemini-3.8-flash | `_BD_MIN` 8 → 4 | 106-150 | -$113 | 0.007 | rejected |
| 30 | Endgame | gpt-6-astra | `_SR_MARGIN` 12 → 16 | 68-53 | +$3 | 0.2 | rejected |
| 31 | Herd | (queue) | cows-only HERD2: `_HD2_OPTIONS` COW, `_HD2_RATIO` 1.0, `_HD2_MIN_GAIN` 0, `_HD2_FUTURE` 1.0 | 34-47 | -$32 | 0.18 | rejected (won back Nikita Makarov's replay, +$4,358, lost 4 other replays) |
| 32 | Crops | claude-opus-5.5 | `V9_FERT_FIRST_DAY` 14 → 12 | 65-52 | -$17 | 0.27 | rejected |
| 33 | Market | gpt-6-astra | `_OR2_SLOT_MARGIN` 12 → 0 | 101-52 | +$2 | 9e-5 | **promoted** |
| 34 | Oracle | gpt-6-astra | guard without MILK (`_OG_ITEMS` WOOL, STRAWBERRY) | 29-41 | $0 | 0.19 | rejected |
| 35 | Opening | gpt-6-luna | `_CXD_BUDGET` 1050 → 1200 | 5-3 | $0 | 0.73 | rejected |
| 36 | Endgame | gpt-6-astra | `_SR_HOURS` 21-23 → 20-23 | 88-83 | +$7 | 0.76 | rejected |
| mix | Herd, Crops, Opening | (mixing, donor Market) | `_OR2_SLOT_MARGIN` 12 → 0 | 101-52 | +$2 | 9e-5 | **promoted** on all three |
| mix | Oracle | (mixing) | `_OR2_SLOT_MARGIN` 12 → 0 and `_CXD_BUDGET` 1050 | 133-62 | +$2 | 4e-7 | **promoted** |
| mix | Endgame | (mixing) | the donor's guard, `_CXD_BUDGET` 1050 and `_OR2_SLOT_MARGIN` 0 | 148-107 | -$1 | 0.012 | rejected (mean < 0) |
| 37 | Herd | (queue) | cows-only HERD2 again, on the new champion | 34-47 | -$29 | 0.18 | rejected |
| 38 | Crops | gpt-6-astra | `_CA_TO` 28 → 24 | 38-153 | -$115 | 8e-16 | rejected |
| 39 | Market | gemini-3.8-flash | `_ADV_LOOK` 3 → 4 | 159-109 | +$28 | 0.003 | **promoted** |
| 40 | Oracle | gemini-3.8-flash | `_OG_KEEP` 2 → 1 | 64-76 | +$6 | 0.35 | rejected |
| 41 | Opening | claude-sonnet-5 | `_CXD_BUDGET` 1050 → 1200, on the new champion | 4-3 | $0 | 1 | rejected |
| 42 | Endgame | gemini-3.8-flash | `_CH_SHED` 100 → 96 | 4-21 | -$4 | 9e-4 | rejected |
| 43 | Herd | gemini-3.8-flash | `_HD2_MIN_GAIN` 600 → 350 | 2-1 | $0 | 1 | rejected |
| 44 | Crops | gemini-3.8-flash | `_CA_FROM` 10 → 9 | 2-3 | -$4 | 1 | rejected |

From iteration 29 the gauntlet has 278 games (258 pool + 20 head-to-head), from iteration 31 302 (282 pool,
24 of them replays, + 20 head-to-head). After iteration 44 Island-Market leads: +$34.6 a game over the seed on
the 282 pool games (162 better, 117 worse, 3 the same; p = 0.008).

Iterations 17 and 18 show that the guard and `_SR_MARGIN` 12 do not add up on the plan's games. Each
helps alone, but together they lose on average against haideptry_shepherd (-$95 a game in iteration 17,
-$119 in 18). In iteration 18 they also lose against leoprovorov_forecast (-$121) and abo_v57 (-$93),
and they still win against tetsutani_demand (+$49) and head-to-head (+$79).

The lost seeds are played from both seats, and the two seats of a seed ended with the same margin in
53 of 60 seeds. Scoring each seed pair once changed no verdict: the four promotions keep p = 0.04,
0.0003, 0.011 and 0.02.

"Changed games" counts the games that got better and worse against the island champion; the
remaining games of the 230 were unchanged.

**The predictor pays.** Our TinyTimeMixer oracle is now in play in two island champions, through
the `oracle_guard` turn stage. It sells milk, wool and strawberries ahead of a predicted rival dump,
into the engine's empty order slots.

Iteration 11 per opponent (changed games W-L-T, mean change):

| opponent | W-L-T | mean change |
|---|---|---|
| mirror (head-to-head) | 12-1-7 | +$89 |
| tetsutani_demand | 11-1-8 | +$54 |
| 13-wheat opener | 13-13-26 | +$59 |
| haideptry_2965 | 14-14-28 | +$25 |
| abo_v57 | 4-4-4 | -$42 |
| pilkwang_sep | 0-2-2 | -$50 |

**Edits of dormant layers are wasted.** The activity map explains the inert iterations:
- The cow-swap and herd-swap layers act in 2 and 6 of 48 games.
- The carrot planner, the race layers and the opening tape never acted.
- The `V9_HERD_*` limits are read only while the module loads.

## Validation (validate1, 780 games, done 12:26 UTC)

`ladder_validate.py` played 3 graphs on the same 260 games each. Report: `~/kagg-evo/runs/ladder1/validation/validate1.json`.
- `seed`: the seed graph, i.e. the plain public engine.
- `opening`: the Opening champion (the guard plus `_CXD_BUDGET` 1050).
- `combo`: the guard plus `_CXD_BUDGET` 1050 plus `_SR_MARGIN` 12.

The games: 12 opponents × 20 arena validation seeds (salt 20260924, never used for selection), plus the
10 new lost seeds (`new_losses.json`) from both seats.

| graph | W-L-T (260) | mean margin | validation seeds (240) | new lost seeds (20) |
|---|---|---|---|---|
| seed | 200-39-21 | +$3,674 | 187-34-19 | 13-5-2, +$931 |
| opening | 217-35-8 | +$3,727 | 202-30-8 | 15-5-0, +$943 |
| combo | 216-41-3 | +$3,731 | 203-34-3 | 13-7-0, +$941 |

Paired on identical games:

| pair | better | worse | same | mean change | p |
|---|---|---|---|---|---|
| opening vs seed | 92 | 49 | 119 | +$53 | 0.0004 |
| combo vs seed | 140 | 68 | 52 | +$57 | 7e-7 |
| combo vs opening | 99 | 30 | 131 | +$4 | 8e-10 |

What it shows:
- **The plain engine already beats the pool.** It wins 200 of 260 games, including 16-4 against abo_v57
  (+$1,058 a game; the V57 family beat Rowan Glen 0-96) and 25-11 against haideptry_2965. On the seeds of
  our 10 newest ladder losses it goes 13-5-2 against the bundles assigned by opening.
- **The predictor guard generalizes.** On seeds never used for selection it adds $53 a game over the plain
  engine. It gains against 8 of the 12 opponents (+$62 to +$139 a game against abo_v55, abo_v57,
  haideptry_2965, pioneers, forecast and the plain engine). Against abo_v43, mohui, pilkwang_sep and
  robust_economy it loses $2-8 a game. Head-to-head against the plain engine (tetsutani_demand, the agent
  that 2,155-2,188-rated ladder players run) it goes 14-0 with 8 ties.
- **`_SR_MARGIN` 12 on top is small.** On validation it is better far more often (99 better, 30 worse), but
  only +$4 a game. In the gauntlet (iteration 17) a few large losses made its mean negative.

The Opening champion is the strongest validated graph. At the user's request it was submitted as
**Alder Ford**: Kaggle submission 56582140, COMPLETE at 14:56 UTC on 09-26. Its graph is
`opening_champion_graph.json`, and the package and manifest are in
`shinka/champions/submissions/alder_ford/`. The Kaggle-style validation passed: 8 of 8 games DONE, the
predictor live on numpy, moves averaging 81-97 ms. Seed 101 against starter ends with the same cash
($181,469) through the Kaggle loader and the arena harness.

**Birch Hollow** is Alder Ford with engine `_ADV_LOOK` 4, the fix from the loss analysis that held on both the
replay backtest and fresh seeds (`shinka/champions/evidence/alder_ford_20260927/README.md`). At the user's
request it was submitted as Kaggle submission 56603928 at 08:50 UTC on 09-27 and was COMPLETE by 08:56 UTC
(starting rating 600). Its graph is `alder_ford_look4_graph.json`; the package and manifest are in
`shinka/champions/submissions/birch_hollow/`. The same validation and fidelity checks passed.
