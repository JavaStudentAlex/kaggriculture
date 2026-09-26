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

Fixes in the 10:49 loop:
- The knowledge file is re-read every iteration.
- Later models see the edits proposed earlier in the same iteration. Before, three models proposed
  the same `_FX_FLOW_MIN` edit and two of them were refused as duplicates.
- A stop that arrives during a gauntlet keeps that iteration's result.

Iterations 1-11 saw a stale engine description: the knowledge file still named haideptry_2965,
while the controls table said tetsutani_demand. The corrected file adds the layer-activity map.

The pool fingerprint was unchanged by the restart (checked), so every cached baseline was reused.

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
