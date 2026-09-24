# Edit-based graph evolution: results of 2026-09-24

Runs of `highcpu_island_evolution.py` on the Brev box `kagg-arena-80` (n2d-highcpu-80,
60 concurrent games; deleted 2026-09-24 after run 3 was stopped). The run directories
themselves (`runs/evolution/`) are git-ignored; this folder keeps what they produced,
except the rebuildable graph bundles and candidate graphs. Evaluation: 40 seeds per pool
opponent (20 per seat, salt 20260925) plus 40 head-to-head games against the island
champion.

## Run 1: v5.0.0 baseline only

The seed graph `policy_graph.json` v5.0.0 (13/9 opening, sells-first, bundle
`g_9a573c4db73fdb3d`) played its 160 pool games. All 6 iterations then failed at once:
every LLM call was refused because the reverse tunnel to the proxy was down, so no edit
was proposed or played.

| opponent | W-L-T | mean cash margin |
|---|---|---:|
| mohui13 (backbone with a 13/9 opening) | 40-0-0 | +$36,317 |
| mohui (plain Mohui v66 backbone, 5/0 opening) | 0-40-0 | **-$18,062** |
| hazel (Hazel Weir, 35/30) | 40-0-0 | +$37,615 |
| willow (Willow Ford, the Harvest Current stand-in, 35/30) | 40-0-0 | +$26,355 |

## Run 2: 6 iterations (one per island)

This run started 06:36 UTC and ended 08:50 UTC. Its 28 mutation calls all succeeded. Eleven
proposals were refused as duplicates of edits already played or proposed.

| island | edit (model) | mean change per game vs v5 (160 pool + 40 head-to-head games) | verdict |
|---|---|---:|---|
| Opening | `_OPENING_SELL_WHEAT_QTY` 9 -> 8 (gpt-6-astra) | -$1,810 (mohui +$18,880, mohui13 +$14,272, hazel -$34,932, willow -$25,529) | rejected |
| Town | `_TOWN_CADENCE_PHASE` 0 -> 3 (gemini-3.8-flash) | -$9 | rejected |
| Shed | `_SHED_PRESSURE_AT` 80 -> 88 (gpt-6-sol) | -$158 | rejected |
| Oracle | `_ORACLE_FRONTRUN_BATCH` 6 -> 4, `_ORACLE_FRONTRUN_PRICE_RATIO` 0.6 -> 0.7 (gpt-6-luna) | +$221 (136W-53L, 11 unchanged, p=1.3e-9) | **promoted** |
| Endgame | `_PRICE_THRESHOLD_RATIO` 0.85 -> 0.80 (claude-sonnet-5) | +$2 (186 of 200 games unchanged) | rejected |
| Dispatch | `_OPENING_SELL_WHEAT_QTY` 9 -> 13 (gpt-6-luna) | -$60,674 (4W-196L) | rejected |

`best_graph.json` is v5 plus the oracle edit, bundle `g_eedbc9c07e2cf465`. Compared with
v5 on the pool it goes 103W-46L-11T, +$173 per game (p=3.5e-6). Absolute results:
mohui13 40-0 +$36,366, mohui 0-40 -$17,795, hazel 40-0 +$37,889, willow 40-0 +$26,458.
It has not been confirmed on the arena validation seeds and is not promoted into
`policy_graph.json`.

Files: `candidates.jsonl` holds every proposal, refusal and gauntlet verdict with the
models' rationales. `console.log` includes the judge votes and supervisor guidance.
`scores/` holds per-game margins per graph bundle, and `games/` holds the raw game records
(gzipped JSONL).

## Run 3: stopped

Run 3 started at 09:40 UTC from run 2's best graph, with the ideas in
`../evolution_ideas.md`. It was stopped at the user's request while replaying its
baseline, before any game finished or any edit was proposed. Nothing of it is kept.

## Why v5 loses to the plain Mohui backbone (`plain_mohui_diagnosis/`)

The folder holds 12 traced games from arena round r2b: seeds 1167109771, 1170730226,
1183470969 and 1209828168, each played as Hazel (35/30), open13 (the 13/9 opening) and
opening_backbone (the opening stage off), all against the plain backbone, candidate in
seat 0. `../arena/replay_trace.py` replays them through an instrumented engine, and every
replay reproduces the recorded cash exactly. `../arena/opening_cash.py` computes the
day-1 hire cash of each opening pair.

- **The 13/9 opening leaves 4 wheat after step 1.** The backbone route needs 5 to feed its
  two cows on day 1. In all 4 open13 games one of our cows escapes at the end of day 1,
  and in 3 of them a second cow escapes on day 8. No animal escapes in the Hazel games, or
  for the backbone in any game.
- **The shortfall compounds.** One cow fewer means less fertilizer to sell. The next cow
  becomes unaffordable at step 88, so we hold 4 animals against the backbone's 6.
- **The town shops change.** The shop added every 3 days comes from the same RNG as the
  weed spawn, which draws once per empty tile, so our different farm opens different
  shops from about day 9.
- **The backbone gains through prices, not volume.** It sells the same units at better
  prices, for example strawberries at $125 instead of $80. It ends about $12k richer than
  in the Hazel games, and we end about $5k poorer.
- **Why 13/9 still beats Hazel and Willow.** Its 9-unit step-1 sell leaves these 35/30
  openers $6 at the step-24 hire, $1 short of their fourth hand, and that dwarfs the lost
  cow.
- **Openings that should get both effects.** 15/10, 16/11, 17/12 and 18/13 keep 5 wheat
  and still leave the 35/30 openers $5-6. This is untested in games; it is the top item
  in `../evolution_ideas.md` for the next run.
