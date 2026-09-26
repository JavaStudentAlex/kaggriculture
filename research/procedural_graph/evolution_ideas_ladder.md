# Ideas for ladder-engine evolution (hypotheses, not facts)

Read by `highcpu_island_evolution.py --ideas evolution_ideas_ladder.md` and shown to every mutating
model; re-read each iteration, so it can be edited while a run is going. The seed graph plays the
public "Demand-Preserving" engine (tetsutani_demand) as-is.

- Aim at layers that act. The knowledge section lists which engine layers changed actions in 48
  games and which never did: an edit of a dormant layer's parameters (carrot planner, race layers,
  opening tape, yarn reorder) or of a load-time table changed 1-3 of 230 games in run ladder1, unless
  it woke the layer (the carrot edit did, and lost).
- Do not spend an edit on copying another island's promoted settings. The loop mixes the islands
  every 12 iterations: every island is offered the champion of the island that gains most over the
  seed, and the gauntlet decides. A copy proposed as an edit is refused anyway when the same settings
  were already played on any island (iteration 15: two models proposed the guard for Market and both
  were refused). Propose settings nobody has played yet.
- Herd island: its own layers barely act (herd swap in 6 of 48 games, cow swap in 2, V9_HERD_* only at
  load; five Herd edits so far changed 1-3 games each). Better uses of a Herd iteration: tune an active
  layer that touches animals (feed/fertilizer economics _R85_*/_R88_*, the _CA_* layer, _SM_MILK_SHOP).
- Promoted so far: the oracle guard on MILK/WOOL/STRAWBERRY at score 0.5, batch 4, keep 2, price ratio
  0.75 (Oracle, +$31 a game), _SR_MARGIN 12 (Endgame, +$9), _CXD_BUDGET 1050 (Opening, +$2). On 260
  fresh validation games the guard with _CXD_BUDGET 1050 gained $53 a game over the plain engine
  (92 games better, 49 worse) and beat the plain engine head-to-head 14-0 with 8 ties.
- Rejected on the stronger line: _SR_MARGIN 12 on top of the guard and _CXD_BUDGET (75 games better,
  42 worse, but mean -$22: a few large losses against haideptry_shepherd), _ADV_FRONT on (51 better,
  177 worse), _CA_BUFFER 12 (22 better, 60 worse), _CH_SHED 90 (6 better, 27 worse).
- Oracle guard, untested: _OG_SCORE 0.55-0.6; _OG_MAX_ORDERS 1 or 3; _OG_TO_STEP later than 696 (the
  last day) or _OG_FROM_STEP later than 256; EGG back in the item list alone; a larger _OG_KEEP for
  WOOL (a yarn store takes 2 a visit). The guard loses a little against V57 (4-4, -$43) and pilkwang:
  a price floor closer to 1.0 may help there.
- Oracle, untested: our full market channel (the champion's oracle_frontrun, town cadence and
  liquidation stages) on top of the engine. It rewrites the engine's order list, so expect large
  changes in both directions.
- Endgame: _SR_MARGIN 8 -> 12 helped; the guard's hours (_SR_HOURS) and margins beyond 12 are untested.
  The capacity harvest/sell layer (_CH_SELL, _CH_SHED) acts in 29 of 48 games.
- Market, active and untested: the order-slot priority (_OR2_*, 24 turns a game), quote reordering
  (_R37_ADAPTIVE, _R37_QUOTE), the model-based lead seller's items (_MPX_ITEMS), hour-window and dawn
  lead sells (_MP_*, _DP_*, _EV_*), ready-stock advancing (_ADV_*).
- Crops, active: input planning (_R51_INPUT_CROPS, _R51_INPUT_MAX_WORKERS, 64 turns a game), feed and
  fertilizer economics (_R85_*, _R88_*), fertilizer timing (V9_FERT_FIRST_DAY 14 -> 16 changed 77 games
  at +$24 mean but lost more games than it won), the _CA_* layer (41 turns a game in 30 of 48 games).
- Opening: the engine's opening comes from _PIPE_MODE ('EarlyCycle'); it beats the 13-wheat openers
  20-0, so do not weaken it. Wheat buy-the-dip (_BD_*) and the cash reserve (_CXD_*) act every game.
