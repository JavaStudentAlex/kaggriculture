# Ideas for ladder-engine evolution (hypotheses, not facts)

Read by `highcpu_island_evolution.py --ideas evolution_ideas_ladder.md` and shown to every mutating
model; re-read each iteration, so it can be edited while a run is going. The seed graph plays the
public "Demand-Preserving" engine (tetsutani_demand) as-is.

- Priority 1 (09-29, the user's request: give the evolution a chance to vary the land it buys): the land_plot
  stage (CONTROLS: LAND PLOT) on the three Island-Land-* islands. Vary when the fourth quadrant is bought
  (_LP_DAY 8-18), what it holds (_LP_USE: GOOSE first, then WHEAT/TOMATO/CARROT), its size (_LP_TILES) and hands
  (_LP_WORKERS; about 4 geese or 8-10 crop tiles per hand), and the money guards (_LP_MIN_MONEY, _LP_RESERVE).
  Count what the plot costs (land, geese, feed, hands) against what it sells by day 29; earlier land means more
  laying days but less cash for the engine's own plan. _LP_MIRROR decides against which rivals land is bought.

## Aspen Vale oracle-required run (supersedes older seed/guard suggestions below for this run)

- Baseline: current Aspen Vale (`tetsutani_demand_0927`, `_SR_MARGIN=12`, rival emulator) with `ttm_c256_h96_ft_2026-09-13` until a hash-checked calibrated 09-26 refit comparison.
- Every candidate must run the predictor with `oracle_guard` on and nonempty `_OG_ITEMS` globally AND per-family; no channel-off or `_OG_ITEMS: []` counter.
- Priority 1: test narrow MILK/WOOL/STRAWBERRY sale-slot mutations against tuned copies; compare moving existing sells vs adding orders under the ten-slot limit.
- `_CA_MARGIN=-22` (ladder1 iteration 92: +$43 a game, p=7e-11) and `_SR_MARGIN=14` (ladder1 iteration 71) are in the seed since the 07:50 UTC 09-29 launch; do not re-propose them.
- Priority 3: test `_CXTB_MIN_REVENUE` 9,000 -> 7,000 for zero-to-ten tomatoes; completed oracle-off tests favored 7,000 and worsened at 6,000. Ichika's 10-to-18 needs separate route ownership.
- Tomato route ownership must budget seed cash, watering, harvest labour and storage; `_CXTB_OUR_UNITS` is not a planting command.
- Priority 4: test geese/egg production separately, net of goose/coop purchases, wheat feed, labour and market proceeds.
- Try EGG/CARROT guard expansion separately from the initial sale-slot and carrot-margin tests; `_OG_BATCH=4` does not cap strong-score sales.
- `score_4` is not a probability; forecast only after 256 completed turns and at most 96 ahead, never from hidden rival state or future shops.
- Use small evidence-led mutations in this same loop, not offsite LLM play/training; fixed replays diagnose but reactive paired games in both seats and fresh seeds decide.
- Aspen Vale's 44 ladder games after 20:16 UTC 09-28 (19W-25L, rating 2,206 -> 2,175), exact money per product
  from the environment's interpreter (0 mismatches; `shinka/champions/evidence/aspen_vale_20260929/`):
  - The 19 wins against strong rivals are mirror races: no tomato, land or hire difference, median +$81.
  - 12 of the 25 losses are by less than $600: milk, strawberry, wool and carrot sale races, 8 of them as seat 1.
    Sale timing and order slots against engine copies decide these, not production.
  - 8 losses by $5k-$24k to rivals rated 2,170-2,560 are economy scale. Per loss over all 25: wheat -$2,044,
    eggs -$1,151, strawberries -$776, tomatoes -$768, carrots -$741, wool -$538; ahead only on melons (+$818),
    and we spent less on hires (+$368), seeds and animals (+$387) and land. In 4 of the 8 the rival sold 89-102
    tomatoes and we sold none; in 2 we bought more land than the rival and still lost.
  - So against this band the engine under-invests: more wheat and geese early and a tomato plot are the
    hypotheses (Crops, Herd, Tomatoes islands). Check every such edit against Aspen's recorded wins in the plan,
    since a production change also moves the mirror races we win now.

- Since iteration 31 the gauntlet also plays each of Alder Ford's 24 lost or tied games against the
  rival's recorded moves (replay_<episode> in the per-opponent results). These games tell whether an
  edit would have saved a real ladder loss.
- The target since 09-26 19:30 UTC: Alder Ford, the Opening champion on the ladder, stands at about
  2,080. Its rivals there play agents close to our engine, and most of its losses were by a few
  hundred dollars. The gauntlet now also plays its 22 lost and 2 tied seeds from both seats (48
  games, 28 of them against tetsutani_demand). What moves the rating is an edit that wins a few
  hundred dollars in these near-mirror games without losing to the V57 family.
- Aim at layers that act. The knowledge section lists which engine layers changed actions in 48
  games and which never did: an edit of a dormant layer's parameters (carrot planner, race layers,
  opening tape, yarn reorder) or of a load-time table changed 1-3 of 230 games in run ladder1, unless
  it woke the layer (the carrot edit did, and lost).
- Do not spend an edit on copying another island's promoted settings. The loop mixes the islands
  every 12 iterations: every island is offered the champion of the island that gains most over the
  seed, and the gauntlet decides. A copy proposed as an edit is refused anyway when the same settings
  were already played on any island (iteration 15: two models proposed the guard for Market and both
  were refused). Propose settings nobody has played yet.
- Herd island: V9_HERD_* can never change play (the layer is never called), so do not edit them. The
  levers are HERD2 (_HD2_*, the tape's goose purchase on days 8-15 turned into cows or sheep by expected
  value; acts in 6 of 48 games at its defaults) and COWSWAP (_CS_*, 2 of 48). Rivals who beat Alder Ford
  bought cows where the engine bought sheep or geese, but a cows-only HERD2 (_HD2_OPTIONS COW, ratio 1.0,
  no minimum gain, future demand counted) lost: -$32 a game in iteration 31 and -$337 a game against the
  recorded rivals of Alder Ford's 72 games. Loosening HERD2's thresholds is refuted. Better uses of a Herd iteration: tune an active
  layer that touches animals (feed/fertilizer economics _R85_*/_R88_*, the _CA_* layer, _SM_MILK_SHOP).
- Promoted so far: the oracle guard on MILK/WOOL/STRAWBERRY at score 0.5, batch 4, keep 2, price ratio
  0.75 (Oracle, +$31 a game), _SR_MARGIN 12 (Endgame, +$9), _CXD_BUDGET 1050 (Opening, +$2). On 260
  fresh validation games the guard with _CXD_BUDGET 1050 gained $53 a game over the plain engine
  (92 games better, 49 worse) and beat the plain engine head-to-head 14-0 with 8 ties.
- Rejected on the stronger line: _SR_MARGIN 12 on top of the guard and _CXD_BUDGET (75 games better,
  42 worse, but mean -$22: a few large losses against haideptry_shepherd), _ADV_FRONT on (51 better,
  177 worse), _CA_BUFFER 12 (22 better, 60 worse) and 6 (4 better, 3 worse), _CA_FEED_DAYS 2 (24
  better, 69 worse, -$100 a game), _CH_SHED 90 (6 better, 27 worse), _BD_CAP 40 (6 better, 6 worse).
- Oracle guard: its _OG_* settings are graph parameters, edited as {"parameters": {"_OG_SCORE": 0.55}},
  not engine_parameters. Its thresholds sit on a flat optimum: _OG_STRONG_SCORE 0.7, _OG_PRICE_RATIO
  0.8 and _OG_TO_STEP 714 changed 2-22 of 230 games with no gain. Do not nudge them further in small
  steps (_OG_KEEP 3 and _OG_TO_STEP 712-716 were proposed again and again). Untested and larger:
  _OG_SCORE 0.55-0.6; _OG_MAX_ORDERS 1 or 3; _OG_FROM_STEP later than 256; EGG back in the item list
  alone. The guard loses a little against V57 (4-4, -$43) and pilkwang.
- Oracle, untested: our full market channel (the champion's oracle_frontrun, town cadence and
  liquidation stages) on top of the engine. It rewrites the engine's order list, so expect large
  changes in both directions.
- Endgame: _SR_MARGIN 8 -> 12 helped; the guard's hours (_SR_HOURS) and margins beyond 12 are untested.
  The capacity harvest/sell layer (_CH_SELL, _CH_SHED) acts in 29 of 48 games.
- Ready-stock advancing (_ADV_*) is the biggest lever found so far. _ADV_LOOK 3 -> 4 changed 197 of
  230 games: +$226 a game head-to-head against the champion and +$124 against tetsutani_demand, which
  is what the near-mirror ladder games need, but about -$95 against abo_v55, abo_v57 and the Forecast
  (p = 0.48). Untested: _ADV_LOOK 4 limited to some items (_ADV_ITEMS) or to part of the game
  (_ADV_FROM / _ADV_TO), _ADV_LOOK 2, _ADV_PROTECT, _ADV_BOOK. _ADV_FRONT on is refuted: it made 176-177
  of about 228 changed games worse, twice.
- Market, active and untested: the order-slot priority (_OR2_*, 24 turns a game), quote reordering
  (_R37_ADAPTIVE, _R37_QUOTE), the model-based lead seller's items (_MPX_ITEMS), hour-window and dawn
  lead sells (_MP_*, _DP_*, _EV_*).
- Crops, active: input planning (_R51_INPUT_CROPS, _R51_INPUT_MAX_WORKERS, 64 turns a game), feed and
  fertilizer economics (_R85_*, _R88_*), fertilizer timing (V9_FERT_FIRST_DAY 14 -> 16 changed 77 games
  at +$24 mean but lost more games than it won), the _CA_* layer (41 turns a game in 30 of 48 games).
- Opening: the engine's opening comes from _PIPE_MODE ('EarlyCycle'); it beats the 13-wheat openers
  20-0, so do not weaken it. Wheat buy-the-dip (_BD_*) and the cash reserve (_CXD_*) act every game.
- Rival counters (knowledge: "The rival counter"). The first counter to try is the one the recorded games
  support: {"channels": {"rival_counter": true}, "counters": {"mirror": {"_EV_H": 12, "_DP_H": 12, "_MP_H": 12,
  "_ADV_LOOK": 4}}} on an old-engine island (leave out _ADV_LOOK where the champion already has 4). Then per
  class: what wins against nsell_openers (they are the Forecast family and the 09-27 engine: sale timing that
  reacts to their selling, the guard's _OG_* per class) and against other openings. Counters change play only
  against their class, so a counter edit's changed games are that class's games.
- The public engine's 09-27 version beats all our graphs 18-2. On the Island-Next-* islands (that engine as
  the backbone): start from its mirror, find its layers that act, and give it our guard and rival counters.
- Tactics (channel tactic; knowledge: "The tactic stage") are for decisions no constant can express. Untested:
  (1) with the rival emulator on, the rival's orders of this turn are known: when it sells a product we hold,
  our sale of it in an earlier slot of the same turn clears before its units land, and a sale of ours right after
  its sale gets the lowered prices (hold it back instead?); (2) the rival's farm is public: its crops (crop,
  planted_day, yield_units) and animals are its coming sales, so our stock of the same product can go first;
  (3) the last day: final cash decides, and stock left in the shed at step 719 is worth nothing; (4) rivals no
  public engine reproduces (classes other_opening, other, nsell_opener; info['rival_family'] with the
  rival_counter channel on): their visible farm is the only forecast. Keep a tactic narrow (a few dozen lines,
  acting only where it means to), and send the whole new source when improving one.
