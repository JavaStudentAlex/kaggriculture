# Verified knowledge for ladder-engine graph evolution

Loaded into the mutation, judge and supervisor prompts by `highcpu_island_evolution.py
--knowledge evolution_knowledge_ladder.md`. Only facts verified in the engine, in ladder replays
or in paired arena games belong here, with sample sizes and dates. Engine: kaggle-environments
1.32.7.

## Aspen Vale evidence and bounded run constraints

For this authorized mutation run the baseline is Aspen Vale's `tetsutani_demand_0927`,
`_SR_MARGIN=12` and rival emulator with the reference 09-13 predictor. Unlike the submitted
Aspen Vale (guard off, zero forecasts), **every candidate must run the predictor and keep
`oracle_guard` on with nonempty `_OG_ITEMS` in all rival families**; the local queue's two
`_OG_ITEMS: []` engine-family overrides are not suitable for this run. The older seed and
oracle-off results below are historical, not permission to remove the guard here. Source:
`shinka/champions/evidence/aspen_vale_20260928/README.md` and `REVIEW_20260929.md`.

- **Sale slots / tuned copies:** 17 of Aspen's first 31 losses were classed as mirrors;
  near-copies often left the public engine in market order placement or quantity. The
  engine interleaves both ten-entry lists by slot, unit by unit; earlier sales can capture
  better prices. The guard only adds sales when free stock and a slot remain; it does not
  optimize existing orders. Its estimated six flips / +$5,983 on 31 selected losses were
  not continuing games: a full ten-slot list produced a credited sale in the estimator
  but no added guard order. Test executed fills and final post-emulator positions.
- **Carrots:** `_CA_MARGIN -15 -> -22` on the oracle-off Aspen line changed 36-31-0 to
  38-29-0 across 67 recorded opponents (+$64.3 mean margin); on 31 lost seeds in both
  seats against substitute live opponents it changed 51-11-0 to 53-9-0 (+$109.7).
  Four wins were gained and two prior wins lost in the 67 replays. These are not
  oracle-on interaction results; the 62-game six-proxy suite started at 51 wins.
- **Tomatoes / geese:** Aspen's 31 losses had net tomato sales -$32,237 (-$1,040/loss)
  and egg sales -$16,085 (-$519/loss) vs rivals; even its 36 wins had egg sales
  -$14,169 (-$394/win). The estimated tomato expansion cost is about $7,500.
  `_CXTB_MIN_REVENUE` controls a zero-to-ten-plant investment and rejects existing
  tomato routes; Ichika's 18 vs 10 plants is a separate route-capacity problem, not a
  lower-gate case. Geese cost $300 each, require a coop and ongoing wheat feed; check
  labour, land and cash before inferring profit from egg volume.
- **Forecast / comparison limits:** the reference 09-13 model forecasts only after
  256 completed turns and at most 96 turns ahead. `score_4` is the maximum predicted
  log1p sale across four horizons (or a calibrated score), not a probability or exact
  volume. `_OG_BATCH=4` is not a hard cap: at score >= `_OG_STRONG_SCORE` (0.6 in the
  reviewed guard) it sells all free stock. Neither replay-only private rival state nor
  future shop draws can enter live decisions. The 09-26 calibration used 600 feed15-era
  games, no held-out games in the all-games fit and no positive egg/tomato examples.
  Round 3 labelled `guard26` reused the reference bundle without calibration; it is
  **not** a refit comparison. Compare distinct verified bundle/checkpoint/calibration
  identities on paired reactive games before switching models.
- **Evidence boundary:** fixed rival action tapes cannot react to changed actions;
  the 158/168 predicted-sale hits are conditional on proposed firings in 31 selected
  losses, not ladder-wide precision or six replayed wins. Changed farm actions can
  also shift later shop RNG draws on the same seed. Require both seats, held-out
  seeds/opponents, errors and gained/lost wins before promotion.

## Engine facts (verified)

### Completed Aspen guard tests, checked 2026-09-29 06:06 UTC

- Reference 09-13 guard ALWAYS on: the 67-game fixed-tape backtest changed 36W-31L-0T
  to 38W-29L-0T, mean margin -167.9 to -145.2; 62 lost-seed proxy games changed
  51W-11L-0T to 53W-9L-0T, mean 709.1 to 758.0.
- Fresh-seed pool including Aspen head-to-head, 180 games per graph: Aspen
  157W-13L-10T vs guard 165W-10L-5T, mean margin 3436.2 to 3455.1. These games
  now inform development; do not claim they remain an untouched final test set.
- Tomato gate 7000 (oracle off) changed backtest to 38W-29L and proxies to 55W-7L.
  Gate 6000 worsened them to 32W-35L and 47W-15L: prioritize 7000, not 6000.
- The completed guard26-labelled results equal the reference because both used the
  same old-model bundle. They are NOT evidence about September 26 weights.
- The oracle-required branch checks live model availability every turn and requires
  forecasts after warmup; a broken predictor makes a game fail instead of silently
  using the backbone. Tactics can economically veto a sale; always-on prediction
  does not mean forced selling. No empty-item family counters or enormous threshold opt-outs.

- 720 turns (30 days x 24 hours); both seats start with $3,000; final cash decides.
- Market orders: at most 10 entries per turn. Both seats' lists are walked index by index: the
  orders at index i of both seats clear unit by unit in lockstep, and index i+1 starts only after
  both finish. An entry the engine cannot parse (an empty `[]`) takes its index and does nothing,
  so the ladder engines use `[]` as a deliberate one-slot delay; dropping one moves every later
  order a slot earlier (measured: a few dollars a game).
- Prices move with the product's market stock (base price at stock 10,000; `engine_contract.py`
  has the curves). Base prices: wheat 25, carrot 35, tomato 60, strawberry 120, melon 250, egg 50,
  milk 160, wool 200, fertilizer 100. Selling adds 1 unit of stock per unit sold (a sale at $1
  adds none). A surplus of about 59 wool, 62 strawberries, 76 milk or 158 melons drives the price
  to $1; wheat never falls below about $17 and eggs still fetch about $38 with 1,000 extra.
- The town takes stock after the market on steps with step % 4 == 0: each open shop takes 1 unit
  of each product it buys (2 for the one-product YARN_STORE and PET_CAFE), and the town centre 1 a
  day of every product except fertilizer, which nobody buys. What each shop buys (engine
  `SHOPS`): BAKERY egg, wheat; PIZZA_SHOP milk, tomato, wheat; BRUNCH_SPOT egg, wheat,
  strawberry; ICE_CREAM_SHOP strawberry, milk, wheat; SMOOTHIE_SHOP strawberry, milk;
  FARMERS_MARKET wheat, carrot, tomato, strawberry; YARN_STORE wool; PET_CAFE carrot. No shop
  buys melon (only the town centre, 1 a day).
- A new shop opens every 3 days (days 3, 6, 9, ..., at most 8), drawn with replacement from a
  per-day RNG that also spawns weeds (one draw per empty tile on both farms), so either farm's
  layout changes the later shops.
- An animal not fed on two consecutive days escapes at the end of the second day. FEED uses 1
  wheat from the acting unit's own bag. Goose $300 (egg daily), cow $400 (milk every 2 days),
  sheep $500 (wool every 3 days).
- Hands are hired daily at Fibonacci cost (1, 1, 2, 3, 5, ... dollars).

## The production engine (the backbone of every candidate)

`tetsutani_demand`: tetsutani's public notebook "Demand-Preserving Turn Sale Timing" (Apache-2.0,
cha22 lineage), the strongest public agent we have (round-robin below). One 7,482-line file: the V5x
base (yhay81 shop-router route tapes on Thomas Tschinkel's chassis; at step 144 a 719-step tape is
picked from the first two shops, `_R110_OLD_SHOPS` / `_V92_TABLE`) with 76 layer functions stacked on
it: each saves the previous `agent` and edits its action. Its opening comes from `_PIPE_MODE`
('EarlyCycle', the first 96 steps); on the ladder its step-0 wheat order is BUY 5. Two of the rivals
who beat Rowan Glen (AkiraIshikawa, 2155, and Shane Thivaharraja, 2188) run exactly this agent: it
reproduces their recorded games move for move.

The seed graph plays this engine as-is (the farmer, hands and market channels pass its action
through, the oracle guard is off): its games equal the public agent's. Our layers act only where a
channel is switched on.

### Which engine layers act (engine_activity.py, 48 games of the gauntlet plan, 2026-09-26)

A layer acts on a turn when the action it returns differs from the one its parent gave it. Per
layer: turns acted per game (mean over 48 games), games in which it acted, the parameters it reads.
- input planning (line 2281): 64 turns/game, 45/48 games: _R51_INPUT_CROPS, _R51_INPUT_MAX_WORKERS, _R37_HINGE_GAIN
- _CA_* layer (line 4254): 41, 30/48: _CA_BUFFER, _CA_CASH, _CA_DROP, _CA_FEED_DAYS, _CA_FROM, _CA_MARGIN, _CA_RESCUE, _CA_TO
- feed/fertilizer economics (line 2611): 35, 48/48: _R85_FEED, _R85_FERT, _R88_HORIZON, _R88_PHASE
- order-slot priority (line 4456): 24, 48/48: _OR2_CAP, _OR2_SLOT_H, _OR2_SLOT_MARGIN, _OR2_SN_H, _OR2_SN_ITEMS, _OR2_SN_K
- quote reordering (line 1886): 22, 48/48: _R37_ADAPTIVE, _R37_QUOTE, _R37_HINGE_GAIN
- tomato/crop worker layer (line 1407): 22, only 6/48: CROP_MIN_PRICE, V13V_SKIP_DAYS
- model-based lead seller MODELPX (line 7063): 19, 48/48: _MPX_ITEMS
- yarn reorder gate (line 5833): 12, 48/48: _V44Y_REORDER_GATE
- best-response order book (line 7242): 11, 48/48: _CXD_BUDGET, _CXD_FIXED, _CXD_FROM
- ready-stock sale advancing (line 6282): 10, 47/48: _ADV_BOOK, _ADV_FROM, _ADV_FRONT, _ADV_ITEMS, _ADV_LOOK, _ADV_PROTECT, _ADV_SUBTRACT_DEBTS, _ADV_TO
- courier (line 3011): 8.5, 48/48: V9_COURIER_FROM_HOUR, V9_COURIER_ITEMS
- fertilizer (line 3319): 6.2, 48/48: V9_FERT_AGES, V9_FERT_CROPS, V9_FERT_FIRST_DAY
- capacity harvest/sell (line 4695): 6.0, 29/48: _CH_SELL, _CH_SHED
- hour-window lead sells (line 6881): 6.0, 48/48: _MP_H, _MP_HOURS, _FX_ITEMS, _FX_QUOTE_WIN
- wheat buy-the-dip (line 6959): 6.0, 40/48: _BD_CAP, _BD_DEADLINE, _BD_ITEM, _BD_MIN, _BD_WIN
- rival-flow / event lead sells FLOWPX (line 6706): 3.8, 41/48: _EV_H, _EV_HOURS, _FX_* (the _FX_ flow trigger is off at _FX_FLOW_MIN 999)
- dawn-window lead sells (line 6808): 3.7, 41/48: _DP_H, _DP_HOURS
- night shed guard (line 4811): 3.2, 36/48: _SR_HOURS, _SR_MARGIN
- herd swap _HD2 (line 5156): 3.0, only 6/48: _HD2_*; cow swap _CS (line 5296): 0.7, 2/48: _CS_*
- rare: _Y_CFG (4/48 games), _SM_MILK_SHOP (13/48), _V231_CAP (2/48)
- Layers with no parameters also act (lines 1511, 6062, 1721, 7467, 7396, 7323: 7-86 turns/game).

Never acted in the 48 games (their parameters change play only if an edit makes the layer act):
the carrot planner V9_CARROT_*, the opening tape V9_OPENING_STEP0 / V9_OPENING_TAPE, the race layers
V9_RACE_*, V9_RACEGATE_*, _RACE_*, the yarn-town reorder _Y_HOURS / _Y_ITEMS / _Y_MARGIN / _Y_MIN_DAY,
the weed-lag replay _E343_WL_*, APPLY_TIMING, _V219_FERTILIZE.
Read only while the module loads (they shape the route tapes and settings, so an edit changes only
the towns whose tape they touch): _R110_OLD_SHOPS, _V92_TABLE, _R42_OPENING, _SETTINGS,
FRONT_RUN_ITEMS, V9_RACEPX_*. V9_HERD_* can never change play: the public agent defines that goose-swap
layer (`_v9_herd`) but its wrapper never calls it.
Herd layers that do act: HERD2 (_HD2_*) turns the tape's goose purchase on days 8-15 into cows or sheep
when their expected value is at least _HD2_RATIO (1.3) times the geese's and $600 more (_HD2_MIN_GAIN),
and only if the farm has no coop yet; COWSWAP (_CS_*) weighs the tape's first cow purchase (days 6-8)
against geese. A sheep purchase cannot be turned into cows by a parameter: the tape harvests sheep every
third day, so cows placed there would lose milk.
Measured in run ladder1: edits of _CS_*, V9_HERD_* (four edits), _FX_FLOW_MIN and
_SETTINGS.terminal_liquidation changed 0-3 of 230 games; V9_CARROT_LAST_DAY 23 -> 25 woke the carrot
planner and lost 24 of the 27 games it changed.

## The gauntlet's opponents (public agents; exact = reproduces a recorded ladder game move for move)

| bundle | what it is | ladder evidence |
|---|---|---|
| tetsutani_demand | the engine itself (mirror): "Demand-Preserving Turn Sale Timing", step-0 wheat BUY 5 | exact: AkiraIshikawa (2155), Shane Thivaharraja (2188); Yusuraume (2055) and BorisV, who beat and tied Alder Ford; ~ J.Moriuchi (372 steps), Shangshang Zhang (226), Igor V (150) |
| haideptry_2965 | "The 2965 Master Hybrid Engine" (the notebook's version of 09-25): V57 + an order-book evaluator that permutes the cash-sale slots; step-0 buy 20 / sell 15 wheat (>= $1,050 left after step 1) | ~ Matin Urdu, Hector Valverde, sneaky6767 (1920) |
| haideptry_2965_0926 | the same notebook's version of 09-26 13:54 UTC, same opening | exact: ADRIANO ALMEIDA (2031), who beat Alder Ford; ~ Rohan L (241 steps), edwinis (159) |
| haideptry_shepherd | "Shepherd's Ledger": V57 family, opening buy 8 / sell 3 | ~ z7777 (2437), Rashid K. (2175) |
| abo_v57 | Ahmed Berat Ozer V57 | exact: PUN |
| abo_v55 | V55 | exact: Ansh Agarwal (1808) |
| abo_v43 | V43 | exact: Dariush Afshar (1453) |
| tetsutani_shape_shop | tetsutani's earlier notebook "Shape the Shop Work the Pasture" (09-06): the 13-wheat opening (buy 13 at step 0, sell 9 at step 1) | exact: wuy1hao, random_numb, phi; ~ YuRuiZe (623 steps), williams (106), all of whom beat Rowan Glen or Linden Brook |
| abo_v57_open13 | V57 with the same 13-wheat opening (our stand-in before tetsutani_shape_shop was found; it matches none of these rivals past step 0) | the 13-wheat losses that no public agent reproduces past step 1 |
| leoprovorov_forecast | "Four-Turn Forecast", opening buy 8 / sell 3 | ~ Rashid K. (2175); King-damon (216 steps) and AI After Hours (150), who beat Alder Ford |
| hanifnoerrofiq_pioneers | "Pioneers of Kaggle Town", opening buy 6 / sell 5 | ~ julien gaza (2091) |
| robust_economy | nihilisticneuralnet "Population-Robust Economy" (Metav4 lineage) | ~ AlexMoura2026 |
| pilkwang_sep | "Structured Economic Policy" | ~ Zhong Lyu, edwinis |
| mohui | plain Mohui v66 (our old backbone) | |

## Measured (Colab arena, 2026-09-26)

Round-robin of the public agents, 20 seeds per pair, seats alternating (W-L, mean margin of the row):
- haideptry_2965 126-14 (+$5.9k), haideptry_shepherd 122-18 (+$5.8k), abo_v57 105-35 (+$5.3k),
  abo_v55 82-58, robust_economy 64-76, abo_v43 41-99, pilkwang_sep 20-120, mohui 0-140 (-$20.7k).
- Among the V57 family the margins are small: 2965 vs Shepherd 11-9 (+$0.2k), 2965 vs V57 17-3
  (+$1.0k), V57 vs V55 20-0 (+$0.5k). Against Mohui every one wins by ~$22k.
- Round-robin 2: tetsutani_demand (the engine) beats 2965 18-2 (+$1.1k), Shepherd 19-1, V57 19-1,
  Four-Turn Forecast 18-1, Pioneers 18-2 and the 13-opening V57 20-0, and Rowan Glen 20-0 (+$17.6k).
- The seed graph (the engine as-is) in this gauntlet: lost ladder seeds 97W-13L-10T (+$4,037 a
  game), random seeds 76W-5L-9T (+$4,043). Its margins against the V57 family are small, so most of
  an edit's effect shows as a few hundred dollars per changed game.

Run ladder1 (2026-09-26, iterations 1-11), promoted edits:
- Oracle island: oracle_guard on for MILK/WOOL/STRAWBERRY, _OG_SCORE 0.5, _OG_BATCH 4, _OG_KEEP 2,
  _OG_PRICE_RATIO 0.75: 76W-47L of 230 games, +$31 a game, p = 0.011 (the looser first try, score 0.45,
  batch 6, keep 0, ratio 0.6: 90W-66L, +$29, p = 0.065). The predictor helps most against the mirror
  (11-1, +$54) and the 13-wheat opener (15-11, +$60), least against V57 (4-4, -$43).
- Endgame island: _SR_MARGIN 8 -> 12 (night shed guard): 81W-38L, +$9 a game, p = 0.0001.
- Opening island: _CXD_BUDGET 800 -> 1050: 40W-22L, +$2 a game, p = 0.03.
Rejected with many changed games: V9_FERT_FIRST_DAY 14 -> 16 (33W-44L, +$24 mean, p = 0.25).
Iterations 12-28 promoted nothing. The first island mixing (after iteration 18) gave Herd, Crops and
Market the Opening champion (+$31 a game over their old champions); Oracle and Endgame rejected it.
Rejected on the stronger line: _ADV_FRONT on (twice: 51W-177L and 52W-176L), _CA_FEED_DAYS 1 -> 2
(24W-69L, -$100 a game), _ADV_LOOK 3 -> 4 (104W-93L, +$13, p = 0.48: +$226 a game head-to-head against
the champion and +$124 against tetsutani_demand, but about -$95 against abo_v55, abo_v57 and
leoprovorov_forecast). Nearly inert (at most 22 of 230 games changed, no gain): _OG_STRONG_SCORE 0.7,
_OG_PRICE_RATIO 0.8, _OG_TO_STEP 714, _BD_CAP 40, _CA_BUFFER 6, V9_HERD_MIN_MILK_SHOPS 2,
V9_HERD_MAX_EGG_SHOPS_COW 1.
Each island keeps its own champion; an edit that adds another island's promoted edit to this
island's champion is valid and tests whether the gains add up.

Rowan Glen (our Mohui-based submission, 2026-09-25) against the pool, 28 lost ladder seeds x 2
seats + 40 random seeds (96 games each): 0-96 against V57, V55, Shepherd, 2965 and Robust Economy
(-$18k to -$20k a game), 7-89 against V43, 16-80 against pilkwang, 74-22 against Mohui.

On the ladder (57 games, 2026-09-25/26) Rowan Glen went 8-23 against the V57 family and 21-5
against everyone else. Seven losses of $27k-$71k came against rivals who bought 13-14 wheat at
step 0: our step-1 purchases left ~$20, the day-1 hire got 2 hands, a cow starved on day 2 and the
farm never recovered. The engine does not have this weakness: it beats the 13-wheat opener 20-0.

## Against the ladder's top players (Kaggle notebooks, 09-27)

The 80 best games of 09-26 (average rating 3,030-3,083), both seats: our graph took one top player's seat on the
game's seed against the other player's recorded moves (160 games per graph; benchmark/results/top_bench_2026-09-26).
The old public engine (and Cedar Ridge, whose counter acts only against mirrors) earned $114.5k a game, a median
$3.4k (mean $8.7k) more than the top player it replaced earned in the real game, and won 93-67 against the recorded
moves; the 09-27 engine earned $111.1k (median +$1.9k), 87-73, and sold fertilizer worth $16k a game at the quote
where the old engine sold $36k. The recorded moves do not react: they lost a median $12.6k against their real game,
so these numbers flatter us. What the farm produces is not where the top players are far ahead of us.

## What the top players' farms look like (plan mining, 2,990 games of 09-22..26)

The ladder's best players (the daily top-game datasets) do not play our route tapes: only 9% of their seats come near
any of the engine's 41 tapes in what they buy by days 6, 12, 20 and 29. Mean per seat, their winners against our own
ladder games at the start of day 20: geese 5.7 against 2.5, tomato tiles 11.0 against 1.6, carrot tiles 5.8 against
0.7, strawberry tiles 23.3 against 33.0, land quadrants 3.5 against 3.2, cows 8.2 against 7.3 (day 25: carrots 12.8
against 9.8, strawberries 10.5 against 17.0). They diversify into geese, tomatoes and carrots where the tapes grow
strawberries; which of these a layer or a tactic can add is untested (benchmark/results/plan_mining_2026-09-26).

## Alder Ford on the ladder (the Opening champion, submitted 2026-09-26 14:50 UTC)

Alder Ford is the Opening island's champion of iteration 11 (the guard on MILK/WOOL/STRAWBERRY plus
_CXD_BUDGET 1050). By 19:00 UTC it had played 70 ladder games: 46W-22L-2T, rating 2,078 (our earlier
best submission reached 1,378). It won its first 17 games, up to a 1,966-rated rival; against rivals
rated 1,900-2,250 it is about even. Those games are near-mirrors: 18 of its 22 losses were by less
than $1,800 (median about $600); the largest were $9,275 (syouya tobita, 2965 opening), $6,771 (Nikita
Makarov, buy 18 / sell 13) and $4,798 (Alexander Sokolov, buy 3 wheat and five hires at step 0).
The 24 lost or tied games by the rival's step-0 opening: buy 5 (tetsutani) 6, buy 20 / sell 15 (2965)
5, buy 8 / sell 3 (Forecast family) 4, buy 8 / sell 8 3, and one each of buy 6 / sell 1, 7 / 2, 10 / 10,
18 / 13, 13 and buy 3 with hires. Three rivals run a pool agent exactly (tetsutani_demand twice,
haideptry_2965_0926 once): even the plain engine beats or ties our champion on some seeds.

After 100 games (22:29 UTC, 58W-31L-11T, rating 2,086; `shinka/champions/evidence/alder_ford_20260927/`):
- 10 of the 11 ties are against the public engine run exactly (all 719 moves). Alder Ford plays it move
  for move, so both farms earn the same every day. A mirror is beaten only by selling first.
- All 15 losses by at most $600 are against rivals with exactly our herd (the same route tape). They are
  sale races, lost after day 20 on milk, strawberries, wool or wheat. The rivals that leave the public
  engine win by selling first or more: 2 wool where the engine sells 1 (test_money, from step 468),
  strawberries ordered before wheat in the same turn (Bhaskar #2, step 385; both seats' orders clear
  slot by slot).
- The 16 losses above $600 are other strategies: 2965-family wheat trading, more geese (eggs), carrots,
  more cows, early hiring. No sale-timing setting changes them; leaning the herd to cows cost more
  elsewhere (-$337 a game on the replays).

## The rival counter: counter-settings per rival class (since 2026-09-27)

The optional `rival_counter` turn stage (channel `rival_counter`; an edit that switches it on inserts it)
compares the rival's public farm with ours every turn. On the same seed a copy of our engine has our money
and our farm step for step, so by step 93 the rival falls into one class, fixed for the game:
- `mirror`: equal to us past step 92. In Alder Ford's 132 classified ladder games: 53 of the 54 copies of the
  public engine (its near-copies depart only at steps 160-661, or never) and 5 others.
- `nsell_opener`: its money first falls behind ours at step 92 by the price of 3 wheat (-$87 to -$90): a
  "buy N / sell N-5" opener that sold its 3 spare wheat at step 0 while our engine sells them at 92. All 23
  Forecast-family games, 13 of 20 2965-family games, 15 others; the public engine's 09-27 version opens so.
- `wheat92_seller`: the reverse (it sells at 92 what we sold at step 0); `other_opening`: differs before
  step 92 (13-wheat, "buy 8 / sell 8", some 2965 variants); `other`.
From its class on, a class's `counters` apply: values for switchable engine constants (118 of the old
engine's 120 parameters are read at call time) and `_OG_*` guard parameters. Until then, and for classes
without counters, the graph's own settings play. Edit: {"channels": {"rival_counter": true},
"counters": {"mirror": {"_EV_H": 12, "_DP_H": 12, "_MP_H": 12}}}.

What the classes are worth (Alder Ford's graph against the recorded moves of its rivals; + lead 12 =
_ADV_LOOK 4 and _EV_H/_DP_H/_MP_H 8 -> 12):
- mirror: + lead 12 gained +$717 a game (20 better, 1 worse; 5-6-10 -> 17-4-0) on 82 games and +$592
  (21/1; 8-10-4 -> 19-3-0) on 49 newer ones; _ADV_LOOK 4 alone +$81 and +$200. Against the public engine in
  the arena + lead 12 wins 20-0 where Alder Ford draws 8 of 20. Most of the win/loss gain is here.
- nsell_opener: + lead 12 +$244 (26/8) and +$195 (5/2) against recorded rivals but no more wins (22-12 ->
  23-11; 2-5 -> 1-6), and against reacting Forecast agents in the arena it lost $78-98 a game.
- other_opening: + lead 12 +$181 (5/3) and +$125 (5/2), no result changed on balance.
So a counter per class can take the mirror gains without the costs against the others.

Iteration 51 played lead 12 for every rival on Island-Market (engine_parameters _EV_H/_DP_H/_MP_H 12): rejected,
164 games better and 130 worse, +$126 a game, p = 0.054. Better where the rival plays our engine
(tetsutani_demand 47 better / 1 worse, +$494; the incumbent 18/2, +$404; 18 of the 24 replays of recorded
rivals; haideptry_2965 40/14, +$138), worse against agents that react (abo_v57_open13 5/37, -$138;
leoprovorov_forecast 5/14, -$146; haideptry_shepherd 3/8, -$104; pilkwang_sep 0/4, -$196; abo_v55 3/12, -$74).
The mirror counter keeps the first group and leaves the second.

The mirror counter was then promoted: iteration 53 on Island-Opening ({"mirror": {"_ADV_LOOK": 4, "_EV_H": 12,
"_DP_H": 12, "_MP_H": 12}}: 134 games better, 6 worse, 259 unchanged, +$153 a game, p = 1e-15) and iteration 54 on
Island-Endgame (the same without _ADV_LOOK: 136 better, 4 worse). Only games against rivals classed as mirrors
changed: tetsutani_demand 93 better / 1 worse (+$448 a game), the incumbent 18/2 (+$404), replays 47/3 (56
unchanged); every other pool agent, the 09-27 engine included (an nsell_opener), played exactly as before. The
mixing carries this counter to the other islands, so do not propose it again. Open questions: the best lead
against mirrors (16 and 20 were played in iterations 57-58, below) and counters for the other classes.
In iteration 51 lead 12 for everyone gained against haideptry_2965 (+$138) and haideptry_2965_0926 (+$96) but
lost against leoprovorov_forecast (-$146); all three are "buy N / sell N-5" openers, so the class alone does not
separate them.

Birch Hollow (Alder Ford + _ADV_LOOK 4, on the ladder since 09-27 08:50) lost 28 of its 84 games by 17:00 UTC
(rating 2,044). By class: 10 against mirrors (all play the old public engine, 169-719 moves: the mirror counter's
target), 11 against nsell_openers (4 play the public engine's 09-27 version, 3 of them all 719 moves, -$834 to
-$1,367; 3 Forecast-like, 2 like statma, 1 like 2965, 1 unknown) and 7 against other openings that no public agent
plays (three of them -$7.8k to -$17k). Since the restart after iteration 55 these 28 are in the plan (both seats
against the stand-in, and their replays). The 09-27 engine is the growing threat: no graph of ours beats it yet,
and a counter against it needs something that separates it from the Forecast family within nsell_opener.

The public engine's 09-27 version (`tetsutani_demand_0927`, a pool opponent since the restart) beats every
graph we had 18-2 in the arena (Alder Ford, _ADV_LOOK 4, + lead 12, the Island-Market champion: $660-710 a
game). It drops the old engine's ready-stock and hour-window lead sells (_ADV_*, _EV_*, _DP_*, _MP_*) and
adds a library-based predictor of the rival's sales (_V92_P_*, _V92_Q_*), more race layers (_RACE_*), a
hybrid opening (_ALT_MODE) and rival-keyed routes. Islands named Island-Next-* evolve on it.

Iterations 55-58 (09-27 evening):
- The lead against mirrors: 12 is the best of 12, 16 and 20. Island-Herd took lead 16 (iteration 57) and
  Island-Crops lead 20 (58), both promoted over champions without a counter. Paired with Island-Opening (lead 12)
  on the same games, lead 16 is 35 games better and 91 worse against tetsutani_demand (-$44 a game), lead 20
  21/105 (-$236); wins against it 122-6 with 12 or 16, 118-10 with 20. Against the recorded moves of Birch
  Hollow's 10 lost mirror games longer leads win more (6, 7 and 9 of 10), but a recorded rival does not answer
  our earlier sales and the public engine does. Leads 10 and 14 are untested.
- The oracle guard on the 09-27 engine (Island-Next-Market, 55): rejected, 50 better / 51 worse, $0.
- _SR_MARGIN 8 -> 12 on the 09-27 engine (Island-Next-Production, 56): promoted, 165 better / 37 worse, +$9.
- The two engines side by side on the plan's 382 games against stand-ins (both seats; the replays are left out
  because a recorded rival fits the engine it played against): Island-Next-Production wins 320, Island-Opening
  323. The 09-27 line wins more against haideptry_2965 (53-3 vs 44-12), leoprovorov_forecast (24-4 vs 18-10),
  haideptry_shepherd (12-0 vs 10-2) and the 09-27 engine (8-2 and 14 ties vs 1-23: the old line loses $989 a
  game to it). The old line wins more against the old public engine (122-6 vs 98-30; +$704 vs +$357 a game), by
  its mirror counter.
- On a 09-27 graph the classes are relative to the 09-27 engine: its copies are `mirror`, and the old public
  engine should be `wheat92_seller` (at step 92 it sells the 3 wheat the 09-27 engine sold at step 0). The 09-27
  engine has none of the old engine's lead-sell constants (_EV_*, _DP_*, _MP_*, _ADV_*), so the old line's mirror
  counter does not carry over. A counter against wheat92_seller on Island-Next-* has to use the 09-27 engine's own
  sale timing: _RACE_* (the race layers, e.g. _RACE_HORIZON_MIRROR and _RACE_HORIZON_ESCALATED, 24), _SR_*, _OR2_*,
  _S738_* (look 4, steps 144-718), _S758_ITEMS, _V92_P_*/_V92_Q_* (horizon 48); all are switchable, so all can be
  counters. The 30 games the 09-27 line loses to the old public engine are the largest open gain there.

## The rival emulator: the rival's orders known before ours (since 09-28)

Every observation shows both farms in full (money, every tile with its yield, positions, land) and the market;
only the shed, the seeds and what each worker carries are private, and the environment is deterministic apart
from weeds and shop unlocks at the end of a day. So when the rival runs a public engine we have, the optional
`rival_emulator` stage (channel `rival_emulator`) runs that engine in the rival's place from our own view: it
rebuilds the rival's private stock with the environment's own rules, checks every step against the next
observation, and drops an engine at its first disagreement. On Birch Hollow's recorded games it emulated all five
rivals that ran a public engine (two the old one, three the 09-27 one) exactly for all 719 moves, and dropped the
one that left the old engine at move 418 at that move. In play it tells the old engine from the 09-27 one at step
92, where they first differ. `_EM_ENGINES` lists the engines (default both), `_EM_LOCK` the matched steps before a
prediction counts (12).
While one engine matches, the rival's orders of the current turn are known before we send ours:
- the rival counter's family `engine:<name>` applies (its own counters for an exact copy of that engine, next to
  the step-93 classes), e.g. {"channels": {"rival_emulator": true, "rival_counter": true}, "counters":
  {"engine:tetsutani_demand": {"_EV_H": 12}}};
- with `_EM_RACE` (default on) our market orders are rearranged against the rival's known orders: both seats'
  lists clear index by index, the two orders at one index unit by unit, so a sale queued behind the rival's sale
  of the same product gets the lower prices. The stage keeps exactly the same orders and moves ours to the slots
  that earn us the most in a simulation of the turn; nothing bought or sold changes.
About 40% of Alder Ford's rivals near 2,000 ran the old public engine exactly for hundreds of moves, and the
09-27 version's copies are spreading.

## The tactic stage: code the evolution writes (since 09-28)

Parameters only tune what the engine already does. The optional `tactic` stage (channel `tactic`, CONTROLS:
TACTIC) runs a Python function given in the edit, `tactic(obs, action, memory, info)`, on our action every turn,
after every other stage and before `sanitize` (legality) and the rival emulator's slot race. It sees the whole
observation (both farms, the market, our private stock), our action as the engine and our layers left it, a
`memory` dict kept for the game, and `info`: the rival counter's class (channel rival_counter), the public engine
the rival emulator matched and, while it matches, the rival's action of this very turn (`info['rival_action']`,
channel rival_emulator). It returns the new action, or None to keep it. Checked on 09-28: in a 40-step game a
tactic that sells the shed's milk first ran on all 39 turns without an error. The sandbox refuses imports,
attributes starting with an underscore, str.format, eval/open/getattr/print and top-level code other than
functions, literal constants and docstrings; a turn may spend 200,000 ticks (loop iterations, comprehension
elements, calls) and 0.25 s. A turn that raises or runs out of budget keeps the action, and after 20 such turns
the tactic stays off for the game. Validation plays the candidate for 30 turns and then calls the tactic at steps
96-719 on the last observation (moved to those steps): any error rejects the edit with the error text, so a retry
can fix it. A tactic is judged like any edit, on the games it changes: one that acts in a few games changes few,
and the first stage stops a candidate that changed at most four of its games.

## Land: the land_plot stage (since 2026-09-29)

- Our engine's 41 route tapes all buy the second quadrant (NE) at step 150 (day 6) and the third (SW) at step 265
  (day 11), in every game, and never the fourth (SE); no engine parameter moves a land purchase. The ladder's top
  teams (DSM 83% wins, DECEM, Vadim Vasilenko, Mother-Goose; tape mining of 09-22..09-28) all build the same bigger
  farm: day 9 = 3 quadrants, 15 wheat, 19 strawberries, 11 melons, 4 geese, 8 cows, 5 sheep, 10 hands; day 12 =
  3.4 quadrants (43% own all four), 28 wheat, 7 geese, first tomatoes. Aspen Vale's big losses were this gap.
- The land_plot stage (channel land_plot) is the lever: from _LP_DAY it buys SE (and SW first, early, when the
  tape has not bought it yet), hires _LP_WORKERS hands a day after the engine's own hires, and farms _LP_TILES SE
  tiles with _LP_USE (a crop, or GOOSE: coop + $300 goose per tile, a wheat a day each). It buys its own seeds,
  geese and feed and sells its produce. Costs: SE $4,000 (+$2,000 for SW early), a goose $300, feed ~$30 a goose a
  day, and each extra hand fib(hires so far today) = $89, $144, $233 with the tape's 10 hands. A goose lays from 4
  days after placing: 1 egg a day, 2 when also cared for, plus 1 fertilizer a day when collected; one hand keeps
  about 4 geese fully (feed, care, harvest, collect).
- Test games (cliproxyapi, 09-29): against the public 0927 engine every goose plot lost $5-24k of margin: that
  engine races a rival it recognises as its own clone (same quadrants, 90-95% of occupied tiles alike), and the
  fourth quadrant ends the recognition, so the rival leaves the race and earns $11-15k more. Hence _LP_MIRROR 0.9:
  no land while the rival's farm is that alike (Mohui-family rivals such as abo_v57 also look alike, so the plot
  stays off against them too). Set _LP_MIRROR 1.01 to buy against every rival. Wheat and tomato plots bought on
  day 12 earned less than the quadrant and the hands cost in those games.

## What a candidate must do to be promoted

It plays the same games as its island champion: every lost ladder seed (Rowan Glen's and Linden
Brook's losses, and since 09-26 19:30 UTC Alder Ford's 22 losses and 2 ties) against the bundle
that plays like the rival who beat us there, from both seats, plus random seeds against the pool,
plus head-to-head games against the champion, and since iteration 31 each of Alder Ford's 24 lost or
tied games against a replay of the rival's recorded moves (below): 282 pool games and 20 head-to-head. Since
the 09-27 restart also Alder Ford's 29 newer lost or tied games (from both seats against stand-ins,
and their replays) and 10 random seeds against the public engine's 09-27 version: 379 pool games; since the restart after
iteration 55 Birch Hollow's 28 lost games as well: 463 pool games. Since 09-28 a candidate is promoted when its
dollar test passes (below) or when the games whose result (win, tie, loss) changed improved significantly, and
never when it turns more results against us than for us: the ladder rates results. On the run's first 81
candidates the results test would have promoted 9 more (among them _ADV_LOOK 4 at iteration 24, results +23/-1,
and global lead 12 at iteration 51, +36/-2). The gauntlet is staged: 30% of the games are
played first, and a candidate that changed at most four of them, or made at least as many worse as better
in dollars and in results, stops there (rejected). Edits of layers that never act are therefore cheap to refute but still wasted. A rival no pool
agent reproduces for 100 moves gets the agent of its opening, else tetsutani_demand (Alder Ford's)
or haideptry_2965 (Rowan Glen's and Linden Brook's). Five 13-wheat losses of Rowan Glen and Linden
Brook moved from abo_v57_open13 to tetsutani_shape_shop on 09-26 at 19:35 UTC, when it was found.

Replay opponents (`shinka/champions/replay_opponents/`, tags replay_<episode>) play the rival's
recorded moves. They do not react, so they measure what a change to our play earns against the moves
that actually beat us. The champion replays the ladder game exactly: in all 24 its margin equals the
ladder margin to the dollar (e.g. -$6,771 against Nikita Makarov's replay, -$1,148 against J.Moriuchi's). The stand-in agents do not: on the
21 seeds without an exact public agent the champion beat them 13 times and lost 3 times. A replay game
therefore shows a candidate's effect on a real loss, and small late changes (sale timing, the guard,
herd choice) are measured most faithfully.

Backtest against the recorded rivals of Alder Ford's 72 games with strong rivals (2026-09-26 22:15 UTC,
one change each on the champion; `shinka/champions/evidence/alder_ford_20260926/README.md`):
_ADV_LOOK 3 -> 4 +$60 a game (49 better, 15 worse, p = 2e-5; +$143 against tetsutani-family rivals, +$64
against the 2965 family), the plain engine -$16, guard without MILK +$11, _OG_SCORE 0.6 +$6, guard stop
on day 21 -$2, cows-only HERD2 -$337. The gauntlet's stand-ins had judged _ADV_LOOK 4 a loss (iteration 24).
On 300 fresh arena games (validate1: 20 validation seeds against each of the 14 pool agents, plus 10 held-out
lost seeds from both seats) _ADV_LOOK 4 on the champion gained +$32 a game (158 better, 111 worse, p = 0.005;
267W-33L-0T against 253W-39L-8T). On the public engine's 20 validation seeds 12-0-8 became 19-1-0 (+$200 a
game, 18 better, 2 worse); abo_v43 +$268 (18 better, 0 worse); the other 12 agents +$2 a game over 240 games
(-$42 to +$46 per agent, none significant); the held-out lost seeds -$8.
Backtest 2 (2026-09-26 23:39 UTC, all 82 recorded games; the champion is 40W-31L-11T on them): _ADV_LOOK 4 plus
the three hour-window lead-sell horizons _EV_H, _DP_H and _MP_H 8 -> 12 went 56W-26L-0T, +$365 a game (64
better, 18 worse): all 11 ties and 7 close losses won, 2 close wins lost. _ADV_LOOK 5 and 6 add nothing over 4
(about as many games worse as better); guard without MILK adds $1 on top of _ADV_LOOK 4. Replays do not react,
so large sale-timing changes can look better on them than live; a fresh-seed validation decides. On the same 300
fresh games (2026-09-27) the lead-sell horizons 12 on top of _ADV_LOOK 4 gained +$37 a game over _ADV_LOOK 4 alone
(145 better, 155 worse, not significant; 269W-31L against 267W-33L): +$345 against tetsutani_demand (19 better, 1
worse), +$450 on the 2965 family's held-out lost seeds (14/2) and +$150 against haideptry_2965 and 2965_0926, but
-$122 against pilkwang_sep (1/19), -$110 against haideptry_shepherd, -$78 against leoprovorov_forecast and -$30
against abo_v57_open13 (4/16). Selling earlier wins the races against the engine's clones and gives away price
against agents that sell later. In the loop, _ADV_LOOK 4 was promoted on Island-Market at iteration 39 (159W-109L,
+$28 a game, p = 0.003); the cows-only HERD2 edit was rejected on Island-Herd at iterations 31 and 37 (-$32, -$29). The mirror of
the engine ties itself, so head-to-head margins start at 0. The dollar test is an exact sign test over the games
whose margin changed (p <= 0.05) with a positive mean; the results test one over the games whose result changed.
Small margins among the V57 family mean most games change by a few hundred dollars: an edit that
gains $300 a game consistently is worth promoting.

### Measured by hand on 2026-09-30 (fresh seeds, not this loop's gauntlet; do not re-propose)

All on the 0927 engine with the oracle guard and KAD off, 200 paired games each (10 pool opponents x 10 fresh seeds x
both seats), against the graph with `_R51_INPUT_CROPS` WHEAT (2,4,8) (the iteration-8 promotion, confirmed on fresh
seeds: without it 78 better / 122 worse, -$75 a game; with the guard and KAD on, 146 / 94, +$79):
- `_R51_INPUT_CROPS` WHEAT cap 10: identical to cap 8 in all 200 games (the planner's gain saturates at 8).
- CARROT cap 6 (on top of wheat 8): 34 better / 152 worse, -$108 a game. Do not raise the carrot cap.
- `_R51_INPUT_MAX_WORKERS` 3: identical in all 200 games.
- `V9_FERT_FIRST_DAY` 10: 102 / 98, +$15 (neutral); 12: 28 / 28, -$20; `V9_FERT_AGES` (1, 2): 78 / 116 (worse).
- `_CA_MARGIN` -30: 66 / 82, +$6 (neutral).
- The engine's hard-coded cash reserves (a modified engine copy, not available here) are never binding except the
  input planner's $3,000: tomato-plot money 12,000 -> 9,000, crop-worker reserve, sheep reserves and the day-11 check
  changed 0 of 200 games; the planner's reserve 3,000 -> 1,500 with ROI 1.5 -> 1.25 gave 114 / 80 but -$39 a game.
