# Verified knowledge for ladder-engine graph evolution

Loaded into the mutation, judge and supervisor prompts by `highcpu_island_evolution.py
--knowledge evolution_knowledge_ladder.md`. Only facts verified in the engine, in ladder replays
or in paired arena games belong here, with sample sizes and dates. Engine: kaggle-environments
1.32.7.

## Engine facts (verified)

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

## What a candidate must do to be promoted

It plays the same games as its island champion: every lost ladder seed (Rowan Glen's and Linden
Brook's losses, and since 09-26 19:30 UTC Alder Ford's 22 losses and 2 ties) against the bundle
that plays like the rival who beat us there, from both seats, plus random seeds against the pool,
plus head-to-head games against the champion, and since iteration 31 each of Alder Ford's 24 lost or
tied games against a replay of the rival's recorded moves (below): 282 pool games and 20 head-to-head. A rival no pool
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
the engine ties itself, so head-to-head margins start at 0. A change is promoted only if an exact
sign test over the games whose result changed is significant (p <= 0.05) with a positive mean.
Small margins among the V57 family mean most games change by a few hundred dollars: an edit that
gains $300 a game consistently is worth promoting.
