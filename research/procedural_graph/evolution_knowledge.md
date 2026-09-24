# Verified knowledge for graph-edit evolution

Loaded into the mutation, judge and supervisor prompts by `highcpu_island_evolution.py`.
Only facts verified in the engine or measured in paired arena games belong here; keep
sample sizes and dates. Engine: kaggle-environments 1.32.7.

## Engine facts (verified)

- 720 turns (30 days x 24 hours); both seats start with $3,000.
- Positions are [x, y]; farm tiles are indexed `tiles[y][x]`.
- Market orders: at most 10 per turn (market channel only; farmer and hand actions are
  separate). Both seats' orders at the same index clear unit by unit in lockstep, and
  index i+1 starts only after both index-i orders finish: a SELL queued behind BUY/HIRE
  orders clears after an opponent's earlier SELL of the same item.
- Town shops consume their stock after the market on steps where step % 4 == 0.
- FEED takes wheat from the acting unit's own bag, not from the shed.
- Hiring 4 hands on day 1 costs 1 + 1 + 2 + 3 = $7 (the backbone's route hires at step 24).
- The engine is seat-symmetric and the agents are deterministic: a swapped-seat replay
  gives identical cash, and a graph playing itself ties exactly.
- An animal not fed on two consecutive days escapes at the end of the second day (the
  structure stays; the animal, its $400/$500 and all its future output are lost). FEED
  uses 1 wheat from the acting unit's bag, once per animal per day.
- The backbone route buys 2 cows and 2 sheep at step 1, leaves the cows unfed on day 0 and
  feeds them on day 1 with wheat carried out of the shed. The shed must hold 5 wheat after
  step 1 (opening buy minus opening sell >= 5); with 4, one cow escapes at the end of day 1.
- The town shop added every 3 days is drawn from the same per-day RNG as the weed spawn,
  which draws once per empty tile on both farms. So the layout of either farm changes
  which shops open for the rest of the game.
- Nothing trades between step 2 and the day-1 hire at step 24, so a seat's cash after
  step 1 is its cash at the hire (4 hires cost $7; with $6 or less the route gets 3 hands).

## Measured in paired arena games (2026-09-24, candidate vs Hazel Weir unless noted)

Seeds alternate the candidate's seat; W-L-T counts games, margin is mean cash difference.

| change (on Hazel's policy) | games | W-L-T | mean margin |
|---|---:|---|---:|
| `_OPENING_BUY_WHEAT_QTY` 35 -> 13, `_OPENING_SELL_WHEAT_QTY` 30 -> 9 | 60 | 60-0-0 | +$25,759 |
| same, vs Mohui13 (backbone with the 13/9 opening) | 60 | 59-1-0 | +$32,639 |
| Hazel itself vs Mohui13 | 60 | 51-9-0 | +$5,941 |
| `opening_scalp` stage disabled (backbone opening) | 60 | 6-54-0 | -$95 |
| same, vs Copper Weir | 60 | 20-39-1 | -$95 |
| same, vs Mohui13 | 60 | 60-0-0 | +$47,421 |
| routine_dispatch `order: sells_first` (source-snippet version) | 40 | 40-0-0 | +$987 |
| `_TOWN_CADENCE_PHASE` 0 -> 3 | 60 | 54-6-0 | +$86 |
| `_TOWN_CADENCE_PHASE` 3 + `sells_first` | 60 | 59-1-0 | +$1,031 |
| `_SHOP_DEMANDS` set to the engine's exact recipes | 60 | 6-2-52 | +$1 |
| surgical `fertilizer_guard` only | 40 | 1-39-0 | -$1,483 |
| surgical `deferred_sales` only | 40 | 10-29-1 | +$3 |
| surgical `predrop_headroom` only | 40 | 2-5-33 | -$51 |
| surgical `luxury_supplement` only | 40 | 11-9-20 | +$28 |
| surgical `idle_dispatch` v1 only | 40 | 2-1-37 | +$1 |
| 2026-09-22 checkpoint + calibration instead of Hazel's 09-13 (not an editable control) | 40 | 2-38-0 | -$307 |
| Hazel vs Copper Weir | 60 | 22-37-1 | -$3 |
| 13/9 opening vs Copper Weir | 60 | 60-0-0 | +$25,782 |
| 13/9 opening vs Orchard Tide | 60 | 60-0-0 | +$25,485 |
| **13/9 opening vs the plain Mohui v66 backbone** | 60 | **0-60-0** | **-$16,415** |
| Hazel vs the plain Mohui v66 backbone | 60 | 55-5-0 | +$886 |
| Hazel vs Orchard Tide | 60 | 60-0-0 | +$914 |
| `opening_scalp` disabled vs the plain Mohui backbone | 60 | 49-11-0 | -$27 |
| graph `order: sells_first` alone (round 3, stopped early) | 7 | 7-0-0 | +$1,427 |
| 13/9 + sells-first vs the plain Mohui backbone (round 3, stopped early) | 11 | 0-11-0 | -$16,597 |
| 13/9 + sells-first vs Mohui13 (round 3, stopped early) | 11 | 10-1-0 | +$33,636 |
| 13/9 + sells-first + cadence phase 3 vs the plain Mohui backbone (round 3) | 10 | 0-10-0 | -$16,629 |

Mechanism of the opening result (traces): Hazel's 35/30 wheat scalp against an opponent
that also buys wheat at step 0 leaves it $6 at the day-1 hire step, one dollar short of
its fourth hand; its route breaks and the loss compounds. The 13/9 opening keeps $52-71.
Hazel's own ladder games show the same short hire in 2 of 20 sampled games.

Why 13/9 loses every game to the plain backbone (verified 2026-09-24 by replaying 4 traced
seeds through an instrumented engine, 4 of 4): 13/9 leaves 4 wheat in the shed after
step 1, the backbone's own 5/0 and Hazel's 35/30 leave 5. On day 1 one of our cows goes
unfed for the second day and escapes at midnight; the backbone keeps all its animals.
With one animal fewer we sell less fertilizer on day 2, miss the next cow at step 88
($73 + $290 of fertilizer < $400) and stay at 4 animals while the backbone reaches 6. Our
farm also changes the town shops from about day 9. The backbone sells the same units but
at better prices (strawberries $125 instead of $80), ending about $12k richer, while we
end about $5k poorer than Hazel's own game. Mohui13 has the same 4-wheat problem, which
is why it is weak.

The 13/9 edge over the 35/30 openers (Hazel, Willow) is a different mechanism: our
9-unit step-1 sell clears in lockstep with their 30 and lowers their proceeds, so they
reach the step-24 hire with $6, one dollar short of their fourth hand.

Cash after step 1 (= cash at the step-24 hire) and our wheat, computed with the engine for
our opening against each opponent opening; `short` = fewer than $7, only 3 hands:

| our buy/sell | our wheat | vs 35/30: us / them | vs 5/0: us / them | vs 13/9: us / them |
|---|---:|---|---|---|
| 13/9 (v5) | 4 | $71 / $6 short | $51 / $22 | $52 / $52 |
| 13/8 | 5 | $40 / $9 | $23 / $22 | $24 / $52 |
| 14/9 | 5 | $41 / $7 | $23 / $22 | $22 / $52 |
| 15/10 | 5 | $43 / $6 short | $24 / $22 | $22 / $54 |
| 16/11 | 5 | $43 / $5 short | $24 / $22 | $20 / $54 |
| 17/12 | 5 | $44 / $6 short | $24 / $22 | $21 / $56 |
| 18/13 | 5 | $43 / $5 short | $24 / $22 | $19 / $56 |
| 20/15 | 5 | $43 / $5 short | $24 / $22 | $17 / $58 |
| 14/10, 15/11 | 4 | $72-74 / $4 short | $51-52 / $22 | $50 / $52-54 |
| 12/7, 11/6, 10/5 | 5 | $35-38 / $10-13 | $23 / $22 | $24-25 / $49-50 |
| 35/30 (Hazel) | 5 | $27 / $27 | $25 / $22 | $6 short / $71 |

An opening edit must be judged across the whole pool, never against one opponent.

The gauntlet pool (run 2 onwards): `mohui13` = the backbone with a 13/9 opening, `mohui` =
the plain Mohui v66 backbone (5/0), `hazel` = Hazel Weir (35/30), `willow` = Willow Ford,
the Harvest Current stand-in (35/30).

## Evolution run 2 (2026-09-24, seed v5.0.0, 40 seeds per opponent + 40 head-to-head)

Paired change against the v5 seed (per opponent: W-L-T, mean change per game):

| edit | pool mean | mohui | mohui13 | hazel | willow | vs v5 | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| `_OPENING_SELL_WHEAT_QTY` 9 -> 8 (13/8) | -$1,810 | 40-0 +$18,880 | 29-11 +$14,272 | 0-40 -$34,932 | 0-40 -$25,529 | 40-0 +$18,258 | rejected |
| `_OPENING_SELL_WHEAT_QTY` 9 -> 13 (13/13, no wheat) | -$60,674 | 0-40 | 4-36 | 0-40 | 0-40 | 0-40 | rejected |
| `_TOWN_CADENCE_PHASE` 0 -> 3 | -$9 | 20-20 -$94 | 27-13 -$1,576 | 29-11 -$312 | 23-17 +$1,386 | 32-8 +$551 | rejected |
| `_SHED_PRESSURE_AT` 80 -> 88 | -$158 | -$89 | -$39 | -$351 | -$94 | -$217 | rejected |
| `_PRICE_THRESHOLD_RATIO` 0.85 -> 0.80 | +$2 | 186 of 200 games unchanged | | | | | rejected |
| `_ORACLE_FRONTRUN_BATCH` 6 -> 4, `_ORACLE_FRONTRUN_PRICE_RATIO` 0.6 -> 0.7 | +$221 | 25-12 +$266 | 26-11 +$49 | 21-15 +$274 | 31-8 +$102 | 33-7 +$412 | **promoted** (p=1e-9) |

13/8 confirms the cow mechanism (it fixes the plain-backbone loss) and the hire mechanism
(it gives up the edge over the 35/30 openers).

Ladder mix (26 of Hazel Weir's ladder games, 2026-09-15..23): the opponent's step-0 wheat
buy was 13 in 12 games (Hazel lost 10 of them), 0 in 3, 7 in 2, 5 (the plain backbone's)
in 1, and 15-174 in the other 8. The Mohui13 opponent is the pool's proxy for the 13-wheat
openers; the real ones earn far more than it does.
