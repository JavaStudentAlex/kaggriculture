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
| 13/9 opening vs Copper Weir (round 2b, partial) | 25 | 25-0-0 | +$25,660 |
| 13/9 opening vs Orchard Tide (round 2b, partial) | 28 | 28-0-0 | +$24,671 |
| **13/9 opening vs the plain Mohui v66 backbone** (round 2b, partial) | 31 | **0-31-0** | **-$15,503** |
| Hazel vs the plain Mohui v66 backbone (round 2b, partial) | 31 | 28-3-0 | +$836 |
| Hazel vs Orchard Tide (round 2b, partial) | 28 | 28-0-0 | +$960 |
| `opening_scalp` disabled vs the plain Mohui backbone (round 2b, partial) | 30 | 25-5-0 | -$123 |

Mechanism of the opening result (traces): Hazel's 35/30 wheat scalp against an opponent
that also buys wheat at step 0 leaves it $6 at the day-1 hire step, one dollar short of
its fourth hand; its route breaks and the loss compounds. The 13/9 opening keeps $52-71.
Hazel's own ladder games show the same short hire in 2 of 20 sampled games.

The opening is opponent-dependent: the same 13/9 change that wins every game against
the 35/30 and 13/9 scalpers loses every game against the plain backbone. An opening
edit must be judged across the whole pool, never against one opponent.
