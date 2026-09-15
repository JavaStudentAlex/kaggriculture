# Why Copper Weir loses on seeds 12316 and 12720 (trace audit, 2026-09-14)

The only genuine pattern in Copper Weir's 104 tournament losses (`copper_weir_losses.txt`): two seat-symmetric
worlds where it earns less than every other non-floor champion — seed 12316 (a $141k world, lost 12 of 14) and
seed 12720 ($92k, lost 10 of 14). Its parent Open Sluice (gen 188; the only code difference is Copper Weir's
value-weighted sell-order eviction, the removal of the `len(mkt) < 10` guards and the day-29 liquidation from
step 693) earns $1.1-1.6k more on the same worlds against the same opponents, so the two were replayed against
the same opponent (Cider Ridge, then Quiet Barley), both on seat 1 as in the tournament, with the full step
record and an exact per-unit market ledger (`audit/trace_exact.py` wraps the engine's `_commit_unit`) and a
shed-overflow counter (`audit/trace_overflow.py` wraps `_apply_unit_action` / `_drop_inventories_to_shed`).

| game (ours on seat 1) | Copper Weir | Open Sluice | diff |
|---|---|---|---|
| 12316 vs Cider Ridge | 140,713 | 141,949 | −1,236 |
| 12316 vs Quiet Barley | 140,395 | 141,547 | −1,152 |
| 12720 vs Cider Ridge | 91,876 | 93,511 | −1,635 |
| 12720 vs Quiet Barley | 91,472 | 92,907 | −1,435 |

## What happens

1. **The farms are identical.** Hands, animals, plants, purchases (seeds, animals, hires: −$7,942 in all four
   games), every farm/hand action — no difference at all. Only the market orders differ, first at step 313
   (fertilizer 16 vs 15 — worth $1) and then at the **hour-0 step of days 21, 23, 25, 26, 28**.
2. **Hour 0 is where the 10-order cap binds.** The backbone re-hires its hands every day at hour 0 and each
   HIRE is a market order, so 8-9 of the 10 slots are taken and 1-2 sells fit. Open Sluice puts its cheap
   *fertilizer* sale in that slot (shed-pressure section) and sells the wool/milk at hour 1, when the slots are
   free. Copper Weir's eviction rule sees a low-value SELL in the list and replaces it with the higher-value
   wool/milk order — which gains nothing (the same units sell an hour later at the same price in the parent's
   game) but leaves the fertilizer in the shed. Both agents keep buying 4 fertilizer a day at hour 1 for the
   fields, so Copper Weir's fertilizer stock climbs from 18 to 37 units (mean 17 over days 21-28 vs the
   parent's 5, max 22), while the fertilizer price collapses from 27 to 1 (nobody draws it).
3. **The shed overflows.** These are high-output worlds (17 animals, 55 plants; ~70 units come in with the
   end-of-day drop). With 20+ units of fertilizer parked in the 100-unit shed, the end-of-day drop of day 23
   (step 575) discards **4 WOOL + 4 WHEAT + 1 STRAWBERRY**, and day 25 (step 623) another **2 WHEAT** — the
   engine's `_drop_inventories_to_shed` silently deletes what does not fit. Open Sluice discards nothing; it
   peaks at 99/100 on the same steps, one unit from the same failure.
4. **The bill.** Exact ledger differences (units sold × realised price, Copper Weir − Open Sluice):

   | seed 12316 vs Cider Ridge | | seed 12720 vs Cider Ridge | |
   |---|---|---|---|
   | WOOL 274 vs 278 units | −$832 | MILK 184 vs 188 units | −$1,193 |
   | WHEAT 2,182 vs 2,188 | −$180 | WOOL 271 vs 278 | −$340 |
   | MILK (price) | −$95 | FERTILIZER (price) | −$87 |
   | FERTILIZER (price) | −$93 | fertilizer bought 50 vs 46 | −$61 |
   | fertilizer bought (price) | −$54 | WHEAT / STRAWBERRY | +$76 |
   | STRAWBERRY | +$55 | | |
   | **total** | **−$1,236** | **total** | **−$1,635** |

   On 12316 the discarded wool and wheat are the whole deficit. On 12720 the wool is cheap ($120) and the lost
   wheat matters differently: the cows are fed from the hands' wheat, and on day 29 the same hand with the same
   actions collects 4 fewer milk units than in the parent's game (steps 706/710: 4 vs 6, 5 vs 9) — the
   underfeeding knock-on of the missing wheat, another ≈ $900 at $150 a unit.

## Fix — applied as **Hazel Weir** (`top/patch_2026-09-14/hazel_weir/`, at the user's word)

The eviction must respect shed headroom: when `sum(shed) + expected inflow` is within ~15 units of
`shedCapacity`, keep the bulk-clearing SELL (or sell the cheapest bulk product first) instead of evicting it,
and clear fertilizer before the end-of-day drop. Two related backbone habits hurt both programs: buying 4
fertilizer a day while 20+ sit in the shed, and spending 8-9 of the 10 market slots on HIRE orders at hour 0
(the hires could go to another hour so hour 0 keeps its sell slots).

## Scope

The overflow only happens on worlds productive enough to fill the shed (both seeds are top-decile cash worlds);
on the other 38 stage-1 seeds Copper Weir's shed never reaches capacity and it wins 88 % of the games. The
parent has the same exposure one unit away, so the fix protects the whole lineage.

Files: `audit/trace_games.py` (full replays; the 32 MB replays on /results were removed 2026-09-15, the script regenerates them),
`audit/trace_steps.py` (step diff), `audit/trace_exact.py` + `audit/ledger_diff.py` (exact ledgers,
`audit/*_ledger_summary.json`), `audit/trace_overflow.py` (discards), `audit/mirror_losses.py` +
`audit/seed_sym.py` (seat-swap replays behind `copper_weir_losses.txt`).

## Patch results (2026-09-14, `audit/challenger.py` + `evolution/eval_once.sh`)

Three edits on Copper Weir: the pre-drop headroom guard (from hour 20 sell the cheapest stock until
shed + carried fits, orders protected from eviction), eviction that defers the evicted order to the following
turns instead of cancelling it, and a day-28 feed reserve of `n_animals + 6` (with the overflow fixed, the patched
program still collected 4 fewer milk / 3 fewer wool: on day 28 hour 0 it sold 21 wheat where the parent's cap
guard had limited it to 13, the last feeding hand found the shed empty and three animals went unfed).

* Seeds 12316 / 12720 vs Cider Ridge / Quiet Barley / Open Sluice: 6-0 (Copper Weir 0-6), no discards.
* Fresh tournament seeds, Copper Weir's seats, vs the 14 other champions: **516-44** vs Copper Weir's 483-77
  (36 games flipped to wins, 3 to losses), higher average cash against every opponent.
* Evaluator (15-pool, crowning disabled): 87.7 % (526-74); vs the 14 others 509-51 against Copper Weir's 504-56 on
  the same seeds (+9 / −4). Mirror vs Copper Weir 17-23 with 16 losses under $25 (the deferred fertilizer sells an
  hour later) and +$1,940 total cash.
