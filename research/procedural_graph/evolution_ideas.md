# Ideas for the next short evolution run

Read by `highcpu_island_evolution.py --ideas evolution_ideas.md` and shown to every
mutating model as hypotheses to test (bullets only; the file is re-read each iteration,
so it can be edited while a run is going). Seed of run 3: run 2's best graph (v5.0.0 with
the oracle front-run edit: batch 4, price ratio 0.7).

- Top priority, Opening: keep the 5th wheat and still starve the 35/30 openers. Use an opening whose buy minus sell is exactly 5 and whose sell is at least 10: 15/10, 16/11, 17/12 or 18/13. The engine table in the knowledge section shows these leave us 5 wheat (no cow escape against any opponent) and leave Hazel/Willow $5-6 at the step-24 hire. 13/9 leaves 4 wheat and loses a cow; 13/8 and 14/9 let the 35/30 openers hire 4 hands.
- Never propose an opening that leaves fewer than 5 wheat after step 1 (buy minus sell < 5: 13/9, 14/10, 15/11, 11/7, 13/13). It costs a cow on day 1 against every opponent.
- After the opening, the remaining gap is the plain backbone and Mohui13. Shop sales and the oracle front-run are the next levers: `_SHOP_SELL_BATCH_MAX` 4, `_MIN_HELD_FOR_SHOP_SALE` 2, `_ORACLE_FRONTRUN_SCORE` 0.30, `_ORACLE_FRONTRUN_MIN_HELD` and `_ORACLE_FROM_STEP` are untested.
- The oracle front-run edit (batch 6 -> 4, price ratio 0.6 -> 0.7) gained about $220 a game against every opponent; neighbouring values (batch 3 or 5, ratio 0.65 or 0.75) are untested.
- Measured, do not repeat: `_TOWN_CADENCE_PHASE` 0 -> 3 (-$9), `_SHED_PRESSURE_AT` 80 -> 88 (-$158), `_PRICE_THRESHOLD_RATIO` 0.85 -> 0.80 (no effect), the surgical fertilizer guard (-$1.5k); the other surgical overrides and the engine-exact `_SHOP_DEMANDS` change nothing.
