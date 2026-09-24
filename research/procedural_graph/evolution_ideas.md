# Ideas for the next short evolution run

Read by `highcpu_island_evolution.py --ideas evolution_ideas.md` and shown to every
mutating model as hypotheses to test (bullets only; the file is re-read each iteration,
so it can be edited while a run is going). Seed of run 1: policy_graph.json v5.0.0
(13/9 opening, sells-first, surgical off, Hazel's predictor).

- The 13/9 opening wins every game against 35/30 and 13/9 scalpers but loses every game (-$16k) to the plain backbone that buys 5 wheat. Look for an opening that keeps the wins and removes that loss: other buy/sell pairs between 5 and 20 (for example 9/5, 11/7, 15/11, 13/5, 17/13), keeping at least $50 at the step-24 hire.
- The backbone opening (opening_scalp stage disabled) lost 6-54 to Hazel by only $95 but won 60-0 by $47k against Mohui13; a middle ground between it and 13/9 may exist.
- Town shops: `_TOWN_CADENCE_PHASE` 3 won 54-6 (+$86) against Hazel; phases 1 and 2 are untested. Shop batch size (`_SHOP_SELL_BATCH_MAX` 4), price threshold (`_PRICE_THRESHOLD_RATIO` 0.85) and minimum held (`_MIN_HELD_FOR_SHOP_SALE` 2) are untested.
- Oracle front-running: score threshold `_ORACLE_FRONTRUN_SCORE` 0.30, batch `_ORACLE_FRONTRUN_BATCH` 6 and price floor `_ORACLE_FRONTRUN_PRICE_RATIO` 0.60 are untested with Hazel's predictor.
- Shed: `_SHED_PRESSURE_AT` 80, `_SHED_PRESSURE_PRICE_RATIO` 0.35 and `_HEADROOM_FROM_HOUR` 20 are untested.
- Measured, do not repeat: the surgical fertilizer guard loses about $1.5k a game; the other surgical overrides and the engine-exact `_SHOP_DEMANDS` change nothing.
