# Surgical graph candidate: scope and acceptance criteria

This update implements the user's requested strategic changes in the active local graph. It is a research candidate, not a proven winning submission. Frozen submitted `champion.py`, opponent packages, archived matches and predictor weights must not be changed.

## Why source-level changes

An extension that runs after Hazel's market stage cannot prevent the original headroom sale or recover an order already evicted. Similarly, an idle-only extension cannot fix an incorrect non-PASS rescue action already produced by transposed coordinates. The fixes therefore belong inside the executable market stages and in replacement idle maintenance dispatch.

## Requested behavior

- Headroom: hour-22 trigger, subtract stock-backed scheduled sales, retain up to two fertilizer units per active plot for additional policy-layer sales, protect feed, prefer excess wheat, do not give headroom orders unconditional eviction protection. Release the fertilizer reserve from step 696. Existing backbone orders are preserved rather than clamped by a global post-processing filter. These thresholds are experimental, not established optima.
- Deferred sales: expire stale entries; suppress replay during morning expansion; append only while enough slots remain, without evicting funding or production orders.
- Idle maintenance: preserve backbone non-PASS actions; use `[x,y]` positions with `tiles[y][x]`; feed only from the actor's own bag; avoid duplicate maintenance assignments.
- Forecast sales: bounded additional wool/milk sales based on available stock and price floors, using short/24-step signals. Raw `score_k` values are ranking signals, not sale probabilities. No automatic seat advantage or guaranteed exact-peak timing.

## Corrections to earlier advice

The previous blanket claim that these changes would guarantee wins was unsupported. Keeping fertilizer is not automatically better: use requires worker transport/actions and an eligible crop window. Wheat can have feed value. A late trigger cannot undo a DROP overflow that happened before trading. A newer predictor is not automatically a stronger trading policy. Shop consumption affects subsequent quotes, not a private same-turn premium.

## Validation

1. Assert active runtime and checkpoint hashes and unchanged submitted champion.
2. Test the actual compiled market generator, not only standalone extension handlers.
3. Check disabled source-overrides equivalence and non-PASS preservation.
4. Run candidate and frozen previous graph on matching opponent/seat/seed jobs with the SAME 09-22 predictor.
5. Record wins/losses/draws and relative cash margins separately; save full traces.
6. Replay a real complete observation stream to detect hidden graph fallbacks and oracle errors. Replay is a runtime-health check, not counterfactual performance evidence.

Artifacts for this iteration belong in `runs/surgical_validation/`. Existing frozen payloads/results are retained and are not rebuilt in place. No Kaggle submission or remote notebook replacement is part of this local update.
