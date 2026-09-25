# Feed-reserve fix, 2026-09-24

Why: the run-2 best graph (13/9 opening scalp) lost 400 of 400 games to plain Mohui. In every
game it starved an animal on day 1 and again on day 8-9: it sold the wheat its herd needed,
and its feed reserve counted only animals already placed, not the ones just bought or about
to be placed on new pastures.

Change: runtime parameter `_FEED_RESERVE_LOOKAHEAD` (`hazel_runtime/graph_runtime.py`, commit
53b21e7). When it is set, the feed reserve also counts animals in the shed, in inventories,
on BUY_ANIMAL orders and on empty pasture or coop tiles. Two graphs, both with it set:

- `feed15_graph.json`: opening 15/10 (keeps 5 wheat)
- `feed17_graph.json`: opening 17/9

Predictor in both: `ttm_c256_h96_ft_2026-09-13` (Hazel Weir's; model sha256 3c94bfc5...).

## Batch 1: feedfix, 800 games on Colab (5 High-RAM VMs, account 2)

Seeds are the first 200 of the old 400-game run vs Mohui, and 40 seeds for each other opponent.
The candidate's seat alternates by seed. `feedfix_results.jsonl`.

| opponent | feed15 | feed17 |
|---|---|---|
| plain Mohui (200) | **122W-77L-1T, +425** | 0W-200L, -13,709 |
| Hazel (40) | 40W-0L, +45,125 | 0W-40L, -9,977 |
| Copper (40) | 40W-0L, +45,172 | 0W-40L, -9,934 |
| Orchard (40) | 40W-0L, +43,416 | 0W-40L, -10,388 |
| Willow (40) | 40W-0L, +41,221 | 1W-39L, -11,581 |
| Mohui13 (40) | 40W-0L, +45,570 | 39W-1L, +29,946 |

The old best graph lost 0W-400L to plain Mohui. feed15 is the new best graph; feed17 is rejected.

## Batch 2: model0923, 200 games: feed15 with the newest predictor vs feed15 with the 09-13 predictor

Built with `payload.py --checkpoint feed15new=models/ttm_c256_h96_ft_2026-09-23`. There are 200
seeds; the 09-23 side plays seat 0 on 100 of them and seat 1 on the other 100. Nothing else
differs. `model0923_results.jsonl`.

| 09-23 side seat | W-L-T | mean margin | median | range |
|---|---|---|---|---|
| 0 | 12-86-2 | -272 | -302 | -1,249 .. +254 |
| 1 | 12-88-0 | -254 | -296 | -1,259 .. +370 |
| total | **24-174-2** | **-263** | -301 | |

The newer predictor loses consistently, on both seats, but by only about 0.3 % of about $84k of
final cash. feed15 keeps the 09-13 predictor. The 09-22 predictor with calibration had also lost
to it earlier (2-38).

Traces for both batches: `runs/arena/feedfix/traces`, `runs/arena/model0923/traces`
(git-ignored). Escape counts from replaying the feed15 traces are not yet computed.

## feed15cal: feed15 with the calibrated 09-23 predictor (saved 2026-09-24)

`feed15cal/` is a complete agent bundle: `main.py` (= `agent_graph.py`), `policy_graph.json`,
and `hazel_runtime/` with the predictor in `hazel_runtime/checkpoint/`. Its graph is feed15's.
The predictor is `ttm_c256_h96_ft_2026-09-23` (model sha256 26fdac50…) with
`calibration/ttm_c256_h96_ft_2026-09-23/calibration.json` (sha256 d0528e31…) next to it; the
oracle scales `score_4`, `score_24` and `units_24` per product by it. All pins in its
`policy_graph.json` match the bundle's own files. The bundle is byte-identical to the one
playing the calibrated rematch (`runs/arena/model0923cal`: feed15cal vs feed15 with the 09-13
predictor, the same 200 seeds as batch 2).

Play it as-is: `arena/payload.py --bundle feed15cal=evolution_results/feed_fix_2026-09-24/feed15cal ...`.
Rebuild it: `arena/payload.py --graph feed15cal=feed15_graph.json --checkpoint
feed15cal=models/ttm_c256_h96_ft_2026-09-23 --calibration
feed15cal=calibration/ttm_c256_h96_ft_2026-09-23/calibration.json`.

## Batches 3 and 4: feed15 with the calibrated 09-23 predictor vs feed15 with 09-13

These use the same 200 seeds and seats as batch 2, so every game pairs with its uncalibrated
counterpart. The files are `model0923cal_results.jsonl` and `model0923own_results.jsonl`.

| 09-23 side | games | W-L-T | mean margin |
|---|---|---|---|
| uncalibrated (batch 2) | 200 | 24-174-2 | −263 |
| + ladder calibration (`calibration/ttm_c256_h96_ft_2026-09-23/calibration.json`, the `feed15cal` bundle; stopped at 160 of 200) | 160 | 19-141-0 | −265 |
| + own-games calibration (`calibration/ttm_c256_h96_ft_2026-09-23/own_games/calibration.json`) | 200 | **41-155-4** | **−195** |

On the 160 seeds all three share: own-games −191, ladder −265, uncalibrated −255. The own-games
calibration recovers about a quarter of the gap (+$68 a game on average; better in 78 games,
worse in 80). The rest is ranking quality: on these opponents the 09-23 model picks worse
moments. At the same firing rate its wheat calls come true 37 % of the time vs 41 %, and no
calibration changes that. feed15 keeps the 09-13 predictor in the arena. Whether the 09-23 or
09-24 refit is better against top ladder players, which it predicts better, only a ladder test
can show.
