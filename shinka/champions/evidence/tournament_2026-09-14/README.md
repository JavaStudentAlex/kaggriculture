# All-champions tournament on fresh seeds (2026-09-14)

Round-robin of the whole active pool (15 champions) to rank them against each other — the evaluator only ever
plays a candidate against the pool, so pool members had no common yardstick beyond the calibration games.

**Verdict: Copper Weir is the best champion** — 696W-104L-0T (87.0 %), Bradley-Terry lead of 222 Elo over the
runner-up Open Sluice, rank 1 in 100 % of 2,000 bootstrap resamples, a winning record against all 14 others
(closest pairing Meadow Lantern 31-9; 83-85 of 100 against each of the other finalists).

## Protocol

* `tournament.py` (run through `run.sh` = the evaluator's environment: CUDA venv, MPS, `kagg_oracle.py` +
  the current checkpoint `evolution/checkpoint/` (256-context, 09-13 refit), Mohui closure). 108 spawned workers,
  4,800 games in 1,519 s, 0 crashes.
* Stage 1: every pairing, 40 games = 20 seeds with A on seat 0 + 20 disjoint seeds with B on seat 0
  (seat-0 seeds 11104..13023, seat-1 seeds 130108..132027) → 105 pairings, 4,200 games, 560 per champion.
  The seeds are fresh: never used by `evaluate.py` (101..2020 / 70102..72021) nor by the run-2 tie-break
  (5104..7023 / 90108..92027), so nothing evolved against them.
* Stage 2: the top 5 by stage-1 rating replay each other on a third fresh block (30 + 30 per pairing; seeds
  23112..26041 / 150114..153043) → 600 games, so every podium pairing rests on 100 games.
* Ranking: Bradley-Terry (MM fit; ties = half a win) over all 4,800 games; `P(#1)` and `E[rank]` from a bootstrap
  that resamples each pairing's games. WR% counts ties for neither side; the 95 % CI is Wilson on wins/games.
* Files: `tournament.txt` (report), `tournament.json` (every game with seeds and cash, aggregates, pair records,
  ratings), `games.jsonl` (the same games in `evaluate.py`'s `champion_vs_champion` record format, so
  `evolution/curate_pool.py` can use them as calibration data).

## Result

```
 # champion            W    L   T    WR%        95% CI  seat0%  seat1%  avg cash  BT-Elo  P(#1) E[rank]
 1 Copper Weir       696  104   0   87.0  84.5-89.2      90.8    83.2    85,797       0 100.0%    1.00
 2 Open Sluice       498  284  18   62.2  58.8-65.5      63.8    60.8    85,927    -222   0.0%    2.32
 3 Mirror Hedgerow   463  266  71   57.9  54.4-61.3      61.8    54.0    85,944    -231   0.0%    2.71
 4 Granary Brook     431  298  71   53.9  50.4-57.3      52.8    55.0    85,947    -262   0.0%    3.98
 5 Cider Ridge       331  228   1   59.1  55.0-63.1      58.6    59.6    84,874    -310   0.0%    5.17
 6 Slate Pasture     315  245   0   56.2  52.1-60.3      55.0    57.5    84,889    -334   0.0%    6.42
 7 Furrow Dawn       305  255   0   54.5  50.3-58.5      55.7    53.2    85,158    -348   0.0%    7.43
 8 Quiet Barley      365  417  18   45.6  42.2-49.1      44.2    47.0    85,947    -349   0.0%    7.38
 9 Clover Bank       288  272   0   51.4  47.3-55.5      49.6    53.2    85,135    -373   0.0%    9.07
10 Meadow Lantern    283  277   0   50.5  46.4-54.7      48.6    52.5    85,674    -380   0.0%    9.50
11 Willow Ford       223  336   1   39.8  35.8-43.9      37.1    42.5    85,352    -468   0.0%   11.23
12 Orchard Tide      212  347   1   37.9  33.9-41.9      37.1    38.6    85,218    -485   0.0%   11.80
13 Amber Loft        187  372   1   33.4  29.6-37.4      32.5    34.3    85,232    -525   0.0%   12.98
14 Birch Hollow       99  460   1   17.7  14.7-21.1      17.9    17.5    85,789    -704   0.0%   14.00
15 First Furrow       12  547   1    2.1   1.2-3.7        1.1     3.2    78,242   -1082   0.0%   15.00
```
(finalists have 800 games — 560 of stage 1 + 240 of stage 2 — the others 560; full tables and the 15×15
head-to-head matrix in `tournament.txt`.)

## Reading

* **Copper Weir** (run 3 gen 198, `value_weighted_order_preemption`) is not a seed artefact: its 90 % crowning
  result on the evaluator's seeds reproduces on seeds it never saw (87.0 % overall, 85.5 % among the oracle-era
  champions, 83.8 % against the other four finalists at 100 games each: 85-15 Open Sluice, 83-17 Mirror
  Hedgerow, 84-16 Granary Brook, 83-17 Quiet Barley). Against the pre-oracle 09-08 champions it is 35-5 / 35-5 /
  34-6 / 34-6 / 39-1, against the pool copies of Orchard Tide and Amber Loft 38-2 / 38-2. It wins a bit more on
  seat 0 (90.8 %) than on seat 1 (83.2 %).
* The next three — **Open Sluice** (gen 188, Copper Weir's parent), **Mirror Hedgerow** (gen 189) and **Granary
  Brook** (gen 122, the ancestor of that lineage) — sit within 40 Elo of each other. Mirror Hedgerow and Granary
  Brook tie 54 of their 100 games with identical final cash, i.e. they play the same game on most seeds.
* The crowning order of run 3 does not survive fresh seeds: **Quiet Barley** (78.9 % when crowned, best score of
  its hour) drops to 45.6 % and 8th, **Furrow Dawn** and **Meadow Lantern** are about even with the pre-oracle
  champions (20-20 / 21-19 against Cider Ridge and Slate Pasture), and the pre-oracle **Cider Ridge** (#5,
  59.1 %) and **Slate Pasture** (#6) still outrank every run-3 champion below the top four. Only the top four
  beat the 09-08 champions convincingly.
* **Orchard Tide** (#12) and **Amber Loft** (#13) are the pool copies, which play with the current 256-context
  checkpoint; Orchard Tide was tuned to the 512-context one and is known to lose ~12 pp against the 8-pool when
  the checkpoint changed (evaluator runs of 2026-09-14, logs since removed: 67.8 % → 55.3 %). Its Kaggle
  bundle carries its own 512-context checkpoint, so the ladder version is stronger than this row; every run-3
  champion still beats the pool copy 37-3 or better, and Copper Weir beat it 36-4 at crowning as well.
* The field is seat-symmetric: seat 0 won 50.0 % of the 4,708 decided games. Average cash is flat across the
  oracle-era champions ($85.7-85.9 k); the ranking is decided by who ends higher, not by cash volume.

## Copper Weir's losses

`copper_weir_losses.txt` lists all 104 with a seats-swapped replay of each: 13 are the seat artefact of seed
11205 (the world gives seat 0 the better farm), 65 are near-clone games against its own lineage decided at the
margin (median $143), and 22 are two high-output worlds (seeds 12316, 12720) where it is worse than every
non-floor champion. `copper_weir_loss_audit.md` traces those two: the sell-order eviction parks fertilizer in the
shed at the hour-0 re-hire step, the 100-unit shed overflows at the end-of-day drop and 4 wool + 6 wheat +
1 strawberry are discarded (≈ the whole deficit); fix = shed-headroom guard on the eviction.
