# Run 2 (2026-09-12) — top programs and the best candidate

Shinka run 2: seed = run-1 gen 62 (`champ_20260912_gen62_avg82350.py`), pool of 7,
200 generations planned, 4 islands, oracle checkpoint `models/ttm_v3_h96_ft_2026-09-11`
(frozen; only the policy evolves). Started 14:31 UTC, **stopped by hand 19:01 UTC**
at 93/200 generations (89 accepted, 4 starter-check failures, $58.74). The run's
database and per-generation directories are in `shinka_results/` (git-ignored home
mirror) and `/results/kagg/shinka_results` (SSD); this directory holds the copies
that matter.

## Best candidate: gen 33 `dynamic_shop_batch_sizing`

`gen_33/main.py` — byte-identical (sha256 `66ad4552b094…`) to
`champions/pool/champ_20260912_161630_avg82113.py`, `champions/roster/…` and
`shinka_results/best/main.py`.

* **Crowned** by evaluate.py at 16:16 UTC: 77.1 % (216W-64L) against the seven-pool,
  $82,113 average cash, seat 0 / seat 1 = 76 % / 78 % (every other top program is
  seat-skewed, 66-69 % / 78-82 %). It is the only program of runs 1 and 2 to pass the
  75 % gate. Its edit: the town-shop demand-harvesting batch is sized from the
  oracle's `units_24` / `score_24` — 6-8 units when the opponent is about to dump
  the same product (drain the shop's cash first), 3 under a monopoly (let the price
  drift up), static default before turn 512 and for MELON.
* **Tie-break on fresh seeds** (`tiebreak_fresh_seeds.txt`; 840 games, seed blocks
  the evolution never saw; gens 33/55/60/81/82/88 + the run-2 seed, 40 games per
  pairing): gen 81 134W, **gen 33 133W**, gen 60 132W, gen 88 123W, gen 55 119W,
  gen 82 105W, seed 94W (of 240 each). Gen 33 is the only program with no losing
  pairing (27-13 seed, 21-19 g55, 21-19 g60, 20-20 g81, 22-18 g82, 22-18 g88).
* Programs evaluated after 16:16 faced the eight-pool (gen 33 included), so their
  `combined_score` is not comparable with gen 33's; the comparable column below is
  the win rate against the seven old champions. On it the top is a statistical tie
  (±2.5 pp at 95 %): gen 88 77.5 %, gen 33 77.1 %, gen 81 77.1 %, gen 55 76.8 %.
  Gen 88's edge vanishes on fresh seeds (51.2 %, loses to 33, 55, 82); gen 82's
  76.4 % becomes 43.8 % — both look fitted to the evaluator's fixed seed blocks.
* Runner-up: **gen 81** `adaptive_fertilizer_and_endgame_liquidation` (island 0 family:
  late-endgame fertilizer monetisation, investment pruning, pre-midnight flush) —
  same fresh-seed record to within one game, 20-20 against gen 33, but 18-22 to gen
  88. Gen 33 and gen 81 come from different islands (33 → 47 → 55 → 82 and
  47 → 57 → 60 → 71 on island 1; 37 → 67 → 81 and 37 → 64 → 86/88 on island 0), so
  their edits are complementary and a run-3 seed could combine them.

## Saved here

`gen_NN/main.py` + `metrics.json` (evaluate.py output) + `summary.json` (parsed
breakdown, lineage, sha256) for every program with ≥ 75 % against the seven old
champions: gens 33, 47, 55, 60, 67, 71, 81, 82, 86, 88. `INDEX.json` has the parsed
metrics of every accepted program. Gens 60 and 71 are behavioural clones (identical
per-champion results); so are the 62/64/74/78 group (not saved, 73.9-74.3 %).

## Every accepted program with combined ≥ 0.70

`pool` = champions faced (7 before the crowning, 8 after); `r2g33` = games against
the crowned gen 33. Cells are W-L (ties marked).

| gen | name | combined | WR vs pool | pool | **WR vs old 7** | seat 0 / 1 | avg cash | seed | 153302 | 150703 | 183514 | 190355 | 191956 | r1g62 | r2g33 | parent |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 88 | `zero_distance_worker_rescue_and_advanc` | 0.771 | 74.1 % | 8 | **77.5 %** | 66 / 82 | $81,774 | 40-0 | 40-0 | 27-13 | 27-13 | 26-14 | 26-14 | 31-9 | 20-20 | 64 |
| 33 | `dynamic_shop_batch_sizing` | 0.791 | 77.1 % | 7 | **77.1 %** | 76 / 78 | $82,114 | 40-0 | 40-0 | 35-5 | 27-13 | 25-15 | 25-15 | 24-15-1T | – | 0 |
| 81 | `adaptive_fertilizer_and_endgame_liquid` | 0.768 | 73.4 % | 8 | **77.1 %** | 68 / 79 | $81,821 | 40-0 | 40-0 | 27-13 | 27-13 | 27-13 | 29-11 | 26-14 | 19-21 | 67 |
| 55 | `proactive_feed_pipeline_architecture` | 0.772 | 74.1 % | 8 | **76.8 %** | 69 / 79 | $81,860 | 38-2 | 38-2 | 29-11 | 30-10 | 27-13 | 27-13 | 26-14 | 22-18 | 47 |
| 82 | `adaptive_fertilizer_and_late_input_mon` | 0.768 | 73.4 % | 8 | **76.4 %** | 69 / 78 | $81,796 | 38-2 | 38-2 | 29-11 | 30-10 | 27-13 | 27-13 | 25-15 | 21-19 | 55 |
| 71 | `dynamic_decaying_endgame_price_floors` | 0.766 | 73.1 % | 8 | **76.1 %** | 69 / 78 | $81,849 | 38-2 | 38-2 | 29-11 | 30-10 | 27-13 | 27-13 | 24-16 | 21-19 | 60 |
| 60 | `oracle_dynamic_fertilizer_and_worker_s` | 0.766 | 73.1 % | 8 | **76.1 %** | 69 / 78 | $81,847 | 38-2 | 38-2 | 29-11 | 30-10 | 27-13 | 27-13 | 24-16 | 21-19 | 57 |
| 47 | `harvest_aware_dynamic_shed_valving_and` | 0.760 | 72.2 % | 8 | **75.4 %** | 68 / 76 | $81,836 | 38-2 | 38-2 | 29-11 | 29-11 | 27-13 | 27-13 | 23-17 | 20-20 | 33 |
| 86 | `phased_pipeline_premium_sell` | 0.755 | 71.2 % | 8 | **75.0 %** | 69 / 74 | $81,827 | 40-0 | 40-0 | 26-14 | 25-15 | 26-13-1T | 28-12 | 25-15 | 18-22 | 64 |
| 67 | `severity_scaled_oracle_undercut` | 0.755 | 71.2 % | 8 | **75.0 %** | 68 / 75 | $81,820 | 40-0 | 40-0 | 27-13 | 25-15 | 26-14 | 28-12 | 24-16 | 18-22 | 37 |
| 9 | `adaptive_endgame_liquidation_windows` | 0.774 | 74.3 % | 7 | **74.3 %** | 66 / 82 | $82,026 | 40-0 | 40-0 | 28-12 | 26-14 | 23-17 | 22-18 | 29-5-6T | – | 0 |
| 78 | `decaying_endgame_price_floors_and_var_` | 0.751 | 70.6 % | 8 | **74.3 %** | 68 / 74 | $81,824 | 40-0 | 40-0 | 26-14 | 25-15 | 26-14 | 28-12 | 23-17 | 18-22 | 74 |
| 64 | `oracle_dynamic_fertilizer_monetization` | 0.749 | 70.3 % | 8 | **73.9 %** | 68 / 73 | $81,826 | 40-0 | 40-0 | 27-13 | 24-16 | 25-15 | 27-13 | 24-16 | 18-22 | 37 |
| 74 | `yield_weighted_farmer_rescue` | 0.749 | 70.3 % | 8 | **73.9 %** | 68 / 73 | $81,824 | 40-0 | 40-0 | 26-14 | 24-16 | 26-14 | 28-12 | 23-17 | 18-22 | 37 |
| 62 | `oracle_dynamic_fertilizer_and_idle_han` | 0.749 | 70.3 % | 8 | **73.9 %** | 66 / 74 | $81,813 | 40-0 | 40-0 | 27-13 | 25-15 | 24-16 | 27-13 | 24-16 | 18-22 | 49 |
| 56 | `spatial_priority_hand_rescues` | 0.747 | 70.0 % | 8 | **73.6 %** | 66 / 74 | $81,817 | 40-0 | 40-0 | 27-13 | 25-15 | 25-15 | 26-14 | 23-17 | 18-22 | 39 |
| 37 | `value_weighted_idle_worker_rescue` | 0.747 | 70.0 % | 8 | **73.2 %** | 66 / 74 | $81,817 | 40-0 | 40-0 | 27-13 | 25-15 | 24-16 | 26-14 | 23-17 | 19-21 | 0 |
| 72 | `severity_scaled_cadence_bypass_undercu` | 0.745 | 69.7 % | 8 | **73.2 %** | 66 / 74 | $81,819 | 40-0 | 40-0 | 27-13 | 24-16 | 25-15 | 26-14 | 23-17 | 18-22 | 56 |
| 75 | `time_decaying_endgame_price_floors` | 0.745 | 69.7 % | 8 | **73.2 %** | 66 / 74 | $81,818 | 40-0 | 40-0 | 27-13 | 24-16 | 25-15 | 26-14 | 23-17 | 18-22 | 72 |
| 65 | `severity_scaled_cadence_undercutting` | 0.742 | 69.1 % | 8 | **72.5 %** | 66 / 72 | $81,812 | 40-0 | 40-0 | 27-13 | 23-17 | 24-16 | 26-14 | 23-17 | 18-22 | 62 |
| 48 | `distance_optimized_rescues` | 0.727 | 66.6 % | 8 | **70.7 %** | 66 / 68 | $81,842 | 40-0 | 39-1 | 26-14 | 22-18 | 24-16 | 25-15 | 22-18 | 15-25 | 27 |
| 59 | `none` | 0.727 | 66.6 % | 8 | **70.7 %** | 66 / 68 | $81,841 | 40-0 | 39-1 | 26-14 | 22-18 | 24-16 | 25-15 | 22-18 | 15-25 | 48 |
| 63 | `severity_scaled_cadence_undercut` | 0.727 | 66.6 % | 8 | **70.7 %** | 66 / 68 | $81,841 | 40-0 | 39-1 | 26-14 | 22-18 | 24-16 | 25-15 | 22-18 | 15-25 | 59 |
| 14 | `fix_liquidation_bug_and_undiversified_` | 0.740 | 68.6 % | 7 | **68.6 %** | 61 / 76 | $82,042 | 40-0 | 40-0 | 28-12 | 23-17 | 22-18 | 22-18 | 17-8-15T | – | 11 |
| 27 | `value_aware_shed_pressure` | 0.740 | 68.6 % | 7 | **68.6 %** | 61 / 76 | $82,042 | 40-0 | 40-0 | 28-12 | 23-17 | 22-18 | 22-18 | 17-8-15T | – | 14 |
| 34 | `oracle_sized_town_shop_batches` | 0.737 | 68.2 % | 7 | **68.2 %** | 61 / 75 | $82,025 | 40-0 | 40-0 | 28-12 | 24-16 | 22-18 | 22-18 | 15-7-18T | – | 10 |
| 61 | `time_decaying_endgame_price_floors_and` | 0.717 | 65.0 % | 8 | **68.2 %** | 58 / 72 | $81,759 | 40-0 | 40-0 | 28-12 | 24-16 | 22-18 | 22-18 | 15-10-15T | 17-22-1T | 34 |
| 39 | `price_elastic_monopoly` | 0.711 | 64.1 % | 8 | **67.5 %** | 56 / 72 | $81,755 | 40-0 | 40-0 | 28-12 | 24-16 | 22-18 | 22-18 | 13-8-19T | 16-23-1T | 0 |
| 42 | `price_elastic_monopoly_batch` | 0.710 | 63.7 % | 8 | **67.5 %** | 56 / 71 | $81,755 | 40-0 | 40-0 | 28-12 | 24-16 | 22-18 | 22-18 | 13-7-20T | 15-24-1T | 34 |
| 53 | `inventory_scaled_cadence_bypass_batchi` | 0.710 | 63.7 % | 8 | **67.5 %** | 56 / 71 | $81,755 | 40-0 | 40-0 | 28-12 | 24-16 | 22-18 | 22-18 | 13-7-20T | 15-24-1T | 42 |
| 46 | `urgency_weighted_adjacent_rescue` | 0.710 | 63.7 % | 8 | **67.1 %** | 58 / 69 | $81,757 | 40-0 | 40-0 | 28-12 | 23-17 | 22-18 | 22-18 | 13-9-18T | 16-23-1T | 34 |
| 10 | `revert_harmful_features_adaptive_endga` | 0.727 | 66.4 % | 7 | **66.4 %** | 59 / 74 | $82,021 | 40-0 | 40-0 | 28-12 | 24-16 | 22-18 | 22-18 | 10-1-29T | – | 5 |
| 51 | `pipeline_market_arbiter_feed_defense` | 0.710 | 63.7 % | 8 | **66.4 %** | 57 / 71 | $81,757 | 40-0 | 33-7 | 25-15 | 22-18 | 23-17 | 23-17 | 20-20 | 18-22 | 39 |
| 85 | `adaptive_fertilizer_and_capital_prunin` | 0.704 | 62.8 % | 8 | **66.4 %** | 59 / 66 | $81,716 | 39-1 | 39-1 | 26-14 | 19-21 | 19-21 | 19-21 | 25-15 | 15-25 | 63 |
| 29 | `price_floor_aware_oracle_cadence_bypas` | 0.725 | 66.1 % | 7 | **66.1 %** | 61 / 71 | $82,040 | 40-0 | 40-0 | 27-13 | 20-20 | 20-20 | 20-20 | 18-7-15T | – | 14 |
| 11 | `item_specific_oracle_frontrun` | 0.725 | 66.1 % | 7 | **66.1 %** | 59 / 74 | $82,020 | 40-0 | 40-0 | 28-12 | 24-16 | 22-18 | 22-18 | 9-1-30T | – | 0 |
| 77 | `adaptive_fertilizer_monetization` | 0.702 | 62.5 % | 8 | **66.1 %** | 57 / 68 | $81,776 | 40-0 | 40-0 | 27-13 | 20-20 | 20-20 | 20-20 | 18-7-15T | 15-25 | 29 |
| 19 | `revert_proactive_idle_add_oracle_caden` | 0.701 | 62.1 % | 7 | **62.1 %** | 52 / 72 | $82,090 | 40-0 | 34-6 | 24-16 | 20-20 | 19-21 | 19-21 | 18-15-7T | – | 16 |
| 17 | `continuous_shed_pressure_surplus_selli` | 0.701 | 62.1 % | 7 | **62.1 %** | 55 / 69 | $82,002 | 40-0 | 40-0 | 27-13 | 22-18 | 22-18 | 22-18 | 1-39 | – | 11 |

## Tie-break on fresh seeds (840 games)

```
Tie-break round-robin on fresh seeds: 840 games, 611 s wall
seat-0 seeds 5104..7023, seat-1 seeds 90108..92027 (20 + 20 per pairing)

program         W    L   T    WR%  seat0%  seat1%  avg cash  BT-Elo crash
g81           134  106   0   55.8    57.5    54.2    88,431       0     0
g33_crowned   133  107   0   55.4    53.3    57.5    88,438      -3     0
g60           132  108   0   55.0    53.3    56.7    88,234      -5     0
g88           123  117   0   51.2    48.3    54.2    88,368     -28     0
g55           119  121   0   49.6    46.7    52.5    88,235     -38     0
g82           105  135   0   43.8    42.5    45.0    88,173     -73     0
r1g62_seed     94  146   0   39.2    36.7    41.7    88,348    -101     0

head-to-head (row vs column, W-L-T from the row's side):
               r1g62_seed  g33_crowned          g55          g60          g81          g82          g88
r1g62_seed              -      13-27-0      16-24-0      16-24-0      18-22-0      17-23-0      14-26-0
g33_crowned       27-13-0            -      21-19-0      21-19-0      20-20-0      22-18-0      22-18-0
g55               24-16-0      19-21-0            -      15-25-0      16-24-0      23-17-0      22-18-0
g60               24-16-0      19-21-0      25-15-0            -      15-25-0      29-11-0      20-20-0
g81               22-18-0      20-20-0      24-16-0      25-15-0            -      25-15-0      18-22-0
g82               23-17-0      18-22-0      17-23-0      11-29-0      15-25-0            -      21-19-0
g88               26-14-0      18-22-0      18-22-0      20-20-0      22-18-0      19-21-0            -
```
