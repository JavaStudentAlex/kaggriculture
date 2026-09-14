# Run 3 (2026-09-14) — crowned champions and the best candidate

Shinka run 3: seed = run-2 gen 33 (**Orchard Tide**, `champ_20260912_161630_avg82113.py`) with the
256-context oracle (`models/ttm_c256_h96_ft_2026-09-13`, copied into `shinka/evolution/checkpoint`) active from
turn 256 instead of 512; pool of 8, 200 generations, 4 islands. Ran 08:47–20:34 UTC to completion:
203 programs (189 correct, 14 incorrect), $48.72, ~208 s per program. The database and per-generation
directories are in `/results/kagg/shinka_results_r3` (SSD) mirrored to `shinka_results_r3/` (ceph, git-ignored);
this directory holds the copies that matter: one directory per crowned champion, named by its **codename**
(the neutral two-word public name `make_submission.py` requires), each with `main.py`, `metrics.json`
(evaluate.py output) and `summary.json` (codename, gen, lineage, per-champion breakdown, sha256).

## Codenames

| codename | gen | directory | pool / roster file |
|---|---|---|---|
| **Meadow Lantern** | 112 | `meadow_lantern/` | `champ_20260914_150730_avg82221.py` |
| **Furrow Dawn** | 114 | `furrow_dawn/` | `champ_20260914_151631_avg81853.py` |
| **Quiet Barley** | 115 | `quiet_barley/` | `champ_20260914_152040_avg82067.py` |
| **Granary Brook** | 122 | `granary_brook/` | `champ_20260914_154316_avg81783.py` |
| **Mirror Hedgerow** | 189 | `mirror_hedgerow/` | `champ_20260914_195349_avg81696.py` |
| **Open Sluice** | 188 | `open_sluice/` | `champ_20260914_195540_avg81696.py` |
| **Copper Weir** | 198 | `copper_weir/` | `champ_20260914_203220_avg81408.py` |

`pool/` and `roster/` keep the evaluator's `champ_<timestamp>_avg<cash>.py` names because `POOL.json` and
`games.jsonl` reference them; the codename ↔ gen ↔ file mapping lives here and in `INDEX.json` (`codenames`).

## Best candidate: **Copper Weir** (gen 198, `value_weighted_order_preemption`)

`copper_weir/main.py` — byte-identical (sha256 `1f3ae68faf78…`) to `champions/pool/champ_20260914_203220_avg81408.py`,
`champions/roster/champ_20260914_203220_avg81408.py` and `shinka_results_r3/gen_198/main.py`.

* **Crowned** by evaluate.py at 20:32 UTC as the second-to-last program of the run: **90.0 %**
  (504W-56L-0T, 0 crashes) against the full 14-pool, $81,409 average cash,
  seat 0 / seat 1 = 92.5 % / 87.5 %. Authored by local/claude-sonnet-5 as a `diff` on Open Sluice (gen 188, codex);
  lineage 198 ← 188 ← 141 ← 122 ← 89 ← 86 ← 60 ← 40 ← 8 ← 0 (198 Copper Weir ← 188 Open Sluice ← 141 ← 122 Granary Brook ← 89 … ← 0 Orchard Tide), island 0 throughout.

* **The edit** (verified in the diff vs Open Sluice): the seed's `_add_sell` silently dropped any new sell once the
  10-order market cap was full, and every sell section (shed pressure, oracle front-run, shop harvesting,
  pre-liquidation, day-29 liquidation) carried a `len(mkt) < 10` guard — so with the cap filled by cheap
  WHEAT/CARROT orders, MELON/WOOL and front-run batches were never placed. Now `_add_sell` evicts the
  lowest-value SELL (qty × price) when the candidate is worth > 1.1× it, the guards are gone, a shared
  `_already_selling` helper replaces the inline sums, and day-29 liquidation starts at step 693 with batch 7.

* **Why it is a real gain, not a pool artifact**: against the four old mid-lineage champions that every earlier
  best only split (Quiet Barley: 20-20 / 21-19 / 21-19 / 25-15; Open Sluice the same) it goes
  **36-4 / 37-3 / 37-3 / 35-5** — the Orchard Tide close-loss cluster (the $120–$350 margins audited in `ideas.md`) is what
  the prompt targeted and what closed.

* **Submitted to Kaggle 2026-09-14 21:24 UTC as submission 56239161** (`submissions/copper_weir/`, `CopperWeir.tar.gz`):
  `check_oracle.py numpy` max diff 1.51e-4, bundle validation PASS in the clean venv (8 games, oracle live on numpy,
  torch never imported), Kaggle validation episode 109042656 COMPLETED both seats, status COMPLETE 21:31 UTC.
* **Confirmed on fresh seeds 2026-09-14 22:09 UTC** (`../../evidence/tournament_2026-09-14/`: round-robin of the whole
  15-pool, 4,800 games on seeds no run ever used): rank 1 with 696W-104L-0T (87.0 %), 222 Elo clear of the
  runner-up Open Sluice, rank 1 in 100 % of 2,000 bootstrap resamples, a winning record against all 14 others
  (85-15 Open Sluice, 83-17 Mirror Hedgerow, 84-16 Granary Brook, 83-17 Quiet Barley at 100 games each; closest
  pairing Meadow Lantern 31-9). The crowning order of the other six does not survive fresh seeds (Quiet Barley falls
  to 8th; Open Sluice / Mirror Hedgerow / Granary Brook are the podium).

### Copper Weir against every champion (40 games each)

| champion | W-L-T | seat 0 | seat 1 | cand cash | champ cash |
|---|---|---|---|---|---|
| `00_initial_seed` | 40-0-0 | 20/20 | 20/20 | $93,257 | $75,077 |
| `150703_avg101195` | 35-5-0 | 17/20 | 18/20 | $81,060 | $80,301 |
| `153302_avg99058` | 40-0-0 | 20/20 | 20/20 | $80,681 | $78,465 |
| `183514_avg93196` | 36-4-0 | 17/20 | 19/20 | $80,056 | $79,478 |
| `190355_avg92490` | 37-3-0 | 18/20 | 19/20 | $79,903 | $79,125 |
| `191956_avg91958` | 37-3-0 | 18/20 | 19/20 | $79,943 | $79,094 |
| `161630_avg82113` | 40-0-0 | 20/20 | 20/20 | $80,831 | $79,352 |
| `gen62_avg82350` | 40-0-0 | 20/20 | 20/20 | $80,865 | $79,384 |
| `150730_avg82221` | 33-7-0 | 18/20 | 15/20 | $80,633 | $79,752 |
| `151631_avg81853` | 38-2-0 | 19/20 | 19/20 | $80,135 | $79,490 |
| `152040_avg82067` | 32-8-0 | 18/20 | 14/20 | $80,601 | $79,774 |
| `154316_avg81783` | 32-8-0 | 18/20 | 14/20 | $80,603 | $79,779 |
| `195349_avg81696` | 31-9-0 | 18/20 | 13/20 | $80,566 | $79,777 |
| `195540_avg81696` | 33-7-0 | 18/20 | 15/20 | $80,587 | $79,760 |

## Crowned during the run (>= 75 % vs the pool at evaluation time) — pool 8 → 15

Programs are evaluated against the pool as it stands, so `combined_score` / WR are only comparable at equal
pool size (column *pool*). Every crowned program is also in `champions/pool/` (active) and `champions/roster/` (archive).

| codename | gen | shinka name | crowned at | combined | WR | pool | seat 0 / 1 | avg cash | LLM | lineage |
|---|---|---|---|---|---|---|---|---|---|---|
| **Meadow Lantern** | 112 | `urgency_rescue_and_selling_tweaks` | 15:07 | 0.7864 | 76.2 % | 8 | 71 / 81 % | $82,222 | claude-opus-5 | 112←86←60←40←8←0 |
| **Furrow Dawn** | 114 | `non_demanded_frontrun_and_cadence_fix` | 15:16 | 0.7841 | 76.1 % | 9 | 77 / 75 % | $81,854 | claude-opus-5 | 114←95←91←87←83←80←53←29←26←21←0 |
| **Quiet Barley** | 115 | `oracle_champion_refined_crossover` | 15:20 | 0.8016 | 78.9 % | 9 | 74 / 83 % | $82,067 | claude-opus-5 | 115←98←89←86←60←40←8←0 |
| **Granary Brook** | 122 | `oracle_surplus_throughput` | 15:43 | 0.8003 | 78.9 % | 11 | 75 / 83 % | $81,784 | claude-opus-5 | 122←89←86←60←40←8←0 |
| **Mirror Hedgerow** | 189 | `seat_asymmetric_oracle_bypass_and_execution_guard` | 19:53 | 0.7793 | 75.4 % | 12 | 74 / 77 % | $81,697 | claude-sonnet-5 | 189←60←40←8←0 |
| **Open Sluice** | 188 | `volume_scaled_uncapped_frontrun_and_urgency_hand_rescue` | 19:56 | 0.7843 | 76.2 % | 12 | 73 / 80 % | $81,697 | headless/codex | 188←141←122←89←86←60←40←8←0 |
| **Copper Weir** | 198 | `value_weighted_order_preemption` | 20:32 | 0.8656 | 90.0 % | 14 | 92 / 88 % | $81,409 | claude-sonnet-5 | 198←188←141←122←89←86←60←40←8←0 |

## What each crowned edit did (from the LLM's own description; Copper Weir's checked against the diff)

* **Meadow Lantern** (gen 112, `urgency_rescue_and_selling_tweaks`) — Four targeted changes addressing both production preservation and revenue capture:  1. **Urgency-weighted rescue sorting** (the recommendation): Sort thirsty/hungry lists by consecutive_unwatered/consecutive_unfed FIRST, then by price. An animal at consecutive_unfed=1 (will escape at day boundary) must be rescued before one at unfed=0 (has a full grace period), regardless of price. This prevents the most critical ass…
* **Furrow Dawn** (gen 114, `non_demanded_frontrun_and_cadence_fix`) — Three targeted fixes for the ~$300 cash gap against the hardest champions:  1. Lower _PREMIUM_SELL_RATIO from 1.20 to 0.80 (captures non-demanded revenue like TOMATO/WOOL/EGG at reasonable prices instead of only at 20% premium). Increase batch from 2 to 3 for more throughput.  2. Reverse the oracle gate for non-demanded goods: instead of skipping when opponent is about to dump (which means we miss selling before the …
* **Quiet Barley** (gen 115, `oracle_champion_refined_crossover`) — Base: Program 2 (champion at 0.79 combined score, 76.25% WR).  Crossover additions: 1. `_already_selling` helper from Program 3 for cleaner code and reduced redundancy 2. `player_idx` in farm_state for metadata 3. Day-29 phased liquidation batch increased from 6 to 7 (compromise between P2's 6 and P3's 8) 4. Conservative quiet-window batch boost: non-demanded items get batch_max=3 (instead of 2) when opponent is very…
* **Granary Brook** (gen 122, `oracle_surplus_throughput`) — Replace fixed quiet-window batches with inventory-sensitive sizing. From turn 256, demanded goods can sell 3–6 additional units when forecast supply is weak and the current quote is at least the base price. Non-demanded goods receive a smaller 3–4-unit allowance only with a particularly quiet forecast. Sizing uses inventory remaining after feed reserves and existing orders; missing forecasts never qualify as evidence…
* **Mirror Hedgerow** (gen 189, `seat_asymmetric_oracle_bypass_and_execution_guard`) — Implement Seat-Asymmetric Oracle Cadence Bypass and robust market order execution to overcome the persistent ~9% win-rate penalty on Seat 0: 1. **Seat-Aware Oracle Cadence Bypass**: Make the dynamic `_bypass_thresh` seat-aware by reducing it by `0.05` across all shed tiers when on Seat 0 (`player_idx == 0`). In simultaneous quote execution, Seat 0 suffers from adverse selection if it waits; lowering the threshold tri…
* **Open Sluice** (gen 188, `volume_scaled_uncapped_frontrun_and_urgency_hand_rescue`) — Implement Volume-Scaled Uncapped Front-Running and restore urgency-first idle hand maintenance: 1. Volume-Scaled Uncapped Front-Running: In previous iterations, the dynamic batch sizing in shop demand harvesting (section 5) and oracle front-running (sections 4c and 5b) rigidly capped sales at 8 units even when the oracle forecast a massive impending opponent dump. In mirror matches or high-production scenarios, this …
* **Copper Weir** (gen 198, `value_weighted_order_preemption`) — Implement Value-Weighted Order Preemption for the 10-order market order limit.  Previously, `_add_sell` checked `if len(orders) < 10:` and silently dropped any new sell order when 10 orders were present. Furthermore, all selling loops (shed pressure valve, oracle front-running, town-shop harvesting, high-value asset pre-liquidation, and phased day-29 liquidation) contained short-circuit checks (`len(mkt) < 10` or `if…

## Run-level notes

* Throughput: 17.3 programs/h, 3.1–3.9 min/gen all day; sampling median 147 s, evaluation median 394 s (p90 548 s,
  3 eval workers; 560-game tournaments at the end took up to 680 s). Evaluation is the bottleneck.
* LLM arms: claude-opus-5 (local) 84 programs / best 0.8016 (Quiet Barley); claude-sonnet-5 (local) 33 / best 0.8656 (Copper Weir);
  gemini-3.8-flash (local) 36 / 0.7710; gemini-3.1-pro 25 / 0.7710 ($7.47); codex gpt-6/gpt-5 21 / 0.7843 (Open Sluice, $26.27).
* The prompt-evolution guard (`run_evo.py`, 7 required anchor phrases) vetoed mutated prompts that dropped anchors; the
  run stayed on the original `shinka_config.yaml` task_sys_msg (oracle facts + Orchard Tide loss audit).
* `INDEX.json` carries the summary of all 184 accepted programs (gen > 0), the crowning order, the codename map and the best candidate.
