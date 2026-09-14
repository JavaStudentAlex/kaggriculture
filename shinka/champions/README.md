# Champions

Every champion has a **codename** (neutral two words — the rule `submissions/make_submission.py` enforces for public
names: letters only, none of gen/score/win/oracle/shinka, no mechanism in the name). `CODENAMES.json` is the registry:
codename → file → origin → sha256. `pool/` and `roster/` keep the evaluator's `champ_<timestamp>_avg<cash>.py` file
names because `pool/POOL.json` and the game log reference them.

## Layout

- `pool/` — the **active pool** `evolution/evaluate.py` plays every candidate against (`POOL.json`: selection rationale,
  hashes). 15 members: the 8 curated on 2026-09-09/12 plus the 7 crowned in run 3. The evaluator crowns a candidate
  into this directory when it wins ≥ 75 % against the pool as it stands. Its game log `pool/games.jsonl` is
  git-ignored (it is regenerated; the 2026-09-14 copy is in `/results/kagg/logs/`).
- `roster/` — archive copy of every champion (same bytes as `pool/` today; keeps members if the pool is pruned later).
- `top/run2_2026-09-12/` — run 2's top programs with the fresh-seed tie-break that chose Orchard Tide (`gen_33/`).
- `top/run3_2026-09-14/` — run 3's crowned champions under their codenames (`<snake_codename>/main.py`, `metrics.json`,
  `summary.json`; `README.md` + `INDEX.json` with all 184 accepted programs).
- `submissions/copper_weir/` — **Copper Weir** as submitted 2026-09-14 (Kaggle 56239161, COMPLETE; `MANIFEST.json` has hashes,
  validation report and the Kaggle record).
- `submissions/orchard_tide/` — **Orchard Tide** packaged for Kaggle's CPU sandbox (bootstrap `main.py`, oracle on the
  numpy backend, its checkpoint); `submissions/make_submission.py` is the standard packager
  (`--champion <file> --name "<Two Words>" --validate`). Built `.tar.gz` bundles are git-ignored.
- `dependencies/mohui_v66/` — the backbone closure every champion imports (Apache LICENSE/NOTICE retained;
  provenance in `RECOVERY_PROVENANCE.json`).
- `evidence/orchard_tide_loss_audit_20260914/` — the audit of Orchard Tide's 97 public games that shaped run 3's prompt.
- `evidence/tournament_2026-09-14/` — the **all-champions tournament on fresh seeds** that ranks the pool
  (`tournament.py` / `run.sh` to rerun it; `tournament.txt` report, `tournament.json` + `games.jsonl` every game).
- `pool.json` — the 2026-09-09 round-robin that selected the original pool.

Removed 2026-09-14 (recovery tarball in `/results/kagg/logs/shinka_pruned_2026-09-14_2108.tar.gz`): the pre-oracle
roster entries (external wrappers, automatylicza replicas, distilled-trajectory agents, non-pool 09-08 seeds), the
pre-oracle `top/gen_*` programs and their `harvest_meridian` / `field_current` bundles, the `zansued` / `tamizh_local`
dependencies only those wrappers used, and the 09-09 repair/validation reports.

## Codenames

| codename | file (`pool/`, `roster/`) | origin | role |
|---|---|---|---|
| **First Furrow** | `champ_00_initial_seed.py` | the initial seed (Mohui v66 backbone + Keiz opening); regression floor of the pool | regression floor: losing to the seed means broken |
| **Cider Ridge** | `champ_20260908_191956_avg91958.py` | 2026-09-08 grandmaster-seed run; roster rank 1 of the 09-09 round-robin (the binding constraint) | strongest; the binding constraint |
| **Slate Pasture** | `champ_20260908_190355_avg92490.py` | 2026-09-08 run; roster rank 2 (only agent that takes 20 % off rank 1) | only agent that takes 20% off #1 |
| **Clover Bank** | `champ_20260908_183514_avg93196.py` | 2026-09-08 run; roster rank 3 (graded difficulty step below rank 2) | graded difficulty step below #2 |
| **Willow Ford** | `champ_20260908_150703_avg101195.py` | 2026-09-08 run; roster rank 8 (takes 10 % off rank 1 despite ranking 8th) | takes 10% off #1 despite ranking 8th |
| **Birch Hollow** | `champ_20260908_153302_avg99058.py` | 2026-09-08 run; roster rank 10 (most behaviourally distinct opponent, 58-68 % agreement) | most behaviourally distinct opponent (58-68% agreement) |
| **Amber Loft** | `champ_20260912_gen62_avg82350.py` | run 1 (2026-09-12) gen 62 oracle_cadence_bypass_and_idle: 73.3 % vs the six-pool, $82,350; seed of run 2 | best program of the 2026-09-12 Shinka run (gen 62, oracle_cadence_bypass_and_idle): 73.3% (176W-64L) vs the si |
| **Orchard Tide** | `champ_20260912_161630_avg82113.py` | run 2 (2026-09-12) gen 33 dynamic_shop_batch_sizing: crowned 77.1 % vs the seven-pool; Kaggle submission 56193386; seed of run 3 | crowned by evaluate.py during run 2 (gen 33, dynamic_shop_batch_sizing, 16:16 UTC): 77.1% (216W-64L) vs the se |
| **Meadow Lantern** | `champ_20260914_150730_avg82221.py` | run 3 (2026-09-14) gen 112 urgency_rescue_and_selling_tweaks: crowned 76.2 % vs the 8-pool at 15:07 UTC | active pool member |
| **Furrow Dawn** | `champ_20260914_151631_avg81853.py` | run 3 (2026-09-14) gen 114 non_demanded_frontrun_and_cadence_fix: crowned 76.1 % vs the 9-pool at 15:16 UTC | active pool member |
| **Quiet Barley** | `champ_20260914_152040_avg82067.py` | run 3 (2026-09-14) gen 115 oracle_champion_refined_crossover: crowned 78.9 % vs the 9-pool at 15:20 UTC | active pool member |
| **Granary Brook** | `champ_20260914_154316_avg81783.py` | run 3 (2026-09-14) gen 122 oracle_surplus_throughput: crowned 78.9 % vs the 11-pool at 15:43 UTC | active pool member |
| **Mirror Hedgerow** | `champ_20260914_195349_avg81696.py` | run 3 (2026-09-14) gen 189 seat_asymmetric_oracle_bypass_and_execution_guard: crowned 75.4 % vs the 12-pool at 19:53 UTC | active pool member |
| **Open Sluice** | `champ_20260914_195540_avg81696.py` | run 3 (2026-09-14) gen 188 volume_scaled_uncapped_frontrun_and_urgency_hand_rescue: crowned 76.2 % vs the 12-pool at 19:56 UTC | active pool member |
| **Copper Weir** | `champ_20260914_203220_avg81408.py` | run 3 (2026-09-14) gen 198 value_weighted_order_preemption: crowned 90.0 % vs the 14-pool at 20:32 UTC | active pool member - best candidate for Kaggle |

## Ranking (all-champions tournament on fresh seeds, 2026-09-14)

Round-robin of the whole pool, 40 games per pairing on seeds no run ever used (4,200), plus 60 more per pairing
among the top 5 (600); Bradley-Terry over all 4,800 games, 0 crashes (`evidence/tournament_2026-09-14/`).

| # | champion | W-L-T | WR | BT-Elo | P(#1) | note |
|---|---|---|---|---|---|---|
| 1 | **Copper Weir** | 696-104-0 | 87.0 % | 0 | 100 % | beats all 14; 83-85 of 100 vs each other finalist; closest 31-9 vs Meadow Lantern |
| 2 | Open Sluice | 498-284-18 | 62.2 % | −222 | 0 % | Copper Weir's parent (gen 188) |
| 3 | Mirror Hedgerow | 463-266-71 | 57.9 % | −231 | 0 % | ties 54 of 100 with Granary Brook (identical cash) |
| 4 | Granary Brook | 431-298-71 | 53.9 % | −262 | 0 % | ancestor of the 141 → 188 → 198 lineage |
| 5 | Cider Ridge | 331-228-1 | 59.1 % | −310 | 0 % | pre-oracle; still even or better vs run-3 ranks 7-10 |
| 6 | Slate Pasture | 315-245-0 | 56.2 % | −334 | 0 % | pre-oracle |
| 7 | Furrow Dawn | 305-255-0 | 54.5 % | −348 | 0 % | |
| 8 | Quiet Barley | 365-417-18 | 45.6 % | −349 | 0 % | 78.9 % when crowned — seed-specific |
| 9 | Clover Bank | 288-272-0 | 51.4 % | −373 | 0 % | pre-oracle |
| 10 | Meadow Lantern | 283-277-0 | 50.5 % | −380 | 0 % | |
| 11 | Willow Ford | 223-336-1 | 39.8 % | −468 | 0 % | pre-oracle |
| 12 | Orchard Tide | 212-347-1 | 37.9 % | −485 | 0 % | pool copy plays with the 256-ctx checkpoint; the Kaggle bundle (512-ctx) is stronger |
| 13 | Amber Loft | 187-372-1 | 33.4 % | −525 | 0 % | |
| 14 | Birch Hollow | 99-460-1 | 17.7 % | −704 | 0 % | pre-oracle |
| 15 | First Furrow | 12-547-1 | 2.1 % | −1082 | 0 % | regression floor |

Finalists have 800 games (560 + 240), the rest 560. Seat 0 won 50.0 % of decided games (no seat bias). Per-champion
ranks are also in `CODENAMES.json` (`tournament_2026_09_14`).

Lineage: First Furrow → (09-08 run) Cider Ridge / Slate Pasture / Clover Bank / Willow Ford / Birch Hollow → run 1
Amber Loft → run 2 **Orchard Tide** (Kaggle 56193386) → run 3 Meadow Lantern, Furrow Dawn, Quiet Barley, Granary Brook,
Open Sluice, Mirror Hedgerow, **Copper Weir** (90 % vs the 14-pool; on Kaggle since 2026-09-14 as submission 56239161). Retired public names: Harvest Meridian, Field Current (pre-oracle bundles).

## Run and evaluate

Champions run in the project environment (`kaggle-environments`, numpy; no Torch needed by the policies — the oracle
loads its checkpoint through `evolution/kagg_oracle.py`). Each champion must execute in its own process because the
shared backbone modules keep state. `evolution/eval_once.sh <program>` evaluates one program against the pool exactly
as the evolution does; `evolution/curate_pool.py` maintains `POOL.json`.
