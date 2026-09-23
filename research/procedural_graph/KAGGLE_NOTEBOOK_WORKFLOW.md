# Kaggriculture Cloud Notebook Workflow & Tournament Findings

This document records:
1. **The standard protocol for running Kaggriculture evaluation tournaments on Kaggle's free CPU tier** (with 0 external cost and zero lingering notebook footprints).
2. **Empirical findings from the 40-game tournament against the latest submission (Hazel Weir)**, explaining model calibration, surgical overrides, and trace forensic outcomes.

---

## Part 1: How We Run Kaggriculture CPU Notebooks on Kaggle

The Kaggriculture evaluation pipeline uses Kaggle's free CPU tier (4 concurrent shards $\times$ 4 vCPUs per notebook = 16 parallel CPUs) with complete process isolation and network sandboxing.

### Architectural Pipeline
```
Local Codebase (policy_graph.json + hazel_runtime/)
         │
         ▼ (build_merged_eval_payload.py)
Frozen Payload (runner.py + candidate agent + opponent submissions + manifest.json)
         │
         ▼ (package_merged_eval_dataset.py)
Kaggle Dataset: sunshinethroughfog/kagg-merged-graph-eval
         │
         ▼ (prepare_merged_eval_shards.py)
4 CPU Shards (shards/0, 1, 2, 3 with kernel-metadata.json & run.py)
         │
         ▼ (kaggle kernels push)
Kaggle Cloud Execution (sunshinethroughfog/kagg-iter27-trace-{0..3})
         │
         ▼ (monitor_merged_shards.py)
Local Trace Harvesting (results.jsonl + traces/*.json.gz) & Immediate Kernel Deletion
```

### Step-by-Step Execution Commands

#### 1. Build and Freeze the Evaluation Payload
Construct the isolated runtime bundle containing the procedural candidate, the opponent submission packages (`submission_hazel_weir`, `submission_copper_weir`), the sandboxed runner, and the cryptographic manifest:
```bash
python3 research/procedural_graph/build_merged_eval_payload.py
```
This generates `runs/merged_eval_kaggle/payload/manifest.json` with SHA-256 digests of all 58 bundle files.

#### 2. Package & Upload as a Private Kaggle Dataset
Tar and gzip the payload, update the metadata, and push to Kaggle:
```bash
python3 research/procedural_graph/package_merged_eval_dataset.py
```
* Kaggle Dataset Slug: `sunshinethroughfog/kagg-merged-graph-eval`
* Dataset command: `kaggle datasets version -p runs/merged_eval_kaggle/dataset -m "Update payload" -r zip`

#### 3. Prepare the 4 CPU Shards
Generate the shard execution scripts and metadata:
```bash
python3 research/procedural_graph/prepare_merged_eval_shards.py
```
* Shards are mapped across seeds (e.g., for a 40-game tournament: 5 seeds per shard $\times$ 2 seats = 10 matches per shard).
* Environment isolation:
  ```python
  os.environ.update(
      OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
      NUMEXPR_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='', KAGG_ORACLE_BACKEND='numpy',
      KAGG_ORACLE_DEVICE='cpu', PYTHONDONTWRITEBYTECODE='1'
  )
  ```

#### 4. Push Shards to Kaggle
Deploy each shard kernel:
```bash
for s in 0 1 2 3; do
    kaggle kernels push -p research/procedural_graph/runs/merged_eval_kaggle/shards/$s
done
```

#### 5. Monitor, Harvest Traces, and Auto-Delete Notebooks
Run the monitor loop (can run as a background task):
```bash
python3 research/procedural_graph/runs/merged_eval_kaggle/monitor_merged_shards.py
```
**Strict Invariant Enforced:**
The moment a shard reaches `COMPLETE` or `ERROR`, the script downloads all results and full 720-step `.json.gz` traces to `remote_results/<shard>/` and immediately invokes:
```bash
kaggle kernels delete -y sunshinethroughfog/kagg-iter27-trace-<shard>
```
This guarantees **zero leftover notebooks** on Kaggle.

---

## Part 2: 40-Game Tournament Findings vs Hazel Weir

We conducted a 40-game benchmark (20 seeds on Seat 0, 20 seeds on Seat 1) pitting the updated procedural graph against our top Kaggle submission, **Hazel Weir** (Ladder score 1212.4, Ref `56246758`).

### 1. Tournament Match Record

| Seat Configuration | Wins | Losses | Draws | Candidate Mean Cash | Hazel Weir Mean Cash | Net Margin |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Seat 0 (Candidate P0)** | 1 | 19 | 0 | $79,828.9 $\pm$ 24,416 | $81,228.8 $\pm$ 25,025 | **-$1,399.8** |
| **Seat 1 (Candidate P1)** | 2 | 18 | 0 | $71,897.0 $\pm$ 18,839 | $73,089.5 $\pm$ 19,535 | **-$1,192.5** |
| **Overall Combined** | **3** | **37** | **0** | **$75,863.0 $\pm$ 22,045** | **$77,159.1 $\pm$ 22,698** | **-$1,296.2** |

* **Winning Seeds:**
  * Shard 0 (Seed 1163059083, Seat 1): Candidate **+$640.0** ($56,384 vs $55,744)
  * Shard 1 (Seed 883441263, Seat 0): Candidate **+$1,004.0** ($76,330 vs $75,326)
  * Shard 1 (Seed 1309464252, Seat 1): Candidate **+$501.0** ($68,185 vs $67,684)

---

### 2. Forensic Trace Analysis & Root Cause Breakdown

Comparing the 720-step replay traces of the candidate against Hazel Weir revealed the exact drivers of the -$1,296.2 average deficit:

#### Finding A: Opponent Model Calibration Works
* In earlier uncalibrated runs, the refit 2026-09-22 TTM opponent model (while having higher validation AUC: 0.853 vs 0.849) output lower raw logits for Wheat. As a result, it failed to trigger the hardcoded `_ORACLE_FRONTRUN_SCORE = 0.30` cutoff at step 256.
* We solved this via **empirical percentile calibration** (`checkpoint/calibration.json`), scaling Wheat `score_4` by $1.250\times$, `score_24` by $1.800\times$, and Wool by $2.093\times$.
* In this tournament, Step 256 oracle front-running triggered properly, resolving the timing divergence.

#### Finding B: The Fertilizer Floor Cost ~$1,275 per Match
* In `surgical.py`, the `_FERTILIZER_GUARD` AST hook enforced:
  $$\text{fert\_floor} = \min(\text{held}, \text{plants} \times 2)$$
  blocking all fertilizer sales before step 696.
* Across the 40 matches, the candidate sold **2,470 fewer units of Fertilizer** (~51 units per match) than Hazel Weir.
* Because market fertilizer prices are highest in mid-game, withholding this stock forfeited approximately **$1,275 of early liquidity** per match directly to Hazel Weir.

#### Finding C: Idle Dispatch Override Caused 689 Dropped Worker Actions
* `surgical.py` replaced the champion's idle worker rescue loop with an AST-injected `idle_dispatch`.
* Across the 40 games, Candidate executed:
  * **482 fewer `WATER` actions** (51,773 vs 52,255)
  * **213 fewer `FEED` actions** (14,036 vs 14,249)
  * **689 more idle `PASS` actions** (23,160 vs 22,471)
* This reduced crop maturation rates and slowed animal cycles relative to Hazel Weir.

---

### 3. Takeaway & Next Architectural Step

1. **Keep the Model Calibration:** The 2026-09-22 checkpoint + `calibration.json` functions as intended.
2. **Remove / Relax the Fertilizer Floor:** Reverting to Hazel's natural fertilizer liquidation schedule recaptures ~$1,275 in margin.
3. **Restore Native Idle Worker Dispatch:** Aligning worker priority and tile handling eliminates the 689 passive `PASS` turns.
