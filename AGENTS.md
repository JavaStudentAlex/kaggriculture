# AGENTS.md — operating guide for the Kaggriculture opponent model

Read this before touching training, fine-tuning, or data. It records what exists,
where it lives, how to run it, and the rules that keep results reproducible.
Last updated 2026-09-12 00:00 UTC. Detailed evaluation methodology and
numbers: `research/opponent_model/EVALUATION.md`.

## 1. What this is

A TinyTimeMixer (`granite-tsfm`, `tsfm_public`) that watches one seat of a
Kaggriculture game and predicts, for each of the 9 products and each of the
next 96 turns (4 game days), how many units the **opponent** will sell.
Its value is in *ranking* (which product-turns carry a sell), not in unit
counts. Input: 141 leak-free features per turn (`features.py`, `extract.py`)
plus the 9 `log1p` targets as channels, context 512 turns; ~1.03 M params.

Current best: **`models/ttm_v3_h96_ft_2026-09-10/`** — v3 (`models/ttm_v3_h96/`) fine-tuned
on the five newest days. Only these two checkpoints are kept; fine-tune runs are
deleted once their best is promoted to `models/` (section 6).
The model is not wired into a playing agent yet (`src/` is a placeholder).

## 2. Repo map

| path | what |
|---|---|
| `models/ttm_v3_h96/` | the base model: `model.safetensors`, `config.json`, **`scaler.npz`** (input mean/std — required at inference), metrics, README |
| `models/ttm_v3_h96_ft_2026-09-10/` | the current best: v3 fine-tuned on 09-06..09-10, recency-weighted (section 6.1); same file layout |
| `replays/kaggriculture-episodes-<date>.zip` | Kaggle's daily replay datasets, 2026-07-30 → 09-10 so far (43 days, 24 GB); not in git — re-download with the command in 4.1 |
| `research/opponent_model/` | all model code: `extract.py` / `extract_parallel.py` (replays → shards), `features.py`, `mechanics.py`, `ttm_dataset.py` (windows, splits, scaler), `train_ttm.py`, `evaluate.py` (the one scorer; uses a checkpoint's `scaler.npz` when present) |
| `research/opponent_model/dataset_v2/` | 42 shards 07-30..09-09 — **frozen**, v3's training set |
| `research/opponent_model/dataset_daily/` | shards for days after the v3 cutoff (09-10 …), one per day |
| `research/opponent_model/runs/<run>/` | working dirs of runs (`best/`, `scores.json`, `scaler.npz`); **not in git** — promoted checkpoints move to `models/`. `runs/ttm_v2/best` (the v2 checkpoint v3 was warm-started from) lives here on ceph only |
| `research/opponent_model/ops/` | job scripts: `env.sh`, `copy.sh` (SSD staging), `train_v3.sh` (base model from scratch), `finetune.sh` (engine) with the named recipes `finetune_1day.sh`, `finetune_4days.sh`, `finetune_forward_check.sh` |
| `research/opponent_model/logs/` | copies of every job log (the live ones are on the SSD); not in git |
| `.agents/skills/` | Kaggle account/API and simulation-competition how-tos |
| `shinka/`, `shinka_results*/` | ShinkaEvolve agent-evolution experiments (separate track) |

## 3. Infrastructure and job conventions

- **Pod**: 256 CPUs, 1 TB RAM, 3× A100 40 GB. `/home/jovyan` is ceph (slow, persistent);
  **`/results/kagg` is a local SSD that is wiped on pod restart** (it vanished on
  2026-09-11). Everything heavy runs from the SSD and is rsynced home at the end.
- **Rebuild after a restart** (each waits for the previous one's marker):
  `ops/env.sh` (uv venv `/results/kagg/venv-cuda`, python 3.11, torch cu126 — driver 535
  cannot run cu13x wheels — `granite-tsfm`, `kaggle==2.2.4`, `kaggle-environments==1.32.7`;
  prints `ENV_EXIT=0`) → `ops/copy.sh` (stages `replays/*.zip`, `dataset_v2`, the v2
  checkpoint onto the SSD; prints `COPY_EXIT=0`). Also re-copy `dataset_daily/` and
  `models/` if a job needs them on the SSD.
- **Long jobs run in tmux**, named `kagg-<job>`, with output teed to
  `/results/kagg/logs/<job>.log` and a final `<NAME>_EXIT=<code>` line
  (`V3_DONE`, `FT_DONE`). Check progress with
  `tr '\r' '\n' < /results/kagg/logs/<job>.log | grep -E "eval_auc|/775200"` (tqdm uses `\r`).
- GPUs go to 0 % for ~2–3 min at every epoch boundary (evaluation + checkpoint
  write + dataloader restart). That is normal, not a hang.
- **Never `pkill -f`**, and when using `ps -eo pid,cmd | grep '[p]attern'` make sure
  the pattern does not also appear elsewhere in the same command line (a kill loop
  killed its own shell that way on 2026-09-11). Prefer `tmux kill-session -t <name>`.
- Kaggle CLI: `/results/kagg/venv-cuda/bin/kaggle` (or `~/.local/bin/kaggle`).
  One account, credentials in `~/.kaggle` (never print or copy them), read-only use
  from scripts, submissions are manual (5/day).

## 4. Data

### 4.1 Source: Kaggle's daily datasets (the only source we use)
- `kaggle/kaggriculture-episodes-<YYYY-MM-DD>` — "top episode replays ranked by
  average agent rating, capped at 20 GiB per day", published **~00:10 UTC the next
  day**. 660–930 episodes/day (657 on 09-10), ~0.5 GB zip, members `<episode_id>.json`
  (~31 MB each: 720 steps, both seats' observations incl. `private`) + `manifest.csv`.
  Index with per-day counts and scores: `kaggle/kaggriculture-episodes-index`.
- Download: `kaggle datasets download kaggle/kaggriculture-episodes-<date> -p replays/`
  (`ops/finetune.sh` does this itself and waits until the day is published).
- The number of distinct teams per day (45–73) is a byproduct of the size cap, not
  a "top-N teams" rule. The datasets contain **none of our own games**.
- The full ladder is ~330k games/day; individual replays can be pulled through the
  API but the endpoint throttles to ~300/h. A stratified API fetcher was built and
  then **removed on 2026-09-11 by decision** — do not rebuild it unless asked.

### 4.2 Shards
- `extract_parallel.py --replays <dir of zips> --out <dir> --workers 16` turns each zip
  into `<zip stem>.npz` (`X` 141 features, `Y` 9 targets, `D` = episode/step/seat/contested,
  `feature_names`, `products`); 1,438 rows per episode (2 seats × 719 turns), ~50 MB/day,
  ~10 min/day. It skips zips whose shard already exists in `--out`.
- `dataset_v2` = 42 shards (29,525 episodes, 59,050 series). **Never add shards to it**:
  the series split, the scaler and every number in EVALUATION.md are derived from that
  exact list. New days go to `dataset_daily/`.
- Loaders glob `*.npz`, so any file name works; the fine-tune script relies on the
  date in the name (`kaggriculture-episodes-YYYY-MM-DD.npz`) for recency weighting.

## 5. Models and how to load them

| model | supervision | data / split | key numbers |
|---|---|---|---|
| `runs/ttm_v2/best` | 24 steps | dataset_v2, series split | honest t+1 (clean, uniform hours): AUC 0.901 / AP 0.423 (EVALUATION.md §5) |
| **`models/ttm_v3_h96`** | 96 steps, warm-started from v2 | dataset_v2, series split seed 0 | val AUC 0.8695 / AP 0.296 pooled over 96 horizons; clean t+0 AUC 0.912 / AP 0.449; day 1–4 AUC 0.892 / 0.891 / 0.866 / 0.830 |

v3 recipe (`ops/train_v3.sh`): 3 GPUs via torchrun, batch 64/GPU, window stride 8,
eval stride 5, lr 1e-4 with reduce-on-plateau (×0.5, patience 2), early stopping
patience 8 on `eval_auc_any_sell` (`--metric-horizon all`), 3,876 steps/epoch,
~8 min/epoch on free GPUs. Best epoch 19, stopped at 27 on 2026-09-11 22:14 UTC.
`RESUME=<checkpoint dir>` resumes weights, optimizer, scheduler and the
early-stopping counter (`train_ttm.py --resume-from`).

**Loading**: `TinyTimeMixerForPrediction.from_pretrained(<dir>)` **plus** `<dir>/scaler.npz`:
inputs are `(x - mean) / std`, concatenated with `log1p(clip(y, 0))` of the 9 targets
as channels 141–149 → `past_values` of shape `(B, 512, 150)`; the model's
`prediction_outputs` is `(B, 96, 9)` in log1p units (`ttm_dataset.OpponentSupplyWindows`).
Every model dir and every run dir written since 2026-09-11 contains its `scaler.npz`;
`evaluate.py` uses it when present and otherwise recomputes the v3 scaler from
`dataset_v2` with the seed-0 split, which only works while `dataset_v2` is unchanged.

## 6. Fine-tuning on new days

Three ready-made recipes in `research/opponent_model/ops/` (each takes `DAY=`, default yesterday UTC):

| script | what |
|---|---|
| `finetune_1day.sh` | newest day only, val = 10 % of its episodes |
| `finetune_4days.sh` | **production recipe**: newest day + 4 previous, recency-weighted, val = 10 % of the newest day |
| `finetune_forward_check.sh` | same training on `DAY`..`DAY-4`, validated on the whole of `DAY+1` (waits for Kaggle to publish it) |

All three call the engine `finetune.sh`, whose settings are environment variables:

| var | meaning | recommended |
|---|---|---|
| `DAY` | newest training day (default: yesterday UTC) | the day Kaggle just published |
| `PREV_DAYS` | days before `DAY` added to training in full | `4` |
| `DECAY`, `STRIDE` | recency weighting: `DAY` is windowed at `STRIDE`, each day further back at `STRIDE·DECAY^age` — with `DECAY=2` every day back contributes half as many windows | `DECAY=2 STRIDE=4` (newest day ≈ 48 % of every epoch with 4 previous days) |
| `VAL_DAY` | validate on this **whole** day (forward in time); `DAY` is then trained on in full. Waits for Kaggle to publish it | the day after `DAY` for the forward check |
| `GPUS`, `LR` | torchrun on N GPUs; LR scales linearly with the batch (2e-5 at 1×64) | `GPUS=3 LR=6e-5` |
| `BASE` | weights + `scaler.npz` to start from | default `models/ttm_v3_h96`; point at a `runs/ft_*/best` to chain |
| `PORT` | torchrun master port, change when two runs overlap | |

Fixed inside: `--split episode` (10 % of `DAY`'s episodes held out unless `VAL_DAY`),
`--eval-on-start` (**epoch 0 in the log = the base model scored on the same val
windows** — the number a fine-tune must beat), the base model's scaler (never refit;
a refit on top-only data shifts the inputs under the weights), `--metric-horizon all`,
30 epochs max (the recency run was still creeping up at 30 — raise `--epochs` in the script if that repeats), early stopping patience 5, plateau ×0.5 patience 2, batch 64/GPU.
Output: `/results/kagg/runs/<TAG>` → `research/opponent_model/runs/<TAG>/{best,scores.json,scaler.npz}`
with `TAG = ft_<DAY>[_p<PREV>][_val<VAL_DAY>][_g<GPUS>][_d<DECAY>][_s<STRIDE>]`; the
day's shard lands in `dataset_daily/` (SSD + home). Intermediate checkpoints are deleted.

### 6.1 What we have measured so far (val = the same 65 held-out episodes of 2026-09-10, 2,990 windows)

| run | training data | best AUC | best AP |
|---|---|---|---|
| v3 untouched (epoch 0) | — | 0.8115 | 0.1396 |
| `ft_2026-09-10` | 09-10 only, 1 GPU, lr 2e-5 | 0.8326 (ep 11) | 0.1605 |
| `ft_2026-09-10_p4_g3` | 09-06..09-10 equal weight, 3 GPUs | 0.8374 (ep 9) | 0.1633 |
| `ft_2026-09-10_p4_g3_d2_s4` → **`models/ttm_v3_h96_ft_2026-09-10`** | 09-06..09-10, recency-weighted (DECAY 2, STRIDE 4) | **0.8655** (ep 24, ran to the 30-epoch cap) | **0.2124** |

Takeaways: v3 scores much lower on the newest day (0.81) than on its own validation
(0.87) — the top of the ladder drifts, which is the reason for daily refits; a few
epochs of fine-tuning recover most of it; and **weighting toward the newest day is
by far the biggest lever** (+0.03 AUC / +0.05 AP over equal weighting).
Caveat: these are within-day tests. The forward-in-time test (train through 09-10,
validate on all of 09-11) is `ft_2026-09-10_p4_val2026-09-11_g3_d2_s4`, launched to
run as soon as the 09-11 dataset is published; read its epoch 0 vs best.

## 7. Daily routine when a new day appears

Kaggle publishes day `D` at ~00:10 UTC on `D+1`. Then, from the repo root:

1. **Forward check of the recipe** (does refitting on `D-1` help on `D`?):
   `tmux new -d -s kagg-ft_val<D> 'DAY=<D-1> bash research/opponent_model/ops/finetune_forward_check.sh 2>&1 | tee /results/kagg/logs/finetune_val<D>.log'`
   It downloads and extracts `D` itself (waiting if not published yet).
   Success = best epoch beats epoch 0 on the whole of `D`.
2. **Production refit** including the new day:
   `tmux new -d -s kagg-ft_<D> 'DAY=<D> bash research/opponent_model/ops/finetune_4days.sh 2>&1 | tee /results/kagg/logs/finetune_<D>.log'`
   (val = 10 % of `D`'s episodes; `BASE=models/<previous fine-tune>` chains from it
   instead of restarting from v3 — untested so far, compare both once).
3. Results: `research/opponent_model/runs/ft_<D>.../{best,scores.json,scaler.npz}`;
   add the row to the table in section 6.1 and to EVALUATION.md. If it is the new best,
   copy `best/*`, `scaler.npz`, `scores.json` to `models/ttm_v3_h96_ft_<D>/` with a README
   and delete the run dirs (home and SSD) — only promoted checkpoints are kept.
4. The two runs need all 3 GPUs each — run them one after the other (each takes
   10–20 min); `ops/finetune.sh` deletes its checkpoints, so nothing to clean up.

## 8. Evaluation — what the numbers mean

- The Trainer's `eval_auc_any_sell` / `eval_ap_any_sell` pool all supervised horizons
  (`--metric-horizon all`, eval stride 5, coprime with the 24-turn day so every hour
  is represented). Use them for model selection; quote headline numbers from
  `evaluate.py` (EVALUATION.md §8) on the clean set — it picks up the checkpoint's
  `scaler.npz` automatically, so it works for `models/*` and for fine-tunes.
- Pitfalls documented in EVALUATION.md: the series split leaks seats (v2/v3 used it
  to keep the warm start valid; new runs use `--split episode`); a window grid aliased
  onto the end-of-day dump flatters AP ~2× (v2's old AUC 0.963 / AP 0.837 are superseded
  by 0.901 / 0.423); predicted magnitudes are shrunk 3–35× and need calibration.
- Sell rate is ~1.2–2 % of (turn, product) cells, with 3–7× spikes at game hour 0;
  any metric that credits zeros is uninformative.

## 9. Rules

1. `dataset_v2` is frozen; new data goes to `dataset_daily`.
2. Never refit or drop the scaler when starting from existing weights; ship
   `scaler.npz` next to every model.
3. Kaggle's daily datasets are the data source. One account, read-only from scripts.
4. Long jobs: tmux + SSD + `tee` + exit marker; rsync results home at the end.
5. Keep `EVALUATION.md` and this file's section 6.1 current when a run finishes.

## 10. Runs and logs (2026-09-11)

| tmux / log | what | state |
|---|---|---|
| `kagg-train_v3` / `train_v3.log`, `train_v3.attempt2.log` | v3 training (attempt 2 epochs 1–6, resumed run 7–27) | done, `V3_DONE` 22:20 UTC |
| `finetune_0910.log` | fine-tune on 09-10 only, 1 GPU | done, run dir deleted (numbers in 6.1) |
| `kagg-ft_p4_g3` / `finetune_p4_g3.log` | 5 days equal weight, 3 GPUs | done, run dir deleted (numbers in 6.1) |
| `kagg-ft_p4_d2` / `finetune_p4_d2.log` | 5 days recency-weighted, 3 GPUs | done → promoted to `models/ttm_v3_h96_ft_2026-09-10` |
| `kagg-ft_val0911` / `finetune_val0911.log` | forward validation on the whole 09-11 | waiting for Kaggle to publish 09-11 |
