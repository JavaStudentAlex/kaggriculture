# AGENTS.md — operating guide for the Kaggriculture opponent model

Read this before touching training, fine-tuning, or data. It records what exists,
where it lives, how to run it, and the rules that keep results reproducible.
Last updated 2026-09-12 18:40 UTC. Detailed evaluation methodology and
numbers: `research/opponent_model/EVALUATION.md`.

## 1. What this is

A TinyTimeMixer (`granite-tsfm`, `tsfm_public`) that watches one seat of a
Kaggriculture game and predicts, for each of the 9 products and each of the
next 96 turns (4 game days), how many units the **opponent** will sell.
Its value is in *ranking* (which product-turns carry a sell), not in unit
counts. Input: 141 leak-free features per turn (`features.py`, `extract.py`)
plus the 9 `log1p` targets as channels; ~1.03 M params. The context length is a
property of each checkpoint (`config.json`): 512 turns for v3 and its
fine-tunes (the oracle is silent until day 21), 256 for the v4 line in training
since 2026-09-12 18:40 UTC (forecasts from day 10 h16) -- section 12.

Current best: **`models/ttm_v3_h96_ft_2026-09-11/`** — the daily chain v3
(`models/ttm_v3_h96/`) → fine-tune on 09-06..09-10 → fine-tune on 09-07..09-11.
Only promoted checkpoints are kept; fine-tune runs are deleted once their best is
promoted to `models/` (section 6).
The model IS wired into a playing agent: the Shinka seed program serves a copy of
it as an in-game "oracle" (section 11). **Known issue with the training labels:
section 4.3.**

## 2. Repo map

| path | what |
|---|---|
| `models/ttm_v3_h96/` | the base model: `model.safetensors`, `config.json`, **`scaler.npz`** (input mean/std — required at inference), metrics, README |
| `models/ttm_v3_h96_ft_2026-09-10/` | v3 fine-tuned on 09-06..09-10, recency-weighted (section 6.1); same file layout |
| `models/ttm_v3_h96_ft_2026-09-11/` | **the current best**: the 09-10 model fine-tuned one day further (09-07..09-11); `labels.json` marks the label alignment (4.3); the copy in play is `shinka/evolution/checkpoint/` |
| `replays/kaggriculture-episodes-<date>.zip` | Kaggle's daily replay datasets, 2026-07-30 → 09-10 so far (43 days, 24 GB); not in git — re-download with the command in 4.1 |
| `research/opponent_model/` | all model code: `extract.py` / `extract_parallel.py` (replays → shards), `features.py`, `mechanics.py`, `ttm_dataset.py` (windows, splits, scaler), `train_ttm.py`, `evaluate.py` (the one scorer; uses a checkpoint's `scaler.npz` when present) |
| `research/opponent_model/dataset_v2/` | 42 shards 07-30..09-09 — **frozen**, v3's training set |
| `research/opponent_model/dataset_daily/` | shards for days after the v3 cutoff (09-10 …), one per day — legacy labels (4.3) |
| `research/opponent_model/dataset_v3/`, `dataset_daily_v3/` | the same replays with the **corrected** labels (`extract.py --alignment next_action`, 4.3): 44 days 07-30..09-11 in `dataset_v3` (v4's training set), later days in `dataset_daily_v3`; each dir carries `labels.json` naming its rule |
| `research/opponent_model/runs/<run>/` | working dirs of runs (`best/`, `scores.json`, `scaler.npz`); **not in git** — promoted checkpoints move to `models/`. `runs/ttm_v2/best` (the v2 checkpoint v3 was warm-started from) lives here on ceph only |
| `research/opponent_model/ops/` | job scripts: `env.sh`, `copy.sh` (SSD staging), `train_v3.sh` (v3 base), `extract_v3.sh` (re-extract every day with corrected labels), `train_v4.sh` (v4 base: context 256, corrected labels; section 12), `finetune.sh` (engine) with the named recipes `finetune_1day.sh`, `finetune_4days.sh`, `finetune_forward_check.sh` |
| `research/opponent_model/logs/` | copies of every job log (the live ones are on the SSD); not in git |
| `.agents/skills/` | Kaggle account/API and simulation-competition how-tos |
| `shinka/evolution/` | the ShinkaEvolve task: `initial.py` (seed agent with the oracle), `kagg_oracle.py` + `checkpoint/` (the model in play), `evaluate.py`, `shinka_config.yaml`, `launch_shinka.sh`, `mps.sh`, `check_oracle.py`; section 11 |
| `shinka/champions/` | the opponent pool (`pool/`), the full roster and their vendored dependencies (Mohui v66 backbone) |
| `shinka_results*/` | Shinka run outputs (not in git) |

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
- `extract_parallel.py --replays <dir of zips> --out <dir> --workers 16 --alignment next_action|legacy`
  turns each zip into `<zip stem>.npz` (`X` 141 features, `Y` 9 targets, `D` = episode/step/seat/contested,
  `feature_names`, `products`); 1,438 rows per episode (2 seats × 719 turns), ~50 MB/day,
  ~10 min/day (one worker per zip; 44 zips at once on the pod). It skips zips whose shard
  already exists in `--out` and writes `<out>/labels.json` with the label rule (refusing to
  mix rules in one dir). `train_ttm.py` reads that marker (through symlinks) and copies it to
  `best/labels.json`; `finetune.sh` reads the base checkpoint's copy to pick its shard dirs.
- `dataset_v2` = 42 shards (29,525 episodes, 59,050 series). **Never add shards to it**:
  the series split, the scaler and every number in EVALUATION.md are derived from that
  exact list. New days go to `dataset_daily/`.
- Loaders glob `*.npz`, so any file name works; the fine-tune script relies on the
  date in the name (`kaggriculture-episodes-YYYY-MM-DD.npz`) for recency weighting.

### 4.3 Known issue: label attribution is off by one step (found 2026-09-12)
`extract.py` labels the market flow of step *t* (`inv[t+1] - inv[t] + town_draw(t)`)
by the orders stored at replay index *t* — but Kaggle stores at index *t* the action
taken **from** observation *t−1* (verified on a live `env.run`: an order placed at
step 5 appears in `steps[6].action`, and 100 % of nonzero flows are explained by
the orders at index *t+1*, 0 % by index *t*). So a step's flow is attributed by the
previous step's orders: the opponent's real sells land on 26.6 % of steps, the shard
labels are nonzero on 9.3 % (mostly sells that followed a sell one step earlier),
and the `oppcum/opplast/oppsince` features are built from the same undercount.
**Every shard in `dataset_v2`/`dataset_daily` and both checkpoints in `models/`
carry this** ("legacy" alignment). Consequences and what was measured:
- Validation numbers in 6.1 / EVALUATION.md are against the legacy labels.
- Against the **true** opponent supply (`shinka/evolution/check_oracle.py truth`, inputs
  rebuilt as a live agent rebuilds them): the 09-10 fine-tune on the whole of 09-11
  (forward in time) — AUC 0.75 pooled over 96 horizons, 0.80 at horizon 0, 0.83 at
  horizon 24; the 09-11 fine-tune on 09-11's 65 held-out games — 0.76 / 0.84 / 0.84
  (the 09-10 model on the same games: 0.75 / 0.84 / 0.84). Useful ranking, but recall is
  low and predicted volumes are ~5× too small; and the daily fine-tune improves the
  legacy-label metric far more than the real one.
- **Fix applied 2026-09-12 18:30 UTC** (user's decision): `extract.py --alignment next_action`
  (default; `legacy` kept to reproduce the old shards) reads `steps[t+1][seat]["action"]`;
  verified equal to `check_oracle.true_supply` on every step of 3 replays (nonzero-step rate
  9 % → 27 %). All 44 days were re-extracted into `dataset_v3/` (`ops/extract_v3.sh`);
  `dataset_v2`/`dataset_daily` stay as they are for the legacy checkpoints. The v4 base
  model (section 12) is trained on `dataset_v3`. **Do not mix** rules: a fine-tune of a
  legacy checkpoint on corrected shards changes the feature semantics under the weights,
  so `finetune.sh` picks `dataset_daily_v3`/`dataset_v3` or `dataset_daily`/`dataset_v2`
  from the base checkpoint's `labels.json` and `train_ttm.py` refuses shards of two rules
  in one run (a *full* retrain warm-started from a legacy checkpoint, as v4 is, is fine --
  it prints a warning and marks the result with the shards' rule). The live oracle reads
  `labels.json` (`"alignment": "legacy" | "next_action"`) next to a checkpoint and
  rebuilds whichever accounting the checkpoint was trained on; `train_ttm.py` writes it
  (plus `scaler.npz` and `val_episodes.json`) into `best/` since 2026-09-12.

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
as channels 141–149 → `past_values` of shape `(B, context_length, 150)` with
`context_length` from the checkpoint's `config.json` (512 for v3, 256 for v4 -- TTM's
patch grid is baked into the weights, so a checkpoint accepts exactly that length and
nothing else; left-padding a shorter history scores AUC 0.61 vs 0.92, measured); the
model's `prediction_outputs` is `(B, 96, 9)` in log1p units (`ttm_dataset.OpponentSupplyWindows`).
`train_ttm.py`, `evaluate.py`, `kagg_oracle.py` and `check_oracle.py` all take the context
from the checkpoint; `train_ttm.py --context-length N --patch-length P` builds a new grid
from a warm start (re-initialising only the patcher, patch mixers and head whose shapes changed).
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
| `BASE` | weights + `scaler.npz` to start from; its `labels.json` selects the shard dirs (4.3) and its `config.json` the context length | default `models/ttm_v3_h96`; point at a `runs/ft_*/best` to chain |
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

Next day, chained (val = 10 % of **2026-09-11**'s episodes, 287,040 cells; not comparable row-to-row with the table above):

| run | training data | epoch 0 (base) | best AUC | best AP |
|---|---|---|---|---|
| `ft_2026-09-11_p4_g3_d2_s4` → **`models/ttm_v3_h96_ft_2026-09-11`** | 09-07..09-11, recency-weighted, `BASE=` the 09-10 fine-tune | 0.8414 | **0.8605** (ep 23, early-stopped at 28) | **0.1745** |

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
6. A checkpoint that goes into play carries `labels.json` (section 4.3); the copy in
   `shinka/evolution/checkpoint/` must stay byte-identical to the `models/` original.
7. Never fine-tune a checkpoint on shards of the other label rule (4.3); `finetune.sh`
   and `train_ttm.py` enforce it, a full retrain is the only way across.

## 10. Runs and logs (2026-09-11)

| tmux / log | what | state |
|---|---|---|
| `kagg-train_v3` / `train_v3.log`, `train_v3.attempt2.log` | v3 training (attempt 2 epochs 1–6, resumed run 7–27) | done, `V3_DONE` 22:20 UTC |
| `finetune_0910.log` | fine-tune on 09-10 only, 1 GPU | done, run dir deleted (numbers in 6.1) |
| `kagg-ft_p4_g3` / `finetune_p4_g3.log` | 5 days equal weight, 3 GPUs | done, run dir deleted (numbers in 6.1) |
| `kagg-ft_p4_d2` / `finetune_p4_d2.log` | 5 days recency-weighted, 3 GPUs | done → promoted to `models/ttm_v3_h96_ft_2026-09-10` |
| `kagg-ft_val0911` / `finetune_val0911.log` | forward validation on the whole 09-11 | stopped on request before 09-11 was published; rerun with `DAY=2026-09-10 bash research/opponent_model/ops/finetune_forward_check.sh` — but read 4.3 first |
| `kagg-dl0911` / `dl0911.log` | download of `kaggriculture-episodes-2026-09-11.zip` (583 MB, 2026-09-12 00:2x UTC) | done: in `replays/` and `/results/kagg/replays/`; **not extracted** (4.3) |
| `kagg-evalseed` / `evalseed.log` | full evaluator pass of the new Shinka seed (section 11) | done, 254 s, 40W-200L |
| `kagg-ft0911` / `finetune_0911_p4_d2.log` | `DAY=2026-09-11 BASE=models/ttm_v3_h96_ft_2026-09-10 finetune_4days.sh` | done 01:30 UTC → promoted to `models/ttm_v3_h96_ft_2026-09-11`; run dir, dataset symlinks (SSD) and the home mirror deleted |

| `kagg-extract_v3` / `extract_v3.log` | `ops/extract_v3.sh`: 44 days → `/results/kagg/dataset_v3` with corrected labels, 44 workers (2026-09-12 18:31 UTC) | see `EXTRACT_EXIT=` |
| `kagg-v4` / `train_v4.log` | `ops/train_v4.sh`: v4 base, context 256, warm start from the 09-11 fine-tune, waits for the extraction, 3 GPUs shared with Shinka run 2 (2026-09-12 18:36 UTC) | see `V4_EXIT=`; section 12 |

## 11. Shinka evolution with the oracle (2026-09-12)

The agent-evolution track (`shinka/`) now plays with the model. Everything below is
also in `shinka/evolution/README.md`; the mutation LLMs get the full oracle
description inside `task_sys_msg` (`shinka_config.yaml`, section "THE OPPONENT
ORDER-FLOW ORACLE").

- **What runs in a game**: `initial.py` (copied by Shinka to `shinka_results/gen_N/main.py`)
  imports `shinka/evolution/kagg_oracle.py`, which loads `shinka/evolution/checkpoint/`
  (byte-identical copy of the newest promoted fine-tune — `models/ttm_v3_h96_ft_2026-09-11`
  since 2026-09-12 07:30 UTC — plus `labels.json`) onto a
  GPU once per process, rebuilds the 150-channel input every turn from public state +
  own shed + own submitted orders (legacy accounting, 4.3), and from turn 512 (day 21)
  forecasts the opponent's supply for the next 96 turns. The policy sees `st["oracle"]`
  (`score_1/4/24/96`, `units_1/4/24/96` per product, raw 96×9 arrays). Before day 21
  the forecast is `None` — padded contexts measured AUC 0.65 vs 0.91 with a full one.
- **Seeded levers** (evolvable): shop sales ordered by predicted opponent supply, front-run
  a predicted dump (`score_4 ≥ 0.30`), optional step-aside (`score_1 ≥ 0.5`, off).
  Seed baseline vs the original 6-champion pool (09-11 checkpoint): 40W-200L (16.7 %),
  combined 0.407, avg cash $76,847 — identical to the 09-10 checkpoint's run; head-to-head
  vs the same policy without the oracle (`champ_00_initial_seed`): 28W-12L. Vs the
  7-champion pool (gen 62 added, see below): 42W-238L (15.0 %), $76,602, 279 s.
- **GPU process model**: evaluator workers are spawned (never forked); 36 workers per
  evaluation job × 3 jobs = 108 CUDA clients spread over the 3 GPUs by pid (~0.5 GB
  each, ~6 ms per forecast). CUDA MPS (`shinka/evolution/mps.sh start`, pipe
  `/results/kagg/mps/pipe`) triples per-GPU throughput (540 → 1,500 forecasts/s) and
  caps clients at 48 per GPU — keep `KAGG_WORKERS × max_evaluation_jobs ≤ ~130`.
  One 300-game evaluation ≈ 250 s wall. Training jobs (torchrun) are unaffected by
  MPS unless they export `CUDA_MPS_PIPE_DIRECTORY`.
- **Venvs**: Shinka runs from `/home/jovyan/shinka_venv` (Python 3.12.13 from
  `~/.local/share/uv/python`, shinka-evolve 0.0.7 — its interpreter was re-pointed
  there on 2026-09-12 after the SSD copy vanished); the evaluator runs under
  `/results/kagg/venv-cuda` (`job.python_executable`).
- **Verify before a run** (all under venv-cuda):
  `check_oracle.py faithful --zip replays/kaggriculture-episodes-2026-09-10.zip` (live
  rebuild vs shard), `check_oracle.py truth --zip replays/kaggriculture-episodes-2026-09-11.zip`
  (forecast quality vs reality), `check_oracle.py game` (one full game, ~25 s, prints
  oracle timing), then a full evaluator pass:
  `bash shinka/evolution/eval_once.sh shinka/evolution/initial.py <out dir>`
  (games logged to a scratch file, never to the pool's `games.jsonl`).
- **Launch**: `tmux new -s kagg-shinka 'GENERATIONS=200 ISLANDS=4 bash shinka/evolution/launch_shinka.sh 2>&1 | tee /results/kagg/logs/shinka.log'`
  (an EMPTY results dir per run: shinka resumes whatever DB it finds in `RESULTS_DIR`, and
  the mirror rsyncs with `--delete` — set `RESULTS_DIR`/`MIRROR_DIR` or delete the old ones first)
  (preflight covers venvs, checkpoint, GPUs, LLM proxy on 8317 and every configured
  model being served (`check_models.py`), Ollama embeddings, npx/codex, pool).
  Results: `RESULTS_DIR` = `/results/kagg/shinka_results` (SSD), mirrored by the launcher
  to `shinka_results/` on home every 10 min and at exit (rsync `--delete`, so the home
  copy is an exact mirror); `GENERATIONS` default 500;
  `ISLANDS` overrides the config's 8 islands — keep ≥ ~50 generations per island (the
  config's own rule), so 100 generations → 2 islands. The supervisor (meta) LLM runs
  after every 5 evaluated programs (`meta_rec_interval: 5`). The prompt states that
  using the oracle is mandatory, and the prompt-evolution guard keeps that phrase.
  The stale run from 09-10 sits in `shinka_results.stale_20260910_004817/`.
- **First run (2026-09-12)**: 07:55 UTC attempt stalled — `headless/codex@gpt-5.6-sol`
  is no longer in the ProxyPal catalog (every proposal: `model_not_found`); replaced by
  `gpt-6-astra` (probed with `codex exec`), relaunched 08:05, restarted once more at
  09:26 after two hung supervisor requests blocked collection for 35 min
  (`local_request_timeout: 600` now caps every LLM call; shinka resumes from its DB).
  Finished 12:56 UTC: 98 programs, 88 accepted, $73.19, best gen 62 at 73.3 % /
  combined 0.769 / $82,350 — nothing crossed the 75 % crowning gate. The ≥ 70 %
  programs are one behaviour (per-champion results identical to within a game), so
  gen 62 alone was hand-promoted into the pool as
  `champ_20260912_gen62_avg82350.py` (+ `roster/`, `POOL.json`); the pool is 7 = 280
  games per evaluation, and an oracle-using champion is verified to play crash-free
  (340 games) on the champion side. Run 1's artifacts were then deleted (user's call).
- **Run 2 (2026-09-12 14:31 UTC)**: launched in tmux `kagg-shinka`; seed = gen 62 (`initial.py`; the run-1 seed kept as
  `initial_v1_oracle_seed.py`), 200 generations, 4 islands, pool of 7 incl. gen 62 itself
  (the seed starts ~50/50 against it, so 75 % means improving on the starting point),
  block-only GPU embeddings. `KAGG_TTM_DIR` is pinned to
  `models/ttm_v3_h96_ft_2026-09-11` (frozen weights and scaler; only the policy evolves).
  Started from an empty results directory. Results `/results/kagg/shinka_results` →
  mirror `shinka_results/`; live log `/results/kagg/logs/shinka.log`, final marker `SHINKA_EXIT=<code>`.
  **Stopped by the user 19:01 UTC** (Ctrl-C in tmux; mirror complete) at 93/200
  generations, 89 accepted, $58.74. **Gen 33 `dynamic_shop_batch_sizing` was crowned
  at 16:16 UTC** — 77.1 % (216W-64L) vs the seven-pool, $82,113, seat 0/1 76/78 % —
  and evaluate.py copied it into the pool as `champ_20260912_161630_avg82113.py`;
  the pool is now 8 = 320 games per evaluation (also in `roster/` + `POOL.json`;
  prompt says "eight at the moment"). Every program after gen 33 was scored against
  the eight-pool, so compare programs on their win rate vs the seven old champions
  (`INDEX.json` field `win_rate_vs_old7`), not on `combined_score`. On that metric
  the top is a statistical tie (±2.5 pp): gen 88 77.5 %, gen 33 77.1 %, gen 81
  77.1 %, gen 55 76.8 %, gen 82 76.4 %, gen 60 ≡ 71 76.1 %; nobody beats gen 33
  head-to-head by more than 22-18. The ten programs at ≥ 75 % are saved with
  metrics, lineage and hashes in `shinka/champions/top/run2_2026-09-12/` (README
  there has the full table); the run itself (DB + gen dirs) lives only in
  `shinka_results/` (git-ignored) and on the SSD. **Best candidate now: gen 33**
  (`top/run2_2026-09-12/gen_33/main.py` == the crowned pool file) — a round-robin of
  gens 33/55/60/81/82/88 + the seed on fresh seed blocks (840 games,
  `top/run2_2026-09-12/tiebreak.py`) puts gens 81/33/60 at 134/133/132 W of 240
  (a tie) but gen 33 is the only one without a losing pairing; gen 88's pool lead
  drops to 51.2 % and gen 82's to 43.8 % on fresh seeds (fitted to the fixed
  evaluator seeds). Runner-up gen 81 (island 0, endgame liquidation) is
  complementary to gen 33 (island 1, oracle-sized shop batches).
- **Submission "Orchard Tide"** (gen 33 packaged in `shinka/champions/submissions/orchard_tide/`
  → `OrchardTide.tar.gz`, `main.py` at the root, private `SUBMISSION_MANIFEST.json` with hashes,
  the gen→name mapping and every attempt; public messages must stay neutral per
  `.agents/skills/kaggle-simulation-competitions`). **Attempt 1 (id 56192942, 21:09 UTC,
  torch + transformers + vendored tsfm) failed Kaggle validation: both seats TIMEOUT at step
  1, agent logs empty** — the first request execs main.py and the torch/transformers import
  did not finish inside the sandbox's budget (~60 s `remainingOverageTime` + 1 s; that import
  is 5.9 s on one core here and Kaggle measured 2-2.5x slower on an earlier accepted agent,
  worse for shared-library-heavy imports). **Attempt 2 (id 56193386, 21:46 UTC)** ships the
  oracle on the **numpy backend**: `shinka/evolution/kagg_ttm_numpy.py` reimplements the
  TinyTimeMixer forward pass (adaptive patching 4/2/1, mix-channel decoder, softmax gates,
  std scaler with float64 statistics) — within 1.6e-5 of the float64 truth where torch-f32
  is 8.5e-4 off; `check_oracle.py numpy` compares them — and `kagg_oracle.py` gained
  `KAGG_ORACLE_BACKEND=auto|torch|numpy` (the evaluator still takes torch on the GPU,
  verified game-identical; a torch import failure now falls back to numpy instead of
  playing oracle-less). Loader facts that shaped the bootstrap, each found by a failed
  validation: kaggle_environments `exec`s `main.py` with **no `__file__`** and takes the
  **last callable**; it APPENDS the bundle dir to `sys.path`; a missing piece makes the
  champion **silently play oracle-less** (`kagg_oracle` needs `features`/`mechanics` from
  `research/opponent_model`, now bundled; validate with an empty `HOME` and `python -I`).
  Validated in `/results/kagg/venv-kaggle-sim`: 8 games both seats, oracle live on numpy
  (torch never imported), import 2.2 s, from turn 512 p99 137 ms vs the 1 s limit,
  reproduces the repo champion to the dollar vs gen 62. `validate.py` in the bundle dir
  re-runs the check (README has the command). **Attempt 2 passed: COMPLETE 21:49 UTC**,
  validation episode 108314881 both seats DONE; Kaggle's timings for the numpy build:
  step 0 = 5.0 s, from turn 512 mean 321 ms / max 404 ms per step (2.5x one pod core).
  **This is the standard procedure for oracle-based submissions from now on** (user's
  decision 2026-09-12): `shinka/champions/submissions/make_submission.py --champion <file>
  --name "<Two Words>" --note "<private provenance>" --validate`, then the manual
  `kaggle competitions submit` line it prints (quota 5/day; errored ones are refunded).
  Written up in `.agents/skills/kaggle-simulation-competitions/references/submission-promotion.md`
  section 6. After a checkpoint change, `check_oracle.py numpy` first.
- **Code embeddings / GPU Ollama** (2026-09-12, after run 1): the novelty check was
  blind — shinka embeds the first 10,000 chars of the file (= the fixed backbone), the
  pod's Ollama 0.33.x runs on the CPU (refuses driver 535) and truncates at 4,096 tokens.
  Now: `embed_evolve_block_only: true` + `embed_max_chars: 28000` (run_evo.py override),
  model `qwen3-embedding:8b-ctx16k` (`ollama create`, num_ctx 16384) served by Ollama
  0.11.11 from `/results/kagg/ollama/` on port 11436 (`shinka/evolution/ollama_gpu.sh`,
  started by the launcher; 2.5 s per block on a GPU vs 45 s on the CPU; its runner
  breaks above ~8,192 tokens, hence the cap), `code_embed_sim_threshold: 0.999`
  (block-only vectors: nearest earlier program median 0.9985, p90 0.9993). Run 1's DB
  was converted with `reembed_db.py`. A permanent fix (driver ≥ 550 or a pod Ollama
  ≤ 0.11.x) is outside the pod; then point the config back at 11434.
- **Run-3 loop changes (2026-09-13, not yet run)**: (1) the LLMs now see the
  evaluation. Shinka only prompts `metrics.json["public"]` (after the score line) and
  `text_feedback` (only with `evo.use_text_feedback: true`, default false); run 2 had
  neither, so all 30 stored prompts described every program as
  `Combined score to maximize: 0.7` and nothing else — the per-champion table was
  computed and dropped. `evaluate.py` now nests win rate / seat split / cash / W-L-T
  under `public` (calibration and starter cash under `private`), and the config sets
  `use_text_feedback: true`. (2) `patch_types: ["diff", "full", "cross"]` at
  0.65/0.20/0.15 — `cross` merges the parent with one inspiration program (same island
  under `enforce_island_separation`; cross-island only via migration). (3)
  `games.jsonl` identity: candidates were all logged as `main.py` (49,480 games of
  runs 1-2 under one name), which merged them into a single Bradley-Terry node and
  compressed the champion ratings; `evaluate.py` now logs
  `<gen dir>/main.py@<sha12>`, and `curate_pool.py` drops the legacy unnamed games,
  rates on the 15,300 champion-vs-champion games, and only lets roster members fill a
  slot (candidates bridge). Its dry run puts the current eight in the right order
  (191956 > 190355 > gen 33 > 183514 > gen 62 > … > 153302 > seed). Pool target is
  10 (`--size 10` default; the prompt already says "top 10"). Scores from a run with
  these changes are not comparable with run 2's.
- **Swapping the checkpoint**: copy the new model dir (with `scaler.npz` and a
  `labels.json`) over `shinka/evolution/checkpoint/`, or point `KAGG_TTM_DIR` at it;
  rerun `check_oracle.py truth` and the seed evaluation before evolving on it.

## 12. v4: the day-10 oracle (context 256, corrected labels) — started 2026-09-12

Why: with a 512-turn context the oracle is silent until day 21, and by then 45 % of the
opponent's sold units are gone (8.6 % before day 10; 09-11 shard). Padding a younger game
to 512 does not work (AUC 0.61 vs 0.92 at origin 240, 09-11 held-out series), and TTM's
patch grid (8 patches × 64 turns: patch mixers, forecast head) is baked into the weights,
so a shorter context means a new checkpoint. Decisions (user, 2026-09-12): context **256**
(day 10 h16), warm start from the checkpoint in play, train on **every day incl. 09-11**,
and fix the labels (4.3) in the same run.

- **Recipe** (`ops/train_v4.sh`, tmux `kagg-v4`, log `/results/kagg/logs/train_v4.log`,
  marker `V4_EXIT=`): `train_ttm.py --context-length 256 --patch-length 32` = 8 patches
  of 32 turns, so from `models/ttm_v3_h96_ft_2026-09-11` only the patcher (32→192,
  6,144 params, 0.6 %) is re-initialised and the rest carries over (a 240/24 grid would
  re-initialise 14 %); scaler kept; `--split episode` over `dataset_v3` (44 days, 10 %
  of episodes held out → `best/val_episodes.json`); window stride **17** (coprime with
  24, so training origins cover every hour — v3's stride 8 only saw hours 0/8/16 while
  the live oracle runs at all 24); 3 GPUs, lr 1e-4, plateau ×0.5/2, patience 6, ≤ 40
  epochs, `--eval-on-start` (epoch 0 = the legacy 09-11 checkpoint with a random
  patcher, not a meaningful baseline). ~1.2 M windows/epoch, expect 15–25 min/epoch
  while Shinka run 2 shares the GPUs. Output `runs/ttm_v4_c256_h96/{best,scores.json,eval.*}`;
  `best/` carries `config.json`, `scaler.npz`, `labels.json` (next_action), `val_episodes.json`.
- **Then**: (1) `DAY=2026-09-11 BASE=research/opponent_model/runs/ttm_v4_c256_h96/best bash
  research/opponent_model/ops/finetune_4days.sh` — the recency-weighted refit on 09-07..09-11
  (`finetune.sh` switches to the `_v3` shard dirs by itself; run tag gets `_na`);
  (2) compare with the 512 model on the same games and origins, against the *true* supply:
  `KAGG_TTM_DIR=<dir> check_oracle.py truth --zip replays/kaggriculture-episodes-2026-09-11.zip --held-out`
  (uses the checkpoint's `val_episodes.json`; `--every 7` scores every hour) and
  `check_oracle.py padding` (per-origin AUC/AP; origins ≥ 512 are the apples-to-apples rows,
  240–511 are the new coverage); forward in time on 09-12 once Kaggle publishes it
  (~00:10 UTC 09-13). Reference: the 09-11 legacy checkpoint scores 0.76 pooled / 0.84 h0
  on 09-11's day-split held-out games (4.3).
- **Going into play** (run 3, not before Shinka run 2 finishes): copy `best/` over
  `shinka/evolution/checkpoint/` (rule 6), rerun `check_oracle.py faithful` (it now compares
  against the shard of the checkpoint's rule) and `game`, and update the two places that
  hardcode day 21: `initial.py`'s evolve block (`if _ENABLE_ORACLE_PRIORITY and step >= 512`)
  and the "None until turn 512 (day 21)" text in `shinka_config.yaml`'s task prompt.
