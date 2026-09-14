# AGENTS.md — operating guide for the Kaggriculture opponent model

Read this before touching training, fine-tuning, or data. It records what exists,
where it lives, how to run it, and the rules that keep results reproducible.
Last updated 2026-09-14 06:20 UTC. Evaluation methodology and numbers:
`research/opponent_model/EVALUATION.md`. Legacy data, checkpoints and scripts (the
512-context v1–v3 line, the pre-fix label rule) were deleted on 2026-09-13; only what
is in use remains, and nothing is version-suffixed any more.

## 1. What this is

A TinyTimeMixer (`granite-tsfm`, `tsfm_public`) that watches one seat of a
Kaggriculture game and predicts, for each of the 9 products and each of the
next 96 turns (4 game days), how many units the **opponent** will sell.
Its value is in *ranking* (which product-turns carry a sell), not in unit
counts. Input: 141 leak-free features per turn (`features.py`, `extract.py`)
plus the 9 `log1p` targets as channels; ~1.03 M params. Context length is a property
of each checkpoint (`config.json`): **256 turns** for the current model (forecasts from
day 10 h16); the retired 512-context line was silent until day 21.

Current model: **`models/ttm_c256_h96/`** — the 256-context base trained on the 44
corrected-label days through 09-11 (section 12; held-out AUC 0.8668, README inside),
committed in git. The copies in play (`shinka/evolution/checkpoint/`, the Kaggle
"Orchard Tide" bundle) are still the retired 512-context `ttm_v3_h96_ft_2026-09-11`
until the play-side steps of section 12 are done. `models/` holds only the promoted
checkpoint; run dirs are deleted once their best is promoted (section 6).
The model IS wired into a playing agent: the Shinka seed program serves a copy of
it as an in-game "oracle" (section 11). **Known issue with the training labels:
section 4.3.**

## 2. Repo map

| path | what |
|---|---|
| `models/ttm_c256_h96/` | the promoted checkpoint (committed): `model.safetensors`, `config.json`, **`scaler.npz`** (input mean/std — required at inference), `labels.json`, `val_episodes.json`, `scores.json`, `eval.json`/`eval.txt`, README |
| `replays/kaggriculture-episodes-<date>.zip` | Kaggle's daily replay datasets, 2026-07-30 → 09-12 so far (45 days, 22 GB); not in git — the 04:30 UTC cron adds each new day (4.1) |
| `datasets/shards/` | **the only shard directory**: one `kaggriculture-episodes-<date>.npz` per day (45 days, 2.3 GB) with `next_action` labels (4.2–4.3) and one `labels.json`; ceph copy of `/results/kagg/datasets/shards`. Not in git. Never inside the code directory |
| `research/opponent_model/` | all model code: `extract.py` / `extract_parallel.py` (replays → shards), `features.py`, `mechanics.py`, `ttm_dataset.py` (windows, episode split, scaler), `metrics.py` (streaming histogram AUC/AP), `train_ttm.py`, `evaluate.py` (the one scorer; uses the checkpoint's `scaler.npz` and `val_episodes.json`) |
| `research/opponent_model/runs/<run>/` | working dirs of runs (`best/`, `scores.json`, `scaler.npz`, `eval.json`); **not in git** — promoted checkpoints move to `models/` |
| `research/opponent_model/ops/` | job scripts: `env.sh` (venv), `copy.sh` (SSD staging after a restart), `extract.sh` (extract every replay day without a shard), `train.sh` (base model: context 256, warm start; section 12), `finetune.sh` (daily refit: newest day + the 4 before it, recency-weighted) |
| `research/opponent_model/logs/` | copies of the job logs (the live ones are on the SSD); not in git |
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
  prints `ENV_EXIT=0`) → `ops/copy.sh` (stages `replays/*.zip`, `datasets/shards` and the
  run dirs onto the SSD; prints `COPY_EXIT=0`). Models are read from ceph directly.
- **Long jobs run in tmux**, named `kagg-<job>`, with output teed to
  `/results/kagg/logs/<job>.log` and a final `<NAME>_EXIT=<code>` line
  (`TRAIN_DONE`, `FT_DONE`, `EXTRACT_DONE`). Check progress with
  `tr '\r' '\n' < /results/kagg/logs/<job>.log | grep -E "eval_auc|/775200"` (tqdm uses `\r`).
- GPUs go to 0 % for a few minutes at every epoch boundary (evaluation + checkpoint
  write + dataloader restart). That is normal, not a hang. Until 2026-09-13 19:50 UTC the
  eval tail was 10–70 min: sklearn sorted all 4e8 (window, step, product) cells on one CPU
  core after every epoch. `metrics.py` (histogram AUC/AP, 2^20 bins, accumulated per batch
  on the GPU via `batch_eval_metrics`) replaced it — same numbers to 1e-5, seconds instead.
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
  (`ops/finetune.sh` does this itself and waits until the day is published). The daily
  04:30 UTC cron (`sync_kaggriculture_replays.py`, outside this repo) downloads the new
  day to `replays/` + `/results/kagg/replays/`; extraction must then be
  `ops/extract.sh` (or `extract_parallel.py --out /results/kagg/datasets/shards --alignment next_action`
  mirrored with `rsync -a` to `datasets/shards/`) — never another directory.
- The number of distinct teams per day (45–73) is a byproduct of the size cap, not
  a "top-N teams" rule. The datasets contain **none of our own games**.
- The full ladder is ~330k games/day; individual replays can be pulled through the
  API but the endpoint throttles to ~300/h. A stratified API fetcher was built and
  then **removed on 2026-09-11 by decision** — do not rebuild it unless asked.

### 4.2 Shards
- `ops/extract.sh` (= `extract_parallel.py --replays <dir of zips> --out <dir> --workers N --alignment next_action`)
  turns each zip into `<zip stem>.npz` (`X` 141 features, `Y` 9 targets, `D` = episode/step/seat/contested,
  `feature_names`, `products`); 1,438 rows per episode (2 seats × 719 turns), ~50 MB/day,
  ~10 min/day (one worker per zip; 44 zips at once on the pod). It skips zips whose shard
  already exists in `--out` and writes `<out>/labels.json` with the label rule (refusing to
  mix rules in one dir). `train_ttm.py` reads that marker (through symlinks) and copies it to
  `best/labels.json`; `finetune.sh` refuses a base checkpoint of another rule.
- One directory for every day, `datasets/shards` (SSD: `/results/kagg/datasets/shards`).
  It may grow: a checkpoint's held-out episodes are recorded in its `best/val_episodes.json`
  and `evaluate.py` scores exactly those, so adding days never changes an old number.
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
**Every shard and checkpoint made before 2026-09-12 carried this** ("legacy"
alignment); all of them were deleted on 2026-09-13 except the checkpoint in play,
`models/ttm_v3_h96_ft_2026-09-11` (`labels.json: legacy`). What was measured:
- Against the **true** opponent supply (`shinka/evolution/check_oracle.py truth`, inputs
  rebuilt as a live agent rebuilds them): the 09-10 fine-tune on the whole of 09-11
  (forward in time) — AUC 0.75 pooled over 96 horizons, 0.80 at horizon 0, 0.83 at
  horizon 24; the 09-11 fine-tune on 09-11's 65 held-out games — 0.76 / 0.84 / 0.84
  (the 09-10 model on the same games: 0.75 / 0.84 / 0.84). Useful ranking, but recall is
  low and predicted volumes are ~5× too small; and the daily fine-tune improves the
  legacy-label metric far more than the real one.
- **Fix applied 2026-09-12 18:30 UTC** (user's decision): `extract.py` reads
  `steps[t+1][seat]["action"]` (`next_action`, the only rule left); verified equal to
  `check_oracle.true_supply` on every step of 3 replays (nonzero-step rate 9 % → 27 %).
  Every day was re-extracted into `datasets/shards`, and the 256-context base model
  (section 12) trains on it. **Do not mix** rules: a fine-tune of a legacy checkpoint on
  corrected shards changes the feature semantics under the weights, so `finetune.sh`
  refuses a `BASE` whose `labels.json` is not `next_action` and `train_ttm.py` refuses
  shards of two rules in one run (a *full* retrain warm-started from the legacy
  checkpoint, as the 256 run is, is fine -- it prints a warning and marks the result with
  the shards' rule). The live oracle reads `labels.json` (`"alignment": "legacy" |
  "next_action"`) next to a checkpoint and rebuilds whichever accounting the checkpoint
  was trained on; `train_ttm.py` writes it (plus `scaler.npz` and `val_episodes.json`)
  into `best/`.

## 5. Models and how to load them

| model | context / labels | status |
|---|---|---|
| `models/ttm_c256_h96` | 256 / next_action | **current**: base trained 09-12 → 09-14 on 44 days (07-30..09-11), best epoch 40 of 40, held-out AUC 0.8668 / AP 0.392 (trainer), canonical pooled 0.8669 / 0.390, day 1–4 AUC 0.873/0.872/0.868/0.855 — README in the dir. The daily chain (section 6) continues from it |
| `ttm_v3_h96_ft_2026-09-11` (512 / legacy) | — | deleted from `models/` 2026-09-14; survives only as the in-play copies (`shinka/evolution/checkpoint/`, the Orchard Tide bundle) until the 256 model replaces them |

Base-model recipe: `ops/train.sh` (section 12). `RESUME=<checkpoint dir>` resumes weights,
optimizer, scheduler and the early-stopping counter (`train_ttm.py --resume-from`).

**Loading**: `TinyTimeMixerForPrediction.from_pretrained(<dir>)` **plus** `<dir>/scaler.npz`:
inputs are `(x - mean) / std`, concatenated with `log1p(clip(y, 0))` of the 9 targets
as channels 141–149 → `past_values` of shape `(B, context_length, 150)` with
`context_length` from the checkpoint's `config.json` (256 now; 512 for the retired line -- TTM's
patch grid is baked into the weights, so a checkpoint accepts exactly that length and
nothing else; left-padding a shorter history scores AUC 0.61 vs 0.92, measured); the
model's `prediction_outputs` is `(B, 96, 9)` in log1p units (`ttm_dataset.OpponentSupplyWindows`).
`train_ttm.py`, `evaluate.py`, `kagg_oracle.py` and `check_oracle.py` all take the context
from the checkpoint; `train_ttm.py --context-length N --patch-length P` builds a new grid
from a warm start (re-initialising only the patcher, patch mixers and head whose shapes changed).
Every model dir and every run dir contains its `scaler.npz` (and, since 2026-09-12,
`labels.json` + `val_episodes.json`); `evaluate.py` and the oracle use them.

## 6. Fine-tuning on new days

One script, `research/opponent_model/ops/finetune.sh`: `BASE=models/<newest> DAY=<day>
bash research/opponent_model/ops/finetune.sh` trains on the newest day + the 4 days before
it, recency-weighted, validated on 10 % of the newest day's episodes. Its settings are
environment variables with the production values as defaults:

| var | meaning | recommended |
|---|---|---|
| `DAY` | newest training day (default: yesterday UTC) | the day Kaggle just published |
| `PREV_DAYS` | days before `DAY` added to training in full | default `4` |
| `DECAY`, `STRIDE` | recency weighting: `DAY` is windowed at `STRIDE`, each day further back at `STRIDE·DECAY^age` — with `DECAY=2` every day back contributes half as many windows | default `DECAY=2 STRIDE=4` (newest day ≈ 48 % of every epoch with 4 previous days) |
| `VAL_DAY` | validate on this **whole** day (forward in time); `DAY` is then trained on in full. Waits for Kaggle to publish it | the day after `DAY` for the forward check |
| `GPUS`, `LR` | torchrun on N GPUs; LR scales linearly with the batch (2e-5 at 1×64) | default `GPUS=3 LR=6e-5` |
| `BASE` | **required**: weights + `scaler.npz` + `labels.json` (must be `next_action`) to start from; its `config.json` sets the context length | the newest promoted `models/*`; a `runs/ft_*/best` to chain unpromoted refits |
| `PORT` | torchrun master port, change when two runs overlap | |

Fixed inside: `--split episode` (10 % of `DAY`'s episodes held out unless `VAL_DAY`),
`--eval-on-start` (**epoch 0 in the log = the base model scored on the same val
windows** — the number a fine-tune must beat), the base model's scaler (never refit;
a refit on top-only data shifts the inputs under the weights), `--metric-horizon all`,
30 epochs max (the recency run was still creeping up at 30 — raise `--epochs` in the script if that repeats), early stopping patience 5, plateau ×0.5 patience 2, batch 64/GPU.
Output: `/results/kagg/runs/<TAG>` → `research/opponent_model/runs/<TAG>/{best,scores.json,scaler.npz}`
with `TAG = ft_<DAY>[_p<PREV>][_val<VAL_DAY>][_g<GPUS>][_d<DECAY>][_s<STRIDE>]`; a
missing day's shard is extracted into `datasets/shards` (SSD, mirrored home). Intermediate
checkpoints are deleted.

### 6.1 What the recipe was tuned on

Measured on the retired 512-context line (legacy labels, deleted 2026-09-13), so only the
qualitative findings carry over: the base scored much lower on the newest day than on its
own validation (the top of the ladder drifts — the reason for daily refits); a few epochs of
fine-tuning recovered most of it; and **weighting toward the newest day was by far the
biggest lever** (+0.03 AUC / +0.05 AP over equal weighting of the same 5 days). Chained
daily refits (BASE = the previous day's fine-tune) kept improving the within-day metric.
Fresh numbers for the 256-context line go here once its first refit has run.

## 7. Daily routine when a new day appears

Kaggle publishes day `D` at ~00:10 UTC on `D+1`; the 04:30 UTC cron downloads it. Then,
from the repo root, with `BASE` = the newest promoted `models/*`:

1. **Refit** including the new day:
   `tmux new -d -s kagg-ft_<D> 'BASE=models/<newest> DAY=<D> bash research/opponent_model/ops/finetune.sh 2>&1 | tee /results/kagg/logs/finetune_<D>.log'`
   (extracts `D` itself if the cron has not; val = 10 % of `D`'s episodes; epoch 0 in the
   log = `BASE` on those windows, the number the refit must beat).
2. Results: `research/opponent_model/runs/ft_<D>.../{best,scores.json,scaler.npz}`;
   add the row to section 6.1. If it beats epoch 0, copy `best/*` to
   `models/<line>_ft_<D>/` with a README and delete the run dirs (home and SSD) — only
   promoted checkpoints are kept. Then the play-side steps of section 11 (copy into
   `shinka/evolution/checkpoint/`, `check_oracle.py truth --held-out`, `eval_once.sh`,
   prompt numbers).
3. The run needs all 3 GPUs (10–20 min); `finetune.sh` deletes its checkpoints, so
   nothing to clean up. A forward-in-time check of the recipe (train through `D-1`,
   validate on the whole of `D`) is the same engine with `VAL_DAY=<D>`; run it only when
   the recipe itself is in question.

## 8. Evaluation — what the numbers mean

- The Trainer's `eval_auc_any_sell` / `eval_ap_any_sell` pool all supervised horizons
  (`--metric-horizon all`, eval stride 5, coprime with the 24-turn day so every hour
  is represented), computed from score histograms (`metrics.SellDetection`, exact to
  ~1e-5). Use them for model selection; quote headline numbers from
  `evaluate.py` (EVALUATION.md §8) on the clean set — it picks up the checkpoint's
  `scaler.npz` automatically, so it works for `models/*` and for fine-tunes.
- Pitfalls documented in EVALUATION.md: a per-seat split leaks (all runs split by
  episode); a window grid aliased onto the end-of-day dump flatters AP ~2× (stride must be
  coprime with 24); predicted magnitudes are shrunk several-fold and need calibration.
- Sell rate is ~1.2–2 % of (turn, product) cells, with 3–7× spikes at game hour 0;
  any metric that credits zeros is uninformative.

## 9. Rules

1. All shards live in `datasets/shards` (one per day) — never inside the code directory,
   never in git. A checkpoint's held-out set is its `val_episodes.json`, so the directory
   may grow. No version-suffixed files or directories (`*_v3`, `*_v4`): the current thing
   has the plain name, history lives in git.
2. Never refit or drop the scaler when starting from existing weights; ship
   `scaler.npz` next to every model.
3. Kaggle's daily datasets are the data source. One account, read-only from scripts.
4. Long jobs: tmux + SSD + `tee` + exit marker; rsync results home at the end.
5. Keep `EVALUATION.md` §4 and this file's section 6.1 current when a run finishes.
6. A checkpoint that goes into play carries `labels.json` (section 4.3); the copy in
   `shinka/evolution/checkpoint/` must stay byte-identical to the `models/` original.
7. Never fine-tune a checkpoint on shards of the other label rule (4.3); `finetune.sh`
   and `train_ttm.py` enforce it, a full retrain is the only way across (the in-play
   legacy checkpoint is therefore frozen).

## 10. Runs and logs

| tmux / log | what | state |
|---|---|---|
| `kagg-train` / `/results/kagg/logs/train.log` | the 256-context base run (`ops/train.sh`, run dir `/results/kagg/runs/ttm_c256_h96`). Started 2026-09-12 18:36 UTC; resumed 09-13 19:25 (renames), 19:50 (streaming metrics) and 23:22 (pod restart wiped the SSD at ~21:22 — rebuilt with `env.sh`, restaged, resumed from the ceph mirror of `checkpoint-203552`) | **done** 2026-09-14 01:21 UTC, `TRAIN_EXIT=0`; canonical eval done 03:02; promoted to `models/ttm_c256_h96` 06:15 |
| `extract.log` | the corrected extraction of all days into `datasets/shards` (2026-09-12) | done |

The SSD run dir (`/results/kagg/runs/ttm_c256_h96`, checkpoints of epochs 39/40 + the
44-day shard farm) is kept only until the "continue past epoch 40?" question is settled;
the ceph mirror was deleted at promotion. Logs of the retired line were deleted on 2026-09-13.

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

## 12. The day-10 oracle: context 256, corrected labels — started 2026-09-12

Why: with a 512-turn context the oracle is silent until day 21, and by then 45 % of the
opponent's sold units are gone (8.6 % before day 10; 09-11 shard). Padding a younger game
to 512 does not work (AUC 0.61 vs 0.92 at origin 240, 09-11 held-out series), and TTM's
patch grid (8 patches × 64 turns: patch mixers, forecast head) is baked into the weights,
so a shorter context means a new checkpoint. Decisions (user, 2026-09-12): context **256**
(day 10 h16), warm start from the checkpoint in play, train on **every day incl. 09-11**,
and fix the labels (4.3) in the same run.

- **Recipe** (`ops/train.sh`; tmux `kagg-train`, log `/results/kagg/logs/train.log`, marker
  `TRAIN_EXIT=`): `train_ttm.py --context-length 256 --patch-length 32` = 8 patches
  of 32 turns, so from `models/ttm_v3_h96_ft_2026-09-11` only the patcher (32→192,
  6,144 params, 0.6 %) is re-initialised and the rest carries over (a 240/24 grid would
  re-initialise 14 %); scaler kept; `--split episode` over the 44 corrected days (10 %
  of episodes held out → `best/val_episodes.json`); window stride **17** (coprime with
  24, so training origins cover every hour — v3's stride 8 only saw hours 0/8/16 while
  the live oracle runs at all 24); 3 GPUs, lr 1e-4, plateau ×0.5/2, patience 6, ≤ 40
  epochs, `--eval-on-start` (epoch 0 = the legacy 09-11 checkpoint with a random
  patcher, not a meaningful baseline). ~1.2 M windows/epoch, ~25 min/epoch on the 3 GPUs.
  Output `runs/ttm_c256_h96/{best,scores.json,eval.*}`; `best/` carries `config.json`,
  `scaler.npz`, `labels.json` (next_action), `val_episodes.json`. **Result** (promoted as
  `models/ttm_c256_h96`, 2026-09-14): held-out AUC 0.63 (epoch 0) → 0.82 (1) → 0.86 (24) →
  0.8633 (30) → 0.8668 (40, the cap, still +0.0005/epoch after the LR halving at epoch 32);
  canonical pooled 0.8669 / AP 0.390, day 1–4 AUC 0.873 / 0.872 / 0.868 / 0.855, no seat
  leak. Magnitudes are ranking-only: a few late-game WHEAT dumps (true up to 1,500 units)
  are over-predicted ~4× in log1p space, which explodes unit-space totals at horizons ≥ 69.
- **Then**: (1) `DAY=2026-09-12 BASE=models/ttm_c256_h96 bash
  research/opponent_model/ops/finetune.sh` — the recency-weighted refit on 09-08..09-12;
  (2) compare with the 512 model on the same games and origins, against the *true* supply:
  `KAGG_TTM_DIR=<dir> check_oracle.py truth --zip replays/kaggriculture-episodes-2026-09-11.zip --held-out`
  (uses the checkpoint's `val_episodes.json`; `--every 7` scores every hour) and
  `check_oracle.py padding` (per-origin AUC/AP; origins ≥ 512 are the apples-to-apples rows,
  240–511 are the new coverage); forward in time on 09-12 once Kaggle publishes it
  (published; the 09-12 shard is already in `datasets/shards`). Reference: the 09-11
  legacy checkpoint scores 0.76 pooled / 0.84 h0 on 09-11's day-split held-out games (4.3).
- **Going into play** (run 3, not before Shinka run 2 finishes): copy `best/` over
  `shinka/evolution/checkpoint/` (rule 6), rerun `check_oracle.py faithful` (it now compares
  against the shard of the checkpoint's rule) and `game`, and update the two places that
  hardcode day 21: `initial.py`'s evolve block (`if _ENABLE_ORACLE_PRIORITY and step >= 512`)
  and the "None until turn 512 (day 21)" text in `shinka_config.yaml`'s task prompt.
