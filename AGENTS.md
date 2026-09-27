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

Current model: **`models/ttm_c256_h96_ft_2026-09-26/`** — the 256-context base
(`ttm_c256_h96`, trained on the 44 corrected-label days through 09-11, section 12)
refit on 09-22..09-26 by the daily recipe (section 6): AUC 0.853 → 0.855 on 09-26's
held-out games; README inside, committed in git. `models/` holds only the promoted
checkpoint; run dirs are deleted once their best is promoted (section 6).
The model IS wired into a playing agent: the Shinka seed program serves a copy of
it as an in-game "oracle" (section 11). **Known issue with the training labels:
section 4.3.**

## 2. Repo map

| path | what |
|---|---|
| `models/ttm_c256_h96_ft_2026-09-26/` | the promoted checkpoint (committed): `model.safetensors`, `config.json`, **`scaler.npz`** (input mean/std — required at inference), `labels.json`, `val_episodes.json`, `scores.json`, README |
| `replays/kaggriculture-episodes-<date>.zip` | Kaggle's daily replay datasets, 2026-07-30 → 09-13 so far (46 days, 22.5 GB); not in git — the 00:30 UTC cron adds each new day (4.1) |
| `datasets/shards/` | **the only shard directory**: one `kaggriculture-episodes-<date>.npz` per day (46 days, 2.4 GB) with `next_action` labels (4.2–4.3) and one `labels.json`; ceph copy of `/results/kagg/datasets/shards`. Not in git. Never inside the code directory |
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

### 3.1 CPU compute for arena games: Google Colab (added 2026-09-24)

Games (graph gauntlets, arena statistics) need about one CPU and 1 GB of RAM each; the
LLM side of an evolution run needs neither and runs where the loop runs. Options:
**Google Colab** (below, the default), a Brev box (`brev create`, about $2/h, delete it after
the run) and Kaggle CPU notebooks (5 sessions × 4 games); the last two are documented in
`research/procedural_graph/arena/README.md`.

- **CLI and accounts.** The Colab CLI (`uv tool install "google-colab-cli==0.7.2"
  --overrides <file with jupyter-kernel-client==0.9.0>`: 0.7.2 pins 0.8, whose client
  lacks `JupyterSubprotocol`, and 1.0.2 lacks `KernelClient`, so code execution crashes
  with either) runs as one Google account per wrapper, `~/.local/bin/colab<N>`. Add an
  account with `~/.local/bin/colab-add-account <N>` (interactive Google sign-in through a
  throwaway gcloud config; the credential is saved to
  `~/.config/colab-cli/acc<N>_adc.json`, mode 600, never in the repo; `--info <N>` shows
  its compute units and GPUs). The CLI's debug log `~/.config/colab-cli/colab.log`
  contains short-lived access tokens: keep it mode 600, never copy it.
- **Machines (measured 2026-09-24).**
  - A standard CPU VM has 2 CPUs and 12.7 GB, costs 0.08 compute units/h, and every
    account can run it.
  - A High-RAM CPU VM (`--high-mem`) has 8 CPUs and 51 GB, costs 0.26 units/h, and
    needs a Colab Pro/Pro+ entitlement. Only the account with ~200 units has it, and up
    to 5 at once work, i.e. 40 CPUs. The other paid account and the free accounts get
    `503 Service Unavailable`.
  - VMs are Intel Xeon 2.2 GHz with 88-206 GB of free disk and Python 3.13. Games still
    run on 3.12, see below.
  - A T4 GPU VM (`--gpu T4 --high-mem`, runner shape `t4hm`) has a Tesla T4 (15 GB), 8 CPUs
    and 50 GB. It comes with CUDA 12.8 and a system Python 3.13 with torch 2.11 and cupy 14.
    Account 2 runs it; its cost in units was not measured. It is used for predictor
    calibration (below).
  - Colab's rules restrict using several accounts to get around resource limits, so rely
    on the paid accounts.
- **The runner runs on cliproxyapi (rule since 2026-09-24).** This PC sleeps, and a sleeping
  PC loses its VMs (below), so every Colab run is driven from `ssh cliproxyapi` (always on),
  in tmux. Layout there, in `~/kagg-colab/`:
  - `home/` is the runner's `HOME`, so its CLI stays apart from the older setup in
    cliproxyapi's own `~/.config/colab-cli`. That older setup (4 accounts under the old
    numbering, CLI 0.6.0) is not ours to change.
  - `home/.config/colab-cli/acc<N>_adc.json` holds all 7 accounts with the same numbering as
    here, mode 600 in a mode-700 directory. They were copied at the user's request on
    2026-09-24. A new account's credential goes there only with the user's OK.
  - `home/.local/bin/colab<N>` are the same wrappers as here.
  - `home/.local/share/uv/tools/google-colab-cli` is a venv with exactly the local CLI's
    packages (`arena/colab_cli_requirements.txt` there: google-colab-cli 0.7.2,
    jupyter_kernel_client 0.9.0).
  - `arena/` holds copies of `colab_run.py`, `colab_readopt.py` and `colab_requirements.txt`.
    Copy them again after changing them here.
  - `runs/<run>/` holds the payload, jobs, `results.jsonl`, `traces/` and the log.
- **Running games.**
  - Build a payload here with `arena/payload.py` (`--graph NAME=FILE` plays a saved graph
    as-is), then `rsync -a <payload dir> cliproxyapi:kagg-colab/runs/<run>/`.
  - Start the runner there:
    `ssh cliproxyapi 'tmux new -d -s kagg-colab-<run> "cd /home/alex/kagg-colab && export
    HOME=/home/alex/kagg-colab/home PATH=/home/alex/kagg-colab/home/.local/bin:\$PATH && python3
    arena/colab_run.py --payload runs/<run>/payload --out runs/<run>/results.jsonl --run-name
    <run> --vm colab2:hm --vm colab2:hm ... > runs/<run>/colab.log 2>&1; echo
    COLAB_RUN_EXIT=\$? >> runs/<run>/colab.log"'`.
  - Once the log shows `COLAB_RUN_EXIT=`, rsync `results.jsonl` and `traces/` back.
  - **Give the user each VM's browser link when the VMs start** (user's request,
    2026-09-24). The runner logs them as `[<session>] colab<N> link: https://colab.research.google.com/notebooks/empty.ipynb?dbu=...`;
    `colab<N> url -s <session>` prints the same link. A link holds no token.
    - To open one, the user must be signed in as that VM's Google account. It attaches a
      notebook to the running VM, where `!tail /content/arena/arena.log` shows progress.
    - "Disconnect and delete runtime" in that notebook deletes the VM.
  - What the runner does:
    - splits the games across the VMs;
    - uploads the payload;
    - builds a Python 3.12.13 venv with the pinned `arena/colab_requirements.txt`;
    - plays the games with `arena.py`, 8 workers per High-RAM VM;
    - watches every VM (next bullet);
    - downloads the results and **stops every VM it created**, also on errors and Ctrl-C;
    - checks every account (`COLAB_VMS_LEFT=0`).
  - Rerunning plays only the missing games. `--jobs FILE` plays only the listed jobs under a
    new `--run-name`. `--attach` finishes the sessions of a runner that was killed, and
    `--cleanup` stops them. Check with `colab<N> sessions`.
- **What deletes a Colab VM (measured 2026-09-24), and what the runner does about it.** Each
  of these cost games that day:
  1. **No keep-alive ping for ~30 min.** `colab new` starts a daemon that pings every 60 s
     from the machine that ran it. While this PC slept (15:20-15:50 CEST), Colab deleted 9
     of 12 VMs, and 299 games were lost. The ping is
     `GET https://colab.research.google.com/tun/m/<endpoint>/keep-alive/` with the account's
     OAuth token and `X-Colab-Tunnel: Google`.
  2. **No command on the VM for ~60 min.** This happens even with working pings and even
     while games run: 4 VMs went 61-62 min after the last command run on them.
  3. **The CLI dropping live VMs.** It deletes a VM's local record and kills its keep-alive
     in two cases:
     - one `sessions` listing misses the VM (seen right after a wake-up);
     - the VM's access token expired. Tokens last 3600 s; exec then gets a 401, and the CLI
       prints "appears to be lost (404/401). Cleaning up.".
     `colab<N> sessions` then lists the VM as `[?] <endpoint>`. Exec, download and stop by
     name fail, and without pings Colab deletes it.
  4. **A create that reports failure** ("…a temporary usage or capacity limit…", 5
     High-RAM VMs at once) can still create the VM.

  The runner handles each case:
  - It polls every VM every 60 s with a command.
  - Every 30 min, `colab_readopt.py` gets each VM a fresh token, puts back a dropped
    record and restarts a dead keep-alive.
  - Every 10 min it pulls results and traces.
  - A VM that Colab deleted is replaced (up to twice), and the replacement plays only the
    games it hadn't finished.
  - It uses a "failed" VM that exists.

  `--start-only` leaves the VMs unwatched: attach within ~45 min, or Colab deletes them.
  By hand: `python3 arena/colab_readopt.py <N> <endpoint>=<name>` re-registers a `[?]` VM
  (or refreshes its token). The CLI's per-session history
  (`~/.config/colab-cli/history/<session>.jsonl`: `keep_alive_error`, `keep_alive_stopped`,
  `session_terminated`) shows what happened. `ps` start times of processes that ran across a
  sleep are shifted by the sleep's length. If a runner ever has to run on this PC,
  `bash arena/windows_awake.sh [hours]` holds a Windows wake lock (tmux `kagg-awake`). It
  prevents idle sleep only; a closed lid still sleeps the laptop.
- **Verified 2026-09-24.** Six run-2 gauntlet games replayed on Colab gave exactly the cash
  recorded on Brev. Game times: about 3 min against the oracle-free opponents and about
  4.5 min against oracle agents (2.3-4.7 min on standard VMs, depending on the VM). A
  240-game batch on 5 High-RAM VMs takes about 30 min and ~0.7 units.
- **Rules.**
  - Starting VMs spends compute units: agree the batch with the user first.
  - Run the runner on cliproxyapi in tmux (above). Never run it on this PC, and never as a
    foreground or background task of an agent session.
  - Prefer the paid account's High-RAM VMs. They mean fewer VMs, faster batches, and no
    spreading work across accounts, which Colab's rules restrict.
  - Remove every VM once its results are pulled. The runner downloads each VM's results,
    then stops it, then checks every account's session list and prints `COLAB_VMS_LEFT=0`
    (a leftover is stopped once more; a `[?]` VM is reported). If it prints anything else,
    stop the listed sessions by hand (`colab<N> stop -s <name>`, after `colab_readopt.py`
    for a `[?]` one). Stopping deletes the VM with its disk; the CLI creates no notebook
    files in Drive. Afterwards delete the finished sessions' history files.
  - Never print the credentials or copy them anywhere else.

## 4. Data

### 4.1 Source: Kaggle's daily datasets (the only source we use)
- `kaggle/kaggriculture-episodes-<YYYY-MM-DD>` — "top episode replays ranked by
  average agent rating, capped at 20 GiB per day", published **~00:10 UTC the next
  day**. 660–930 episodes/day (657 on 09-10), ~0.5 GB zip, members `<episode_id>.json`
  (~31 MB each: 720 steps, both seats' observations incl. `private`) + `manifest.csv`.
  Index with per-day counts and scores: `kaggle/kaggriculture-episodes-index`.
- Download: `kaggle datasets download kaggle/kaggriculture-episodes-<date> -p replays/`
  (`ops/finetune.sh` does this itself and waits until the day is published). The daily
  00:30 UTC cron (`sync_kaggriculture_replays.py`, outside this repo) downloads the new
  day to `replays/` + `/results/kagg/replays/`; extraction must then be
  `ops/extract.sh` (or `extract_parallel.py --out /results/kagg/datasets/shards --alignment next_action`
  mirrored with `rsync -a` to `datasets/shards/`) — never another directory.
- The number of distinct teams per day (45–73) is a byproduct of the size cap, not
  a "top-N teams" rule. The datasets contain **none of our own games**.
- The full ladder is ~330k games/day; individual replays can be pulled through the
  API but the endpoint throttles. A stratified API fetcher was built and then
  **removed on 2026-09-11 by decision**; on 2026-09-14 a narrower one came back at the
  user's request (4.1.1), because the published zips hold only games whose two agents
  average ≥ ~2,950 (for the leader that is 83 % of his games; for a 2,500-rated
  opponent none) and the oracle scored AUC 0.70 on our own Kaggle games vs 0.88 on
  top-vs-top ones — the mid-ladder band is the missing data.

### 4.1.1 Second source: the top-100 teams' games below the cutoff (`ops/fetch_top.sh`)

`DAY=<D> TOP=100 bash research/opponent_model/ops/fetch_top.sh` (tmux; log
`/results/kagg/logs/fetch_top100_<D>.log`, marker `FETCH_EXIT=`) runs
`research/opponent_model/fetch_top_teams.py` in three steps and then the shard step:

1. `enumerate`: leaderboard → the top-N team ids → `team-submissions` → `episodes`
   (read-only API; paced, backs off on 429) → every COMPLETED public game of day `D`
   that is **not** in `replays/kaggriculture-episodes-<D>.zip`, with both teams'
   current ratings → `/results/kagg/replays_top/<D>/episodes.json`, sorted by the
   pair's mean rating (09-13: 6,882 games for the top 100; weaker side 2,500–2,950 in
   4,107 of them, 2,000–2,500 in 1,450, < 2,000 in 1,285). A submission's episode
   listing is capped at ~200 rows, so enumerate **early on D+1** (the 00:30 UTC slot
   is fine; a team with > 200 games in the window loses its oldest ones).
2. `download`: 3 workers, resumable (`<id>.json.gz` on the SSD), 429 back-off; the
   endpoint gives a burst (~3,500/h) and then a few per minute, so a full day takes
   hours — stopping early keeps the highest-rated games.
3. `pack --mirror`: `replays/kaggriculture-top<N>-<D>.zip` on the SSD and on ceph,
   same member layout as Kaggle's (`<id>.json` + `manifest.csv` with both ratings).
4. `ops/extract.sh`: the new zip becomes `kaggriculture-top<N>-<D>.npz` in
   `/results/kagg/datasets/shards`, rsync'd to `datasets/shards/`. `finetune.sh`
   treats every `kaggriculture-*-<D>.npz` as day `D` (published + top-N together);
   `train_ttm.py` reads the whole directory anyway.

Rules: one account, read-only endpoints, never more than 3 concurrent replay
requests, replays only under `replays/` (ceph) and `/results/kagg/replays/`, shards
only under the two `datasets/shards` dirs.

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
| `models/ttm_c256_h96_ft_2026-09-26` | 256 / next_action | **current**: the daily refit of the base through 09-26 (`train_5days.py` dual-GPU, 2026-09-27, early-stopped at epoch 10, best 5): on 09-26's held-out episodes AUC 0.853 → **0.855**, AP 0.395 → **0.400** (all 96 steps pooled); README in the dir. |
| `ttm_c256_h96_ft_2026-09-25` (256 / next_action) | — | previous refit through 09-25; in git history (commit a376de4) |
| `ttm_c256_h96_ft_2026-09-24` (256 / next_action) | — | previous refit through 09-24; in git history (commit 4e1039a) |
| `ttm_c256_h96_ft_2026-09-23` (256 / next_action) | — | previous refit through 09-23; in git history (commit 5e40c2f) |
| `ttm_c256_h96_ft_2026-09-22` (256 / next_action) | — | previous refit through 09-22; in git history (commit edf06d0) |
| `ttm_c256_h96_ft_2026-09-21` (256 / next_action) | — | previous refit through 09-21; in git history (commit 566df12) |
| `ttm_c256_h96_ft_2026-09-20` (256 / next_action) | — | previous refit through 09-20; in git history (commit 6e48c85) |
| `ttm_c256_h96_ft_2026-09-19` (256 / next_action) | — | previous refit through 09-19; in git history (commit a7049a9) |
| `ttm_c256_h96_ft_2026-09-18` (256 / next_action) | — | previous refit through 09-18; in git history (commit facdddc) |
| `ttm_c256_h96_ft_2026-09-17` (256 / next_action) | — | previous refit through 09-17; in git history (commit 3f6ea9e) |
| `ttm_c256_h96_ft_2026-09-16` (256 / next_action) | — | previous refit through 09-16; in git history (commit c0aab28) |
| `ttm_c256_h96_ft_2026-09-15` (256 / next_action) | — | previous refit through 09-15; in git history (commit ecdf38f) |
| `ttm_c256_h96_ft_2026-09-13` (256 / next_action) | — | previous refit through 09-13; in git history (commit 376717e) |
| `ttm_c256_h96` (256 / next_action) | — | the base: trained 09-12 → 09-14 on 44 days (07-30..09-11), best epoch 40 of 40, held-out AUC 0.8668 / AP 0.392, canonical pooled 0.8669 / 0.390, day 1–4 AUC 0.873/0.872/0.868/0.855. Removed from `models/` at the 09-13 promotion (only the newest promoted checkpoint is kept); in git history (commit c5b8684) |
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
First refit of the 256-context line (2026-09-14, `DAY=2026-09-13 BASE=models/ttm_c256_h96`,
defaults): 49 % of the 222,570 training windows from 09-13; on 09-13's 65 held-out episodes
the base scored AUC 0.859 / AP 0.357 (epoch 0; its own held-out was 0.867 — the drift), the
refit 0.878 / 0.381 at epoch 16, early-stopped at 21 (LR halved twice), 36 min on 3 GPUs.
Promoted as `models/ttm_c256_h96_ft_2026-09-13`.

| refit | train days | epoch 0 (base) AUC / AP | best AUC / AP (epoch) | stopped |
|---|---|---|---|---|
| `ft_2026-09-13` from `ttm_c256_h96` | 09-09..09-13 | 0.859 / 0.357 | 0.878 / 0.381 (16) | early, epoch 21 |
| `ft_2026-09-15` from `ttm_c256_h96_ft_2026-09-13` | 09-11..09-15 | 0.845 / 0.335 | 0.849 / 0.346 (7) | early, epoch 12 |
| `ft_2026-09-16` from `ttm_c256_h96_ft_2026-09-15` | 09-11..09-16 | 0.852 / 0.339 | 0.854 / 0.350 (10) | early, epoch 15 |
| `ft_2026-09-17` from `ttm_c256_h96_ft_2026-09-16` | 09-12..09-17 | 0.848 / 0.360 | 0.848 / 0.362 (15) | early, epoch 20 |
| `ft_2026-09-18` from `ttm_c256_h96_ft_2026-09-17` | 09-13..09-18 | 0.835 / 0.332 | 0.841 / 0.343 (19) | early, epoch 24 |
| `ft_2026-09-19` from `ttm_c256_h96_ft_2026-09-18` | 09-14..09-19 | 0.842 / 0.345 | 0.847 / 0.356 (19) | early, epoch 24 |
| `ft_2026-09-20` from `ttm_c256_h96_ft_2026-09-19` | 09-15..09-20 | 0.843 / 0.345 | 0.850 / 0.355 (9) | early, epoch 14 |
| `ft_2026-09-21` from `ttm_c256_h96_ft_2026-09-20` | 09-16..09-21 | 0.849 / 0.355 | 0.854 / 0.363 (9) | early, epoch 14 |
| `ft_2026-09-22` from `ttm_c256_h96_ft_2026-09-21` | 09-18..09-22 | 0.849 / 0.362 | 0.853 / 0.371 (18) | early, epoch 23 |
| `ft_2026-09-23` from `ttm_c256_h96_ft_2026-09-22` | 09-19..09-23 | 0.853 / 0.391 | 0.855 / 0.398 (13) | early, epoch 18 |
| `ft_2026-09-24` from `ttm_c256_h96_ft_2026-09-23` | 09-20..09-24 | 0.852 / 0.405 | 0.853 / 0.411 (7) | early, epoch 12 |
| `ft_2026-09-25` from `ttm_c256_h96_ft_2026-09-24` | 09-21..09-25 | 0.852 / 0.408 | 0.854 / 0.414 (2) | early, epoch 7 |
| `ft_2026-09-26` from `ttm_c256_h96_ft_2026-09-25` | 09-22..09-26 | 0.853 / 0.395 | 0.855 / 0.400 (5) | early, epoch 10 |

## 7. Daily routine when a new day appears

Kaggle publishes day `D` at ~00:10 UTC on `D+1`; the 00:30 UTC cron downloads it. Then,
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
4. **Calibrate the refit for our agent before it plays** (user's decision, 2026-09-24). It is
   one command, run on this PC in tmux:
   `python3 research/procedural_graph/arena/calibrate.py --refit models/<line>_ft_<D>`
   (about 40 min on one Colab T4, a little under 1 compute unit). Then commit
   `research/procedural_graph/calibration/<refit>/own_games/` and run the rematch. What it does,
   how to read the result and the rematch: section 13.

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
| `kagg-train` / `/results/kagg/logs/train.log` | the 256-context base run (`ops/train.sh`, run dir `/results/kagg/runs/ttm_c256_h96`). Started 2026-09-12 18:36 UTC; resumed 09-13 19:25 (renames), 19:50 (streaming metrics) and 23:22 (pod restart wiped the SSD at ~21:22 — rebuilt with `env.sh`, restaged, resumed from the ceph mirror of `checkpoint-203552`) | **done** 2026-09-14 01:21 UTC, `TRAIN_EXIT=0`; canonical eval done 03:02; promoted to `models/ttm_c256_h96` 06:15; run dir and logs deleted 06:40 |
| `kagg-finetune` / `/results/kagg/logs/finetune_2026-09-13.log` | first refit of the 256 line (`ops/finetune.sh`, `DAY=2026-09-13 BASE=models/ttm_c256_h96`) | **done** 2026-09-14 07:04 UTC, `FT_EXIT=0`, best epoch 16; promoted to `models/ttm_c256_h96_ft_2026-09-13` 07:20, run dirs deleted |
| `extract.log` | the corrected extraction of all days into `datasets/shards` (2026-09-12) | done |

No run dirs exist (SSD or ceph): both promotions deleted theirs. Logs of the retired line
were deleted on 2026-09-13, the base run's log on 2026-09-14.

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
  section 6. **Procedural-graph agents** (a graph + the Hazel runtime + a predictor, e.g. feed15)
  are packaged with `research/procedural_graph/make_graph_submission.py` (same file, section 7). After a checkpoint change, `check_oracle.py numpy` first.
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
- **Then**: (1) ~~the first refit~~ done 2026-09-14: `DAY=2026-09-13 BASE=models/ttm_c256_h96`
  → `models/ttm_c256_h96_ft_2026-09-13` (section 6.1); (2) compare with the 512 model on the same games and origins, against the *true* supply:
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

## 13. Calibrating the agent's predictor after a refit (for the agent that fine-tunes)

Every promoted refit gets calibrated for our agent before it plays (user's decision,
2026-09-24). This section is the whole procedure. Run it after step 2 of the daily routine
(section 7).

**What calibration changes.** Neither the model's weights nor the agent. It adds one file,
`calibration.json`, next to the predictor in the agent's bundle
(`hazel_runtime/checkpoint/calibration.json`). The file holds one multiplier per product for
three numbers the oracle gives the agent every turn:
- `score_4`: the largest predicted opponent sale over the next 4 turns, in log1p units;
- `score_24`: the same over 24 turns;
- `units_24`: the predicted opponent units summed over 24 turns.

`kagg_oracle.forecast()` multiplies them before the policy reads them. The raw 96×9 forecast and
the other horizons are untouched.

**Why.** The agent (the feed15 graph) acts on fixed thresholds:
- front-run when `score_4 ≥ 0.30` (`_ORACLE_FRONTRUN_SCORE`), with bigger batches at 0.45 and 0.60;
- `score_24 ≥ 0.45` and `units_24 ≥ 2.0`;
- the town-shop cadence bypass at `score_4` 0.25, 0.40 and 0.50.

They were tuned while the agent played with the **reference** predictor,
`ttm_c256_h96_ft_2026-09-13` (`research/procedural_graph/hazel_runtime/checkpoint`, model sha256
3c94bfc5…). A refit's scores sit at other levels, so the same thresholds fire at other moments.
For each product, the calibration picks the multiplier that makes the refit cross those
thresholds as often as the reference did, on the same turns of our own arena games. The factor
is the geometric mean over the thresholds of threshold / (the refit's score at the reference's
firing rate). The fit has guards: a threshold counts only if the reference fires on at least
100 origins and the product sells on at least 0.5 % of them (otherwise the factor is 1), and
factors are clipped to 0.5–2.

A calibration therefore belongs to one model, one set of thresholds and one reference. Refit it
when any of them changes: a new refit, an evolution run that moves `_ORACLE_FRONTRUN_SCORE` or
another oracle threshold, or a policy re-tuned on a newer predictor (which then becomes the
reference). The threshold list is `METRICS` in `arena/calib_fit.py`, copied from
`hazel_runtime/champion.py` (4c front-run, 5 cadence bypass). Keep the two in step.

**How to run it.** On this PC, which has the `cliproxyapi` ssh alias, python3 with numpy, and
rsync:

```sh
tmux new -d -s kagg-calib-<D> 'python3 research/procedural_graph/arena/calibrate.py \
    --refit models/<line>_ft_<D> 2>&1 | tee /tmp/kagg-calib-<D>.log'
```

What it does:
1. Builds a payload from the committed **game set** (`research/procedural_graph/calibration/game_set`),
   the reference and the refit. The game set is 600 of our arena games, 800 seat-views:
   - feed15 with the reference against plain Mohui (200 games), and against Hazel, Copper,
     Orchard, Willow and Mohui13 (40 each), scored from feed15's side;
   - feed15 with the 09-23 model against feed15 with the reference (200 games), scored from
     both sides.
2. Starts `colab_run.py` on cliproxyapi in tmux `kagg-colab-calib-<refit>` on one T4 High-RAM VM
   (`colab2:t4hm`), and prints the VM's browser link; give it to the user (section 3.1).
3. The VM replays each game from its trace through the engine. The job fails unless the replay
   reproduces the recorded final cash. Both models then run as the torch TinyTimeMixer on every
   turn from 256 to 714, after a check against the numpy port (within 1e-5).
4. Waits, requires `COLAB_VMS_LEFT=0` and every game scored, pulls the results and deletes the
   session history.
5. Writes `research/procedural_graph/calibration/<refit>/own_games/`:
   - `calibration.json`, fitted on all games;
   - `calibration_pool.json` and `calibration_mirror.json`, the two subsets;
   - `fit_report*.txt`;
   - `jobs.json`;
   - a README with each product's AUC for both models and the factors.

If it stops (sleep, network), run the same command again: it resumes, pulling a finished remote
run or waiting for a running one. Commit the output directory.

**Reading the result.** The README table gives each product's `score_4` AUC for the reference and
the refit on our games. **AUC does not change under any calibration.** If the refit ranks our
opponents' sales worse on the products the agent trades most, calibration cannot recover that,
and the agent with the refit will still lose to the agent with the reference. Those products
are wheat, milk, strawberry, fertilizer and wool. Measured on 2026-09-25:
- The 09-23 refit ranks top ladder players better (held-out ladder AUC up almost everywhere).
- But it ranks our Mohui-family opponents worse: `score_4` AUC for wheat 0.682 → 0.642, strawberry
  0.805 → 0.777, fertilizer 0.786 → 0.769.
- Calibrated on ladder games, it still lost to the reference 19W-141L (−$265 a game) inside
  feed15; uncalibrated it lost 24W-174L.

Report the table to the user with the refit's usual metrics. A refit can be better on the ladder
and worse in our arena at the same time.

**Rematch (acceptance check).** Before a calibrated refit replaces the reference in any agent,
play the same graph with the refit and the calibration against the same graph with the
reference. Use the arena validation seeds (200 = 100 per seat, the same seeds as every earlier
rematch, so the results pair up game by game):

```sh
K=research/procedural_graph/arena G=research/procedural_graph/evolution_results/feed_fix_2026-09-24/feed15_graph.json
R=models/<line>_ft_<D> C=research/procedural_graph/calibration/<line>_ft_<D>/own_games/calibration.json
python3 $K/payload.py --eval-id rematch-<D> --graph cal=$G --graph ref=$G --checkpoint cal=$R \
    --calibration cal=$C --pairs cal:ref --seeds 200
# then the launch of section 3.1 with 5 x --vm colab2:hm (about 25 min, about 0.5 units)
```

The refit plays only if the calibrated side does not lose the rematch. Otherwise it stays
behind the reference, and the user decides.

**Cost and rules.** A calibration (one T4, about 40 min) plus a rematch (5 CPU VMs, about 25 min)
comes to roughly 1.5 compute units per refit. These two batches are the standard routine; any
other batch needs the user's OK (section 3.1). The runner stops its VMs itself. Check
`COLAB_VMS_LEFT=0` in its log, and `colab2 sessions` if in doubt.

**Other options.**
- Calibrating on ladder games instead of our own: `calib_payload.py --zip <day zip>`, which uses
  the refit's held-out games plus a sample of the rest. It is useful for comparison only, since
  it did not help in the arena.
- A new game set, when the agent or its opponent pool changes: `calib_payload.py --trace
  'runs/arena/<run>/traces/<tag>*@ours|@both'`, then copy the payload's `traces.zip` and write
  `games.json` like the committed one.
- Details and file formats: `research/procedural_graph/arena/README.md` ("Predictor calibration").

## 14. The ladder pool and ladder-engine graph evolution (2026-09-26)

**Why.** The opponents that beat Linden Brook and Rowan Glen run public Kaggle notebooks. Recovered
into `shinka/champions/ladder/` (README there), they beat our Mohui-based agent in almost every game
(Rowan Glen 0-96 against the V57 family, -$18k to -$20k a game), and the strongest of them, tetsutani's
"Demand-Preserving" agent, beats even the V57 family. Parameter evolution of the Mohui graph cannot
close that gap (Mohui's routes have no geese or tomatoes and a step-1 cash crunch), so the graph now
runs a ladder agent as its production engine and evolution tunes that engine and our layers on it.

- **The pool.** `shinka/champions/ladder/build_ladder_pool.py` pulls the notebooks read-only and
  recovers their agents statically (no notebook code runs); `match_ladder_games.py` checks a bundle
  move for move against a recorded ladder game (5 exact matches, including the 2155- and 2188-rated
  players). Arena opponents by name: `payload.py --pairs x:tetsutani_demand` or `--bundle`.
- **Ladder graphs.** `make_ladder_graph.py --engine <bundle>` copies the agent into
  `hazel_runtime/engines/<bundle>/` and writes a graph whose `backbone` node has `engine`; with the
  farmer, hands and market channels off it ties the public agent to the dollar. `engine_parameters`
  set the engine's literal constants (`hazel_runtime/engines.py` catalogs them; game-rule tables are
  excluded); the optional `oracle_guard` turn stage adds the predictor's front-run sells into the
  engine's empty order slots (`_OG_*` parameters). Ladder engines put deliberate empty `[]` entries in
  their order lists (both seats clear index by index): a passed-through market keeps them, and
  anything that rewrites orders must too. With every channel and the guard off the oracle is not run.
- **Colab VM pool.** `arena/colab_pool.py` keeps VMs between batches: a client writes
  `<pool>/inbox/<name>.json` ({name, root, jobs}), the pool uploads each (content-addressed) bundle to
  a VM once, plays, and writes `<pool>/outbox/<name>.jsonl|.log|.done`; `graph_gauntlet.ColabPoolExecutor`
  is that client. `STOP` in the pool dir ends it; idle VMs stop after `--idle-stop`. Same watch rules as
  colab_run.py (keep-alive polls, token refresh, replacement), same `COLAB_VMS_LEFT=0` check. A slot gets
  `--replacements` (2) new VMs for VMs that die young; a VM that served `--long-life` (3 h) before it went
  away does not count, so a pool that runs for days keeps its slots when Colab ends long-running VMs.
- **Evolution run.** On cliproxyapi (the LLM proxy is its localhost:8317): code copy in
  `~/kagg-evo/repo` (rsync of `research/procedural_graph` without runs/calibration, `shinka/champions/ladder`,
  `shinka/champions/submissions/hazel_weir/mohui_v66`, `shinka/evolution/pool_upgrade_bundle_agent.py`),
  venv `~/kagg-evo/venv` (python3.12, kaggle-environments 1.32.7, numpy), pool `~/kagg-evo/pool`
  (tmux `kagg-colab-evo`), loop in tmux `kagg-evo-<run>`:
  `highcpu_island_evolution.py --executor colab-pool --pool_dir ~/kagg-evo/pool --islands ladder
  --plan evolution_results/ladder_2026-09-26/plan.json --knowledge evolution_knowledge_ladder.md
  --ideas evolution_ideas_ladder.md --seed_graph evolution_results/ladder_2026-09-26/seed_graph.json
  --seeds_per_opponent 20 --mix_interval 12 --queue evolution_queue_ladder.json --stage_fraction 0.3 --prefetch`.
  The plan (`ladder_seed_plan.py`) is every lost ladder seed
  against the bundle that plays like the rival who beat us there, from both seats, plus 10 random seeds
  against each pool member; `--seeds_per_opponent` then only sizes the head-to-head block. The bandit
  pulls only the models the proxy serves.
- **Adding a submission's new losses** (done 09-26 for Alder Ford, 22 losses and 2 ties): list its games and
  download the lost replays (`fetch_games.py`-style, read-only API, here), rsync them to cliproxyapi, run
  `match_ladder_games.py` there for every pool bundle (plus candidates recovered from newer public
  notebooks; `build_ladder_pool.py` reports a notebook that changed), add a bundle that matches a rival to
  the pool, then `ladder_seed_plan.py --extend <plan> --with-ties --losses <index>=<replays> --evidence
  <matches> --fallback tetsutani_demand --replay-opponents shinka/champions/replay_opponents` (its old
  entries stay as they are) and restart the loop. Only the new games are played for the existing champions.
  Record: `shinka/champions/evidence/alder_ford_20260926/` (its README also has the loss analysis).
- **Replay opponents** (`shinka/champions/replay_opponents/`, README there). Build them first with
  `make_replay_opponents.py <index> <replay dir>` (on cliproxyapi, where the replays are). The bundle
  `replay_<episode>` plays the rival's recorded moves of one lost game. Against it the champion replays the
  ladder game to the dollar (all 24 of Alder Ford's), whereas our stand-in agents were beaten on 13 of 21
  lost seeds. Use them only for losses of an agent close to the candidates (Alder Ford's). They are one-seat
  jobs next to the stand-in's both-seat jobs.
- **Queued edits.** `--queue evolution_queue_ladder.json` (a JSON list of {island, edit, rationale}, re-read
  every iteration): an island's next iteration plays its first queued edit not played yet, instead of the
  models' proposals, with the same gauntlet and promotion rule (`"model": "queue"`, no bandit update). Use it
  to test a specific hypothesis; edit the file in `~/kagg-evo/repo` after editing here. An entry is re-applied to
  the island's current champion every iteration, so once a mixing changes that champion a played entry counts
  as new and plays again: remove entries once played.
- **Staged gauntlet and proposals ahead** (since 09-27). `--stage_fraction 0.3` plays 30% of a candidate's games
  first (the same games for every candidate, `graph_gauntlet.first_stage`) and stops a candidate that changed at
  most four of them or made at least as many worse as better (`stopped after n of m games` in the log; never
  promoted, the seen list and the bandit treat it as rejected). On the run's first 71 candidates it would have
  stopped 35, none of the 20 promoted, and saved 32% of the games
  (`evolution_results/ladder_2026-09-26/stage_replay.py` re-checks this on a run). `--prefetch` has the models
  propose the next island's candidate while a gauntlet plays (`[NEXT]` in the log), except before a mixing or a
  meta-supervisor run, for an island with queued edits, or when that island's baseline still has games to play;
  the next iteration takes it only if the island's champion is unchanged and its settings are still new.
  Neither is in the gauntlet fingerprint (only `hazel_runtime/`, `agent_graph.py`, the harness, seeds and engine
  are), so switching them keeps the cache.
- **Island mixing** (user's requirement, 2026-09-26). The islands evolve separately, and an edit whose
  settings were played on any island is refused as a repeat, so the models cannot pass a champion to
  another island. Every `--mix_interval` iterations (12 = two rounds of the six ladder islands; a run
  that turns it on mid-block mixes at once) the island whose champion gains most over the seed is the
  donor, and every other island is offered its champion plus the donor's changes. A setting the island
  changed itself keeps its own value (`graph_edits.migration_edit`). The usual gauntlet decides, and pool
  games a graph already played are reused, so a champion moving to another island costs only the 20
  head-to-head games. The log shows `MIXING after iteration n` and one `[MIX]` line per island; the
  records carry `"model": "mixing"` and `"donor"`. A stop during a mixing resumes with the islands not yet
  offered, with the same donor.
- **Steering a running loop.** The `--ideas` and `--knowledge` files are both re-read every iteration.
  Edit them in `~/kagg-evo/repo` (after editing here) and they take effect on the next iteration. Later
  models in an iteration see the edits the earlier ones proposed.
- **Stopping or restarting.** A stop (Ctrl-C, or SIGTERM to the python PID) ends the loop after the
  current iteration and saves it. The checkpoint is written before `ITERATION n` is logged, so a loop
  that runs old code can also be killed right after that line. `~/kagg-evo/restart_ladder1.sh
  <iterations> [--pool]` (run it in tmux; a copy is in `evolution_results/ladder_2026-09-26/`) does that at
  the next boundary and resumes in a new tmux session that appends to the same log. With `--pool` it also
  restarts the idle pool (new pool code): the old pool must report `COLAB_VMS_LEFT=0` first, and the new
  one makes new VMs on the loop's first request (give the user their links).
  - The loop's wrapper touches the pool's STOP file when the loop ends, so the script kills the
    wrapper first. Killing only the python process would stop the pool and its VMs.
  - Run ladder1 has a budget of 200 iterations since 12:35 UTC 09-26. With the predictor in both graphs an
    iteration takes about 30 min, about 38 with the 302-game gauntlet since iteration 31, so the rest takes
    about 4 days and 120-140 compute units at 1.3 units/h (the account had 175.7 units at 18:00 UTC 09-26).
  - Pool margins are cached per game (`<run>/scores/<bundle>.json`). A margin stays valid while the
    gauntlet fingerprint (everything under `hazel_runtime/`, `agent_graph.py`, the harness, the
    head-to-head seeds and the engine version) and the digest of its opponent's bundle are unchanged. The
    plan is not in the fingerprint, so a plan that grows plays only its new games; loop, prompt and
    knowledge files are in neither. Check before restarting: `Gauntlet(...).prepare()` must equal the
    `fingerprint` in `<run>/scores/*.json`.
  - `BEFORE_START='<command>' restart_ladder1.sh ...` runs a command after the kill and before the new start;
    if it fails, the loop is not restarted. The 09-26 19:35 restart used it for `upgrade_scores.py`, which
    carried the cache over to the per-game format (the old fingerprint had covered the plan).
- **Aim edits at layers that act.** `engine_activity.py` shows which of an engine's layers change its
  actions in play and which parameters each layer reads. It wraps each saved parent `agent`, and the
  recorded games equal unwrapped ones. For tetsutani_demand, 34 of 76 layers never acted in 48 games,
  and edits of their parameters changed 1-3 of 230 gauntlet games. The result is in
  `evolution_results/ladder_2026-09-26/engine_activity.json`, and the knowledge file summarises it.
  Run it on Colab or cliproxyapi, never on this PC.
- **Validation.** `ladder_validate.py` plays graphs from the run's checkpoint through the same pool:
  - which graphs: `seed`, `island:<name>`, `merge:<A>+<B>` (island A plus what B changed relative to
    the seed), or a file;
  - which games: arena validation seeds (salt 20260924) against every opponent, plus held-out lost
    seeds (`--lost`) from both seats, plus `--replays <replay_opponents dir>`: every replay opponent once,
    from our ladder seat (sets `replay:W|L|T`);
  - the report pairs every graph with the first one.

  Its request queues in the pool between gauntlets and delays the loop by its own length. Games are cached
  under the run name (`<run>/games/<name>.jsonl`): a rerun plays only missing games, and copying an earlier
  run's file to the new name reuses its games for the same labels.
  - **Ladder backtest** (`--seeds 0 --replays ...`): every graph against the recorded moves of all our
    replayed ladder games, won ones included, so it shows which lost games a change wins and which won ones
    it gives away. Changes a rival would not answer (sale timing, the guard) are measured most faithfully.
    A candidate for submission gets a backtest and then a fresh-seed validation. Scripts on cliproxyapi:
    `~/kagg-evo/backtest2.sh`, `validate_best.sh <label> <graph>`; results in
    `shinka/champions/evidence/alder_ford_20260927/README.md`.
  - The game cache is keyed by label, opponent, seed and seat, not by the graph: reuse a file only with the
    same labels for byte-identical graphs (a dry run prints each label's bundle id; compare them).
- **Rival counters (since 2026-09-27).** The optional `rival_counter` turn stage (first in the chain) runs
  `hazel_runtime/rival_model.MirrorTracker`: every observation shows the rival's whole farm, and on the same
  seed a copy of our engine has our money and farm step for step. By step 93 the rival is `mirror`,
  `nsell_opener` (money first behind ours at step 92 by the price of 3 wheat: buy N / sell N-5 openers such as
  the Forecast family, most 2965 agents and the public engine's 09-27 version), `wheat92_seller`,
  `other_opening` or `other`; from then on the node's `counters` for that class apply ({class: {NAME: value}}:
  engine constants read at call time, `engines.switchable`, and `_OG_*` guard parameters), set in the engine's
  module namespace, and the graph's own values otherwise. Edits: `{"channels": {"rival_counter": true},
  "counters": {...}}` (switching the channel on inserts the node, its concept node and its pin); graphs
  without the node play exactly as before. Evidence and the per-class value of each change:
  `shinka/champions/evidence/alder_ford_20260927_0940/README.md`; offline tools in
  `research/procedural_graph/rival/` (`rival_features.py`, `class_value.py`); tests `test_rival_counter.py`.
- **Islands on another engine.** An island may carry its own `seed` (`add_islands.py` adds one,
  `convert_island.py` moves an existing island onto another island's champion and seed in its place in the
  rotation, as Island-Oracle became Island-Next-Counter on 09-27); it mixes only with
  islands on the same engine (`EditEvolution.donors`). `graph_edits.validate_graph` checks a copy re-pinned to
  the runtime on disk (as `write_graph_bundle` does in every bundle), so a runtime change does not invalidate the
  graphs being evolved. A runtime change still changes the gauntlet fingerprint: check that unchanged graphs
  replay their cached games, then `carry_fingerprint.py`.
- Results and the iteration table: `evolution_results/ladder_2026-09-26/README.md`.
