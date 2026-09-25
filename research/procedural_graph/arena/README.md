# Arena: graph variants vs submissions on Kaggle CPU notebooks

Head-to-head games between graph variants (graph edits of `../policy_graph.json`) and
saved submissions, played on Kaggle's free CPU notebooks.

```sh
K=research/procedural_graph/arena
python $K/payload.py   --eval-id r2 --pairs cadence3:hazel,edges:hazel,hazel:mohui13 --seeds 40
python $K/notebooks.py upload --eval-id r2          # private dataset kagg-arena-r2
python $K/notebooks.py push   --eval-id r2          # 5 private CPU script kernels, 4 games each at a time
python $K/notebooks.py wait                          # returns when every shard is COMPLETE/ERROR
python $K/notebooks.py fetch  --eval-id r2          # results_*.jsonl + traces/ under runs/arena/r2/results
python $K/notebooks.py delete                        # leave no kernels behind
python $K/report.py research/procedural_graph/runs/arena/r2/results --traces
```

Large batches run on **Google Colab VMs** (AGENTS.md section 3.1: accounts, machine sizes,
rules). The runner itself runs in tmux on cliproxyapi, which is always on:

```sh
python $K/payload.py --eval-id s1 --graph best=<best_graph.json> --pairs best:hazel,best:mohui --seeds 40
rsync -a research/procedural_graph/runs/arena/s1/payload cliproxyapi:kagg-colab/runs/s1/
ssh cliproxyapi 'tmux new -d -s kagg-colab-s1 "cd /home/alex/kagg-colab && export HOME=/home/alex/kagg-colab/home \
    PATH=/home/alex/kagg-colab/home/.local/bin:\$PATH && python3 arena/colab_run.py --payload runs/s1/payload \
    --out runs/s1/results.jsonl --run-name s1 --vm colab2:hm --vm colab2:hm > runs/s1/colab.log 2>&1; \
    echo COLAB_RUN_EXIT=\$? >> runs/s1/colab.log"'
# when runs/s1/colab.log shows COLAB_RUN_EXIT=:
rsync -a cliproxyapi:kagg-colab/runs/s1/results.jsonl cliproxyapi:kagg-colab/runs/s1/traces research/procedural_graph/runs/arena/s1/
```

Each `--vm colab<N>:hm|std` is one VM: High-RAM (8 CPUs, 8 workers) or standard (2 CPUs).
The runner uploads the payload, builds a Python 3.12 venv with the pinned
`colab_requirements.txt`, plays the games, downloads the results and stops every VM it
created. Rerunning plays only the missing games; `--attach` finishes a killed runner's
sessions; `--cleanup` stops them. Colab reproduces Brev's results to the dollar (checked
on six run-2 games, 2026-09-24).

The runner itself runs on cliproxyapi, in `~/kagg-colab/`, and not on this PC, which sleeps.
Colab deletes a VM after ~30 min without keep-alive pings, or ~60 min without a command.
Its access tokens last 1 h, and the CLI then drops live VMs. The runner polls every VM every
minute, refreshes tokens every 30 min (`colab_readopt.py`), pulls results every 10 min, and
replaces a VM that Colab deleted. AGENTS.md section 3.1 has the layout, the launch command
and the measurements. `colab_readopt.py` re-registers a running VM that the CLI lists as
`[?]`. `colab_run.py --jobs FILE` replays only the listed games. `windows_awake.sh` keeps
this PC awake, for a runner that has to run here.

## Predictor calibration (after every refit)

The full procedure, including why, how to read the result and the rematch, is AGENTS.md
section 13. In short: the graph's oracle thresholds (front-run `score_4` 0.30, the batch steps,
`score_24` 0.45, `units_24` 2.0, the cadence bypass) were tuned with the **reference** predictor,
`ttm_c256_h96_ft_2026-09-13` (`hazel_runtime/checkpoint`). A refit's scores sit at other levels.
`calibration.json` holds per-product multipliers on `score_4`, `score_24` and `units_24`, so the
refit crosses each threshold as often as the reference did on the same turns of our own games.
The oracle applies it when it sits next to the checkpoint. The weights and the agent are unchanged,
and so is ranking (AUC).

```sh
# one command (run in tmux, ~40 min on one T4): payload, Colab run on cliproxyapi, pull, fit
python3 research/procedural_graph/arena/calibrate.py --refit models/ttm_c256_h96_ft_<D>
# -> research/procedural_graph/calibration/ttm_c256_h96_ft_<D>/own_games/ (commit it)
# play it: payload.py --graph G=<graph> --checkpoint G=models/... --calibration G=<that dir>/calibration.json
```

The pieces, for other data or a manual run:
- `calib_payload.py` builds the payload from three kinds of games:
  - `--game-set calibration/game_set`: the committed 600 own games (traces.zip + games.json);
  - `--trace 'GLOB[@ours|theirs|both]'`: other arena traces;
  - `--zip <Kaggle day zip>`: ladder games (the refit's held-out ones plus a sample of the rest).
- `calib_worker.py` ships in the payload as `arena.py`, so `colab_run.py` drives it like a game
  batch.
  - A trace is replayed through the engine with its recorded actions and seed. The job fails
    unless the replay reproduces the recorded final cash.
  - On a T4 the worker installs torch and granite-tsfm (about 40 s) and checks the torch model
    against the numpy port (within 1e-5). It then forecasts every origin in batches (about 330
    forecasts/s; about 15 own games a minute for two models) while 8 processes rebuild the
    features. Without a GPU it falls back to the numpy port, which is about 100x slower.
  - Each game's npz records the product order.
- `calib_fit.py` fits and reports (plain numpy).
  - `--group REGEX` fits on a subset; `--held-out-only` fits on ladder held-out games only.
  - Guards: at least 100 reference firings per threshold, a product base rate of at least 0.5 %,
    and factors clipped to 0.5–2.
  - Its `METRICS` threshold list mirrors `hazel_runtime/champion.py`. Update it when the
    policy's thresholds change.
- `calibrate.py` chains them and resumes a run that was interrupted.

Results so far are in `calibration/ttm_c256_h96_ft_2026-09-23/`: the top level is the ladder fit,
and `own_games/` is the fit on our own games; their READMEs have the findings.

A Brev CPU box is the alternative (after `brev login`; delete it when done, it bills by the hour):

```sh
brev create kagg-arena-80 --type n2d-highcpu-80      # 80 vCPU / 80 GB, ~$2/h; ~60 games at a time
brev refresh                                          # adds the ssh alias
tar -czf payload.tar.gz -C research/procedural_graph/runs/arena/r2 payload
scp payload.tar.gz $K/remote_run.sh kagg-arena-80:~/ && ssh kagg-arena-80 'bash ~/remote_run.sh ~/payload.tar.gz 60 r2'
ssh kagg-arena-80 'grep -c rewards ~/arena/r2/arena.log'          # progress; ARENA_EXIT= at the end
scp -r kagg-arena-80:~/arena/r2/results.jsonl kagg-arena-80:~/arena/r2/traces research/procedural_graph/runs/arena/r2/
brev delete kagg-arena-80
```

Analysis: `replay_trace.py <trace.json.gz>...` replays `--trace-dir` traces through an instrumented
engine (every executed unit and price, town consumption, per-day animals, escapes and empty tiles;
it checks that the recorded cash is reproduced). `opening_cash.py` prints the day-1 hire cash and the
wheat left for step-0/1 wheat openings against each opponent opening.

`payload.py` writes to `research/procedural_graph/runs/arena/<eval-id>/` (git-ignored).
Needs the `kaggle` CLI with credentials in `~/.kaggle` (see `.agents/skills/kaggle-account-access`)
and, for local runs, `kaggle-environments==1.32.7`:
`python $K/arena.py --jobs <payload>/jobs.json --root <payload> --out results.jsonl --workers 2`.

## Design

- **One game per seed.** The engine clears both seats' orders in lockstep and the agents are
  deterministic: replaying a seed with the seats swapped gives identical cash (checked on
  seed 424242, $70,217 vs $71,219 both ways). The candidate's seat alternates with the seed
  index; every variant uses the same seed list, so variants are paired.
- **Mirror check.** Variant `mirror` (graph with every override off, Hazel's checkpoint)
  must tie Hazel exactly in every game. It does: the compiled graph is a byte-for-byte policy
  clone, so any margin a variant shows comes from its graph edits alone.
- **Variants are graph edits** (`variants.json`): the surgical switches, the checkpoint, and
  node attributes the runtime executes: `parameters` (champion EVOLVE-block constants),
  `enabled` on market stages, `order` on `routine_dispatch`. No policy code per variant.
- **Opponents**: `hazel`, `copper`, `orchard` (the submitted packages' archive files),
  `mohui` (the bundled Mohui v66 backbone) and `mohui13` (the backbone with the 13/9 opening
  wheat scalp seen in Hazel's largest ladder losses; a weak ladder proxy: the real opponents
  earn far more).
- **Isolation**: each agent runs in its own interpreter (`shinka/evolution/pool_upgrade_bundle_agent.py`),
  a fresh pair per game; ~1 GB RAM per game (two ~400 MB agents + the engine).

## Limits measured on 2026-09-24

- 5 concurrent batch CPU sessions per account; a 6th push is rejected, not queued.
- A kernel's log and output are readable only after it finishes: size shards to finish
  in a few hours and keep results resumable.
- Brev n2d-highcpu-80 with 60 concurrent games: ~320 s per oracle-vs-oracle game, ~190 s
  against the oracle-free Mohui opponents; ~55 GB RAM in use. Results are identical to the
  same games played locally (deterministic across machines).

## Backbone patch: yarn store as the second shop (2026-09-25)

`yarn_second_fix.py <bundle> <out>` copies a graph bundle and adds one routing rule to its
Mohui v66 backbone (`candidate_v66_meta_closed_loop.py`, step 144): a YARN_STORE opened second,
after a shop other than YARN_STORE / PET_CAFE, with no other route active, switches to the
`bakery_yarn` route (the default route until step 144, the sheep plan after it). The graph's
runtime pins are recomputed; pass the result to `payload.py --bundle NAME=DIR`.
`yarnfix_report.py <run dir>` scores such a bundle against the unpatched one: repeats, the
mirror property of seeds where the rule never fires, and the games where it does. Result
against Linden Brook on its 59 ladder seeds: `shinka/champions/evidence/linden_brook_loss_audit_20260925/`.
