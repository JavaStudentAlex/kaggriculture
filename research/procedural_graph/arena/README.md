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
