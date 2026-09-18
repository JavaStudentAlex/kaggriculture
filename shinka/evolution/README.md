# `shinka/evolution/` — the Kaggriculture Digital Red Queen task

The ShinkaEvolve task directory. `shinka_run` requires exactly two files here
(`evaluate.py` + `initial.py`); everything else supports them.

| file | role | mutable? |
|---|---|---|
| `initial.py` | seed program — **Hazel Weir**, the last champion added (2026-09-15, Kaggle 56246758) | the EVOLVE-BLOCK is rewritten by the LLMs |
| `evaluate.py` | the DRQ evaluator: 20 seeds seat 0 + 20 disjoint seeds seat 1 per champion, 75 % crowning gate | fixed |
| `shinka_config.yaml` | models, novelty, islands, the task prompt | fixed |
| `launch_shinka.sh` | preflight + launch | fixed |
| `kagg_oracle.py`, `kagg_ttm_numpy.py` | the opponent order-flow oracle (read-only copies of the submission bundle) | fixed |
| `checkpoint/` | byte-identical in-play copy of `models/ttm_c256_h96_ft_2026-09-17` (AGENTS.md rule 6); git-ignored, then staged automatically by the launcher from the tracked source model | fixed |

## What is evolvable

The EVOLVE-BLOCK spans the **whole agent**, not just the market channel:

* reference data (`_SHOP_DEMANDS`, `_BASE_PRICE`) and **every** hyperparameter
* `farm_state` — the derived-state / feature builder
* `evolve_market_orders`, `evolve_farmer_action`, `evolve_hand_actions` — all three action channels
* `policy_sanitize` — the policy's own validity pass
* `decide` — top-level arbitration: whether to consult the Mohui backbone at all, how to blend or
  override its channels, ordering, phase- and seat-specific strategy switching

Functions inside the block may be added, removed, renamed, split or merged.

Immutable (outside the block): the backbone import, the oracle glue, `_hard_sanitize`, and the public
entry point **`agent(obs, configuration)`** — name and signature must not change. Any exception raised
inside the block degrades that turn to the backbone action instead of crashing the game.

## Evaluation protocol

Per candidate, against **every** champion in `shinka/champions/pool/`:

```
20 games, candidate on seat 0, seeds   101 ..  2020  (101 * i)
20 games, candidate on seat 1, seeds 70102 .. 72021  (70001 + 101 * i)
= 40 games per champion
```

The two blocks are disjoint, so a seat-specific overfit cannot be laundered into the other seat, and
both are fixed across generations so every candidate in a run is graded on identical worlds. They are
the same blocks the recorded champions were scored on, so numbers stay comparable across runs.

`combined_score` = overall win rate. **Ties are not wins.** A candidate is crowned into the pool — and
so becomes an opponent for every later candidate — only when it reaches **75 %** overall *and* crashes
zero times *and* passes the starter sanity game. Crowning appends to `pool/CROWNED.jsonl` and copies the
program into `pool/` and `champions/roster/`.

With the 15-champion pool that is 15 × 40 = **600 games per candidate**.

## Two levels of novelty inspection

1. **Embeddings** — every proposal is embedded with `qwen3-embedding:8b` (Ollama, 4096-dim). Cosine
   similarity ≥ `code_embed_sim_threshold` (0.985) against an existing program flags it as a probable
   near-duplicate.
2. **LLM judge** — `gemini-3.1-pro-preview` then reads the flagged candidate against the programs it
   collided with and rules on whether the change is genuinely novel. Rejected proposals are resampled,
   up to `max_novelty_attempts` (3) times.

## Models

| role | model(s) |
|---|---|
| mutation pool (UCB-sampled) | `gpt-6-astra`, `gpt-6-luna`, `gemini-3.1-pro-preview`, `gemini-3.8-flash`, `claude-opus-5`, `claude-sonnet-5` |
| supervisor / meta | `gpt-6-astra`, every **5** generations (`meta_rec_interval: 5`) |
| novelty judge (level 2) | `gemini-3.1-pro-preview` |
| embeddings (level 1) | `qwen3-embedding:8b` via Ollama |

All served through the local proxy as `local/<model>@http://localhost:8317/v1`.

## Run it

```bash
tmux new -s kagg-shinka \
  'GENERATIONS=500 bash shinka/evolution/launch_shinka.sh 2>&1 | tee /results/kagg/logs/shinka.log'
```

The launcher preflights the interpreter, `kaggle-environments`, torch/CUDA, the checkpoint, the champion
pool, the LLM proxy and the embedding model, starts CUDA MPS when `mps.sh` is present, then execs
`shinka_run`. Knobs: `GENERATIONS`, `RESULTS_DIR`, `EVAL_WORKERS`, `EVAL_JOBS`, `PROPOSAL_JOBS`, `PYTHON`.

This `shinka_run` build has **no `--resume` flag** — point `RESULTS_DIR` at an existing run directory to
continue from the programs already in its database.

### GPU inference

The oracle runs on GPU when torch+CUDA are available (`KAGG_ORACLE_DEVICE=auto`), falling back to the
numpy backend on CPU otherwise — the fallback is automatic and the agent still plays, just slower
(~130 ms/forecast on CPU). Evaluator workers are **spawned, never forked**, so each loads the checkpoint
exactly once; CUDA MPS lets the many small clients share the GPUs.

### Evaluate one program without crowning

```bash
KAGG_HISTORY_DIR=$PWD/shinka/champions/pool \
python shinka/evolution/evaluate.py --program_path <prog.py> --results_dir <dir> --no-induct
```

## Tuning knobs (env)

`KAGG_CROWN_THRESHOLD` (0.75) · `KAGG_GAMES_PER_SEAT` (20) · `KAGG_EVAL_WORKERS` ·
`KAGG_HISTORY_DIR` · `KAGG_GAME_LOG` · `KAGG_NO_INDUCT=1` · `KAGG_ORACLE_DEVICE` · `KAGG_TTM_DIR`
