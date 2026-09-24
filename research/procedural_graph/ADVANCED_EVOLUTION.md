# Advanced graph evolution: edit-based SIFT islands

Entry point: `bash run_advanced_evolution.sh [options]`, which runs
`highcpu_island_evolution.py`. Older single-candidate and notebook launchers are not
replacements for it.

```sh
# games on this machine (about 1 GB RAM per concurrent game)
bash run_advanced_evolution.sh --workers 60
# games on a Brev box; the loop, the LLM proxy (:8317) and the run directory stay local
bash run_advanced_evolution.sh --executor ssh --host kagg-arena-80 --workers 60
```

Options:

- `--iterations`, and `--run_dir` (default `runs/evolution`, git-ignored; resumes from
  its `checkpoint.json`).
- `--seeds_per_opponent`: default 40, which is 20 games per seat on 40 different seeds,
  the minimum.
- `--opponents`: default `mohui13,mohui,hazel,willow`, a ladder-like pool (see below).
- `--seed_graph`: default `policy_graph.json`; for a follow-up run, pass the previous
  run's `best_graph.json`.
- `--ideas`: a markdown bullet list injected into every mutation prompt, re-read each
  iteration (`evolution_ideas.md`).
- `--alpha` (default 0.05), `--candidates`, `--supervisor_interval`.

Guided short runs: run 6 iterations (one per island), read `candidates.jsonl` and
`best.json`, update `evolution_ideas.md` and `evolution_knowledge.md`, and start the next
run from the best graph.

## What evolves

The merged graph (`runtime: hazel_merged_v1`) executes a fixed champion program. Its
concept descriptions and edge guidance are documentation, so rewriting them changes
nothing in play. A mutation is therefore an **edit of the executable controls**
(`graph_edits.py`):

- `parameters` on turn/market chain nodes: 37 champion EVOLVE-block constants the code
  reads, plus the runtime's `_TOWN_CADENCE_PHASE`. The same JSON type as the submitted
  value is required, and `null` restores the submitted value. Four constants the
  champion defines but never reads are refused.
- `stages`: turn a market stage off or on (`market_setup` and `routine_dispatch` always run).
- `dispatch_order`: `submitted` or `sells_first` for the routine_dispatch node.
- `surgical`: the switches of `hazel_runtime/surgical.py`.

`apply_edit` places each parameter on the stage whose code reads it and fails closed on
unknown names, wrong types and edits that change nothing. `validate_graph` builds the
runtime and plays the first 30 turns in a child process; a rejected graph or any stage
fallback is an error.

## One iteration

1. **SIFT stage 1.** A UCB1 bandit over the seven proxy models (`shinka_graph_mab.py`)
   picks three distinct models.
   - Each gets the controls table with the island champion's current values, the island
     focus, the champion's per-opponent results and worst games, the last 12 edits
     played in the run on any island with their per-opponent results, the supervisor's
     guidance and `evolution_knowledge.md` (verified engine facts and measured arena
     results).
   - Each returns `{"rationale", "edit"}` (`BAMGraphMutator.mutate_edit`).
   - An edit whose normalized settings were already evaluated anywhere in the run is
     refused before any game.
   - Every failure (type, runtime, duplicate) goes back to the model, up to three attempts.
2. **SIFT stage 2.** `gpt-6-astra` compares the viable edits pairwise, shown as their
   control changes (`SIFTGraphJudge.rank_edits`). A Bradley-Terry fit ranks them, and a
   reply without a valid verdict is no vote, not a default winner. Only the winner is
   played.
3. **Paired gauntlet** (`graph_gauntlet.py`, games in process-isolated agents via
   `arena/arena.py`).
   - The winner plays the run's seeds against the opponent pool, plus head-to-head games
     against the island champion. Its seat alternates with the seed index.
   - The default pool is Mohui13 (the 13-wheat openers are 12 of 26 opponents in Hazel's
     ladder games), the plain Mohui v66 backbone, Hazel Weir and Willow Ford (the
     Harvest Current stand-in).
   - Copper and Orchard are also available but nearly duplicate Hazel's policy.
   - Each game's cash margin is compared with the champion's margin on the same game;
     head-to-head is compared with 0, because a graph playing itself ties exactly.
   - The winner replaces the champion only if an exact sign test over the changed games
     gives p <= alpha and the mean change is positive. A game error, a missing game or
     any graph fallback makes the evaluation invalid.
   - Per-game pool results are cached per graph and evaluation fingerprint (opponent,
     runtime and harness bytes, seeds, engine), so a champion is never replayed.
4. Every `--supervisor_interval` iterations the meta-supervisor
   (`shinka_graph_supervisor.py`) reviews the champions' controls and recent results.
   Its recommendations go into the next mutation prompts, and an unparseable reply
   gives none.

Six islands, each focused on one group of controls: Opening, Town, Shed, Oracle,
Endgame and Dispatch.

## Seeds and outputs

- **Seeds.** Evolution seeds (salt 20260925) are disjoint from the arena validation seeds
  (salt 20260924, `arena/payload.py`), so a result can be confirmed on seeds that played
  no part in selecting it.
- **Run directory.** Everything is written under `--run_dir`:
  - `checkpoint.json` (islands, seen settings, guidance)
  - `candidates.jsonl` (every proposal, failure and gauntlet verdict)
  - `best_graph.json` and `best.json` (the island champion with the largest mean gain
    over the seed on the pool games, with its changes and paired statistics)
  - `status.json`
  - `bundles/`, `jobs/`, `games/`, `logs/` and `scores/`
- **The committed graph is never written.** `policy_graph.json` is read once as the seed.
  Promoting a result means a reviewed commit, after confirmation on the arena
  validation seeds.

## Novelty

Two graphs with the same normalized executable settings play identically, so the
duplicate check is exact, and it runs on those settings rather than on the prose. The
Qwen3-Embedding / semantic-judge modules (`shinka_graph_novelty.py`,
`qwen_novelty_archive.py`) compare whole prose graphs. They stay in the repository with
their tests but are not used for edits: every edited graph is at least 0.999 cosine
similar to its parent.

## Earlier runs

The prose-mutation runs up to 2026-09-22 did not evolve the executed policy:

- The mutator asked for `{version, name, description, nodes, edges}`. `agent_graph.py`
  loads the preserved pre-merge agent for any graph without the merged runtime, so
  candidates were a different agent.
- The judge saw the first 6,000 characters of a 44 kB graph, which is before any node
  or edge.
- Three graph champions inducted into `shinka/champions/pool` came from few-step
  games: 83-100% win rates with a mean cash of $1,769. They remain there because the
  frozen evaluator uses `champ_evo_0027` as its self-control, but this pipeline no
  longer reads that pool.

`mutate_graph` now refuses merged graphs, and `rank_candidates` shows the JSON paths
where two graphs differ.

## Verification

`python -m unittest test_graph_evolution test_surgical_graph` covers the following,
with fake models, fake judge replies and fake game results:

- edits reach execution;
- type, stage and duplicate refusals;
- the judge seeing the changes, and no-vote handling;
- the paired statistics;
- gauntlet caching;
- a promoting iteration that writes nothing outside the run directory.
