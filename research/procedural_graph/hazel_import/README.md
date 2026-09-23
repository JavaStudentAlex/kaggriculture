# Hazel Weir → executable procedural graph

This is a **separate candidate**, not a replacement of the active master graph or
an edit to any ongoing frozen evaluation. Its source is the latest Kaggle
submission returned by the API during this work: **Hazel Weir, 56246758**.

## What changed

The older graph imports the development `initial.py`, arbitrates only market
orders, and can silently suppress routine orders when a target is absent.
This candidate instead retains the exact submitted `champion.py`, Mohui bundle,
NumPy oracle, feature extractor, and checkpoint. The submitted package remains
unchanged. A new entry point executes an explicit graph over that package.

- **8 turn stages**: production backbone → oracle observation → derived state →
  farmer maintenance → hand maintenance → market subgraph → sanitization →
  final-action oracle recording.
- **15 market stages**: setup/consolidation → opening scalp → deferred sells →
  wage liquidity → investment freeze → optional cash floor → optional livestock →
  feed reserve → shed pressure → pre-drop headroom → oracle front-running →
  town demand/fertilizer → early liquidation → final liquidation → routine dispatch.
- Every stage has a real executable binding. Missing bindings, missing routine
  dispatch, duplicate nodes, disconnected/reordered dependencies, and mismatched
  source hashes are rejected during construction.
- The market function's original AST is retained; generator checkpoints expose
  its ordered source blocks without rewriting expressions or dropping closures,
  deferred-sale state, or early returns. Inactive conditions pass through normally.
- The underlying production planner and maintenance functions remain **atomic
  inherited routines**. This is not a claim that every embedded planner branch
  has become independently mutable graph code.

`policy_graph.json` describes the execution schema. Thresholds/conditions are
source-bound in the preserved routines, not inferred from prose. Arbitrary
reordering is deliberately rejected because of state dependencies. A later
mutation system must update executable code/schema together and rerun parity or
competitive tests; this graph is not a drop-in JSON mutation for the legacy engine.

## Important corrections to previous summaries

Actual opening trades are **BUY 35 WHEAT / SELL 30 WHEAT**, not one unit each.
Optional cash-floor and livestock-topup guards are disabled in the submission;
this import does not silently enable them. Final full-stock sales still require a
price of at least 1. Existing market order multiplicity/order and legal `DROP`
actions are preserved rather than naively deduplicated or filtered.

## Files

- `graph_runtime.py`: validated runtime and source-preserving market compilation.
- `policy_graph.json`: executable stage graph with source hash and line mappings.
- `POLICY_MAP.md`: detailed source audit and limitations.
- `build.py`: reproduces a new payload from saved submitted packages, refusing to
  overwrite an existing payload.
- `payload/agents/candidate_hazel_graph/main.py`: runnable standalone entry point.
- `payload/manifest.json`: file fingerprints and diagnostic evaluation design.
- `test_graph.py`: structural, stateful market equivalence, sanitation and fallback tests.
- `verify_replay.py`: compares complete actions from isolated graph/submission
  processes on the same saved observation stream.
- `results/`: isolated full-game diagnostic results and gzipped traces.

## Verification commands

From the repository root:

```sh
.venv/bin/python research/procedural_graph/hazel_import/test_graph.py
.venv/bin/python -B research/procedural_graph/hazel_import/verify_replay.py
.venv/bin/python -B research/procedural_graph/hazel_import/payload/runner.py \
  --root research/procedural_graph/hazel_import/payload \
  --manifest research/procedural_graph/hazel_import/payload/manifest.json \
  --output research/procedural_graph/hazel_import/results \
  --shards 1 --seed-count 1 --workers 3 --save-traces \
  --match-timeout 900 --deadline-seconds 1800
```

The diagnostic includes Hazel, Copper, and the frozen old iteration-27 graph,
with one seed in each seat. These are known regression seeds, **not a held-out
benchmark**. `benchmark_pass` from the runner means complete/valid execution,
not competitive superiority. Use `overall.planned` / `selected_jobs`: the inherited
runner's `full_manifest_jobs` field incorrectly assumes 15 opponents.

No competition submission, live-evolution restart, paid compute, or replacement
of the master graph is performed by this import. The first objective is faithful
Hazel behavior, not an unsupported claim to outperform Hazel.
