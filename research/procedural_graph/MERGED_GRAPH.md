# Current procedural graph merged with Hazel Weir

## Active local entry point

`research/procedural_graph/agent_graph.py` now reads the **current**
`research/procedural_graph/policy_graph.json` and runs the merged graph.
This supersedes the earlier separate-candidate-only status in `hazel_import/README.md`.
The remote evolution runner and already-uploaded Kaggle notebooks were **not**
restarted, repackaged, or modified. No competition submission was made.

## Merge contents

All 16 concepts from the previous graph remain, with source-bound `bindings`,
corrected descriptions, and `previous_description` provenance. Additional
execution nodes give 38 top-level nodes total. Every edge endpoint resolves.

The execution subgraphs contain eight turn stages and fifteen market stages.
The complete submitted bundle is preserved in `hazel_runtime/`, including:

- Mohui v66 production, routing, plot/animal management, hires, land purchases,
  base actions and all bundled inherited strategy modules;
- farmer and hand action channels, including idle-only maintenance;
- market setup, consolidation and value-weighted eviction;
- opening scalp, deferred reissue, wage purchase filter, investment freeze;
- feed reserves, shed pressure and protected carried-harvest headroom;
- forecast-driven front-running, adaptive town demand and fertilizer sales;
- early/endgame/final liquidation and routine market dispatch;
- submitted sanitizer, NumPy oracle, checkpoint, observation and action recording.

Low-level inherited planner branches remain source-preserved **atomic routines**,
not newly reimplemented graph primitives. The merge contains their full code and
calls them, rather than pretending a short prose description implements them.

## Conflicts resolved explicitly

The old graph contained intent that its runtime did not implement, as well as
contradictions with the actual submission. Every old node has an entry in
`merge_resolutions`:

- opening scalp uses buy35 / sell30, not one-unit trades;
- no imaginary watering-can/well observations or actions;
- the ten-order cap is for the market channel, with separate farmer/hand actions;
- no arbitrary nine-worker cap, day13 hiring freeze, or static shop reserves;
- submitted narrow wage guard retained; disabled static cash-floor/topup stay off;
- exact pressure predicates and final-sale price>=1 checks retained;
- routine actions cannot disappear due to missing decorative edge targets.

This is a **behavior-preserving executable baseline merge**, not a claim of new
strategic superiority to Hazel. Future changes can use this complete policy as
their baseline and must be evaluated.

## Safety / compatibility

- Source fingerprints pin every bundled runtime file. No new model checkpoint is
  substituted mid-comparison.
- Graph integrity rejects unknown bindings, missing stages, dangling endpoints,
  disconnected execution, source mismatches and illegal reorderings.
- Use process-isolated agents. Foreign preloaded modules are rejected rather than
  silently contaminating the comparison.
- Backups of the exact prior graph, agent and engine are in `hazel_import/premerge/`.
- Explicit `KAGG_GRAPH_PATH` to a historical graph retains the preserved legacy
  agent behavior. Frozen payloads were not changed.
- The old `ProceduralGraphEngine` rejects the merged schema to prevent silent
  market-only interpretation. Old prose-only mutation/packaging scripts are **not
  compatible** with the merged schema; their runtime+bundle contracts must be
  upgraded before restarting evolution. Existing scores are not transferable.

## Reproducible checks

```sh
.venv/bin/python research/procedural_graph/hazel_import/test_graph.py
.venv/bin/python -B research/procedural_graph/hazel_import/test_merged.py
.venv/bin/python -B research/procedural_graph/hazel_import/verify_replay.py
```

The frozen copy of the current entrypoint in `hazel_import/merged_payload/` is used
by the existing isolated frozen evaluator. Its diagnostic is six full games:
one seed per seat against Hazel, Copper and the old iteration27 candidate.
These are paired known regression seeds, **not a broad held-out benchmark**.

See `hazel_import/replay_equivalence.json`, `hazel_import/merged_results/` and
`hazel_import/MERGE_VERIFICATION.json` for actual measured evidence after completion.
