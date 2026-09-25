# Promoting Evolved Agents to Kaggle Submissions

Use this workflow when promoting one or more candidates from an evolutionary run while a separate tournament or search job may still be active.

## 1. Freeze artifact identity

1. Resolve each requested generation to its exact immutable candidate file; do not package a draft evaluator, current working tree, or similarly named champion.
2. Hash the candidate bytes and compare them with any frozen champion snapshot expected to represent that generation.
3. Record a local manifest containing candidate hash, archive hash, generation-to-neutral-name mapping, and archive size. Keep generation metadata private rather than putting it in the public submission message.

## 2. Build a portable archive

- Work in a separate staging directory per candidate.
- Copy the full transitive dependency closure, including package `__init__.py` files and required data/weights.
- Include upstream `LICENSE` and `NOTICE` files when redistributing upstream code.
- Put `main.py` at archive root and exclude `__pycache__`, temporary files, credentials, test data, and absolute-path assumptions.
- Compile every staged Python file before archiving.
- Because `kaggle-environments` can select the last newly bound callable, make the final definition in `main.py` an explicit submission wrapper:

```python
def kaggle_submission_agent(obs, config=None):
    return agent(obs, config)
```

- Open the completed tarball and assert exactly one root-level `main.py` plus every required dependency and notice.

## 3. Validate the artifact, not the source tree

1. Extract the completed archive into a new temporary directory.
2. Start a fresh Python process with the original repository absent from `sys.path` (`PYTHONPATH=` is a useful guard).
3. Load the extracted root `main.py` by file path, then run complete episodes over multiple seeds in both seats.
4. Require terminal `DONE` statuses, finite rewards, no stderr/tracebacks, and acceptable runtime. Compare mirrored-seat rewards when determinism is expected.
5. Validate each candidate independently; a shared mutable imported backbone can make in-process candidate comparisons misleading.

**Episode-length pitfall:** In `kaggle-environments`, `len(env.steps)` may equal configured `episodeSteps` itself. For Kaggriculture with `episodeSteps=720`, a clean complete game has 720 stored step states, not necessarily 721. Treat terminal statuses and the pinned engine's observed convention as authoritative; do not create a false failure with an assumed `episodeSteps + 1` assertion.

After copying an archive across hosts/pods, recompute SHA-256 and require it to match the build manifest before upload.

## 4. Submit and verify lifecycle

1. Check `kaggle competitions submission-limits <slug>` immediately before consuming quota.
2. Use neutral public names that reveal neither generation, internal fitness, tournament win rate, nor exploit details.
3. Submit candidates sequentially and preserve the CLI acknowledgement and remaining-quota count.
4. Poll `kaggle competitions submissions <slug> --csv`; capture each registered submission ID, filename, message, and status.
5. Distinguish states precisely:
   - CLI upload acknowledgement: bytes accepted.
   - Row with ID and `PENDING`: registered and awaiting validation/grading.
   - `COMPLETE`: server-side validation finished.
   - Error: inspect logs before spending another quota slot.

Do not report `PENDING` as completed evaluation, and do not wait indefinitely merely to turn a successful registered submission into `COMPLETE`; monitoring can continue separately.

## 5. Operational isolation and cleanup

- Submission work does not authorize restarting, stopping, or mutating an unrelated evolution run or championship. Verify the long-running job remains healthy without changing it.
- Keep staging and validation paths distinct from checkpoints and tournament snapshots.
- Remove extracted temporary validation directories, copied transfer files, bytecode caches, and helper scripts after verification. Retain only deliberate archives/manifests in the project artifact directory.

## 6. Oracle-based champions (the standard procedure since 2026-09-12)

Champions evolved on the `shinka/evolution/initial.py` lineage carry the opponent
order-flow oracle (`kagg_oracle.py` + a TTM checkpoint). They are packaged with
`shinka/champions/submissions/make_submission.py`, which encodes everything
learned from the "Orchard Tide" attempts (`submissions/orchard_tide/README.md`):

```bash
/home/jovyan/shinka_venv/bin/python shinka/champions/submissions/make_submission.py \
  --champion shinka/champions/top/<run>/gen_<N>/main.py --name "<Two Words>" \
  --note "<private provenance: run, generation, eval numbers>" --validate
kaggle competitions submission-limits kaggriculture          # right before spending quota
kaggle competitions submit -c kaggriculture -f shinka/champions/submissions/<TwoWords>.tar.gz -m "<Two Words> - Adaptive production and trade"
kaggle competitions submissions kaggriculture --csv           # PENDING -> COMPLETE (~3 min); ERROR -> episodes/logs/replay
```

What the builder does and why each part is mandatory:

1. **Bootstrap `main.py`.** `kaggle_environments` execs the submitted file in a
   bare namespace (no `__file__`), takes the LAST callable, and APPENDS the
   bundle directory to `sys.path` only while the file runs. The champion calls
   `Path(__file__)` at import, so `main.py` locates the bundle (frame filename),
   sets `KAGG_MOHUI_DIR` / `KAGG_ORACLE_SRC` / `KAGG_TTM_DIR` / `KAGG_OPP_MODEL_SRC`,
   inserts its directory at the FRONT of `sys.path` unconditionally, imports the
   byte-identical champion as `champion.py`, and ends with `kaggle_submission_agent`.
2. **numpy oracle backend.** `main.py` sets `KAGG_ORACLE_BACKEND=numpy`;
   `kagg_ttm_numpy.py` runs the checkpoint without torch/transformers. The first
   request to a Kaggle agent is where `main.py` is exec'd and it has the 60 s
   `remainingOverageTime` bank + 1 s; a torch import there produced TIMEOUT at
   step 1 on both seats (submission 56192942). Kaggle measures ~2.5x one pod core:
   the numpy build imports in ~5 s and forecasts in ~320 ms per step there.
   After any checkpoint change run `check_oracle.py numpy` (torch vs numpy on
   real rows; expect <= ~1e-3, which is torch's own float32 error).
3. **Complete closure.** `checkpoint/` (4 files), `opponent_model/features.py` +
   `mechanics.py` (the oracle imports them), `mohui_v66/` with LICENSE + NOTICE.
   A missing piece does not crash the champion: it silently plays oracle-less
   (`ORACLE_STATS["errors"]`), which is why the validation must run with the
   repository unreachable.
4. **Validation of the extracted archive** (`--validate`, or the bundle's
   `validate.py`): clean venv `/results/kagg/venv-kaggle-sim`, `python -I`, empty
   `HOME`, no `KAGG_*`, cwd `/`, `main.py` passed as a path string, one core;
   both seats x starter/random x seeds; requires DONE statuses, finite rewards,
   no stderr traceback, oracle live on numpy with torch never imported, and
   every module resolved from inside the extracted directory. Add a fidelity
   replay against a pool champion when the numerics changed (the bundle must
   reproduce the repo champion's cash to the dollar).
5. **Names and records.** Public name = neutral two-word codename (the builder
   rejects gen/score/oracle words); `MANIFEST.json` in the staging dir keeps the
   candidate/archive hashes, the mapping and the validation report. Errored
   submissions are refunded by Kaggle; an ERROR means a validation episode with
   empty agent logs and `TIMEOUT`/`ERROR` statuses in its replay -- read it
   before spending another slot.

## 7. Procedural-graph agents (graph + Hazel runtime + predictor), since 2026-09-25

A graph agent is the arena bundle `research/procedural_graph/arena/payload.write_graph_bundle`
builds. It holds `policy_graph.json`, the runtime `hazel_runtime/` (Hazel Weir's champion as
executable stages, the numpy oracle, `checkpoint/` and an optional `calibration.json`) and the
entrypoint `agent_graph.py`. The entrypoint calls `Path(__file__)` at import, so the bundle
cannot be submitted as it is. Package it with:

```bash
python3 research/procedural_graph/make_graph_submission.py --graph <graph.json> \
  --checkpoint models/<refit> --calibration research/procedural_graph/calibration/<refit>/calibration.json \
  --name "<Two Words>" --note "<private provenance>" --validate --fidelity 101 \
  --python <clean venv python with kaggle-environments==1.32.7 and numpy, no torch>
kaggle competitions submission-limits kaggriculture
kaggle competitions submit -c kaggriculture -f shinka/champions/submissions/<TwoWords>.tar.gz -m "<Two Words> - Adaptive production and trade"
```

What it adds on top of section 6:
1. **The same bundle as the arena.** The graph is re-pinned to the copied runtime, with the
   predictor and its calibration in `hazel_runtime/checkpoint/`. The graph's own entrypoint is
   kept byte-identical as `agent_graph.py`, so its fingerprint pin still holds.
2. **Bootstrap `main.py`.** It finds the bundle (frame filename or `sys.path`), forces the numpy
   backend, and imports `agent_graph.py` as a real module. It then builds the engine at once:
   the engine verifies every runtime file against the graph's pins, and a mismatch fails loudly
   at import. The file ends with `kaggle_submission_agent`.
3. **Validation** is section 6's validator (both seats × starter/random × seeds, `python -I`,
   empty HOME, cwd `/`, one core) plus graph checks: `agent_graph` resolved inside the extracted
   directory, the engine built, the checkpoint inside, and the calibration present exactly when
   one was packaged.
4. **Fidelity** (`--fidelity SEED`): the same seed against starter, once through the Kaggle
   loader (`main.py` path) and once through the arena harness (process-isolated `BundleAgent`
   on `agent_graph.py`). The cash must be identical, so the package is exactly the agent the
   arena tested.
5. **Names.** The same neutral codename rules as section 6, also rejecting "feed" and "graph".

Timing, measured 2026-09-25 on one laptop core:
- The cold start (exec of `main.py` including the engine build) takes 5.5–6.8 s, against about
  2.2 s for Hazel Weir's champion bundle on a pod core. Kaggle measured 5 s for Hazel, so expect
  roughly 7–9 s at step 0, taken from the 60 s overage bank.
- Moves average 165–175 ms and peak at about 0.5 s on an idle machine.
- Validate on an idle machine: concurrent jobs inflate the step times and the first move (one
  contended test showed 14.6 s).
