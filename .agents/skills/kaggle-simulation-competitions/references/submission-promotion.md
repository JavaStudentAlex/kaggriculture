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
