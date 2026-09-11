---
name: kaggle-simulation-competitions
description: Develop, audit, evaluate, and package Kaggle simulation agents in this repository. Use for Kaggriculture champion inspection, replay analysis, reciprocal-seat evaluation, submission validation, and ladder-result interpretation.
---

# Kaggle simulation competitions

## Scope and permissions

Answer the requested question first. A read-only audit does not authorize policy edits, expensive tournaments, evolution restarts, uploads, or account changes. Do not start evolution or spend submission quota without an explicit request. Keep credentials outside the repository and out of logs.

## Repository layout

- `shinka/champions/roster/`: the active last-run champions (inspect the live roster; it may have been curated); this includes pre-existing agents that participated in that run.
- `shinka/champions/top/gen_*/main.py`: exact named copies of the 12 new champions.
- `shinka/champions/submissions/`: Gen 60 and Gen 58 packages with dependency closure and license notices.
- `shinka/champions/dependencies/mohui_v66/`: recovered shared backbone dependencies.
- `shinka/champions/manifest.json`: current/original source hashes and provenance.

Do not reintroduce earlier-run archive directories, databases, or logs. Preserve the current roster's source bytes when auditing or transferring it. Missing dependencies must be reported, not replaced by a guessed upstream revision. Repository contents are authoritative; do not assume an evaluator, checkpoint, or launcher exists just because a champion archive exists.

## Workflow

1. Inspect the exact requested candidate and its dependency closure; verify hashes against the manifest. Confirm interpreter, package versions, resolved environment configuration, and live observation schema.
2. For mechanics questions, inspect the installed engine source and cite its version and source lines. Historical policy comments, strategy dossiers, and replay impressions are hypotheses, not rules.
3. For evaluation, use reciprocal seats on the same seeds, isolated policy state per seat and game, and a frozen opponent roster. Shared imported backbone modules may contain mutable globals even if entrypoint names differ. Avoid simultaneous imports of competing champions in one process.
4. Distinguish fixed regression seeds from rotating development scenarios and truly held-out seeds/opponents. More historical opponents do not necessarily mean more independent scenarios. Compare champions on a common schedule before ranking them; earlier rolling-pool fitness is not directly comparable to later fitness.
5. Report W/L/T, errors, completed games, seat breakdown, and actual rewards. Count errors separately from draws. Explain whether a percentage is outright wins or wins plus half ties. Cash is not Kaggle rating and is not proof of policy superiority.
6. Inspect legal observation inputs when learning from traces. Replay-only private state and future events may be labels, not deployed model inputs. Submitted HIRE/SELL requests may be no-ops; trace executed state changes. Fixed opponent action tapes are diagnostic counterfactuals, not reactive opponent evaluations. Same seed need not preserve later random events when actions alter RNG consumption.
7. Package exact artifacts and validate full extracted submissions in fresh processes with source paths unavailable before any authorized upload. Preserve all upstream attribution and LICENSE/NOTICE. Read [submission promotion](references/submission-promotion.md).
8. For account access or installation, use the dedicated `$kaggle-account-access` project skill. CLI installation does not install the SDK in every project Python environment.

## CLI and API pitfalls

- Check installed `kaggle --help` and subcommand help before relying on remembered syntax. There is no general competition `describe` command in the inspected CLI.
- CLI episode JSON may have trailing prose. Prefer SDK objects or parse only the JSON document, not all stdout.
- Replay download `--path` is a directory. Logs download to a file; CLI stdout is not the log JSON.
- SDK submission fields use snake_case such as `public_score`. Identify our seat by exact submission ID, not a guessed seat or team-name substring.
- Count completed PUBLIC episodes only for ladder W/L/T; exclude validation and missing/nonfinite results. Sort explicitly for recent-window records and state API coverage limitations.
- A public HTTP 200 is not proof of authentication. An upload acknowledgement is not proof of successful validation. `PENDING` is queued; `COMPLETE` indicates validation finished.
- The inspected file loader chooses the last newly bound callable. End a packaged `main.py` with an explicit final `kaggle_submission_agent` wrapper and validate the exact loader behavior.
- Do not assume `len(env.steps) == episodeSteps + 1`; inspect terminal statuses and the pinned engine convention.

## Interpretation safeguards

Do not promise rating convergence after a fixed number of games, a rating ceiling, or a specific rating algorithm without checking published competition rules. Different opponent schedules prevent attributing rating differences solely to matchmaking or cash averages. Never publish internal generation markers, fitness scores, or exploit details in submission messages; choose neutral names.

## References

- [Evaluation fidelity and rule transfer](references/evaluation-fidelity.md)
- [Submission promotion](references/submission-promotion.md)

These instructions are a project-focused adaptation of the existing Kaggle simulation workflow. They intentionally omit account secrets, private infrastructure, previous-run status logs, and unsupported historical strategic assertions.

## Restored-agent runtime pitfalls

A Torch import or a weights-loading helper does not prove the policy uses a neural model. Inspect the actual entrypoint call graph before installing Torch or seeking weights. Remove only proven-unreachable scaffolding, preserve retained AST nodes, and verify full reciprocal-seat games. Never invent weights or use random initialization as a repair. Resolve recovered dependencies relative to `__file__`, and adapt one-argument upstream agents explicitly to evaluator `(obs, config)` calls. Keep each policy in a fresh process. Read `shinka/champions/RECOVERY_PROVENANCE.json` for pinned-source identity limitations; the Tamizh source is local-only and Git-ignored pending licensing review. Source repair hashes and runtime checks belong beside the current artifacts, not in old-run histories.
