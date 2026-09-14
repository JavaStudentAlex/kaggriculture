# Orchard Tide — submission bundle (run-2 gen 33 with the oracle, CPU-only)

Public name **Orchard Tide** (`../OrchardTide.tar.gz`). Internally: the best
candidate of the 2026-09-12 Shinka runs, run-2 gen 33 `dynamic_shop_batch_sizing`,
crowned as `champions/pool/champ_20260912_161630_avg82113.py`
(`../../top/run2_2026-09-12/` has its metrics and the tie-break). Packaged for
Kaggle's sandbox: no GPU, no repository, 1000 ms per step, ~60 s for the first
request. `SUBMISSION_MANIFEST.json` records hashes, the generation-to-name
mapping, every attempt and the validation.

```
main.py              bootstrap (the only new code): locates the bundle, sets the env overrides
                     the champion honours, imports it, ends with kaggle_submission_agent()
champion_gen33.py    the champion, byte-identical to shinka_results gen_33/main.py (sha256 66ad4552…)
kagg_oracle.py       byte-identical to shinka/evolution/kagg_oracle.py
kagg_ttm_numpy.py    byte-identical to shinka/evolution/kagg_ttm_numpy.py: the checkpoint's
                     network in numpy (main.py sets KAGG_ORACLE_BACKEND=numpy)
checkpoint/          models/ttm_v3_h96_ft_2026-09-11: model.safetensors (4 MB), config, scaler, labels
opponent_model/      features.py + mechanics.py (research/opponent_model): the oracle's input builder
mohui_v66/           the Mohui v66 backbone closure (Apache-2.0; LICENSE, NOTICE inside)
validate.py          the check below (not in the archive); SUBMISSION_MANIFEST.json, this README
```

Only numpy is needed from Kaggle's image.

## What Kaggle's sandbox taught us (each item cost a failed run)

1. `kaggle_environments` `exec`s a submitted `main.py` in a bare namespace — **no
   `__file__`** — and takes the **last callable** left in it. The champion calls
   `Path(__file__)` at import, so `main.py` is a bootstrap that imports it as a
   module (restoring `__file__`) and ends with `kaggle_submission_agent`.
2. The loader appends the bundle directory to the END of `sys.path` (and pops it
   after the file runs); the bootstrap inserts its directory at the front
   unconditionally — a "not already present" guard left the bundle behind
   `sys.path[0]`.
3. Missing pieces do not crash the champion — it **silently plays oracle-less**
   (`ORACLE_STATS["errors"]`). `kagg_oracle` imports `features` / `mechanics` from
   `research/opponent_model`, which the champion's `Path.home()` fallback had been
   finding in the repository during local validation; hence `opponent_model/` in
   the bundle and an empty `HOME` in the validation.
4. **The first request has ~60 s and the sandbox is slow at importing.** Attempt 1
   (submission 56192942, 21:09 UTC, torch + transformers + vendored tsfm) ended with
   both seats **TIMEOUT at step 1** of the validation episode, agent logs empty:
   the exec of `main.py` (torch/transformers import + model load, 5.9 s on one
   core here) did not finish in time there. Calibration from an earlier accepted
   submission: Kaggle is ~2-2.5x slower than one core of this pod for the plain
   Python closure (import 4.9 s vs 2.5 s, per step 17.5 vs 6.9 ms); the shared-
   library-heavy torch import scales worse. The fix is `kagg_ttm_numpy.py`: the
   TinyTimeMixer forward pass in numpy (adaptive patching, mix-channel decoder,
   gated attention, std scaler — verified against the torch model: within 1.6e-5
   of the float64 truth where torch-f32 itself is 8.5e-4 off), ~120-150 ms per
   forecast on one core, import 2.2 s.

## Validation (2026-09-12, see SUBMISSION_MANIFEST.json)

Clean venv `/results/kagg/venv-kaggle-sim` (numpy 2.4.6, kaggle-environments 1.32.7;
torch present but never imported), the extracted archive at a scratch path,
`python -I` (no script dir, no PYTHONPATH, no user site), `HOME` = an empty
directory, every `KAGG_*` unset, cwd `/`, `main.py` passed to `kaggle_environments`
as a path string, one core, one BLAS thread:

* 8 games — seats 0 and 1 × starter/random × seeds 101, 70102 — all `DONE`, finite
  rewards, no traceback in the agent's stderr.
* Oracle live on the numpy backend: 1,656 forecasts, 0 errors, ~120 ms per
  forecast; `torch`/`transformers` absent from `sys.modules`; champion, backbone,
  oracle, numpy network, checkpoint, features, mechanics all resolved from inside
  the extracted directory.
* Timing (loader's own per-step measurement): step 0 = 2.2 s (closure import +
  weights), charged to the episode's overage bank (`remainingOverageTime` = 60 s
  in the spec: a step is fatal only when its excess over 1 s exceeds what is
  left); from turn 512 (forecast every turn) p99 137 ms, max 495 ms. At Kaggle's
  ~2.5x this is ~5 s + ~350 ms steps.
* Fidelity: vs `champ_20260912_gen62_avg82350.py` through the same loader the
  bundle scores $76,816 (seed 101) and $64,197 (seed 70102) — exactly what the
  repo champion scores on the GPU through the torch checkpoint.

Re-run:

```bash
rm -rf /tmp/x /tmp/h && mkdir -p /tmp/x /tmp/h && tar xzf OrchardTide.tar.gz -C /tmp/x
cd / && env -u KAGG_REPO -u KAGG_MOHUI_DIR -u KAGG_ORACLE_SRC -u KAGG_TTM_DIR -u KAGG_OPP_MODEL_SRC -u KAGG_ORACLE_BACKEND \
  HOME=/tmp/h taskset -c 1 /results/kagg/venv-kaggle-sim/bin/python -I \
  /home/jovyan/kaggriculture/shinka/champions/submissions/orchard_tide/validate.py /tmp/x 101,70102
```

## Outcome

Attempt 2 (submission **56193386**, 21:46 UTC) passed Kaggle's validation:
`COMPLETE` at 21:49 UTC, self-play episode 108314881 both seats `DONE`, 720
steps, $58,039 each. Kaggle's own per-step measurement of this build: step 0 =
5.0 s (4 s of the 60 s bank), before turn 512 ~17 ms, from turn 512 mean 321 ms,
p99 ~380 ms, max 404 ms — 2.5x one pod core, as calibrated; the jump at turn 512
is the forecast, i.e. the oracle is live there; stderr empty.

After a submission: `kaggle competitions submissions kaggriculture --csv` for the
status, `kaggle competitions episodes <id> --csv` for the validation episode,
`kaggle competitions logs <episode> <seat>` / `replay <episode>` for the agent's
stderr (look for `[kagg_oracle]` lines) and the terminal statuses.
