# Predictor-required Aspen evolution

Authorized by the user on 2026-09-29: incorporate the predictor into the current
Aspen line and evolve tactics using the loss evidence. This is a separate,
bounded branch of the existing edit/SIFT/paired-gauntlet process on cliproxyapi.
It shares the existing Colab pool and does not create or own additional VMs.

**Launched 07:50 UTC 09-29, replacing ladder1** (user: "stop the previous mutation and start a new one
absorbing the best from the previous one but using predictor"). ladder1 was stopped at 07:45 UTC at its
iteration 93, abandoned; its checkpoint stays at next iteration 93. Before the launch the seed was rebuilt on
ladder1's newest Aspen-line champion (below), the Crops queue entry that the seed now contains was dropped,
the budget was raised to 40 iterations (`ITERATIONS=` in `launch.sh`), and 110 tests passed on the final
code on cliproxyapi. Seed bundle `g_2e6e1664c98cc2a9` (engine settings, guard, `require_oracle` and
model sha `3c94bfc5d5c8` checked inside the bundle); first request: 763 baseline games.

## Seed and experiment

- Parent: ladder1's `Island-Next-Counter` champion at its next iteration 93
  (bundle `g_99508e7525447d95`); `parent_graph.json` preserves it exactly (checked
  equal to the ladder1 checkpoint's graph; `parent.json` has the canonical hash).
  It is Aspen's 09-27 engine with the emulator, `_SR_MARGIN=12`, `_S809_LOOK=4` and
  `_CA_MARGIN=-22`, crowned at ladder1 iteration 92: 304 of 467 changed games better,
  +$43 a game, p=6.7e-11, results +22/-12.
- `seed_graph.json` absorbs the best of ladder1's other 09-27 island, Next-Market:
  `_SR_MARGIN=14` (its iteration 71: 133 better / 42 worse, +$6.3 a game) and the
  oracle guard, here enabled for every rival family, on milk, wool and
  strawberries (score 0.5, strong score 0.6, batch 4, keep 2, price floor 0.75).
  Next-Market's guard played with the guard off against the two public-engine
  families and still gained results +15/-1 (its iteration 87).
  Predictor: reference `ttm_c256_h96_ft_2026-09-13`, SHA-256 `3c94bfc5d5c8…`.
  No calibration. `deployment.json` records the first (pre-rebuild) seed's hashes,
  bundle name and evaluation fingerprint.
- The runtime collects observations from turn 0 and predicts from turn 256.
  `require_oracle=true` rejects a missing/failed model and missing forecasts after
  warmup instead of silently falling back. An invalid game cannot be promoted.
- Six islands: **Crops, Oracle, Tactics, Tomatoes, Herd, Endgame**. The carrot
  margin -22 is in the seed; the first Oracle queue item adds egg/carrot to its
  products; Tomatoes tests the 7,000 investment gate (ladder1 iteration 88 rejected
  it without the predictor: +$87 a game, p=0.19, results +16/-16). Other
  iterations use the existing model bandit and SIFT judge to propose code/parameter
  edits. Tactics receive `info['forecast']` and emulator information.
- Limit: **40 iterations** (first planned as 12), three model proposals when no queue entry
  applies, 30% first-stage screening, 20 head-to-head games, mixing and supervisor
  every six iterations, proposal prefetch enabled. Ordinary gauntlet promotion
  remains unchanged; oracle-on is a requirement, not a claim that this seed has
  already beaten the parent on the full development schedule.
- `plan.json`: **763 development jobs**, the existing 727 ladder1 jobs plus
  **36 recorded Aspen wins** to penalize regressions while addressing its losses.
  Opponent bundles are copied into this run and frozen. These cases are not an
  untouched final test set. Final selection still needs new seeds and reactive
  opponents, with errors and reciprocal-seat results reported separately.

## Required-oracle contract

The run flag is persisted in `checkpoint.json`, so resuming without repeating
`--require_oracle` cannot remove it. The seed, queued edits, model proposals and
mixing candidates must keep `oracle_guard` enabled and nonempty known products
for every active rival-family counter. Zero batch/order limits, impossible shed
reserves and wholly unusable time windows are refused before scheduling games.
The bounded search allows `_OG_SCORE` in [0.05, 1] and `_OG_PRICE_RATIO` in [0, 1.5],
including family overrides, to reject enormous-threshold opt-outs.

The requirement is **working, available predictions**, not compulsory selling.
A tactic may veto a guard sale for a concrete economic reason. A correct sale
forecast may still favor waiting for town consumption or protecting inputs.
The mutation guidance says to preserve production and deliberate empty order
slots, use executed outcomes, and treat scores as ranking signals, not calibrated
probabilities or exact quantities. Strong-score sales are not capped at four.

## Evidence given to the mutation models

`evolution_ideas_ladder.md` and `evolution_knowledge_ladder.md` contain the Aspen
loss audit and completed tests checked at 06:06 UTC on 09-29:

- The reference guard changed the 67-game replay result from 36W-31L to 38W-29L,
  and proxy lost-seed games from 51W-11L to 53W-9L.
- The 180-game fresh pool comparison changed 157W-13L-10T to 165W-10L-5T.
- The tomato gate at 7,000 improved both sets; 6,000 worsened both. The latter
  is not queued. The zero-to-ten investment gate cannot solve Ichika's
  ten-versus-eighteen tomato capacity gap; a route change must own watering,
  harvesting, labour and storage, over an eight-day first-yield period.
- Carrots, sale-order timing against tuned engine copies, and geese/feed
  economics are separate hypotheses. Gains from individually tested changes
  must be rechecked when combined with the predictor.
- The stepwise +$5,983/six-flip estimate is not six replayed wins, and its
  full-order-list bug is documented. The old `guard26` experiment reused the
  reference bundle, so its matching result is not a September 26 model test.

The original ladder1 loop also received these hot-read guidance files. Its
Next-Market queue now explicitly removes the two `_OG_ITEMS: []` overrides
using null resets; merely omitting them from the edit would leave an existing
champion's suppressions intact. Its process, checkpoint, runtime and live pool
were not restarted or overwritten.

## Deployment and monitoring

Remote code: `/home/alex/kagg-evo/oracle-20260929/repo`.
Run state: `/home/alex/kagg-evo/oracle-20260929/run`.
Pool: `/home/alex/kagg-evo/pool` (the original pool owner retains lifecycle control).
Launcher: this directory's `launch.sh`, in tmux **`kagg-evo-oracle`**.

```sh
ssh cliproxyapi 'tail -n 25 /home/alex/kagg-evo/oracle-20260929/run/evolution.log'
ssh cliproxyapi 'cat /home/alex/kagg-evo/oracle-20260929/run/status.json'
```

`status.json` is written after an iteration; while the first baseline runs, read
`evolution.log`, `checkpoint.json` and the pool log. The initial seed must obtain
its full baseline before its first candidate can be compared. Sharing the pool
means a request can wait behind the existing evolution's current batch.

The new code fixes bundle identity using effective graph/runtime/checkpoint/
calibration bytes, verifies legacy content before reuse, and publishes from
unique temporary directories. This branch uses isolated result files and frozen
opponents; no old reference results are relabelled as a refit. The fix is deployed
in this branch's code snapshot; the running ladder1 code is unchanged.

`launch.sh` does not touch the shared pool's STOP file on exit. It ends after the
bounded run and writes `ORACLE_EVOLUTION_EXIT`. It never submits to Kaggle.

## Verification

- 76 targeted/regression tests passed in the existing remote venv, including
  required-oracle rejection/resume/model-health cases, bundle identity/concurrent
  builders, graph evolution, surgical behavior, guard rules and emulation.
- The seed's ordinary 30-turn graph validation completed with zero fallbacks.
- `smoke.sh` runs one full 720-state game per seat against the exact 09-27 engine,
  with the rival in its own RPC process. `smoke_oracle.py` asserts 463 forecasts,
  zero oracle errors/fallbacks, more than 600 emulator predictions and no wrong
  predictions. It records guard changes, cash and tail latency; those two games
  are runtime checks, not win-rate evidence.
- Standard-library syntax compilation, shell syntax and whitespace checks are
  run locally. Ruff, mypy and bandit are absent from the available environments;
  no new tools/dependencies were installed for this task.

Full model-refit comparisons, final held-out promotion, and a new Kaggle
submission are outside this launch. The predictor's weights are fixed; the
mutation process evolves how the agent uses predictions and its production logic.
