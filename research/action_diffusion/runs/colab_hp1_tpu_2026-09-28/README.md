# KAD-HP-1 on a Colab TPU v5e-1 (account colab4), 2026-09-28 18:52 to 2026-09-29 06:52 UTC

Continuation of the Kaggle GPU run in `../kaggle_hp1_2026-09-28/` (resumed from its `last.pt` at step 150,000).
Driven by `research/procedural_graph/arena/colab_train.py` (runner on cliproxyapi, tmux `kagg-colab-kadtpu`,
`~/kagg-colab/runs/kad_hp1_colab/`). 59 days (2026-07-30..09-27; 09-26 is not downloadable), validation on 09-25's
held-out games, batch 64 on one chip (~320 turns/s), peak lr 2e-4, cosine over a 28.9 h schedule.

## How it ended
Colab deleted the VM at 06:52:40 UTC, 12 h 00 min 33 s after it was created (18:52:07). The keep-alive pings
were answered 200 every minute until 06:51:40 and 404 at 06:52:40, and the runner polled every minute, so this
is not a missed ping or an idle timeout: it matches Colab's 12-hour VM lifetime. The runner ended with
`COLAB_TRAIN_VM_LOST`, `COLAB_VMS_LEFT=0`, `COLAB_TRAIN_EXIT=3` (see `runner.log`). colab4: 81.74 units at the
start, 46.88 after (about 35 used).

The runner pulls `last.pt` hourly (last pull 05:57 UTC) and `scores.json` every 10 min. Everything the VM did
after 05:57 is lost: steps 340k to about 356k, including the step-350k checkpoint.

## Files
| file | what |
|---|---|
| `best.pt` | **step 180,000**, weights only. Lowest weighted validation loss of the whole run: 3.2187 (unweighted 2.807), first-command accuracy 72.3%. **Use this one.** |
| `last.pt` | step 340,000 (trained_s 63,618), model + optimizer + scaler, saved at that step's evaluation: weighted val loss 3.720, first-command 66.8%. **Post-divergence, worse than `best.pt`**; a resume point only. |
| `metrics.jsonl.gz` | every train event (every 20 steps) and validation event, steps 0 to 341,660 (0 to 150k are the GPU run's) |
| `scores.json` | the last evaluation seen, step 350,000: weighted val loss 4.195, first-command 65.9%, `complete: false` |
| `launch.log.gz` | the VM's log up to 05:57 UTC |
| `runner.log` | the runner's log to the end |
| `config.json`, `vm_started` | run config; VM creation time (epoch seconds) |

## The divergence
At step 305,480 (03:56 UTC 09-29; lr 8.6e-5, gradient clipping at 1.0 on, no NaN, nothing in the VM log) the
training loss went from 1.4 to about 16 within 100 steps and the model lost nearly everything it had learned.
Cause unknown.

| step | weighted val loss | first-command accuracy |
|---|---|---|
| 145k (GPU run's best) | 3.281 | 73.2% |
| 180k (`best.pt`) | 3.219 | 72.3% |
| 290k | 3.516 | 73.8% |
| 300k (last evaluation before the spike) | 3.484 | 73.5% |
| 310k | 16.62 | 7.0% |
| 320k | 5.98 | 39.5% |
| 330k | 6.25 | 42.3% |
| 340k (`last.pt`) | 3.72 | 66.8% |
| 350k (`scores.json`) | 4.19 | 65.9% |

Before the spike, first-command accuracy had been flat at 73-74% since step ~150k while the validation loss
slowly fell. The pre-spike `last.pt` (step ~305k) was overwritten by the hourly pull at 04:57 UTC, so there is no
rollback point between `best.pt` (180k) and the divergence.
