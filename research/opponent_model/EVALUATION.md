# Opponent order-flow model: evaluation methodology and results

Living document. Update the results tables every time a checkpoint finishes;
keep the methodology section in sync with `evaluate.py`.

| | |
|---|---|
| Last updated | 2026-09-11 |
| Latest checkpoints | `models/ttm_v3_h96` (96-step supervision, done 2026-09-11) and `models/ttm_v3_h96_ft_2026-09-10` (its recency-weighted daily fine-tune) -- see `AGENTS.md` §5-6; section 6 below is still to be filled with `evaluate.py` numbers |
| In flight | forward-in-time check of the daily fine-tune (train through 09-10, validate on all of 09-11), tmux `kagg-ft_val0911` |
| Canonical scorer | `research/opponent_model/evaluate.py` (the ad-hoc trio `eval_clean.py` / `analyze_v2.py` / `horizon_probe.py` it replaced has been deleted; it now uses a checkpoint's own `scaler.npz` when present) |

## 1. TL;DR

The model predicts, for each of the 9 products and each of the next `H` turns,
how many units the opponent will sell. Its value is almost entirely in
**ranking** (which product-turns will carry a sell), not in **sizing** (how
many units).

Honest headline for `ttm_v2` -- episode-disjoint validation, every hour of the
day represented equally, next-turn (t+1) detection:

| | AUC | AP | AP lift over base rate (1.9%) |
|---|---|---|---|
| **model (ttm_v2)** | **0.901** | **0.423** | **22x** |
| best hand-written baseline (turns since last sell) | 0.857 | 0.127 | 6.7x |
| cumulative opponent volume | 0.843 | 0.117 | 6.2x |
| price / base | 0.637 | 0.042 | 2.2x |

The top decile of model scores contains 82% of all units the opponent sells in
the next turn. Ranking quality is nearly flat across the 24 supervised steps
(AP 0.42 at t+1, 0.35 at t+24). Predicted *magnitudes* are shrunk 3-35x toward
zero depending on product and must not be used as unit counts without
recalibration (section 5.3).

The numbers previously circulated for v2 (**AUC 0.963 / AP 0.837** in
`runs/ttm_v2/scores.json` and `clean_scores.txt`) are **superseded**: they were
computed on a window grid aliased onto three hours of the day -- the same three
hours the model was trained on (section 4.2). Same checkpoint, same episodes,
honest grid: AP 0.42.

## 2. What is being predicted, and why it is hard

- **Target.** `y[t, p]` = units of product `p` the opponent net-supplies to the
  market at turn `t`, recovered exactly from public inventory accounting
  (`extract.py`: `inv[t+1] - inv[t] + town_draw[t]`), attributed to the
  opponent when only one seat requested that product and split pro-rata when
  both did (`contested`). Net buys (WHEAT/FERTILIZER only) are clipped to 0 by
  `log1p(clip(y, 0))`, so the target is *gross-ish supply*, not net flow.
- **Sparsity.** Over all turns, 1.2% of (turn, product) cells are nonzero; in
  the band the model is actually scored on (game turns 512-623, see 4.3) it is
  1.9%. Per product it ranges from 0.03% (CARROT, TOMATO) to 7.6% (WHEAT).
  Any metric that counts zeros correctly is uninformative: an all-zero
  predictor has MAE 0.25 units and "accuracy" 98%.
- **Strong clock structure.** Sell rate by hour of the game day (turns 512-623,
  six dataset_v2 shards spread over the date range):

  | hour | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|
  | rate % | **5.4** | 2.8 | 2.3 | 1.3 | 0.8 | 1.3 | 0.9 | 1.7 | 1.3 | 1.9 | 1.3 | 1.2 |
  | hour | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 |
  | rate % | 1.3 | 1.4 | 1.9 | 1.3 | 1.3 | 1.9 | 1.9 | 1.7 | 2.4 | **3.4** | 2.0 | 1.9 |

  Hour 0 is the town-centre draw and the end-of-day shed dump; it carries
  3-7x the rate of a quiet hour. Any evaluation whose rows are not spread
  evenly over the day measures a different, easier problem.
- **Labels are approximate on contested rows.** 2% of rows on 2026-07-30 but
  20-30% of rows from August onward have at least one product where both seats
  traded that turn; those cells are pro-rata guesses. Not yet an eval axis (see
  7).
- **Distribution shift across the ladder.** Daily positive rate moved from 0.46%
  (07-30) to 1.7% (08-20..27) to 1.2% (09-03). The current random split mixes
  dates, so reported numbers are interpolation, not forward generalisation
  (see 7).

## 3. Data and splits

- `dataset_v2`: 42 daily shards, 2026-07-30 .. 2026-09-09, 29,525 episodes,
  59,050 (episode, seat) series, every series exactly 719 turns (30 game days
  of 24 turns, minus the final step which has no successor). 141 leak-free
  features + 9 targets per turn.
- **Training split (both v2 and v3): `--split series`**, seed 0, 10% val:
  53,145 train / 5,905 val series. This is a shuffle of *series*, so the two
  seats of one episode usually land on different sides. (`ttm_dataset.split_series`'s
  docstring claims "episode-disjoint"; it is not -- `split_by_episode` is. Fix
  the docstring once v3 is done and the file is no longer being imported by a
  live job.)
- **Clean set = val series whose episode has no seat in train.** With a 10%
  series split that is ~10% of val episodes: **298 episodes / 596 series**.
  Every number in this document is from that set unless labelled *leaky*.
- Seat leak, measured: on the (aliased, stride-32) grid the un-cleaned val gave
  AUC 0.973 / AP 0.831 vs clean AUC 0.963 / AP 0.837. The leak is small next to
  the grid effect, but the clean set is 10x smaller than it needs to be. Next
  training run should use `--split episode` (already the `train_ttm.py`
  default; v2/v3 overrode it to warm-start from a series-split run), which
  makes all of val clean.

## 4. Methodology

### 4.1 What is measured

All on the clean set, all from one run of `evaluate.py`:

| block | metric | why |
|---|---|---|
| **detection** | AUC, AP, AP-lift (= AP / base rate) per forecast step, pooled per game day, pooled over the first 24 steps, pooled over all emitted steps | AP is the number that tracks precision at the operating point a policy would use; AUC saturates. Lift makes slices with different base rates comparable. |
| **baselines** | same AUC/AP/lift for recency (4, 24 turns), cumulative volume, turns since last sell, opponent ripe stock, price/base, and a *clock-only* hour x product base-rate table fitted on train | tells you how much of the score is free. The clock baseline is new: it bounds what an aliased grid can hand out. |
| **magnitude** | MAE in units vs all-zero; units predicted vs actual *conditional on a sell*; the ratio (shrinkage); per-day total-volume correlation and bias | separated from detection because it is far weaker and would otherwise hide inside a blended number |
| **calibration** | realised sell rate and mean units per prediction decile; share of next-turn volume in the top decile | what a threshold on the score actually buys |
| **per product** | rate, AUC, AP, lift, units actual vs predicted | products differ 200x in base rate |
| **behaviour** | rate and mean prediction bucketed by price/base, by opponent ripe units, by target hour; cold-start (product quiet for 24 turns) vs warm | does the firing pattern follow the mechanics or only the recency features |
| **leak gap** (`--include-leaky`) | clean vs un-cleaned val at t+1 | keeps the seat-leak size visible |

### 4.2 The window grid, and why it decides the number

Windows are `context 512 | horizon 96` slices of a 719-turn series, so only
112 forecast origins exist per series (turns 512..623) and the origin's hour
of day is `(start + 512) mod 24 = (start + 8) mod 24`. Any stride that shares
a factor with 24 pins the origins to a few hours:

| eval stride | origin hours hit | windows (clean) | AUC | AP | source |
|---|---|---|---|---|---|
| 32 | {0, 8, 16} | 2,384 | 0.963 | **0.837** | `runs/ttm_v2/clean_scores.txt` (28-37 is the only stride range consistent with its 4 windows/series; the session used 32) |
| 4 | {0, 4, 8, 12, 16, 20} | 16,688 | 0.943 | **0.684** | `runs/insights.json` (`analyze_v2.py` default) |
| 1 | all 24 | 66,752 | 0.901 | **0.423** | `runs/insights_stride1.json` |

Same checkpoint, same 298 episodes, same t+1 target. The baselines barely move
between these grids (turns-since-last-sell AP 0.127 -> 0.127, cumulative
0.110 -> 0.117), so this is not a "mixture of easy hours" artifact of AP: **the
model itself is much better at hours 0, 8 and 16**. The reason is the
*training* stride: `--window-stride 8` gives 14 origins per series at
`start in {0, 8, ..., 104}` (53,145 x 14 = 744,030, exactly the trainer's
window count), all at hours `(start + 8) mod 24 in {0, 8, 16}`. v2 and v3 were
trained on 3 of the 24 phases the policy will query; stride-32 eval happens to
test exactly those 3.

Rules now enforced by `evaluate.py`:

1. `gcd(stride, turnsPerDay) == 1` or the script exits.
2. With a coprime stride the origin hour cycles with period 24, so each
   series' window list is trimmed to a whole number of days -> exactly equal
   counts per hour (`grid.origin_hour_counts` in the JSON proves it). Default
   stride 1 gives 96 origins per series, 4 per hour.
3. The grid (stride, balanced flag, origin step range, hour counts) is written
   into every report so a number can never be quoted without it.

Status of the other scripts' defaults: `train_ttm.py --eval-stride 5`,
`eval_clean.py --stride 5`, `horizon_probe.py --stride 5` are coprime (23
origins, 23 of 24 hours -- acceptable, slightly uneven). `analyze_v2.py
--stride 4` is aliased and should not be used for headline numbers.
`train_ttm.py --window-stride 8` (training) is the real defect -- see 7.

### 4.3 What the grid cannot cover

Because the context is 512 of 719 turns, **every train and eval window's
forecast origin lies in game turns 512-623 (days 22-26 of 30)**. Nothing in
this document says anything about the model's behaviour in the first 21 days
of a game, which is where a policy would use it most. Fix is a shorter context
or left-padding (7).

### 4.4 Horizon semantics: v2 vs v3 are not the same metric

- `ttm_v2`: head emits 96 steps, but `prediction_filter_length=24` masked the
  loss to the first 24, and model selection used `--metric-horizon 0` (t+1
  only). Its checkpoint emits 24 steps unless unmasked.
- `ttm_v3_h96`: loss on all 96 steps, model selection on
  `--metric-horizon all` (AUC pooled over 96 x 9 cells, eval stride 5).

The trainer's `scores.json` for the two runs are therefore different metrics on
different grids and **must not be compared**. Compare them only through
`evaluate.py`, which always reports the *first-24-steps pooled* block (both
models supervised it) and the *per-day* blocks (v2's days 2-4 only when run
with `--unmask-horizon`, and then labelled as unsupervised).

### 4.5 Known limits of the metrics themselves

- AP on 66k windows x 9 products is computed on ~600k cells with ~11k
  positives; per-product AP for CARROT/TOMATO/EGG rests on 21-50 positives and
  is noise-level. Treat anything with `n_pos < 200` as indicative only.
- Windows from one series overlap (stride 1 -> 96 near-duplicate contexts), so
  effective sample size for confidence intervals is closer to the number of
  series (596) than the number of windows. No CIs are reported yet; the
  per-series bootstrap is the right tool when two checkpoints are within a few
  AP points (7).
- `train_ttm.evaluate()` labels its mean baseline `mae_train_mean` but computes
  the mean on the *val* rows; harmless, but do not quote it as a baseline.
- All reported numbers are unweighted over episodes; the data is Kaggle's
  top-rated daily sample, so they describe top-ladder opponents only (7).

## 5. Results: `ttm_v2`

Checkpoint `runs/ttm_v2/best` (1.03M params, TTM-r1 warm-started from the
stopped 14-epoch long run, series split, 24-step supervision, trained
2026-09-11 00:35-04:04 on 3 GPUs). All numbers: clean set, 298 episodes, 596
series, stride 1, 66,752 windows, from `runs/insights_stride1.json`
(`analyze_v2.py --stride 1`). Base rate at t+1: 1.90%.

### 5.1 Detection vs baselines (t+1)

| predictor | AUC | AP | lift |
|---|---|---|---|
| **ttm_v2** | **0.901** | **0.423** | **22.2x** |
| turns since last sell (negated) | 0.857 | 0.127 | 6.7x |
| cumulative opponent volume | 0.843 | 0.117 | 6.2x |
| sold in last 24 turns | 0.673 | 0.064 | 3.4x |
| sold in last 4 turns | 0.574 | 0.050 | 2.6x |
| price / base | 0.637 | 0.042 | 2.2x |
| opponent ripe units | 0.663 | 0.038 | 2.0x |

AUC alone would say "0.90 vs 0.86, modest"; AP says the model triples the
precision of the best single feature. That is the number to track.

Cold start vs warm (t+1): on product-turns where the opponent has **not** sold
that product in the last 24 turns (81% of cells, base rate 0.93%) the model
reaches AP 0.315 (34x lift); on warm cells (base rate 6.0%) AP 0.497 (8.3x).
It is calling *new* sells, not only continuing a run.

### 5.2 Horizon

| step | t+1 | t+2 | t+3 | t+4 | t+6 | t+8 | t+12 | t+16 | t+20 | t+24 |
|---|---|---|---|---|---|---|---|---|---|---|
| AUC | 0.901 | 0.896 | 0.885 | 0.890 | 0.881 | 0.897 | 0.878 | 0.900 | 0.893 | 0.888 |
| AP | 0.423 | 0.411 | 0.389 | 0.379 | 0.346 | 0.369 | 0.343 | 0.372 | 0.355 | 0.346 |

Ranking quality decays only from t+1 to ~t+5 and is then flat through the end
of day 1. The model is estimating a per-product *propensity* over the coming
day more than the exact turn. Days 2-4 of v2's head were never supervised;
they are not reported here (run `evaluate.py --unmask-horizon` if a
diagnostic is wanted, and label it as such).

### 5.3 Magnitude (the weak half)

Units predicted vs actual, conditional on a sell actually happening (t+1):

| product | base rate | AUC | AP | lift | units sold | units predicted | shrinkage |
|---|---|---|---|---|---|---|---|
| WHEAT | 7.56% | 0.890 | 0.611 | 8x | 9.1 | 2.5 | 3.6x |
| STRAWBERRY | 3.06% | 0.777 | 0.262 | 9x | 8.0 | 2.7 | 2.9x |
| FERTILIZER | 3.41% | 0.842 | 0.375 | 11x | 3.8 | 1.1 | 3.5x |
| MILK | 1.68% | 0.807 | 0.224 | 13x | 4.0 | 0.43 | 9x |
| WOOL | 0.96% | 0.783 | 0.120 | 12x | 4.1 | 0.25 | 17x |
| MELON | 0.31% | 0.871 | 0.521 | 170x | 4.6 | 1.2 | 4x |
| EGG | 0.07% | 0.969 | 0.419 | 560x | 5.2 | 0.40 | 13x |
| CARROT | 0.03% | 0.957 | 0.173 | 520x | 7.5 | 0.29 | 26x |
| TOMATO | 0.03% | 0.881 | 0.136 | 430x | 7.7 | 0.22 | 35x |

(EGG/CARROT/TOMATO: 21-50 positives, indicative only.)

The log1p+MSE loss on a 98%-zero target pulls every conditional mean toward
zero; shrinkage is worst where the product is rarest. Consequences:

- **Do not use raw outputs as unit forecasts.** Use them as a ranking score,
  or pass them through a calibration map fitted on clean val (isotonic on
  `expm1(pred)` -> `E[units | score]`, or a two-stage
  hurdle: P(sell) from the score, units-given-sell from a per-product table).
  This is a **follow-up**, not something to fold into the headline metric.
- Aggregate volume is more usable than per-cell volume: the next-turn top
  decile of scores holds **82.4%** of all units sold.

### 5.4 Calibration by decile (t+1, all products pooled)

| decile | 1 | 2-4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|
| realised sell rate | 0.61% | 0.02-0.05% | 0.19% | 0.37% | 0.59% | 1.13% | 2.03% | **14.0%** |
| realised mean units | 0.038 | ~0.001 | 0.010 | 0.017 | 0.025 | 0.055 | 0.086 | **1.09** |
| predicted mean units | -0.009 | ~0 | 0.000 | 0.003 | 0.008 | 0.015 | 0.029 | 0.54 |

Two things to notice. Decile 10 is the whole signal (14% rate against 1.9%
base). And decile 1 is **non-monotone**: the lowest-scored 10% of cells sell
more often than deciles 2-8. Those are cells where the model outputs a
negative log-count -- most likely the WHEAT/FERTILIZER buy-back regime, whose
labels are clipped to 0 -- so a policy thresholding on "score < 0 means quiet"
would be wrong there. Worth a look before deployment; cheap to confirm by
splitting decile 1 by product.

### 5.5 Behaviour probes (t+1)

Bucketed by feature value at the last observed turn; `p(sell)` is the realised
rate, `pred` the mean predicted units.

**Price / base** -- the response is U-shaped and the model reproduces it,
including the dip:

| price/base | <0.6 | 0.6-0.85 | 0.85-1.0 | 1.0-1.15 | 1.15-1.4 | >1.4 |
|---|---|---|---|---|---|---|
| p(sell) | 1.66% | 1.79% | 2.00% | **0.31%** | 0.50% | **4.38%** |
| pred units | 0.033 | 0.036 | 0.042 | 0.009 | 0.014 | 0.169 |

Sells cluster both at a glut (price well below base, i.e. someone just
dumped) and at scarcity (price >1.4x, the incentive to sell); the band just
above base is quiet. The model's prediction tracks the realised rate across
all six buckets with a roughly constant 2-4x magnitude shrinkage.

**Opponent ripe units** (public farm state) -- monotone and tracked:

| ripe units | 0 | 1-2 | 3-5 | 6-10 | 11-20 | >20 |
|---|---|---|---|---|---|---|
| p(sell) | 1.15% | 1.51% | 1.66% | 1.73% | 2.26% | **5.19%** |
| pred units | 0.028 | 0.030 | 0.051 | 0.042 | 0.057 | 0.206 |

**Hour of the target turn** (target = last observed hour + 1):

| target hour | 1-4 | 5-8 | 9-12 | 13-16 | 17-20 | 21-24 (incl. 0) |
|---|---|---|---|---|---|---|
| p(sell) | 2.03% | 1.24% | 1.37% | 1.56% | 2.04% | **3.16%** |
| pred units | 0.067 | 0.031 | 0.036 | 0.040 | 0.038 | 0.140 |

The model has the end-of-day / hour-0 dump. What this table cannot show --
and `evaluate.py`'s `by_target_hour` block does -- is AUC/AP *within* each
hour, which is where the trained-phase problem of 4.2 will appear as a 3-hour
comb.

## 6. Results: `ttm_v3_h96` -- canonical numbers PENDING

Training finished 2026-09-11 22:14 UTC (`ops/train_v3.sh`: warm start from v2
best, 96-step supervision, `--metric-horizon all`, eval stride 5, plateau LR,
patience 8; best epoch 19, early-stopped at 27). The checkpoint is
`models/ttm_v3_h96/` (with its `scaler.npz`). Numbers so far come from the
trainer and the now-deleted ad-hoc scripts, so they are **not** on this
document's grid: trainer val AUC 0.8695 / AP 0.296 pooled over 96 horizons
(leaky series split); clean t+0 AUC 0.912 / AP 0.449; day 1-4 AUC
0.892 / 0.891 / 0.866 / 0.830. Daily fine-tunes of it are summarised in
`AGENTS.md` §6.1.

To fill this section in on the canonical grid:

1. Run the recipe in section 8 on `models/ttm_v3_h96` (plain, and once
   with `--include-leaky`). Do **not** quote its `scores.json`: that number is
   AUC pooled over 96 steps at eval stride 5 on the leaky split, and v2's
   `scores.json` is t+1 at stride 32 -- neither matches this document.
2. Re-run the same command on `runs/ttm_v2/best` **with `--unmask-horizon`**
   so both checkpoints produce 4 per-day blocks on identical rows.
3. Fill the table:

   | block | v2 (24-step sup.) | v3 (96-step sup.) |
   |---|---|---|
   | t+1 AUC / AP / lift | 0.901 / 0.423 / 22x | |
   | day 1 pooled AP (first24) | | |
   | day 2 pooled AP | *(unsupervised)* | |
   | day 3 pooled AP | *(unsupervised)* | |
   | day 4 pooled AP | *(unsupervised)* | |
   | day-total volume corr, day 1 / 2 / 3 / 4 | | |
   | shrinkage when sold (all products) | | |
   | top-decile volume recall | | |
   | best baseline AP (t+1) | 0.127 | (should be identical rows -> identical) |

   The questions v3 exists to answer, in order: (a) did days 2-4 become
   usable (day-2..4 AP lift clearly above the clock baseline)? (b) did it cost
   anything at t+1 / day 1 compared with v2 on the same rows? (c) did
   magnitude shrinkage change with 4x more nonzero supervision per window?
4. If v3's day-1 numbers regress vs v2 by more than the seat-bootstrap noise
   (4.5), the 96-step MSE is over-weighting far horizons; a horizon-weighted
   loss is the obvious next run, not a longer one.
5. Whichever wins, the next training run should also fix the two training
   defects in section 7 (window stride, episode split); those are
   independent of the 24-vs-96 question.

## 7. Defects and follow-ups, ranked

1. **Training window stride 8 aliases training onto 3 of 24 phases** (4.2).
   Affects v2 and v3. Fix in the next run: `--window-stride 7` (or 5, 11,
   13), or a random per-epoch offset. Expected effect: honest-grid AP rises
   toward the aliased-grid figure; the ~0.42 -> ~0.84 gap is the upper bound
   on what is recoverable. Do not touch `train_ttm.py` while `kagg-train_v3`
   is running -- it imports from the repo path.
2. **Nothing is trained or scored before game turn 512** (4.3). Fix: a
   shorter context (the released TTM checkpoints we use start at 512; either
   pick a shorter-context variant from the TTM-r2 family or left-pad early
   windows with a mask), then re-run this document's tables at origins spread
   across the whole game.
3. **Series split leaks seats and shrinks the clean set 10x** (3). Fix:
   `--split episode` next run; `evaluate.py --split episode` then scores all
   of val. Also fix the `split_series` docstring.
4. **No forward-in-time holdout.** Shards are daily; hold out the last 5-7 days
   as val (`--shard-limit` plus an explicit date split in `load_shards`) to
   measure what the daily-refit pipeline will actually face. Report the random
   and the temporal number side by side.
5. **Magnitude calibration** (5.3). Fit isotonic / hurdle on clean val, report
   the calibrated shrinkage separately; never inside the detection headline.
6. **Contested rows** (2). Add `--uncontested-only` to `evaluate.py` (the
   `D[:, 3]` flag is already in the shards; it needs to be carried through
   `load_shards`) and report both.
7. **Confidence intervals.** Per-series bootstrap of AP (596 series is enough
   for a usable CI); needed before declaring any v2-vs-v3 difference under
   ~0.03 AP.
8. **Rating-band breakdown**: Kaggle's daily datasets carry only top-rated
   games, so per-band numbers need replays from elsewhere; deferred.
9. **Decile-1 non-monotonicity** (5.4): split by product; if it is the clipped
   buy-back regime, either keep net flow as a signed target or add a buy
   indicator channel.
10. Minor: `train_ttm.evaluate()` `mae_train_mean` is a val mean;
    `analyze_v2.py --stride 4` default is aliased; `eval_clean.py`,
    `analyze_v2.py` and `horizon_probe.py` hard-code the series split and
    `max_episodes=100000`, so they silently produce a *different* split (and
    scaler) if a run used `--split episode` or a smaller `--max-episodes`.
    `evaluate.py` takes both as arguments and prints the series count to diff
    against the training log.

## 8. How to evaluate the next checkpoint

Run from the repo root in the training venv (needs `tsfm_public`, `sklearn`;
GPU optional, CPU works for the 596-series clean set):

```bash
/results/kagg/venv-cuda/bin/python research/opponent_model/evaluate.py \
    --checkpoint research/opponent_model/runs/<run>/best \
    --dataset   research/opponent_model/dataset_v2 \
    --split series --max-episodes 100000 \
    --stride 1 --include-leaky \
    --out research/opponent_model/runs/<run>/eval.json
```

Checklist:

1. `--split` and `--max-episodes` must equal the training run's values;
   confirm the printed `series=... episodes=...` line matches the training
   log's, or the split (and the scaler) is not the one the model saw.
2. Keep `--stride 1` (or any coprime stride that still yields >= 24 origins
   per series, i.e. <= 4). The script refuses non-coprime strides and warns if
   it cannot balance hours.
3. For a 24-step-supervised checkpoint add `--unmask-horizon` only when you
   need days 2-4 for a side-by-side, and label them unsupervised.
4. Paste into this document: the `DETECTION` per-day and `pooled first24`
   rows, the `BASELINES` table, the `MAGNITUDE` shrinkage line, top-decile
   recall, and the per-product table. Keep the JSON next to the checkpoint.
5. Quote nothing from the trainer's `scores.json` or from
   `eval_clean.py`/`analyze_v2.py` output in this document.
6. Bump the header table (date, latest checkpoint), and move the previous
   checkpoint's section down rather than deleting it.

Dry run (no model, any numpy env) to confirm the split and grid before
spending GPU time:

```bash
.venv/bin/python research/opponent_model/evaluate.py --checkpoint x \
    --dataset research/opponent_model/dataset_v2 --shard-limit 2 --dry-run
```

## Appendix: where the numbers in this document come from

| file | what | grid | status |
|---|---|---|---|
| `runs/ttm_v2/scores.json` | trainer's final val, t+1, leaky split | stride 32 (aliased) | superseded |
| `runs/ttm_v2/clean_scores.txt` | `eval_clean.py` full vs clean, t+1 | stride 32 (aliased) | superseded; kept as the leak-gap measurement |
| `runs/insights.json` | `analyze_v2.py` clean, t+1..t+24 | stride 4 (aliased) | superseded |
| `runs/insights_stride1.json` | `analyze_v2.py --stride 1` clean | all hours, ~uniform (112 origins, 4-5 per hour) | **source of section 5** |
| `runs/ttm_v3_h96/*.txt` | tracebacks from a failed post-run | -- | no v3 result yet |
| `logs/train_v2.log`, `logs/train_v3.log` | window counts used to reconstruct the strides in 4.2 | | |
