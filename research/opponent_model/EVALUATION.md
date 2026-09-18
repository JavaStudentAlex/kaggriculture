# Opponent order-flow model: evaluation methodology and results

Living document. Update the results table every time a checkpoint finishes; keep
the methodology in sync with `evaluate.py` (the one scorer).

| | |
|---|---|
| Last updated | 2026-09-18 (refit) |
| Scorer | `research/opponent_model/evaluate.py` |
| Data | `datasets/shards/` — one shard per Kaggle day, `next_action` labels (AGENTS.md 4.2–4.3) |
| Results | section 4 — `models/ttm_c256_h96_ft_2026-09-17` (promoted 2026-09-18; earlier checkpoints kept for reference) |

## 1. What is being predicted, and why it is hard

- **Target.** `y[t, p]` = units of product `p` the opponent net-supplies to the
  market at turn `t`, recovered exactly from public inventory accounting
  (`extract.py`: `inv[t+1] - inv[t] + town_draw[t]`, explained by the orders
  Kaggle stores at replay index `t+1`), attributed to the opponent when only one
  seat requested that product and split pro-rata when both did (`contested`).
  Net buys (WHEAT/FERTILIZER only) are clipped to 0 by `log1p(clip(y, 0))`, so
  the target is *gross-ish supply*, not net flow.
- **Sparsity.** Sells land on ~28 % of steps but on a small fraction of
  (turn, product) cells; per product the base rate spans two orders of
  magnitude. Any metric that counts zeros correctly is uninformative: an
  all-zero predictor nearly wins MAE and has "accuracy" ≈ 98 %.
- **Strong clock structure.** Hour 0 of the game day (the town-centre draw and
  the end-of-day shed dump) carries 3–7× the sell rate of a quiet hour. Any
  evaluation whose rows are not spread evenly over the day measures a
  different, easier problem (section 2.2).
- **Labels are approximate on contested rows** (20–30 % of rows have at least
  one contested product from August on); those cells are pro-rata guesses.
- **Distribution shift.** The data is Kaggle's daily top-rated sample, so every
  number describes top-ladder opponents, and the ladder drifts from day to day —
  the reason for the daily refits (AGENTS.md 6).

## 2. Methodology (`evaluate.py`)

### 2.1 Which rows are scored

Only episodes the checkpoint never trained on. `train_ttm.py` splits by
**episode** (both seats of a game on the same side — seats share a market
trajectory and see each other's farm, so a per-seat split leaks) and writes the
held-out ids to `best/val_episodes.json`; `evaluate.py` reads that file, so the
shard directory may keep growing without changing a checkpoint's held-out set.
Without the file it rebuilds the seed-0 episode split over the shards it is
given, and then `--dataset` / `--max-episodes` must equal the training run's.
`--include-leaky` also scores the un-cleaned split, so a seat leak can never pass
unnoticed.

### 2.2 The window grid

Windows are `context C | horizon 96` slices of a 719-turn series (C = 256 for
the current line: 360 origins per series, game turns 256–615, days 11–26). The
origin hour is `(start + C) mod 24`, so a stride sharing a factor with 24 pins
every origin onto a few hours; measured on an unchanged checkpoint that moved AP
by 2× (0.84 at stride 32, 0.42 at stride 1). The training stride has the same
arithmetic (stride 8 = hours 0/8/16 only), which is why training uses stride 17.
Rules enforced by the scorer:

1. `gcd(stride, turnsPerDay) == 1` or it exits.
2. Each series' window list is trimmed to a whole number of days, so every hour
   gets exactly equal counts (`grid.origin_hour_counts` in the JSON proves it).
   Default stride 1 = 15 origins per hour per series.
3. The grid (stride, balanced flag, origin range, hour counts) is written into
   every report, so a number can never be quoted without it.

### 2.3 What is measured

| block | metric | why |
|---|---|---|
| **detection** | AUC, AP, AP-lift (= AP / base rate) per forecast step, pooled per game day, pooled over all 96 steps | AP tracks precision at the operating point a policy would use; AUC saturates. Lift makes slices with different base rates comparable. |
| **baselines** | the same for recency (4, 24 turns), cumulative volume, turns since last sell, opponent ripe stock, price/base, and a clock-only hour × product base-rate table fitted on train | how much of the score is free |
| **magnitude** | MAE in units vs all-zero; units predicted vs actual *conditional on a sell* (shrinkage); per-day total-volume correlation and bias | far weaker than detection and kept apart so it cannot hide inside a blended number |
| **calibration** | realised sell rate and mean units per prediction decile; share of next-turn volume in the top decile | what a threshold on the score actually buys |
| **per product** | rate, AUC, AP, lift, units actual vs predicted | products differ 100× in base rate |
| **behaviour** | rate and mean prediction by price/base, by opponent ripe units, by target hour; cold-start vs warm | does the firing pattern follow the mechanics or only the recency features |
| **leak gap** (`--include-leaky`) | clean vs un-cleaned val at t+1 | keeps the seat-leak size visible |

### 2.4 Limits of the metrics

- Per-product AP for the rare products rests on tens of positives; treat
  anything with `n_pos < 200` as indicative only.
- Windows from one series overlap (stride 1 → 360 near-duplicate contexts), so
  the effective sample size is closer to the number of series than of windows.
  No confidence intervals yet; a per-series bootstrap is the right tool when two
  checkpoints are within a few AP points.
- The trainer's `eval_auc_any_sell` / `eval_ap_any_sell` (eval stride 5, all 96
  steps pooled) are for model selection only; quote headline numbers from
  `evaluate.py`.
- Against the *true* in-game supply the right check is
  `shinka/evolution/check_oracle.py truth --held-out` (inputs rebuilt as the live
  agent rebuilds them, origins at every hour) — that is the number that matters
  for play.

## 3. How to evaluate a checkpoint

Run from the repo root in the training venv (`ops/train.sh` does this at the end
of every base run; GPU optional):

```bash
/results/kagg/venv-cuda/bin/python research/opponent_model/evaluate.py \
    --checkpoint research/opponent_model/runs/<run>/best \
    --dataset   datasets/shards --stride 1 --include-leaky \
    --out research/opponent_model/runs/<run>/eval.json
```

Checklist:

1. The printed `split=episode from .../val_episodes.json` line must appear; if
   it says `rebuilt`, the checkpoint predates 2026-09-12 and `--dataset` /
   `--max-episodes` must match its training run.
2. Keep `--stride 1` (or any coprime stride that still yields ≥ 24 origins per
   series). The script refuses non-coprime strides and warns if it cannot
   balance hours.
3. Paste into section 4: the per-day and pooled detection rows, the best
   baseline, the magnitude shrinkage line, top-decile recall, per-product AP.
   Keep the JSON next to the checkpoint.
4. Quote nothing from the trainer's `scores.json` here.

Dry run (no model, any numpy env) to confirm the split and grid before spending
GPU time:

```bash
.venv/bin/python research/opponent_model/evaluate.py --checkpoint <dir> \
    --dataset datasets/shards --shard-limit 2 --dry-run
```

## 4. Results

| checkpoint | held-out | t+1 AUC / AP / lift | day 1–4 pooled AP | best baseline AP | shrinkage when sold | top-decile volume recall |
|---|---|---|---|---|---|---|
| `models/ttm_c256_h96` (2026-09-14; `eval.json` in the dir, 2.22 M windows, stride 1) | 3,083 held-out episodes of 44 days | 0.873 / 0.403 / 9.6× (day 1 pooled) | 0.403 / 0.401 / 0.389 / 0.368 | clock hour × product 0.151 (3.6×); best history feature 0.074 | 2–20× per product at t+1; late-game WHEAT dumps over-predicted (see README) | 0.704 |

| `models/ttm_c256_h96_ft_2026-09-13` (2026-09-14; `eval.json` in the dir, 46,800 windows, stride 1) | 65 held-out episodes of 09-13 (the refit's own val split) | 0.885 / 0.425 / 9.2× | 0.407 / 0.397 / 0.376 / 0.347 | clock hour × product 0.141 (3.0×); best history feature 0.075 | ~7× pooled (6.1 u actual vs 0.86 predicted) | 0.729 |
| `models/ttm_c256_h96_ft_2026-09-15` (2026-09-16; `scores.json` in the dir, 923,520 windows, dual-GPU) | held-out episodes of 09-15 (the refit's own val split) | pooled AUC 0.849 / AP 0.346 / 5.5× | 0.346 pooled | baseline rate 0.062 (AP 5.5× lift) | mae 4.32 when sold vs 4.89 all-zero | — |
| `models/ttm_c256_h96_ft_2026-09-16` (2026-09-17; `scores.json` in the dir, 895,104 windows, dual-GPU) | held-out episodes of 09-16 (the refit's own val split) | pooled AUC 0.854 / AP 0.350 / 6.1× | 0.350 pooled | baseline rate 0.058 (AP 6.1× lift) | mae 4.74 when sold vs 5.34 all-zero | — |
| `models/ttm_c256_h96_ft_2026-09-17` (2026-09-18; `scores.json` in the dir, 923,520 windows, dual-GPU) | held-out episodes of 09-17 (the refit's own val split) | pooled AUC 0.848 / AP 0.362 / 5.6× | 0.362 pooled | baseline rate 0.064 (AP 5.6× lift) | mae 4.22 when sold vs 4.82 all-zero | — |

`ttm_c256_h96` per product at t+1 (AUC / AP): WHEAT 0.847 / 0.475, CARROT 0.951 / 0.211, TOMATO 0.956 / 0.494,
STRAWBERRY 0.853 / 0.434, MELON 0.857 / 0.214, EGG 0.940 / 0.602, MILK 0.808 / 0.388,
WOOL 0.777 / 0.246, FERTILIZER 0.807 / 0.428. Leak check: clean AP 0.400 = un-cleaned AP 0.400.

`ttm_c256_h96_ft_2026-09-13` per product at t+1 (AUC / AP): WHEAT 0.829 / 0.373, CARROT 0.871 / 0.339,
TOMATO 0.940 / 0.501, STRAWBERRY 0.865 / 0.438, MELON 0.869 / 0.361, EGG 0.913 / 0.562, MILK 0.829 / 0.389,
WOOL 0.859 / 0.315, FERTILIZER 0.844 / 0.500. Leak check: clean AP 0.425 = un-cleaned AP 0.425. The two rows
are on different held-out sets (44 days vs one day) and are not a comparison between the checkpoints; the
like-for-like number is the refit's epoch 0 (0.859 / 0.357) vs its best (0.878 / 0.381) on the same windows.
