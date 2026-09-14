# ttm_c256_h96 — base opponent sell model, context 256, corrected labels

TinyTimeMixer (`tsfm_public`), 96-step (4 game days) supervision, **context 256 turns**
(8 patches of 32) so the in-game oracle forecasts from day 10 h16 instead of day 21.
Trained by `research/opponent_model/ops/train.sh` on the 44 corrected-label days
2026-07-30 .. 2026-09-11 of `datasets/shards` (`labels.json`: next_action), episode
split seed 0 with 10 % of episodes held out (`val_episodes.json`, 3,083 episodes),
warm start from `models/ttm_v3_h96_ft_2026-09-11` (its scaler reused unchanged, only
the patcher re-initialised), 3×A100, batch 64/GPU, window stride 17, lr 1e-4 with
reduce-on-plateau (×0.5, patience 2; halved once after epoch 32), early-stop patience 6.
Started 2026-09-12 18:36 UTC, resumed three times (renames, the streaming-metric fix,
a pod restart at epoch 33) and finished 2026-09-14 01:21 UTC at the 40-epoch cap —
**best = epoch 40**, still improving by ~0.0005 AUC per epoch at the end.

| eval | AUC | AP |
|---|---|---|
| trainer val, all 96 steps pooled (`scores.json`, eval stride 5) | 0.8668 | 0.392 |
| canonical (`eval.json`: 2.22 M windows, stride 1, hour-balanced, held-out episodes only), pooled | 0.8669 | 0.390 (lift 8.7×) |
| canonical day 1 / 2 / 3 / 4 | 0.873 / 0.872 / 0.868 / 0.855 | 0.403 / 0.401 / 0.389 / 0.368 |
| canonical t+1 per product (AUC) | WHEAT 0.85, CARROT 0.95, TOMATO 0.96, STRAWBERRY 0.85, MELON 0.86, EGG 0.94, MILK 0.81, WOOL 0.78, FERTILIZER 0.81 | |

Held-out curve: 0.63 (epoch 0, warm start) → 0.82 (1) → 0.85 (10) → 0.86 (20) →
0.8633 (30) → 0.8668 (40). No seat leak (clean AP = un-cleaned AP 0.400 at t+1).
Not comparable with the legacy-label numbers of the 512 line (different labels and
origins); the apples-to-apples test is `check_oracle.py padding` / `truth --held-out`
against the true supply (AGENTS.md 12).

**Magnitudes are for ranking only.** At t+1 predicted units when a sell happens are
shrunk 2–20× per product (e.g. WHEAT 9.5 → 1.9). A handful of late-game WHEAT cells
(horizon ≥ 69 from origins ≥ 556, true dumps of up to 1,500 units) are over-predicted
by ~4× in log1p space, which explodes in unit space (pooled `units_when_sold_pred`
842 vs actual 6.3; day-3/4 volume totals off by 300×). Use the scores, not the units,
at long horizons.

Load with `TinyTimeMixerForPrediction.from_pretrained("models/ttm_c256_h96")` plus
`scaler.npz` (input mean/std, `(x - mean) / std`, targets as `log1p` channels 141–149;
`past_values` of shape `(B, 256, 150)`). `labels.json` tells the oracle which
accounting to rebuild (next_action).
