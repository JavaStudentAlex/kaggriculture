# ttm_v3_h96_ft_2026-09-10 — first daily fine-tune of v3

Base: `models/ttm_v3_h96` (weights + its `scaler.npz`, reused unchanged).
Data: Kaggle daily datasets 2026-09-06 .. 2026-09-10, recency-weighted — the newest
day windowed at stride 4, each day further back at double the stride (DECAY=2), so
09-10 is ~48 % of every epoch and every day back counts half as much. Validation:
10 % of 09-10's episodes held out (65 episodes, 2,990 windows, episode-disjoint).
Recipe: `research/opponent_model/ops/finetune.sh` with
`DAY=2026-09-10 PREV_DAYS=4 DECAY=2 STRIDE=4 GPUS=3 LR=6e-5` (torchrun, batch 64/GPU,
plateau ×0.5 patience 2, early stopping patience 5, `--metric-horizon all`).
Trained 2026-09-11 23:26–23:45 UTC; best epoch 24 of the 30-epoch cap.

| on 09-10's held-out episodes (all 96 horizons pooled) | AUC | AP |
|---|---|---|
| v3 untouched | 0.8115 | 0.1396 |
| **this model** | **0.8655** | **0.2124** |

Forward-in-time check (train through 09-10, validate on all of 09-11): see
`research/opponent_model/runs/ft_2026-09-10_p4_val2026-09-11_g3_d2_s4/` once it has run.
Load like v3: `TinyTimeMixerForPrediction.from_pretrained(<this dir>)` + `scaler.npz`.
