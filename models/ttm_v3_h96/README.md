# ttm_v3_h96 — opponent sell model, best checkpoint

TinyTimeMixer (`tsfm_public`), 96-step supervision, warm-started from `runs/ttm_v2/best`.
Trained by `research/opponent_model/ops/train_v3.sh` on `dataset_v2` (42 daily shards,
2026-07-30..09-09, 29,525 episodes), series split seed 0, 3×A100, lr 1e-4 with
reduce-on-plateau (×0.5, patience 2), early stopping patience 8 on `eval_auc_any_sell`.
Finished 2026-09-11 22:20 UTC: best epoch 19 (`checkpoint-73644`), stopped at epoch 27.

| eval | AUC | AP |
|---|---|---|
| training val, all 96 horizons pooled (`scores.json`) | 0.8695 | 0.296 |
| clean 298-episode set, t+0 (episode-disjoint, ad-hoc scorer) | 0.912 | 0.449 |
| clean, day 1 / 2 / 3 / 4 (ad-hoc probe) | 0.892 / 0.891 / 0.866 / 0.830 | 0.350 / 0.314 / 0.282 / 0.231 |

Canonical numbers on the hour-balanced grid (`research/opponent_model/evaluate.py`) are still to be produced; see EVALUATION.md §6.

Load with `TinyTimeMixerForPrediction.from_pretrained("models/ttm_v3_h96")`.

`scaler.npz` holds the per-feature input mean/std the weights were trained with
(first 200 train series of the seed-0 series split of `dataset_v2`; recomputed
identically by `eval_clean.py`/`horizon_probe.py`). Pass it as
`train_ttm.py --init-from models/ttm_v3_h96 --scaler models/ttm_v3_h96/scaler.npz`
when fine-tuning; a scaler refit on a different dataset would shift the inputs.
