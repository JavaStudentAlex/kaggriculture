"""Fine-tune IBM Granite TinyTimeMixer on opponent market supply.

Channels are laid out as [141 exogenous features | 9 targets]; TTM forecasts
only the target channels (`prediction_channel_indices`) while attending to the
exogenous ones via channel mixing.

The targets are extremely sparse -- 0.3-3% of turns carry a nonzero sell -- so a
plain regression score is uninformative on its own. Every run is therefore
reported against two trivial baselines (all-zero, per-product train mean) and
scored at forecast horizon 1, which is what the policy actually consumes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ttm_dataset import (
    OpponentSupplyWindows,
    fit_scaler,
    load_shards,
    split_by_episode,
    split_series,
)

MODEL_ID = "ibm-granite/granite-timeseries-ttm-r1"


def shard_date(path):
    """Date of a `kaggriculture-episodes-YYYY-MM-DD.npz` shard (or the newest one in a dir)."""
    import datetime as dt
    import re

    path = Path(path)
    names = [p.name for p in path.glob("*.npz")] if path.is_dir() else [path.name]
    dates = [dt.date.fromisoformat(m.group(1)) for n in names for m in [re.search(r"(\d{4}-\d{2}-\d{2})", n)] if m]
    return max(dates) if dates else None


def build_model(n_channels, target_idx, prediction_filter_length, revision, init_from=None):
    from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction

    if init_from:
        # Warm start from one of our own checkpoints: same channel layout, so
        # the trained channel mixers carry over instead of re-initialising.
        # The checkpoint carries its own filter length, so pass ours explicitly --
        # otherwise a run that means to supervise all 96 steps silently inherits
        # the 24-step mask and days 2-4 keep getting no gradient.
        model = TinyTimeMixerForPrediction.from_pretrained(init_from)
        flt = prediction_filter_length or None
        if flt == model.config.prediction_length:
            flt = None
        model.config.prediction_filter_length = flt
        model.prediction_filter_length = flt
        model.head.prediction_filter_length = flt
        return model

    kwargs = dict(
        num_input_channels=n_channels,
        prediction_channel_indices=target_idx,
        decoder_mode="mix_channel",
        ignore_mismatched_sizes=True,
    )
    if prediction_filter_length:
        kwargs["prediction_filter_length"] = prediction_filter_length
    if revision:
        kwargs["revision"] = revision

    model = TinyTimeMixerForPrediction.from_pretrained(MODEL_ID, **kwargs)
    return model


def make_metric_hooks(target_idx, horizon=0):
    """Trainer hooks that score horizon-`horizon` sell detection every epoch.

    `horizon=None` pools every forecast step instead, which is what model
    selection needs once the loss supervises the whole forecast -- otherwise a
    run training four days still early-stops on how well it calls the next turn.

    The Trainer concatenates every eval batch in memory, so the logits hook
    keeps only the slice we score instead of the full forecast plus backbone
    states.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    def preprocess_logits(logits, labels):
        pred = logits[0] if isinstance(logits, (tuple, list)) else logits
        pred = pred.detach().float()
        return pred if horizon is None else pred[:, horizon, :]

    def compute_metrics(eval_pred):
        pred = np.asarray(eval_pred.predictions, dtype=np.float32)
        labels = eval_pred.label_ids
        labels = labels[0] if isinstance(labels, (tuple, list)) else labels
        labels = np.asarray(labels, dtype=np.float32)
        true = labels[..., target_idx] if horizon is None else labels[:, horizon, :][:, target_idx]
        positive = (np.expm1(true) > 0.5).ravel()
        flat = pred.ravel()
        out = {"positive_rate": float(positive.mean())}
        if 0 < positive.sum() < len(positive):
            out["auc_any_sell"] = float(roc_auc_score(positive, flat))
            out["ap_any_sell"] = float(average_precision_score(positive, flat))
        else:
            out["auc_any_sell"] = 0.5
            out["ap_any_sell"] = float(positive.mean())
        return out

    return preprocess_logits, compute_metrics


@torch.no_grad()
def evaluate(model, loader, target_idx, device, horizon=0):
    """Report loss plus horizon-`horizon` scores against trivial baselines."""
    model.eval()
    preds, trues = [], []
    for batch in loader:
        past = batch["past_values"].to(device)
        future = batch["future_values"].to(device)
        out = model(past_values=past, future_values=future)
        p = out.prediction_outputs.detach().float().cpu().numpy()
        t = future[:, :, target_idx].detach().float().cpu().numpy()
        if horizon is None:
            preds.append(p.reshape(-1, p.shape[-1]))
            trues.append(t.reshape(-1, t.shape[-1]))
        else:
            preds.append(p[:, horizon, :])
            trues.append(t[:, horizon, :])

    pred = np.concatenate(preds)
    true = np.concatenate(trues)
    # Back out of log1p to report errors in units the game actually uses.
    pred_units = np.expm1(np.clip(pred, 0, None))
    true_units = np.expm1(true)

    zero_mae = np.abs(true_units).mean()
    mean_pred = true_units.mean(axis=0, keepdims=True)
    mean_mae = np.abs(true_units - mean_pred).mean()
    model_mae = np.abs(true_units - pred_units).mean()

    positive = true_units > 0
    scores = {
        "n": len(true),
        "positive_rate": float(positive.mean()),
        "mae_model": float(model_mae),
        "mae_all_zero": float(zero_mae),
        "mae_train_mean": float(mean_mae),
        "skill_vs_zero": float(1.0 - model_mae / zero_mae) if zero_mae else 0.0,
    }
    # MAE alone is nearly unbeatable on a 99.7%-zero target, so also report
    # whether the model *ranks* sell-turns above quiet ones (AUC / AP), and how
    # close it gets on magnitude when a sell actually happens.
    from sklearn.metrics import average_precision_score, roc_auc_score

    flat_pos, flat_pred = positive.ravel(), pred.ravel()
    if 0 < flat_pos.sum() < len(flat_pos):
        scores["auc_any_sell"] = float(roc_auc_score(flat_pos, flat_pred))
        scores["ap_any_sell"] = float(average_precision_score(flat_pos, flat_pred))
        scores["ap_baseline_rate"] = float(flat_pos.mean())
        scores["mae_when_sold"] = float(np.abs(true_units - pred_units)[positive].mean())
        scores["mae_when_sold_all_zero"] = float(true_units[positive].mean())
    per_product = {}
    for j in range(true.shape[1]):
        pj, tj = pred[:, j], positive[:, j]
        if 0 < tj.sum() < len(tj):
            per_product[j] = {"auc": float(roc_auc_score(tj, pj)), "n_pos": int(tj.sum())}
    scores["per_product_auc"] = per_product
    return scores


def main():  # noqa: C901 -- one linear CLI flow
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="research/opponent_model/dataset")
    ap.add_argument("--out", default="research/opponent_model/runs/ttm")
    ap.add_argument("--max-episodes", type=int, default=4000)
    ap.add_argument("--shard-limit", type=int, default=None)
    ap.add_argument("--context-length", type=int, default=512)
    ap.add_argument("--prediction-length", type=int, default=96)
    ap.add_argument("--prediction-filter-length", type=int, default=24)
    ap.add_argument("--window-stride", type=int, default=8)
    ap.add_argument("--eval-stride", type=int, default=5, help="must be coprime with turnsPerDay; see val_ds below")
    ap.add_argument(
        "--metric-horizon",
        default="0",
        help="forecast step to score for early stopping, or 'all' to pool every supervised step",
    )
    ap.add_argument(
        "--epochs", type=float, default=3.0, help="epoch cap; with --plateau the schedule does not depend on it"
    )
    ap.add_argument("--patience", type=int, default=3, help="early-stopping patience (epochs)")
    ap.add_argument(
        "--plateau", action="store_true", help="ReduceLROnPlateau on the AUC metric instead of linear decay"
    )
    ap.add_argument("--plateau-factor", type=float, default=0.5)
    ap.add_argument("--plateau-patience", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--revision", default=None)
    ap.add_argument(
        "--init-from", default=None, help="warm-start from a local checkpoint dir instead of the HF weights"
    )
    ap.add_argument(
        "--resume-from",
        default=None,
        help="HF Trainer checkpoint-N dir to resume (weights, optimizer, "
        "scheduler, early-stopping counter); pass the run dir to pick the latest",
    )
    ap.add_argument(
        "--val-dataset",
        default=None,
        help="dir of shards used as the whole val set (e.g. the next day); "
        "--dataset is then trained on in full instead of being split",
    )
    ap.add_argument(
        "--extra-train-decay",
        type=float,
        default=1.0,
        help="recency weighting of --extra-train: a shard dated d days before the newest "
        "--dataset shard gets window stride window_stride*decay**d, so each day back "
        "contributes 1/decay as many windows (2.0 = one-day half-life; 1.0 = flat)",
    )
    ap.add_argument(
        "--extra-train",
        default=None,
        help="dir of shards added to the training set in full, with no share in val "
        "(e.g. the days before the --dataset day), so val stays the --dataset split",
    )
    ap.add_argument(
        "--scaler",
        default=None,
        help="scaler.npz (mean/std) saved next to a model; use it when fine-tuning "
        "so inputs stay in the space the warm-start weights were trained on",
    )
    ap.add_argument(
        "--split",
        choices=["episode", "series"],
        default="episode",
        help="series = the long run's split (required when warm-starting from it)",
    )
    ap.add_argument(
        "--eval-on-start",
        action="store_true",
        help="evaluate before the first step (epoch 0 = the warm-start weights' "
        "score on this val set, the number a fine-tune has to beat)",
    )
    ap.add_argument("--probe", action="store_true", help="print model config and exit")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading shards from {args.dataset} ...", flush=True)
    series, feature_names, products, episode_ids = load_shards(
        args.dataset, args.max_episodes, args.shard_limit, return_ids=True
    )
    print(
        f"series={len(series)} episodes={len(set(episode_ids))} features={len(feature_names)} products={len(products)}",
        flush=True,
    )

    if args.val_dataset:
        train_series = series
        val_series, _, _ = load_shards(args.val_dataset, args.max_episodes)
        print(
            f"split=none: --dataset in full train_series={len(train_series)}; "
            f"val_series={len(val_series)} from {args.val_dataset}",
            flush=True,
        )
    elif args.split == "episode":
        train_series, val_series = split_by_episode(series, episode_ids, val_fraction=0.1)
        print(f"split={args.split} train_series={len(train_series)} val_series={len(val_series)}", flush=True)
    else:
        train_series, val_series = split_series(series, val_fraction=0.1)
        print(f"split={args.split} train_series={len(train_series)} val_series={len(val_series)}", flush=True)
    # (series list, window stride) per training block; the newest day is block 0
    extra_blocks = []
    if args.extra_train:
        newest = shard_date(args.dataset)
        for shard in sorted(Path(args.extra_train).glob("*.npz")):
            block, _, _ = load_shards(shard, args.max_episodes)
            age = (newest - shard_date(shard)).days if newest and shard_date(shard) else 1
            stride = max(1, round(args.window_stride * args.extra_train_decay ** max(age, 0)))
            extra_blocks.append((block, stride))
            print(f"extra train {shard.name}: series={len(block)} age={age}d stride={stride}", flush=True)
    if args.scaler:
        with np.load(args.scaler) as sc:
            mean, std = sc["mean"], sc["std"]
        print(f"scaler loaded from {args.scaler}", flush=True)
    else:
        mean, std = fit_scaler(train_series + [s for b, _ in extra_blocks for s in b])
    np.savez(
        out_dir / "scaler.npz", mean=mean, std=std, feature_names=np.array(feature_names), products=np.array(products)
    )

    train_ds = OpponentSupplyWindows(
        train_series, args.context_length, args.prediction_length, mean, std, args.window_stride
    )
    if extra_blocks:
        parts = [train_ds] + [
            OpponentSupplyWindows(b, args.context_length, args.prediction_length, mean, std, stride)
            for b, stride in extra_blocks
        ]
        print("train windows by block (newest first): " + ", ".join(f"{len(d):,}" for d in parts), flush=True)
        train_ds = torch.utils.data.ConcatDataset(parts)
        train_ds.n_channels = parts[0].n_channels
        train_ds.target_channel_indices = parts[0].target_channel_indices
    # Eval stride must be coprime with turnsPerDay (24): a stride sharing a factor
    # with the day pins every window to the same few hours, and the end-of-day shed
    # dump is far easier to call than the rest of the day, so an aliased grid
    # flatters AP badly (stride 32 -> AP 0.83 vs 0.42 on a uniform hour grid).
    val_ds = OpponentSupplyWindows(val_series, args.context_length, args.prediction_length, mean, std, args.eval_stride)
    target_idx = train_ds.target_channel_indices()
    print(
        f"train windows={len(train_ds):,} val windows={len(val_ds):,} "
        f"channels={train_ds.n_channels} targets={len(target_idx)}",
        flush=True,
    )

    model = build_model(train_ds.n_channels, target_idx, args.prediction_filter_length, args.revision, args.init_from)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,}", flush=True)
    print(f"config: {json.dumps(model.config.to_dict(), default=str)[:600]}", flush=True)
    if args.probe:
        return

    from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

    metric_horizon = None if args.metric_horizon == "all" else int(args.metric_horizon)
    preprocess_logits, compute_metrics = make_metric_hooks(target_idx, horizon=metric_horizon)
    sched = {}
    if args.plateau:
        sched = dict(
            lr_scheduler_type="reduce_lr_on_plateau",
            lr_scheduler_kwargs={"factor": args.plateau_factor, "patience": args.plateau_patience, "mode": "max"},
        )
    targs = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        # Pooling the metric over every forecast step (--metric-horizon all)
        # makes each eval batch's logits (batch, prediction_length, n_targets)
        # instead of (batch, n_targets) -- 96x larger at prediction_length=96.
        # Uncapped, the Trainer concatenates all of them on GPU across the
        # whole eval set before computing metrics, which OOM'd on a GPU shared
        # with another job on 2026-09-11 (crashed 26 min into epoch 1, no
        # checkpoint saved). Offload every 50 batches instead.
        eval_accumulation_steps=50,
        learning_rate=args.lr,
        eval_strategy="epoch",
        eval_on_start=args.eval_on_start,
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="auc_any_sell",
        greater_is_better=True,
        label_names=["future_values"],
        logging_steps=100,
        bf16=torch.cuda.is_available(),
        dataloader_num_workers=8,
        report_to=[],
        ddp_find_unused_parameters=False,
        **sched,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        preprocess_logits_for_metrics=preprocess_logits,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )
    resume = args.resume_from
    if resume and not (Path(resume) / "trainer_state.json").exists():
        from transformers.trainer_utils import get_last_checkpoint

        resume = get_last_checkpoint(resume)
        print(f"resuming from latest checkpoint: {resume}", flush=True)
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(str(out_dir / "best"))

    device = next(model.parameters()).device
    loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, num_workers=4)
    scores = evaluate(model, loader, target_idx, device, horizon=metric_horizon)
    label = "all horizons pooled" if metric_horizon is None else f"horizon t+{metric_horizon}"
    print(f"\n=== validation ({label}) ===")
    for k, v in scores.items():
        print(f"  {k}: {v}")
    (out_dir / "scores.json").write_text(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
