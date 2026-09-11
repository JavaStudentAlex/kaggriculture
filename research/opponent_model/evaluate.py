"""One canonical evaluation for an opponent-supply checkpoint.

This replaces the ad-hoc trio (eval_clean.py / analyze_v2.py / horizon_probe.py),
which disagreed on the two settings that decide whether a number means anything:

1. *Which rows are scored.* The trainer's own val score is computed on a
   series-level split, so one seat of an episode can be in train while the other
   is in val -- they share a market trajectory and the public view of both farms.
   Every number here is computed on the episode-disjoint subset (no seat of the
   episode anywhere in train). `--include-leaky` also reports the un-cleaned
   split so the size of the leak stays visible instead of being assumed small.

2. *Where in the day the forecast origin sits.* Episodes are 719 usable turns and
   the context is 512, so only ~112 window origins exist per series and their
   hour cycles with period turnsPerDay=24. A stride sharing a factor with 24
   pins every origin onto a handful of hours -- and since hour-0 (the town-centre
   draw and the end-of-day shed dump) carries 3-4x the sell rate of a quiet hour,
   an aliased grid changes the headline AP by 2x on an unchanged checkpoint
   (measured on ttm_v2: AP 0.84 at stride 32, 0.68 at stride 4, 0.42 at stride 1).
   Worse, the *training* stride has the same arithmetic: window-stride 8 puts
   every training origin on hour 0, 8 or 16, so an aliased eval grid also happens
   to test only the phases the model was trained on. Here the stride is asserted
   coprime with turnsPerDay and the grid is then trimmed to an exact whole number
   of days per series, so every hour is represented equally.

Detection quality and magnitude accuracy are reported separately on purpose. The
target is ~98% zeros and the loss is MSE on log1p, so predicted magnitudes are
shrunk several-fold toward zero; the model ranks far better than it sizes. Any
headline that mixes the two hides which half is working.

Everything is scored against baselines computed on the same rows -- recency,
cumulative volume, time-since-last-sell, opponent ripe stock, price/base, and a
clock-only (hour x product base rate, fitted on train) predictor -- because on a
2%-positive target an impressive-looking AUC can be almost entirely free.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ttm_dataset import OpponentSupplyWindows, fit_scaler, load_shards

BASELINES = [
    ("recent_4turns", "opplast4_{p}", 1.0),
    ("recent_24turns", "opplast24_{p}", 1.0),
    ("cumulative", "oppcum_{p}", 1.0),
    ("turns_since_last_sell", "oppsince_{p}", -1.0),
    ("opp_ripe_units", "opppend_{p}", 1.0),
    ("price_over_base", "pxbase_{p}", 1.0),
]


def rebuild_split(series, episode_ids, split, val_fraction, seed):
    """Reproduce the trainer's split, then keep only episode-disjoint val series.

    `split` must match the training run's --split. Getting it wrong silently
    scores the checkpoint on episodes it was trained on, so it is printed and
    recorded in the report rather than defaulted quietly.
    """
    rng = np.random.default_rng(seed)
    if split == "series":
        idx = rng.permutation(len(series))
        cut = int(len(series) * (1.0 - val_fraction))
        train_idx, val_idx = idx[:cut], idx[cut:]
    else:
        episodes = np.array(sorted(set(episode_ids)))
        val_eps = set(rng.choice(episodes, size=int(len(episodes) * val_fraction), replace=False).tolist())
        val_idx = np.array([i for i, e in enumerate(episode_ids) if e in val_eps])
        train_idx = np.array([i for i, e in enumerate(episode_ids) if e not in val_eps])

    train_eps = {episode_ids[i] for i in train_idx}
    clean_idx = np.array([i for i in val_idx if episode_ids[i] not in train_eps])
    return train_idx, val_idx, clean_idx


def hour_balance(ds, turns_per_day, context_length):
    """Trim each series' window list to a whole number of days of origins.

    With a stride coprime to turnsPerDay the origin hour cycles with period
    turnsPerDay, so keeping a multiple of turnsPerDay consecutive origins per
    series gives an exactly uniform hour grid. Origin step is `start + context`
    because extract.py emits one row per turn from step 0.

    A stride large enough that a series yields fewer than turnsPerDay origins
    cannot be balanced this way -- episodes are only 719 turns and the context
    eats 512 of them. The trim is then skipped and the caller is told, rather
    than silently emptying the set or silently keeping a lopsided grid.
    """
    per_series = {}
    for s, start in ds.index:
        per_series.setdefault(s, []).append(start)
    shortest = min(len(v) for v in per_series.values())
    exact = shortest >= turns_per_day
    if exact:
        kept = []
        for s, starts in per_series.items():
            n = (len(starts) // turns_per_day) * turns_per_day
            kept.extend((s, st) for st in starts[:n])
        ds.index = kept
    else:
        print(
            f"WARNING: only {shortest} window origins per series at this stride; "
            f"cannot cover all {turns_per_day} hours evenly -- lower --stride",
            flush=True,
        )
    return Counter((st + context_length) % turns_per_day for _, st in ds.index), exact


def clock_baseline_table(train_series, n_products, turns_per_day, n_sample=4000):
    """P(sell) per (hour, product) from train rows -- the free part of the signal."""
    hits = np.zeros((turns_per_day, n_products))
    seen = np.zeros(turns_per_day)
    onehot = None
    for _, y in train_series[:n_sample]:
        if onehot is None or onehot.shape[1] != len(y):
            hour = np.arange(len(y)) % turns_per_day
            onehot = (hour[None, :] == np.arange(turns_per_day)[:, None]).astype(np.float32)
        hits += onehot @ (y > 0).astype(np.float32)
        seen += onehot.sum(axis=1)
    return hits / np.maximum(seen[:, None], 1.0)


def collect(model, ds, target_idx, n_features, mean, std, device, batch_size):
    import torch

    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=4)
    preds, trues, lasts = [], [], []
    with torch.no_grad():
        for b in loader:
            past = b["past_values"]
            out = model(past_values=past.to(device))
            preds.append(out.prediction_outputs.float().cpu().numpy())
            trues.append(b["future_values"][..., target_idx].numpy())
            lasts.append(past[:, -1, :n_features].numpy())
    return (np.concatenate(preds), np.concatenate(trues), np.concatenate(lasts) * std + mean)


def auc_ap(score, positive, min_pos=5):
    # Imported lazily so --dry-run (split and grid diagnostics) runs in a plain
    # numpy environment, without the training venv.
    from sklearn.metrics import average_precision_score, roc_auc_score

    if positive.sum() < min_pos or positive.all():
        return None, None, None
    ap = float(average_precision_score(positive, score))
    return (float(roc_auc_score(positive, score)), ap, float(ap / positive.mean()) if positive.mean() else None)


def detection_block(pred, pos, turns_per_day):
    """AUC / AP / lift over base rate, per forecast step and pooled per day.

    AP is the metric that matters on a ~2%-positive target; AUC saturates near 1
    while the precision the policy would actually see is far worse. Lift (AP over
    the base rate) is reported so the numbers stay comparable across slices whose
    base rates differ.
    """
    horizon = pred.shape[1]
    out = {"horizon": int(horizon), "per_step": [], "per_day": [], "pooled": {}}
    for h in range(horizon):
        a, p, lift = auc_ap(pred[:, h].ravel(), pos[:, h].ravel())
        out["per_step"].append({"h": h, "auc": a, "ap": p, "ap_lift": lift, "positive_rate": float(pos[:, h].mean())})
    for d in range(horizon // turns_per_day):
        lo, hi = d * turns_per_day, (d + 1) * turns_per_day
        a, p, lift = auc_ap(pred[:, lo:hi].ravel(), pos[:, lo:hi].ravel())
        out["per_day"].append(
            {
                "day": d + 1,
                "turns": f"t+{lo}..t+{hi - 1}",
                "auc": a,
                "ap": p,
                "ap_lift": lift,
                "positive_rate": float(pos[:, lo:hi].mean()),
            }
        )
    # `first24` exists so a 24-step-supervised checkpoint and a 96-step one can be
    # compared on identical rows; `all` covers whatever the head emits.
    for tag, sl in (("first24", slice(0, min(24, horizon))), ("all", slice(0, horizon))):
        a, p, lift = auc_ap(pred[:, sl].ravel(), pos[:, sl].ravel())
        out["pooled"][tag] = {"auc": a, "ap": p, "ap_lift": lift, "positive_rate": float(pos[:, sl].mean())}
    return out


def magnitude_block(pred, true, turns_per_day):
    """Units accuracy, kept apart from detection because it is much weaker.

    log1p + MSE on a 98%-zero target drives the conditional mean down, so the
    useful question is not "what is the MAE" (all-zero nearly wins that) but
    "when they do sell, how large is the prediction relative to the truth".
    """
    units_pred = np.expm1(np.clip(pred, 0, None))
    units_true = np.expm1(true)
    sold = units_true > 0.5
    out = {
        "mae_model": float(np.abs(units_true - units_pred).mean()),
        "mae_all_zero": float(np.abs(units_true).mean()),
        "skill_vs_all_zero": float(1.0 - np.abs(units_true - units_pred).mean() / max(np.abs(units_true).mean(), 1e-9)),
        "mae_when_sold": float(np.abs(units_true - units_pred)[sold].mean()),
        "mae_when_sold_all_zero": float(units_true[sold].mean()),
        "units_when_sold_actual": float(units_true[sold].mean()),
        "units_when_sold_pred": float(units_pred[sold].mean()),
    }
    out["shrinkage"] = float(out["units_when_sold_pred"] / max(out["units_when_sold_actual"], 1e-9))
    out["per_day_total"] = []
    for d in range(pred.shape[1] // turns_per_day):
        lo, hi = d * turns_per_day, (d + 1) * turns_per_day
        pu = units_pred[:, lo:hi].sum(axis=(1, 2))
        tu = units_true[:, lo:hi].sum(axis=(1, 2))
        out["per_day_total"].append(
            {
                "day": d + 1,
                "corr": float(np.corrcoef(pu, tu)[0, 1]),
                "true_mean_units": float(tu.mean()),
                "pred_mean_units": float(pu.mean()),
            }
        )
    return out


def bucket_table(key, actual_pos, actual_units, pred_score, pred_units, edges, labels):
    rows = []
    idx = np.digitize(key, edges)
    for b, label in enumerate(labels):
        m = idx == b
        if m.sum() < 50:
            continue
        a, p, lift = auc_ap(pred_score[m], actual_pos[m])
        rows.append(
            {
                "bucket": label,
                "n": int(m.sum()),
                "actual_p_sell": float(actual_pos[m].mean()),
                "actual_units_when_sold": float(actual_units[m][actual_pos[m]].mean()) if actual_pos[m].any() else 0.0,
                "pred_units_mean": float(pred_units[m].mean()),
                "auc": a,
                "ap": p,
                "ap_lift": lift,
            }
        )
    return rows


def behaviour_blocks(report, pred, pos, true, last, feat_names, products, turns_per_day):
    """Does the firing pattern track the mechanics, or only the recency features?"""
    fi = {n: i for i, n in enumerate(feat_names)}
    score = pred[:, 0].ravel()
    hit = pos[:, 0].ravel()
    units_true = np.expm1(true[:, 0]).ravel()
    units_pred = np.expm1(np.clip(pred[:, 0], 0, None)).ravel()

    def col(tmpl):
        return np.stack([last[:, fi[tmpl.format(p=p)]] for p in products], axis=1).ravel()

    report["by_price"] = bucket_table(
        col("pxbase_{p}"),
        hit,
        units_true,
        score,
        units_pred,
        [0.6, 0.85, 1.0, 1.15, 1.4],
        ["<0.6", "0.6-0.85", "0.85-1.0", "1.0-1.15", "1.15-1.4", ">1.4"],
    )
    report["by_ripe_stock"] = bucket_table(
        col("opppend_{p}"),
        hit,
        units_true,
        score,
        units_pred,
        [0.5, 2.5, 5.5, 10.5, 20.5],
        ["0", "1-2", "3-5", "6-10", "11-20", ">20"],
    )
    # Keyed by the hour of the t+1 target turn (last observed turn + 1), so that
    # hour 0 is the town-centre draw / end-of-day dump turn in every table.
    hour = np.repeat((np.rint(last[:, fi["hour"]]) + 1) % turns_per_day, len(products))
    report["by_target_hour"] = bucket_table(
        hour,
        hit,
        units_true,
        score,
        units_pred,
        list(np.arange(0.5, turns_per_day)),
        [str(h) for h in range(turns_per_day)],
    )

    # Cold start: can it call a sell in a product that has been quiet for a day?
    recent = col("opplast24_{p}")
    for tag, mask in (("cold_start", recent < 0.5), ("warm", recent >= 0.5)):
        a, p, lift = auc_ap(score[mask], hit[mask])
        report[tag] = {
            "n": int(mask.sum()),
            "positive_rate": float(hit[mask].mean()),
            "auc": a,
            "ap": p,
            "ap_lift": lift,
        }


def calibration_block(pred, true):
    """Decile of prediction vs realised rate, plus how much volume the top decile holds."""
    score = pred[:, 0].ravel()
    units_true = np.expm1(true[:, 0]).ravel()
    units_pred = np.expm1(np.clip(pred[:, 0], 0, None)).ravel()
    hit = units_true > 0.5
    order = np.argsort(score)
    deciles = []
    for d in range(10):
        sel = order[d * len(order) // 10 : (d + 1) * len(order) // 10]
        deciles.append(
            {
                "decile": d + 1,
                "pred_units_mean": float(units_pred[sel].mean()),
                "actual_p_sell": float(hit[sel].mean()),
                "actual_units_mean": float(units_true[sel].mean()),
            }
        )
    top = order[9 * len(order) // 10 :]
    return {"deciles": deciles, "top_decile_volume_recall": float(units_true[top].sum() / max(units_true.sum(), 1e-9))}


def write_summary(path, r):  # noqa: C901 -- one linear report
    L = []
    g = r["grid"]
    L.append(f"checkpoint       : {r['checkpoint']}")
    L.append(f"dataset          : {r['dataset']}")
    L.append(f"split            : {r['split']} (rebuilt), episode-disjoint clean subset")
    L.append(f"supervised steps : {r['supervised_horizon']}   emitted steps: {r['detection']['horizon']}")
    L.append(
        f"grid             : stride={g['stride']} hour-balanced={g['hour_balanced']} "
        f"origin steps {g['origin_step_min']}-{g['origin_step_max']}"
    )
    L.append(
        f"scale            : {g['n_episodes']} episodes / {g['n_series']} series / "
        f"{g['n_windows']} windows  base rate {g['positive_rate']:.4f}"
    )
    lk = r.get("leak_gap") or {}
    if lk.get("clean_ap") is not None and lk.get("leaky_ap") is not None:
        L.append(f"leak check (t+1) : clean AP {lk['clean_ap']:.4f} vs un-cleaned val AP {lk['leaky_ap']:.4f}")
    L.append("")
    L.append("DETECTION (clean, hour-balanced)")
    for row in r["detection"]["per_day"]:
        if row["ap"] is None:
            continue
        L.append(
            f"  day {row['day']} {row['turns']:>14}  auc {row['auc']:.4f}  "
            f"ap {row['ap']:.4f}  lift {row['ap_lift']:6.1f}x  rate {row['positive_rate']:.4f}"
        )
    for tag, row in r["detection"]["pooled"].items():
        if row["ap"] is None:
            continue
        L.append(f"  pooled {tag:<8}          auc {row['auc']:.4f}  ap {row['ap']:.4f}  lift {row['ap_lift']:6.1f}x")
    L.append("")
    L.append("BASELINES at t+1 (same rows)")
    for name, row in r["baselines"].items():
        if row["ap"] is None:
            continue
        L.append(f"  {name:<24} auc {row['auc']:.4f}  ap {row['ap']:.4f}  lift {row['ap_lift']:6.1f}x")
    L.append("")
    m = r["magnitude"]
    L.append("MAGNITUDE (units, after undoing log1p)")
    L.append(
        f"  mae model {m['mae_model']:.4f} vs all-zero {m['mae_all_zero']:.4f} (skill {m['skill_vs_all_zero']:+.3f})"
    )
    L.append(
        f"  when a sell happens: actual {m['units_when_sold_actual']:.2f} u, "
        f"predicted {m['units_when_sold_pred']:.2f} u  -> shrinkage {m['shrinkage']:.2f}x"
    )
    for row in m["per_day_total"]:
        L.append(
            f"  day {row['day']} total volume corr {row['corr']:.3f}  "
            f"true {row['true_mean_units']:.1f} u  pred {row['pred_mean_units']:.1f} u"
        )
    L.append("")
    L.append(f"top-decile volume recall: {r['calibration']['top_decile_volume_recall']:.3f}")
    L.append("")
    L.append("PER PRODUCT at t+1")
    for row in r["by_product"]:
        if row["ap"] is None:
            continue
        L.append(
            f"  {row['product']:<12} rate {row['positive_rate']:.4f}  auc {row['auc']:.4f}  "
            f"ap {row['ap']:.4f}  units {row['units_when_sold']:.1f} -> "
            f"{row['pred_units_when_sold']:.2f}"
        )
    Path(path).write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():  # noqa: C901 -- one linear CLI flow
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", default="research/opponent_model/dataset_v2")
    ap.add_argument(
        "--split",
        choices=["series", "episode"],
        default="series",
        help="MUST match the training run's --split; v2 and v3 used 'series'",
    )
    ap.add_argument(
        "--max-episodes",
        type=int,
        default=100000,
        help="MUST match the training run; a different cap gives a different split",
    )
    ap.add_argument("--shard-limit", type=int, default=None)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--context-length", type=int, default=512)
    ap.add_argument("--prediction-length", type=int, default=96)
    ap.add_argument("--turns-per-day", type=int, default=24)
    ap.add_argument("--stride", type=int, default=1, help="window stride; asserted coprime with --turns-per-day")
    ap.add_argument(
        "--no-hour-balance", action="store_true", help="skip the whole-day trim (leaves the hour grid slightly uneven)"
    )
    ap.add_argument(
        "--max-series",
        type=int,
        default=None,
        help="subsample clean series (whole series, so the hour grid stays balanced)",
    )
    ap.add_argument(
        "--unmask-horizon",
        action="store_true",
        help="drop prediction_filter_length so a 24-step checkpoint also emits "
        "days 2-4; those steps got no gradient, mark them as diagnostic",
    )
    ap.add_argument(
        "--include-leaky", action="store_true", help="also score the un-cleaned val split, to size the seat leak"
    )
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None, help="JSON report path (default <checkpoint>/../eval.json)")
    ap.add_argument("--summary", default=None, help="text summary path (default alongside --out)")
    ap.add_argument("--dry-run", action="store_true", help="build the split and grid, print diagnostics, load no model")
    args = ap.parse_args()

    if math.gcd(args.stride, args.turns_per_day) != 1:
        raise SystemExit(
            f"--stride {args.stride} shares a factor with turnsPerDay={args.turns_per_day}: "
            "every window origin would land on the same few hours of the day and the "
            "reported AP would be inflated. Use 1, 5, 7, 11 or 13."
        )

    series, feat_names, products, ep_ids = load_shards(
        args.dataset, args.max_episodes, args.shard_limit, return_ids=True
    )
    print(
        f"series={len(series)} episodes={len(set(ep_ids))} features={len(feat_names)} "
        f"products={len(products)}  <- must match the training log",
        flush=True,
    )

    train_idx, val_idx, clean_idx = rebuild_split(series, ep_ids, args.split, args.val_fraction, args.seed)
    train_series = [series[i] for i in train_idx]
    clean_series = [series[i] for i in clean_idx]
    n_clean_eps = len({ep_ids[i] for i in clean_idx})
    print(
        f"split={args.split}  train={len(train_idx)} val={len(val_idx)} "
        f"clean(no seat in train)={len(clean_idx)} from {n_clean_eps} episodes",
        flush=True,
    )
    if not clean_series:
        raise SystemExit("no episode-disjoint val series -- check --split / --max-episodes")

    if args.max_series and len(clean_series) > args.max_series:
        rng = np.random.default_rng(args.seed)
        pick = rng.choice(len(clean_series), size=args.max_series, replace=False)
        clean_series = [clean_series[i] for i in sorted(pick)]

    scaler_path = Path(args.checkpoint) / "scaler.npz"
    if scaler_path.exists():
        with np.load(scaler_path) as sc:
            mean, std = sc["mean"], sc["std"]
        print(f"scaler: {scaler_path}", flush=True)
    else:
        mean, std = fit_scaler(train_series)
        print("scaler: refit from the training split (no scaler.npz next to the checkpoint)", flush=True)
    ds = OpponentSupplyWindows(clean_series, args.context_length, args.prediction_length, mean, std, args.stride)
    if args.no_hour_balance:
        hours = Counter((st + args.context_length) % args.turns_per_day for _, st in ds.index)
        balanced = False
    else:
        hours, balanced = hour_balance(ds, args.turns_per_day, args.context_length)
    origins = [st + args.context_length for _, st in ds.index]
    grid = {
        "stride": args.stride,
        "hour_balanced": balanced,
        "n_episodes": n_clean_eps,
        "n_series": len(clean_series),
        "n_windows": len(ds),
        "origin_step_min": int(min(origins)),
        "origin_step_max": int(max(origins)),
        "origin_hour_counts": {str(h): int(hours.get(h, 0)) for h in range(args.turns_per_day)},
    }
    print(
        f"windows={len(ds):,}  origin steps {grid['origin_step_min']}-{grid['origin_step_max']}  "
        f"hours covered={sum(1 for v in hours.values() if v)}/{args.turns_per_day}",
        flush=True,
    )
    if args.dry_run:
        print(json.dumps(grid, indent=2))
        return

    import torch
    from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    model = TinyTimeMixerForPrediction.from_pretrained(args.checkpoint).to(device).eval()
    supervised = model.config.prediction_filter_length or model.config.prediction_length
    if args.unmask_horizon:
        model.prediction_filter_length = None
        model.head.prediction_filter_length = None
        model.config.prediction_filter_length = None

    tidx = ds.target_channel_indices()
    pred, true, last = collect(model, ds, tidx, ds.n_features, mean, std, device, args.batch_size)
    pos = np.expm1(true) > 0.5
    grid["positive_rate"] = float(pos[:, 0].mean())

    report = {
        "checkpoint": str(args.checkpoint),
        "dataset": str(args.dataset),
        "split": args.split,
        "products": list(products),
        "supervised_horizon": int(supervised),
        "unmasked_horizon": bool(args.unmask_horizon),
        "grid": grid,
        "detection": detection_block(pred, pos, args.turns_per_day),
        "magnitude": magnitude_block(pred, true, args.turns_per_day),
        "calibration": calibration_block(pred, true),
    }

    # Baselines on exactly the rows the model was scored on.
    fi = {n: i for i, n in enumerate(feat_names)}
    hit = pos[:, 0].ravel()
    base = {}
    for tag, tmpl, sign in BASELINES:
        cols = np.stack([last[:, fi[tmpl.format(p=p)]] for p in products], axis=1).ravel()
        a, p, lift = auc_ap(sign * cols, hit)
        base[tag] = {"auc": a, "ap": p, "ap_lift": lift}
    clock = clock_baseline_table(train_series, len(products), args.turns_per_day)
    # `last` is the final *observed* turn, so the t+1 target sits one hour later;
    # it is de-standardised, so round rather than truncate back to int.
    hour_col = (np.rint(last[:, fi["hour"]]).astype(int) + 1) % args.turns_per_day
    a, p, lift = auc_ap(clock[hour_col].ravel(), hit)
    base["clock_hour_x_product"] = {"auc": a, "ap": p, "ap_lift": lift}
    a, p, lift = auc_ap(pred[:, 0].ravel(), hit)
    base["model"] = {"auc": a, "ap": p, "ap_lift": lift}
    report["baselines"] = base

    report["by_product"] = []
    units_true = np.expm1(true)
    for j, name in enumerate(products):
        sold = pos[:, 0, j]
        a, p, lift = auc_ap(pred[:, 0, j], sold)
        report["by_product"].append(
            {
                "product": name,
                "n_pos": int(sold.sum()),
                "positive_rate": float(sold.mean()),
                "auc": a,
                "ap": p,
                "ap_lift": lift,
                "units_when_sold": float(units_true[:, 0, j][sold].mean()) if sold.any() else 0.0,
                "pred_units_when_sold": float(np.expm1(pred[:, 0, j])[sold].mean()) if sold.any() else 0.0,
                "pred_units_when_idle": float(np.expm1(pred[:, 0, j])[~sold].mean()),
            }
        )
    behaviour_blocks(report, pred, pos, true, last, feat_names, products, args.turns_per_day)

    if args.include_leaky:
        leaky = [series[i] for i in val_idx]
        lds = OpponentSupplyWindows(leaky, args.context_length, args.prediction_length, mean, std, args.stride)
        if not args.no_hour_balance:
            hour_balance(lds, args.turns_per_day, args.context_length)  # same grid as clean
        lpred, ltrue, _ = collect(model, lds, tidx, lds.n_features, mean, std, device, args.batch_size)
        lpos = (np.expm1(ltrue[:, 0]) > 0.5).ravel()
        la, lp, _ = auc_ap(lpred[:, 0].ravel(), lpos)
        report["leak_gap"] = {
            "leaky_windows": len(lds),
            "leaky_auc": la,
            "leaky_ap": lp,
            "clean_auc": base["model"]["auc"],
            "clean_ap": base["model"]["ap"],
        }

    out = Path(args.out or (Path(args.checkpoint).parent / "eval.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    write_summary(args.summary or out.with_suffix(".txt"), report)
    print(f"\nwrote {out} and {args.summary or out.with_suffix('.txt')}")


if __name__ == "__main__":
    main()
