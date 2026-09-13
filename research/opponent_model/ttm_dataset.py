"""Turn extracted .npz shards into windowed series for TinyTimeMixer.

TTM consumes `past_values` of shape (context_length, n_channels). We lay each
(episode, seat) out as a contiguous [features | targets] series and slide a
window over it, so exogenous channels and the forecast targets stay aligned on
the same time axis.

Targets are opponent executed sell counts: sparse (0.3-3% nonzero) and heavy
tailed, so they are log1p-compressed before the regression loss sees them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# Diagnostics column layout written by extract.py.
D_EPISODE, D_STEP, D_SEAT, D_CONTESTED = 0, 1, 2, 3


def load_shards(dataset_dir, max_episodes=None, shard_limit=None, return_ids=False):
    """Group rows into per-(episode, seat) series ordered by step.

    With `return_ids`, also return the episode id of every series, so callers
    can build episode-disjoint splits.
    """
    shards = [Path(dataset_dir)] if Path(dataset_dir).is_file() else sorted(Path(dataset_dir).glob("*.npz"))
    if shard_limit:
        shards = shards[:shard_limit]
    if not shards:
        raise FileNotFoundError(f"no .npz shards in {dataset_dir}")

    series, feature_names, products = [], None, None
    episode_ids = []
    seen_episodes = set()

    for shard in shards:
        with np.load(shard, allow_pickle=True) as data:
            X, Y, D = data["X"], data["Y"], data["D"]
            if feature_names is None:
                feature_names = [str(v) for v in data["feature_names"]]
                products = [str(v) for v in data["products"]]

        keys = D[:, D_EPISODE].astype(np.int64) * 4 + D[:, D_SEAT].astype(np.int64)
        order = np.lexsort((D[:, D_STEP], keys))
        X, Y, D, keys = X[order], Y[order], D[order], keys[order]
        boundaries = np.flatnonzero(np.diff(keys)) + 1

        for lo, hi in zip(np.r_[0, boundaries], np.r_[boundaries, len(keys)], strict=True):
            episode = int(D[lo, D_EPISODE])
            if max_episodes and episode not in seen_episodes and len(seen_episodes) >= max_episodes:
                continue
            seen_episodes.add(episode)
            series.append((X[lo:hi], Y[lo:hi]))
            episode_ids.append(episode)

    if return_ids:
        return series, feature_names, products, episode_ids
    return series, feature_names, products


def fit_scaler(series, n_sample=200):
    """Per-feature mean/std over a sample of series, for input standardisation."""
    sample = np.concatenate([x for x, _ in series[:n_sample]], axis=0)
    mean = sample.mean(axis=0)
    std = sample.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


class OpponentSupplyWindows(Dataset):
    """Sliding windows of [scaled features | log1p targets] over one series each."""

    def __init__(self, series, context_length, prediction_length, mean, std, stride=1):
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.mean, self.std = mean, std
        self.index = []
        self.series = []

        span = context_length + prediction_length
        for x, y in series:
            if len(x) < span:
                continue
            xs = (x - mean) / std
            ys = np.log1p(np.clip(y, 0, None)).astype(np.float32)
            self.series.append(np.concatenate([xs, ys], axis=1).astype(np.float32))
            last = len(x) - span
            for start in range(0, last + 1, stride):
                self.index.append((len(self.series) - 1, start))

        self.n_features = series[0][0].shape[1]
        self.n_targets = series[0][1].shape[1]
        self.n_channels = self.n_features + self.n_targets

    def target_channel_indices(self):
        return list(range(self.n_features, self.n_channels))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        s, start = self.index[i]
        block = self.series[s]
        past = block[start : start + self.context_length]
        future = block[start + self.context_length : start + self.context_length + self.prediction_length]
        return {
            "past_values": torch.from_numpy(np.ascontiguousarray(past)),
            "future_values": torch.from_numpy(np.ascontiguousarray(future)),
        }


def split_by_episode(series, episode_ids, val_fraction=0.1, seed=0, return_ids=False):
    """Episode-disjoint split: both seats of an episode land on the same side.

    Seats of one episode share a market trajectory and see each other's public
    farm, so splitting by series lets the model glimpse val episodes from the
    other seat. Splitting by episode removes that. With `return_ids`, the sorted
    val episode ids come back too (saved next to a checkpoint as
    `val_episodes.json`, so later scorers can stay out of sample).
    """
    rng = np.random.default_rng(seed)
    episodes = np.array(sorted(set(episode_ids)))
    val_eps = set(rng.choice(episodes, size=int(len(episodes) * val_fraction), replace=False).tolist())
    train = [s for s, e in zip(series, episode_ids, strict=True) if e not in val_eps]
    val = [s for s, e in zip(series, episode_ids, strict=True) if e in val_eps]
    if return_ids:
        return train, val, sorted(val_eps)
    return train, val
