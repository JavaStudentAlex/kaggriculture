"""Streaming AUC / AP for sell detection, from score histograms.

The ranking metrics that select checkpoints (`auc_any_sell`, `ap_any_sell`) are
pooled over every (window, forecast step, product) cell -- 4e8 cells per pass over
the validation set. sklearn sorts all of them on one CPU core, which cost the
trainer 10-70 minutes after every epoch, with the GPUs idle. Quantising the
scores to 2^20 bins over a fixed range (width 1.1e-5 in log1p units) turns the
same numbers into two histograms (positives, negatives per bin) that accumulate
batch by batch on whatever device the predictions are on; AUC and AP then come
out of the cumulative sums in milliseconds. Binning only merges scores that are
within one bin width of each other, so the result differs from the exact metric
by less than 1e-5 on this problem (checked against sklearn).
"""

from __future__ import annotations

import numpy as np
import torch

BINS, LO, HI = 1 << 20, -2.0, 10.0  # log1p units; the model's outputs lie well inside


class SellDetection:
    """Accumulate sell-detection AUC / AP over any number of batches.

    `add(score, positive)` takes tensors (any device) or numpy arrays of equal
    shape; `result()` returns the pooled metrics and `reset()` clears the counts.
    """

    def __init__(self, bins=BINS, lo=LO, hi=HI, device=None):
        self.bins, self.lo, self.scale = bins, lo, (bins - 1) / (hi - lo)
        self.device = torch.device(device) if device is not None else None
        self.pos = self.neg = None

    def reset(self):
        self.pos = self.neg = None

    def add(self, score, positive, chunk=50_000_000):
        score = torch.as_tensor(np.asarray(score) if not torch.is_tensor(score) else score).reshape(-1)
        positive = torch.as_tensor(np.asarray(positive) if not torch.is_tensor(positive) else positive).reshape(-1)
        device = self.device or score.device
        if self.pos is None:
            self.pos = torch.zeros(self.bins, dtype=torch.int64, device=device)
            self.neg = torch.zeros(self.bins, dtype=torch.int64, device=device)
        for start in range(0, score.numel(), chunk):  # chunked so 4e8 cells never need a 3 GB index tensor
            s = score[start : start + chunk].to(device, torch.float64)
            p = positive[start : start + chunk].to(device, torch.bool)
            idx = ((s - self.lo) * self.scale).round_().clamp_(0, self.bins - 1).long()
            self.pos += torch.bincount(idx[p], minlength=self.bins)
            self.neg += torch.bincount(idx[~p], minlength=self.bins)

    def result(self):
        """auc, ap, positive_rate; AUC 0.5 / AP = base rate when one class is absent."""
        pos = self.pos.cpu().numpy().astype(np.float64)
        neg = self.neg.cpu().numpy().astype(np.float64)
        n_pos, n_neg = pos.sum(), neg.sum()
        rate = float(n_pos / (n_pos + n_neg)) if n_pos + n_neg else 0.0
        if n_pos == 0 or n_neg == 0:
            return {"auc": 0.5, "ap": rate, "positive_rate": rate, "n_pos": int(n_pos), "n": int(n_pos + n_neg)}
        # AUC = P(score_pos > score_neg) + 0.5 P(tie): negatives strictly below each bin, half of those in it.
        neg_below = np.cumsum(neg) - neg
        auc = float((pos * (neg_below + 0.5 * neg)).sum() / (n_pos * n_neg))
        # AP = sum over thresholds (descending) of recall gained x precision there, sklearn's definition.
        pos_d, neg_d = pos[::-1], neg[::-1]
        tp, fp = np.cumsum(pos_d), np.cumsum(neg_d)
        ap = float((pos_d / n_pos * (tp / np.maximum(tp + fp, 1.0))).sum())
        return {"auc": auc, "ap": ap, "positive_rate": rate, "n_pos": int(n_pos), "n": int(n_pos + n_neg)}


def auc_ap(score, positive):
    """One-shot (auc, ap, positive_rate) for arrays that are already in memory."""
    det = SellDetection()
    det.add(score, positive)
    r = det.result()
    return r["auc"], r["ap"], r["positive_rate"]
