"""Fit a calibration.json that makes a new predictor's scores cross the policy's thresholds as
often as the old predictor's do, per product (quantile matching), and report precision and AUC.

    python arena/calib_fit.py runs/arena/calib0923/traces [--held-out-only] [--out calibration.json]

Input: the per-episode npz files of calib_worker.py (models named `old` and `new`). The factor of
a product and metric is the geometric mean, over the policy's thresholds for that metric, of
old_threshold / (the new model's score at the old model's firing rate); kagg_oracle.forecast()
multiplies score_k / units_k by it when calibration.json sits next to the checkpoint.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'opponent_model'))
from mechanics import PRODUCTS  # noqa: E402 -- kagg_oracle's product order (needs kaggle_environments)

ap = argparse.ArgumentParser()
ap.add_argument('traces', help='directory of cal_<episode>.npz')
ap.add_argument('--held-out-only', action='store_true', help="only the new model's held-out episodes")
ap.add_argument('--out', help='write calibration.json here')
a = ap.parse_args()
files = sorted(Path(a.traces).glob('cal_*.npz'))
loaded = [dict(np.load(f)) for f in files]
d = {k: np.concatenate([x[k] for x in loaded]) for k in loaded[0]}
if a.held_out_only:
    d = {k: v[d['held_out']] for k, v in d.items()}
out = a.out
print(f"{len(files)} episodes ({len(set(d['episode'].tolist()))} used"
      f"{', held-out only' if a.held_out_only else ''})")
# metric -> (array key, thresholds used by the policy)
METRICS = {'score_4': ('s4', [0.25, 0.30, 0.40, 0.45, 0.50, 0.60]),
           'score_24': ('s24', [0.45]),
           'units_24': ('u24', [2.0])}
truth = {'s4': d['truth4'] > 0, 's24': d['truth24'] > 0, 'u24': d['truth24'] > 0}
print(f"origins {len(d['step'])}, steps {d['step'].min()}..{d['step'].max()}")


def auc(score, y):
    order = np.argsort(score, kind='mergesort')
    r = np.empty(len(score)); r[order] = np.arange(1, len(score) + 1)
    pos = y.sum(); neg = len(y) - pos
    return (r[y].sum() - pos * (pos + 1) / 2) / (pos * neg) if pos and neg else float('nan')


factors = {}
for metric, (key, taus) in METRICS.items():
    print(f'\n== {metric} (thresholds {taus})')
    factors[metric] = {}
    for i, p in enumerate(PRODUCTS):
        o, n, y = d[f'old_{key}'][:, i], d[f'new_{key}'][:, i], truth[key][:, i]
        ratios, lines = [], []
        for tau in taus:
            rate = (o >= tau).mean()
            if rate <= 0 or rate >= 1:
                lines.append(f'    tau {tau}: old never/always fires ({rate:.3%})')
                continue
            n_tau = float(np.quantile(n, 1 - rate))
            prec_o = y[o >= tau].mean()
            prec_n_same = y[n >= tau].mean() if (n >= tau).any() else float('nan')
            prec_n = y[n >= n_tau].mean() if (n >= n_tau).any() else float('nan')
            if n_tau > 0:
                ratios.append(tau / n_tau)
            lines.append(f'    tau {tau}: old fires {rate:6.2%} prec {prec_o:.2f} | new at same tau fires '
                         f'{(n >= tau).mean():6.2%} prec {prec_n_same:.2f} | new tau for same rate {n_tau:.3f} prec {prec_n:.2f}')
        f = float(np.exp(np.mean(np.log(ratios)))) if ratios else 1.0
        factors[metric][p] = round(f, 4)
        after = ' '.join(f'{(n * f >= t).mean():.2%}/{(o >= t).mean():.2%}' for t in taus)
        print(f'  {p:11s} base {y.mean():5.1%}  AUC old {auc(o, y):.3f} new {auc(n, y):.3f}  factor {f:.3f}  '
              f'fire new*f/old: {after}')
        for line in lines:
            print(line)

if out:
    json.dump({'factors': factors,
               'note': 'quantile-matched to ttm_c256_h96_ft_2026-09-13 on the 60 held-out 2026-09-23 episodes '
                       'of ttm_c256_h96_ft_2026-09-23 plus others; per-product factor = geometric mean of '
                       'old_threshold / new_quantile over the policy thresholds (arena/calib_fit.py)'},
              open(out, 'w'), indent=1)
    print('\nwrote', out)
