"""Fit a calibration.json that makes a new predictor's scores cross the policy's thresholds as
often as the reference predictor's do, per product (quantile matching), and report precision and AUC.

    python arena/calib_fit.py runs/arena/<eval-id>/traces --out calibration.json [--held-out-only]
        [--group REGEX]   # only games whose group (the arena tag, or `ladder`) matches

Input: the per-episode npz files of calib_worker.py and, for provenance, the payload's jobs.json
(default: <traces>/../payload/jobs.json). --reference is the checkpoint the policy's thresholds
were tuned with (default `old`), --calibrated the checkpoint to calibrate (default `new`).
The factor of a product and metric is the geometric mean, over the policy's thresholds for that
metric, of threshold / (the calibrated model's score at the reference model's firing rate);
kagg_oracle.forecast() multiplies score_k / units_k by it when calibration.json sits next to the
checkpoint (it reads only `factors`; the other keys record where the numbers came from).
"""
import argparse
import collections
import json
import re
import sys
from pathlib import Path

import numpy as np

# metric -> (npz key, the thresholds the policy compares it with: hazel_runtime/champion.py,
# 4c front-run (_ORACLE_FRONTRUN_SCORE 0.30, batch steps 0.45 / 0.60, score_24 0.45, units_24 2.0)
# and 5 the town-shop cadence bypass (0.25 / 0.40 / 0.50)). Update them when the policy changes.
METRICS = {'score_4': ('s4', [0.25, 0.30, 0.40, 0.45, 0.50, 0.60]),
           'score_24': ('s24', [0.45]),
           'units_24': ('u24', [2.0])}
MIN_FIRES = 100          # a threshold counts only if the reference fires on this many origins
MIN_BASE = 0.005         # ... and the product is sold at >= 0.5 % of origins (no evidence: factor 1)
FACTOR_RANGE = (0.5, 2.0)


def auc(score, y):
    order = np.argsort(score, kind='mergesort')
    r = np.empty(len(score))
    r[order] = np.arange(1, len(score) + 1)
    pos = y.sum()
    neg = len(y) - pos
    return (r[y].sum() - pos * (pos + 1) / 2) / (pos * neg) if pos and neg else float('nan')


def products_of(loaded):
    """The per-product column order: recorded in every npz since 2026-09-25; older files fall back to
    mechanics.PRODUCTS (kagg_oracle's order), which needs kaggle_environments."""
    if 'products' in loaded[0]:
        return [str(p) for p in loaded[0]['products']]
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'opponent_model'))
    try:
        from mechanics import PRODUCTS
        return list(PRODUCTS)
    except ImportError:   # no kaggle_environments here: the engine's order, checked 2026-09-25
        return ['WHEAT', 'CARROT', 'TOMATO', 'STRAWBERRY', 'MELON', 'EGG', 'MILK', 'WOOL', 'FERTILIZER']


def fit(d, ref, cal, products):
    """(factors, report lines) for arrays `d` of calib_worker.py npz files."""
    truth = {'s4': d['truth4'] > 0, 's24': d['truth24'] > 0, 'u24': d['truth24'] > 0}
    factors, report = {}, []
    for metric, (key, taus) in METRICS.items():
        report.append(f'\n== {metric} (thresholds {taus})')
        factors[metric] = {}
        for i, p in enumerate(products):
            o, n, y = d[f'{ref}_{key}'][:, i], d[f'{cal}_{key}'][:, i], truth[key][:, i]
            ratios, lines = [], []
            for tau in taus:
                rate = (o >= tau).mean()
                if (o >= tau).sum() < MIN_FIRES or rate >= 1 or y.mean() < MIN_BASE:
                    lines.append(f'    tau {tau}: skipped ({ref} fires on {(o >= tau).sum()} origins, '
                                 f'{rate:.3%}; base rate {y.mean():.2%})')
                    continue
                n_tau = float(np.quantile(n, 1 - rate))
                prec_o = y[o >= tau].mean()
                prec_same = y[n >= tau].mean() if (n >= tau).any() else float('nan')
                prec_n = y[n >= n_tau].mean() if (n >= n_tau).any() else float('nan')
                if n_tau > 0:
                    ratios.append(tau / n_tau)
                lines.append(f'    tau {tau}: {ref} fires {rate:6.2%} prec {prec_o:.2f} | {cal} at same tau fires '
                             f'{(n >= tau).mean():6.2%} prec {prec_same:.2f} | {cal} tau for same rate {n_tau:.3f} '
                             f'prec {prec_n:.2f}')
            f = float(np.exp(np.mean(np.log(ratios)))) if ratios else 1.0
            if not FACTOR_RANGE[0] <= f <= FACTOR_RANGE[1]:
                lines.append(f'    factor {f:.3f} clipped to {FACTOR_RANGE}')
                f = min(max(f, FACTOR_RANGE[0]), FACTOR_RANGE[1])
            factors[metric][p] = round(f, 4)
            after = ' '.join(f'{(n * f >= t).mean():.2%}/{(o >= t).mean():.2%}' for t in taus)
            report.append(f'  {p:11s} base {y.mean():5.1%}  AUC {ref} {auc(o, y):.3f} {cal} {auc(n, y):.3f}  '
                          f'factor {f:.3f}  fire {cal}*f/{ref}: {after}')
            report.extend(lines)
    return factors, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('traces', help='directory of cal_<episode>.npz')
    ap.add_argument('--manifest', help='the payload jobs.json (default: <traces>/../payload/jobs.json)')
    ap.add_argument('--reference', default='old', help='model the policy thresholds were tuned with')
    ap.add_argument('--calibrated', default='new', help='model to calibrate')
    ap.add_argument('--held-out-only', action='store_true', help="only the calibrated model's held-out episodes")
    ap.add_argument('--group', help='only games whose group (arena tag, or `ladder`) matches this regex')
    ap.add_argument('--out', help='write calibration.json here')
    a = ap.parse_args()
    files = sorted(Path(a.traces).glob('cal_*.npz'))
    loaded = [dict(np.load(f)) for f in files]
    products = products_of(loaded)
    for x in loaded:
        x.pop('products', None)
    d = {k: np.concatenate([x[k] for x in loaded]) for k in loaded[0]}
    if 'group' not in d:   # npz written before groups existed: Kaggle replays
        d['group'] = np.full(len(d['step']), 'ladder')
    if a.held_out_only:
        d = {k: v[d['held_out']] for k, v in d.items()}
    if a.group:
        keep = np.array([bool(re.search(a.group, str(g))) for g in d['group']])
        d = {k: v[keep] for k, v in d.items()}
    episodes = len(set(d['episode'].tolist()))
    games = collections.Counter(g for g, _ in set(zip(d['group'].tolist(), d['episode'].tolist())))
    print('games by group:', dict(sorted(games.items())))
    print(f"{len(files)} episodes ({episodes} used{', held-out only' if a.held_out_only else ''}), "
          f"origins {len(d['step'])}, steps {d['step'].min()}..{d['step'].max()}")
    factors, report = fit(d, a.reference, a.calibrated, products)
    print('\n'.join(report))
    if a.out:
        manifest_path = Path(a.manifest) if a.manifest else Path(a.traces).parent / 'payload' / 'jobs.json'
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        sources = manifest.get('model_sources', {})
        doc = {'factors': factors,
               'reference': sources.get(a.reference, a.reference),
               'calibrated': sources.get(a.calibrated, a.calibrated),
               'replays': manifest.get('zip'), 'traces': manifest.get('traces'), 'group_filter': a.group,
               'games_by_group': dict(sorted(games.items())), 'episodes': episodes,
               'held_out_episodes': int(len(set(d['episode'][d['held_out']].tolist()))),
               'origins': int(len(d['step'])), 'stride': manifest.get('stride'),
               'thresholds': {m: taus for m, (_, taus) in METRICS.items()},
               'guards': {'min_fires': MIN_FIRES, 'min_base_rate': MIN_BASE, 'factor_range': FACTOR_RANGE},
               'method': 'per product and metric: geometric mean over the thresholds of threshold / the '
                         'calibrated score at the reference firing rate (arena/calib_fit.py)'}
        Path(a.out).write_text(json.dumps(doc, indent=1) + '\n')
        print('\nwrote', a.out)


if __name__ == '__main__':
    main()
