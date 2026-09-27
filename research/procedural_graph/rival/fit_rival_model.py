"""Fit the rival model file (hazel_runtime/rival_model.py) from labelled rival trajectories.

    python fit_rival_model.py <out.json> <features dir> <match dir> <families.json> [...] [--min-games N]

Labels as in cluster_speed.py (the public bundle matching the rival's moves longest, else the opening).
Clusters with fewer than --min-games games (default 3) merge into 'other'. Every game becomes one
prototype per checkpoint (standardized trajectory features); the file also holds each checkpoint's
mean and sd. `RivalModel` classifies a live rival against these prototypes.
"""
import argparse
import collections
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hazel_runtime.rival_model import FIELDS, trajectory_features  # noqa: E402
from cluster_speed import CHECKPOINTS, DROP, labels  # noqa: E402


def build(feats, cluster, relative=True):
    """The model dict from {game: features} and {game: cluster}."""
    games = sorted(feats)
    model = {'clusters': sorted(set(cluster.values())), 'checkpoints': list(CHECKPOINTS), 'drop': sorted(DROP),
             'relative': relative,
             'games': {c: sum(1 for g in games if cluster[g] == c) for c in sorted(set(cluster.values()))},
             'scale': {}, 'prototypes': {}}
    for T in CHECKPOINTS:
        X = {g: trajectory_features(feats[g]['rival'], T, DROP, feats[g]['ours'] if relative else None) for g in games}
        dims = len(X[games[0]])
        mean = [sum(X[g][d] for g in games) / len(games) for d in range(dims)]
        sd = [max(1e-6, math.sqrt(sum((X[g][d] - mean[d]) ** 2 for g in games) / len(games))) for d in range(dims)]
        model['scale'][str(T)] = {'mean': [round(m, 6) for m in mean], 'sd': [round(v, 6) for v in sd]}
        model['prototypes'][str(T)] = [[cluster[g], [round((X[g][d] - mean[d]) / sd[d], 4) for d in range(dims)]]
                                       for g in games]
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('features')
    ap.add_argument('matches')
    ap.add_argument('families', nargs='+')
    ap.add_argument('--labels', default='opening', help="'opening' (the rival's opening family) or 'match'")
    ap.add_argument('--min-games', type=int, default=3)
    ap.add_argument('--absolute', action='store_true', help='the rival alone, not the rival minus us')
    args = ap.parse_args()
    feats = {int(p.stem): json.load(open(p)) for p in Path(args.features).glob('*.json')}
    lab = labels(args.matches, args.families, args.labels)
    games = sorted(g for g in feats if g in lab)
    counts = collections.Counter(lab[g][0] for g in games)
    cluster = {g: lab[g][0] if counts[lab[g][0]] >= args.min_games else 'other' for g in games}
    assert list(feats[games[0]]['fields']) == list(FIELDS), 'features were built with other fields'
    model = build({g: feats[g] for g in games}, cluster, not args.absolute)
    Path(args.out).write_text(json.dumps(model, separators=(',', ':')))
    print(f"{len(games)} games -> {args.out}: " + ', '.join(f'{c} {n}' for c, n in model['games'].items()))


if __name__ == '__main__':
    main()
