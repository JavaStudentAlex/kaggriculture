"""The rival model as it runs in play, on recorded games held out one at a time.

    python eval_online.py <features dir> <match dir> <families.json> [...] [--labels opening|match]
                          [--confidence 0.8] [--tau 1.0] [--min-games 3]

For each game, a model is fitted on the other games (fit_rival_model.build) and RivalModel observes the
held-out rival step by step. Per checkpoint: the share of games where the model has decided (top belief
>= confidence) and is right, decided and wrong, or is undecided; then the same per cluster.
"""
import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hazel_runtime.rival_model import RivalModel  # noqa: E402
from cluster_speed import labels  # noqa: E402
from fit_rival_model import build  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('features')
    ap.add_argument('matches')
    ap.add_argument('families', nargs='+')
    ap.add_argument('--labels', default='opening')
    ap.add_argument('--confidence', type=float, default=0.8)
    ap.add_argument('--tau', type=float, default=1.0)
    ap.add_argument('--min-games', type=int, default=3)
    ap.add_argument('--absolute', action='store_true')
    args = ap.parse_args()
    feats = {int(p.stem): json.load(open(p)) for p in Path(args.features).glob('*.json')}
    lab = labels(args.matches, args.families, args.labels)
    games = sorted(g for g in feats if g in lab)
    counts = collections.Counter(lab[g][0] for g in games)
    truth = {g: lab[g][0] if counts[lab[g][0]] >= args.min_games else 'other' for g in games}
    print(f"{len(games)} games:", ', '.join(f'{c} {n}' for c, n in collections.Counter(truth.values()).most_common()))
    stats = collections.defaultdict(collections.Counter)       # checkpoint -> outcome counts
    per = collections.defaultdict(collections.Counter)         # (checkpoint, cluster) -> outcome counts
    for g in games:
        model = build({h: feats[h] for h in games if h != g}, {h: truth[h] for h in games if h != g},
                      not args.absolute)
        rm = RivalModel(model, tau=args.tau)
        for t, vec in enumerate(feats[g]['rival']):
            rm.observe(t, vec, feats[g]['ours'][t])
            if t in rm.checkpoints:
                decision = rm.decide(args.confidence)
                outcome = 'undecided' if decision is None else 'right' if decision == truth[g] else 'wrong'
                stats[t][outcome] += 1
                per[(t, truth[g])][outcome] += 1
    clusters = [c for c, _ in collections.Counter(truth.values()).most_common()]
    print(f"confidence {args.confidence}, tau {args.tau}: per checkpoint decided-right / decided-wrong / undecided")
    for t in sorted(stats):
        n = sum(stats[t].values())
        row = f"day {t // 24:2d} (step {t:3d}): right {stats[t]['right'] / n:.2f} wrong {stats[t]['wrong'] / n:.2f} " \
              f"undecided {stats[t]['undecided'] / n:.2f} |"
        for c in clusters:
            k = per[(t, c)]
            row += f" {c} {k['right']}/{k['wrong']}/{k['undecided']}"
        print(row)


if __name__ == '__main__':
    main()
