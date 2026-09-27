"""How early does the rival's public state tell its family? (labels from move matching, leave-one-out)

    python cluster_speed.py <features dir> <match dir> <families.json> [<families.json> ...]

Labels: the public bundle that reproduces the rival's recorded moves longest (match_ladder_games.py output
under <match dir>/<batch>/<bundle>.jsonl), mapped to a cluster by CLUSTER_OF when it matches at least
MIN_MOVES moves; otherwise 'other: <opening>' from the families files (the rival's step-0 wheat orders).
Features at checkpoint T: the rival's vector (rival_features.RIVAL_FIELDS without money and farmer
position) every 24 steps up to T, plus log money. 1-nearest-neighbour on standardized features, each
game held out in turn: accuracy per cluster, and how often the true cluster is among the 2 nearest
clusters (the candidate list).
"""
import collections
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hazel_runtime.rival_model import trajectory_features  # noqa: E402

MIN_MOVES = 100
CLUSTER_OF = {
    'tetsutani_demand': 'engine_old', 'lynn_wheat': 'engine_old', 'haideptry_shepherd_new': 'engine_old',
    'guru_master_v3': 'engine_old', 'statma_tetsutani': 'engine_old',
    'tetsutani_demand_0927': 'engine_new', 'lynn_idle': 'engine_new', 'haodou_ledger': 'engine_new',
    'statma_submit': 'statma', 'statma_ca20': 'statma', 'statma_ca25': 'statma',
    'leoprovorov_forecast': 'forecast', 'ars_herdsafe_v3': 'forecast',
    'haideptry_2965': '2965', 'haideptry_2965_0926': '2965', 'haideptry_2965_new': '2965',
    'tetsutani_shape_shop': 'shape_shop', 'tetsu_shape_shop': 'shape_shop',
    'abo_v43': 'abo', 'abo_v55': 'abo', 'abo_v57': 'abo', 'abo_v57_open13': 'abo', 'abo_v54': 'abo', 'abo_v56': 'abo',
}
CHECKPOINTS = (24, 48, 72, 96, 144, 192, 216, 288, 360)
DROP = {'money', 'farmer_x', 'farmer_y', 'hires_today'}


OPENING = {'tetsutani': 'tetsutani_like', '2965': '2965', 'forecast': 'forecast_like', 'shape shop': 'shape_shop'}


def labels(match_dir, family_files, mode='match'):
    """{episode: (cluster, best bundle, its matched moves)}. mode 'opening': the cluster is the rival's
    opening family (OPENING, else 'other'), whatever bundle matches it."""
    fam = {}
    for f in family_files:
        for name, (family, _margin, _team) in json.load(open(f)).items():
            fam[int(name.split('_')[1])] = family.split(' (')[0]
    best = {}
    for path in Path(match_dir).rglob('*.jsonl'):
        for line in open(path):
            r = json.loads(line)
            n = r.get('equal_steps', 0) or 0
            if n > best.get(r['episode'], (0, None))[0]:
                best[r['episode']] = (n, r['bundle'])
    out = {}
    for ep, family in fam.items():
        n, bundle = best.get(ep, (0, None))
        if mode == 'opening':
            out[ep] = (OPENING.get(family, 'other'), bundle, n)
            continue
        cluster = CLUSTER_OF.get(bundle) if n >= MIN_MOVES else None
        out[ep] = (cluster or f'other: {family}', bundle, n)
    return out


def main():
    feats = {int(p.stem): json.load(open(p)) for p in Path(sys.argv[1]).glob('*.json')}
    lab = labels(sys.argv[2], sys.argv[3:])
    games = [g for g in feats if g in lab]
    counts = collections.Counter(lab[g][0] for g in games)
    print(f'{len(games)} games; clusters:', ', '.join(f'{c} {n}' for c, n in counts.most_common()))
    for T in CHECKPOINTS:
        X = {g: trajectory_features(feats[g]['rival'], T, DROP) for g in games}
        dims = len(next(iter(X.values())))
        mean = [sum(X[g][d] for g in games) / len(games) for d in range(dims)]
        sd = [max(1e-6, math.sqrt(sum((X[g][d] - mean[d]) ** 2 for g in games) / len(games))) for d in range(dims)]
        Z = {g: [(X[g][d] - mean[d]) / sd[d] for d in range(dims)] for g in games}
        hit, top2 = collections.Counter(), collections.Counter()
        for g in games:
            dist = sorted((sum((a - b) ** 2 for a, b in zip(Z[g], Z[h])), lab[h][0]) for h in games if h != g)
            nearest = []
            for _, c in dist:
                if c not in nearest:
                    nearest.append(c)
                if len(nearest) == 2:
                    break
            hit[lab[g][0]] += nearest[0] == lab[g][0]
            top2[lab[g][0]] += lab[g][0] in nearest
        big = [c for c, n in counts.most_common() if n >= 3]
        print(f'step {T:3d} (day {T // 24:2d}): 1-NN {sum(hit.values()) / len(games):.2f}, true cluster in top 2 '
              f'{sum(top2.values()) / len(games):.2f} | ' + ', '.join(f'{c} {hit[c]}/{counts[c]}' for c in big))


if __name__ == '__main__':
    main()
