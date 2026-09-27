"""The rival_counter classes on Alder Ford's recorded games, and what each backtested variant gained per class.

    python class_value.py <features dir> <families.json> [...] --backtest <run>/games/<name>.jsonl:<base label> [...]

For each game (features from rival_features.py) MirrorTracker runs over both farms step by step, as in
play: its class and the step it was decided. Then, per backtest file and base label, each other label's
change against the base on the same replays, per class (mean, better / worse, W-L-T of both).
"""
import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hazel_runtime.rival_model import MirrorTracker  # noqa: E402


def classify(game):
    mt = MirrorTracker()
    for t, (r, o) in enumerate(zip(game['rival'], game['ours'])):
        c = mt.observe(t, r[:-2], o[:-2])
        if c is not None:
            return c, t
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('features')
    ap.add_argument('families', nargs='+')
    ap.add_argument('--backtest', nargs='*', default=[])
    args = ap.parse_args()
    fam = {}
    for f in args.families:
        for name, (family, _m, _t) in json.load(open(f)).items():
            fam[int(name.split('_')[1])] = family.split(' (')[0]
    cls = {}
    table = collections.defaultdict(collections.Counter)
    for p in Path(args.features).glob('*.json'):
        g = json.load(open(p))
        c, t = classify(g)
        cls[int(p.stem)] = c
        table[fam.get(int(p.stem), '?')][c] += 1
    classes = sorted({c for c in cls.values()}, key=str)
    print(f'{len(cls)} games: rival opening family x class')
    print(f"{'family':12s} " + ' '.join(f'{str(c):>14s}' for c in classes))
    for f, row in sorted(table.items()):
        print(f"{f:12s} " + ' '.join(f'{row[c]:14d}' for c in classes))
    rec = lambda v: f"{sum(x > 0 for x in v)}-{sum(x < 0 for x in v)}-{sum(x == 0 for x in v)}"
    for spec in args.backtest:
        path, base = spec.rsplit(':', 1)
        m = collections.defaultdict(dict)
        for line in open(path):
            r = json.loads(line)
            rw = r.get('rewards')
            if r.get('errors') or not rw or None in rw or r.get('statuses') != ['DONE', 'DONE']:
                continue
            label, opp = r['tag'].split('@', 1)
            if opp.startswith('replay_'):
                s = r['a_seat']
                m[label][int(opp.split('_')[1])] = rw[s] - rw[1 - s]
        print(f"\n{Path(path).name} against {base}: change a game per class (better/worse; record base -> label)")
        for label in sorted(m):
            if label == base:
                continue
            row = f"  {label:14s}"
            for c in classes:
                eps = [e for e in m[label] if e in m[base] and cls.get(e) == c]
                if not eps:
                    continue
                d = [m[label][e] - m[base][e] for e in eps]
                row += (f" | {c}: {sum(d) / len(d):+.0f} ({sum(x > 0 for x in d)}/{sum(x < 0 for x in d)}; "
                        f"{rec([m[base][e] for e in eps])}->{rec([m[label][e] for e in eps])})")
            print(row)


if __name__ == '__main__':
    main()
