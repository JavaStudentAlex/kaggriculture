"""Lost/tied games of our last five submissions, replayed against each candidate (fast graphs): wins per original agent
and paired comparisons. usage: lost5_report.py INDEX label=jsonl[,jsonl...] ..."""
import json, sys, collections
idx = {g["id"]: g for g in json.load(open(sys.argv[1]))}
res = {}
for spec in sys.argv[2:]:
    label, files = spec.split("=", 1)
    for f in files.split(","):
        for l in open(f):
            r = json.loads(l)
            lab, opp = r["tag"].split("@", 1)
            if lab != label.split(":")[-1] or not opp.startswith("replay_") or r.get("errors") or not r.get("rewards"):
                continue
            eid = int(opp[len("replay_"):])
            if eid not in idx:
                continue
            s = int(r["a_seat"])
            res.setdefault(label.split(":")[0], {})[eid] = r["rewards"][s] - r["rewards"][1 - s]
names = {"av": "Aspen Vale", "jk": "Juniper Knoll", "hd": "Hawthorn Dale", "lf": "Laurel Field", "elm": "Elm Crossing"}
labels = list(res)
print(f"{'lost by':15s} {'games':>5s} " + " ".join(f"{l:>10s}" for l in labels))
for o in ["av", "jk", "hd", "lf", "elm", None]:
    ids = [i for i in idx if idx[i]["res"] in "LT" and (o is None or idx[i]["ours"] == o) and all(i in res[l] for l in labels)]
    if not ids:
        continue
    row = " ".join(f"{sum(res[l][i] > 0 for i in ids):>6d} won" for l in labels)
    print(f"{names.get(o, 'all'):15s} {len(ids):5d} {row}")
if len(labels) >= 2:
    a = labels[0]
    for b in labels[1:]:
        ids = [i for i in idx if i in res[a] and i in res[b]]
        better = sum(res[b][i] > res[a][i] for i in ids); worse = sum(res[b][i] < res[a][i] for i in ids)
        flips = sum(res[b][i] > 0 >= res[a][i] for i in ids); back = sum(res[a][i] > 0 >= res[b][i] for i in ids)
        mean = sum(res[b][i] - res[a][i] for i in ids) / max(1, len(ids))
        print(f"{b} vs {a}: {len(ids)} games, {better} better / {worse} worse, mean {mean:+.0f}; wins gained {flips}, lost {back}")
