"""Per-opponent W-L (full-graph fresh arena, 12 engines x 20) of Laurel, Elm, F, F2, F2t from the cached games file and
any extra pool outbox files. usage: wl2.py [extra.jsonl ...]"""
import json, sys, collections
files = ["/home/alex/kagg-evo/kad-20260929/run/games/fresh0930.jsonl"] + sys.argv[1:]
labels = ("wheat8", "candB", "candF", "candF2", "candF2t")
names = {"wheat8": "Laurel", "candB": "Elm", "candF": "F", "candF2": "F2", "candF2t": "F2t"}
wl, marg = collections.defaultdict(collections.Counter), collections.defaultdict(dict)
for f in files:
    for l in open(f):
        r = json.loads(l)
        lab, opp = r["tag"].split("@", 1)
        if lab not in labels or not r.get("rewards") or r.get("errors"):
            continue
        s = r["a_seat"]; m = r["rewards"][s] - r["rewards"][1 - s]
        key = (opp, r["seed"], s)
        if key in marg[lab]:
            continue
        marg[lab][key] = m
        wl[opp][lab + ("W" if m > 0 else "L" if m < 0 else "T")] += 1
present = [l for l in labels if marg[l]]
fam = {"tetsutani_demand", "tetsutani_demand_0927", "haodou_ledger_0928", "leo_pi"}
print("%-24s" % "opponent" + "".join("%9s" % names[l] for l in present))
tot = {"0927 family": collections.Counter(), "other engines": collections.Counter()}
for opp in sorted(wl, key=lambda o: (o not in fam, o)):
    c = wl[opp]
    print("%-24s" % opp + "".join("%9s" % ("%d-%d" % (c[l + "W"], c[l + "L"])) for l in present))
    for l in present:
        g = "0927 family" if opp in fam else "other engines"
        tot[g][l + "W"] += c[l + "W"]; tot[g][l + "L"] += c[l + "L"]
for g, t in tot.items():
    print("%-24s" % g + "".join("%9s" % ("%d-%d" % (t[l + "W"], t[l + "L"])) for l in present))
print("%-24s" % "games / mean margin" + "".join("%9s" % ("%d/%+.0f" % (len(marg[l]), sum(marg[l].values()) / max(1, len(marg[l])))) for l in present))
for a, b in (("candF2", "candF2t"), ("candB", "candF2"), ("candB", "candF2t")):
    if marg[a] and marg[b]:
        ks = [k for k in marg[b] if k in marg[a]]
        up = sum(marg[b][k] > marg[a][k] for k in ks); dn = sum(marg[b][k] < marg[a][k] for k in ks)
        print(f"{names[b]} vs {names[a]}: {len(ks)} games, {up} better / {dn} worse, mean {sum(marg[b][k] - marg[a][k] for k in ks) / max(1, len(ks)):+.0f}")
