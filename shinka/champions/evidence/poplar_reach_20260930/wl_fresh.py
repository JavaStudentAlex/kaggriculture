"""Per-opponent W-L of Laurel (wheat8), Elm (candB), F (candF) and F2 (candF2) in the full-graph fresh arena games."""
import json, collections
rows = [json.loads(l) for l in open("/home/alex/kagg-evo/kad-20260929/run/games/fresh0930.jsonl")]
labels = ("wheat8", "candB", "candF", "candF2")
wl = collections.defaultdict(collections.Counter)
for r in rows:
    lab, opp = r["tag"].split("@", 1)
    if lab not in labels or not r.get("rewards"):
        continue
    s = r["a_seat"]; m = r["rewards"][s] - r["rewards"][1 - s]
    wl[opp][lab + ("W" if m > 0 else "L" if m < 0 else "T")] += 1
fam = {"tetsutani_demand", "tetsutani_demand_0927", "haodou_ledger_0928", "leo_pi"}
print("%-26s %8s %8s %8s %8s" % ("opponent", "Laurel", "Elm", "F", "F2"))
tot = {"0927 family": collections.Counter(), "other engines": collections.Counter()}
for opp in sorted(wl, key=lambda o: (o not in fam, o)):
    c = wl[opp]
    print("%-26s %8s %8s %8s %8s" % (opp, *("%d-%d" % (c[l + "W"], c[l + "L"]) for l in labels)))
    g = "0927 family" if opp in fam else "other engines"
    for l in labels:
        tot[g][l + "W"] += c[l + "W"]; tot[g][l + "L"] += c[l + "L"]
for g, t in tot.items():
    print("%-26s %8s %8s %8s %8s" % (g, *("%d-%d" % (t[l + "W"], t[l + "L"]) for l in labels)))
