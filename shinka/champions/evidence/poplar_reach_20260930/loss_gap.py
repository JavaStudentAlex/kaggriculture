"""Where F (the promoted champion) still loses: per-product sales gap (rival - us) in F's lost gauntlet games,
from the pool's game rows (ledger: submitted sell units and value at the quoted price, per seat)."""
import json, glob, collections
O = "/home/alex/kagg-evo/pool/outbox/"
rows = []
for f in glob.glob(O + "g_38e773f84d01d17a_vs_g_05d727c95b0055e8-*.jsonl"):
    rows += [json.loads(l) for l in open(f)]
rows = [r for r in rows if r.get("rewards") and not r.get("errors") and "g_38e773f84d01d17a" in r["a"]]
print(len(rows), "F games")
margins = {}
for d in ("/home/alex/kagg-evo/lost5/bundles", "/home/alex/kagg-evo/replays_all0930"):
    for fn in glob.glob(d + "/replay_*/SOURCE.json"):
        s = json.load(open(fn)); margins[s["name"]] = s.get("recorded_margin_for_us") or 0
def margin(r):
    s = r["a_seat"]; return r["rewards"][s] - r["rewards"][1 - s]
lost = [r for r in rows if margin(r) < 0]; won = [r for r in rows if margin(r) > 0]
print("F lost", len(lost), "won", len(won))
bins = collections.Counter()
for r in lost:
    m = -margin(r)
    bins["<250" if m < 250 else "<500" if m < 500 else "<1000" if m < 1000 else "<2000" if m < 2000 else "<5000" if m < 5000 else ">=5000"] += 1
print("F loss margins:", dict(sorted(bins.items())))
def gap(rs, key):
    tot = collections.Counter()
    for r in rs:
        s = r["a_seat"]; L = r.get("ledger") or [{}, {}]
        mine, theirs = L[s].get(key, {}), L[1 - s].get(key, {})
        for p in set(mine) | set(theirs):
            tot[p] += (theirs.get(p, 0) or 0) - (mine.get(p, 0) or 0)
    return tot
for name, rs in (("close losses (<$1000)", [r for r in lost if margin(r) > -1000]), ("big losses (>=$1000)", [r for r in lost if margin(r) <= -1000]), ("wins", won)):
    if not rs: continue
    v, u = gap(rs, "submitted_sell_value_at_quote"), gap(rs, "submitted_sell_units")
    print(f"\n{name}: {len(rs)} games; mean rival-minus-us per game  (value $ | units)")
    for p in sorted(v, key=lambda p: -abs(v[p])):
        print(f"  {p:11s} {v[p] / len(rs):+8.0f} | {u[p] / len(rs):+6.1f}")
