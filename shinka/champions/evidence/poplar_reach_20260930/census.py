"""Farm census of our lost ladder games (rival minus us) at days 5/10/15/20/25, from the recorded replays.
    python census.py INDEX GZ_DIR [GZ_DIR ...]"""
import gzip, json, sys, collections
from pathlib import Path
idx = [g for g in json.load(open(sys.argv[1])) if g["res"] in "LT"]
dirs = [Path(d) for d in sys.argv[2:]]
DAYS = (5, 10, 15, 20, 25)


def census(farm):
    c = collections.Counter()
    c["money"] = farm.get("money", 0)
    c["quadrants"] = len(farm.get("unlocked_quadrants") or [])
    c["hands"] = len(farm.get("hands") or [])
    for row in farm.get("tiles") or []:
        for t in row:
            if isinstance(t, dict):
                if t.get("crop"): c["crop:" + t["crop"]] += 1
                if t.get("animal"): c["animal:" + t["animal"]] += 1
                if t.get("kind") and not t.get("crop") and not t.get("animal"): c["kind:" + t["kind"]] += 1
    return c


out = []
for g in idx:
    f = next((d / f"episode-{g['id']}-replay.json.gz" for d in dirs if (d / f"episode-{g['id']}-replay.json.gz").exists()), None)
    if f is None:
        continue
    r = json.loads(gzip.decompress(f.read_bytes()))
    me = g["seat"]
    rec = {"id": g["id"], "diff": g["diff"], "ours": g["ours"], "days": {}}
    for day in DAYS:
        t = min(day * 24, len(r["steps"]) - 1)
        farms = r["steps"][t][me]["observation"]["farms"]
        a, b = census(farms[me]), census(farms[1 - me])
        rec["days"][day] = {k: b.get(k, 0) - a.get(k, 0) for k in set(a) | set(b)}
        rec["days"][day]["_us_quadrants"], rec["days"][day]["_rival_quadrants"] = a["quadrants"], b["quadrants"]
    out.append(rec)
print(len(out), "games")
buckets = {"close (> -$500)": lambda d: d > -500, "mid (-$500..-$2000)": lambda d: -2000 < d <= -500, "big (<= -$2000)": lambda d: d <= -2000}
for name, sel in buckets.items():
    rs = [x for x in out if sel(x["diff"])]
    if not rs:
        continue
    print(f"\n== {name}: {len(rs)} games (rival minus us, mean)")
    keys = set()
    for x in rs:
        for d in DAYS: keys |= set(x["days"][d])
    for k in sorted(keys):
        vals = [sum(x["days"][d].get(k, 0) for x in rs) / len(rs) for d in DAYS]
        if max(abs(v) for v in vals) >= (100 if k == "money" else 0.5):
            print(f"  {k:22s} " + " ".join(f"d{d}:{v:+8.1f}" for d, v in zip(DAYS, vals)))
json.dump(out, open("/home/alex/kagg-evo/lost5/census.json", "w"))
