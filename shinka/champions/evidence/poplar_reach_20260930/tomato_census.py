"""Tomatoes in our ladder games: rivals who beat us vs rivals we beat, from the recorded replays.
Per seat: tomato tiles at days 10/15/20/25 and the first day a tomato tile appears."""
import gzip, json, statistics as st
from pathlib import Path
K = Path("/home/alex/kagg-evo")
dirs = [K / "lost5/gz", K / "today0930/gz", K / "replays_jk2/gz"]
games = {}
for f in (K / "lost5/index_lost5.json", Path("/tmp/index_today_all.json"), Path("/tmp/index_jk2_new.json")):
    if f.exists():
        for g in json.load(open(f)):
            games[g["id"]] = g


def tomatoes(farm):
    return sum(1 for row in farm.get("tiles") or [] for t in row if isinstance(t, dict) and t.get("crop") == "TOMATO")


rows = []
for gid, g in games.items():
    f = next((d / f"episode-{gid}-replay.json.gz" for d in dirs if (d / f"episode-{gid}-replay.json.gz").exists()), None)
    if f is None:
        continue
    r = json.loads(gzip.decompress(f.read_bytes()))
    me = g["seat"]; steps = r["steps"]
    first = [None, None]; at = {}
    for t in range(0, len(steps), 12):
        farms = steps[t][me]["observation"]["farms"]
        for s in (0, 1):
            if first[s] is None and tomatoes(farms[s]) > 0:
                first[s] = t // 24
        if t // 24 in (10, 15, 20, 25) and t % 24 == 0:
            at[t // 24] = (tomatoes(farms[me]), tomatoes(farms[1 - me]))
    rows.append({"res": g["res"], "diff": g["diff"], "us_first": first[me], "rival_first": first[1 - me], "at": at})


def report(name, rs):
    if not rs:
        return
    n = len(rs)
    rt = sum(r["rival_first"] is not None for r in rs); ut = sum(r["us_first"] is not None for r in rs)
    print(f"\n{name}: {n} games | rival grew tomatoes in {rt} ({100 * rt / n:.0f}%), we did in {ut} ({100 * ut / n:.0f}%)")
    rf = [r["rival_first"] for r in rs if r["rival_first"] is not None]; uf = [r["us_first"] for r in rs if r["us_first"] is not None]
    if rf: print(f"  first tomato day: rival median {st.median(rf):.0f} (25%: {sorted(rf)[len(rf) // 4]}), us median {st.median(uf) if uf else '-'}")
    for d in (10, 15, 20, 25):
        v = [r["at"][d] for r in rs if d in r["at"]]
        if v: print(f"  day {d}: tomato tiles us {sum(a for a, b in v) / len(v):5.1f} | rival {sum(b for a, b in v) / len(v):5.1f}")


report("LOST, margin > -$500", [r for r in rows if r["res"] == "L" and r["diff"] > -500])
report("LOST, -$500..-$2000", [r for r in rows if r["res"] == "L" and -2000 < r["diff"] <= -500])
report("LOST, <= -$2000", [r for r in rows if r["res"] == "L" and r["diff"] <= -2000])
report("WON", [r for r in rows if r["res"] == "W"])
