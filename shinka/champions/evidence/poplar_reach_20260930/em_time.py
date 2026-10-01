"""Per-step cost of RivalEmulator.begin() (verify + repairs + every live engine's move) over recorded games.
    python em_time.py HAZEL_RUNTIME_DIR INDEX_JSON GZ_DIR REPAIR EMULATOR_DIR"""
import gzip, json, sys, time
from pathlib import Path
rt = Path(sys.argv[1]); sys.path.insert(0, str(rt)); sys.path.insert(0, sys.argv[5])
import rival_emulator as R
index, gz, repair = json.load(open(sys.argv[2])), Path(sys.argv[3]), int(sys.argv[4])
names = "tetsutani_demand,tetsutani_demand_0927,haodou_ledger_0928,leo_pi".split(",")
em = R.RivalEmulator(rt / "engines", names, 12, True, repair)
allt = []
for g in index:
    f = gz / f"episode-{g['id']}-replay.json.gz"
    if not f.exists():
        continue
    r = json.loads(gzip.decompress(f.read_bytes()))
    me, cfg = g["seat"], r.get("configuration") or {}
    ts = []
    for t in range(len(r["steps"]) - 1):
        obs = dict(r["steps"][t][me]["observation"]); obs["player"] = me; obs.setdefault("step", t)
        t0 = time.perf_counter()
        em.begin(obs, cfg)
        ts.append((time.perf_counter() - t0) * 1000)
        em.finish(r["steps"][t + 1][me].get("action") or {})
    ts0 = ts[1:]
    allt += ts0
    print(json.dumps({"id": g["id"], "max_ms": round(max(ts0)), "p99_ms": round(sorted(ts0)[int(len(ts0) * .99)]),
                      "over100": sum(x > 100 for x in ts0), "over200": sum(x > 200 for x in ts0),
                      "sum_s": round(sum(ts0) / 1000, 1), "wide": getattr(em, "wide_used", None)}), flush=True)
allt.sort()
print("ALL", json.dumps({"steps": len(allt), "max_ms": round(allt[-1]), "p99_ms": round(allt[int(len(allt) * .99)]),
                         "p999_ms": round(allt[int(len(allt) * .999)]), "over100": sum(x > 100 for x in allt),
                         "over200": sum(x > 200 for x in allt)}))
