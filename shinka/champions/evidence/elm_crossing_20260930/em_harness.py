"""Drive RivalEmulator over recorded games from our own seat (repair off vs on) and score it against the rival's
recorded actions.   python em_harness.py HAZEL_RUNTIME_DIR INDEX_JSON GZ_DIR REPAIR [ENGINES,...]"""
import gzip, json, sys, time
from pathlib import Path
rt = Path(sys.argv[1]); sys.path.insert(0, str(rt)); sys.path.insert(0, str(Path(__file__).parent))
import rival_emulator as R                       # the patched module next to this file
index, gz, repair = json.load(open(sys.argv[2])), Path(sys.argv[3]), int(sys.argv[4])
names = (sys.argv[5] if len(sys.argv) > 5 else "tetsutani_demand,tetsutani_demand_0927,haodou_ledger_0928,leo_pi").split(",")
em = R.RivalEmulator(rt / "engines", names, 12, True, repair)
tot = {"steps_alive": 0, "pred": 0, "pred_ok": 0, "games": 0}
for g in index:
    f = gz / f"episode-{g['id']}-replay.json.gz"
    if not f.exists():
        continue
    r = json.loads(gzip.decompress(f.read_bytes()))
    me, cfg = g["seat"], r.get("configuration") or {}
    t0 = time.time(); alive_until = 0; pred = ok = 0
    for t in range(len(r["steps"]) - 1):
        obs = dict(r["steps"][t][me]["observation"]); obs["player"] = me; obs.setdefault("step", t)
        em.begin(obs, cfg)
        if em.alive():
            alive_until = t
        pr = em.prediction()
        truth = r["steps"][t + 1][1 - me].get("action")
        if pr is not None:
            pred += 1
            ok += json.dumps(R._plain(pr), sort_keys=True) == json.dumps(R._plain(truth), sort_keys=True)
        em.finish(r["steps"][t + 1][me].get("action") or {})
    names_alive = [(h.name, h.matched, h.repairs, h.dropped_at) for h in em.hypotheses if h.matched > 0]
    print(json.dumps({"id": g["id"], "diff": g["diff"], "alive_until": alive_until, "predicted_steps": pred,
                      "prediction_exact": ok, "repairs": em.repairs, "sec": round(time.time() - t0, 1),
                      "hyp": names_alive}), flush=True)
    tot["steps_alive"] += alive_until; tot["pred"] += pred; tot["pred_ok"] += ok; tot["games"] += 1
print("TOTAL", json.dumps(tot))
