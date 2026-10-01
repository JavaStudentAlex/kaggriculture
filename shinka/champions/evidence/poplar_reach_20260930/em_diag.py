"""Where does the rival emulator lose a rival for good? For each recorded game, the step at which the last live
hypothesis died (repair on), and how the best hypothesis's predicted action differed from the rival's recorded one.
    python em_diag.py HAZEL_RUNTIME_DIR INDEX_JSON GZ_DIR REPAIR EMULATOR_DIR"""
import copy, gzip, json, sys
from pathlib import Path
rt = Path(sys.argv[1]); sys.path.insert(0, str(rt)); sys.path.insert(0, sys.argv[5])
import rival_emulator as R
index, gz, repair = json.load(open(sys.argv[2])), Path(sys.argv[3]), int(sys.argv[4])
names = "tetsutani_demand,tetsutani_demand_0927,haodou_ledger_0928,leo_pi".split(",")
em = R.RivalEmulator(rt / "engines", names, 12, True, repair)


def short(v, n=260):
    s = json.dumps(R._plain(v), sort_keys=True, separators=(",", ":"))
    return s if len(s) <= n else s[:n] + "..."


for g in index:
    f = gz / f"episode-{g['id']}-replay.json.gz"
    if not f.exists():
        continue
    r = json.loads(gzip.decompress(f.read_bytes()))
    me, cfg = g["seat"], r.get("configuration") or {}
    before = None
    for t in range(len(r["steps"]) - 1):
        obs = dict(r["steps"][t][me]["observation"]); obs["player"] = me; obs.setdefault("step", t)
        live_before = [(h, copy.deepcopy(h.predicted)) for h in em.alive()]
        em.begin(obs, cfg)
        if live_before and not em.alive() and t > 0:
            h, pred = max(live_before, key=lambda x: x[0].matched)
            truth = R._plain(r["steps"][t][1 - me].get("action") or {})   # the rival's action from obs t-1
            pred = pred if isinstance(pred, dict) else {}
            keys = sorted(set(pred) | set(truth))
            diff = {k: (short(pred.get(k)), short(truth.get(k))) for k in keys
                    if json.dumps(R._plain(pred.get(k)), sort_keys=True) != json.dumps(R._plain(truth.get(k)), sort_keys=True)}
            print(json.dumps({"id": g["id"], "diff": g["diff"], "lost_at": t - 1, "best": h.name, "matched": h.matched,
                              "repairs": h.repairs, "differs": diff}), flush=True)
            break
        em.finish(r["steps"][t + 1][me].get("action") or {})
    else:
        print(json.dumps({"id": g["id"], "diff": g["diff"], "lost_at": None if em.alive() else "never_matched"}), flush=True)
