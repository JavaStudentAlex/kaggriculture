"""At the step where the emulator loses a rival for good: does the rival's TRUE recorded action (whole, or its true
market list with the predicted farm actions) reproduce the observation from the best hypothesis's rebuilt state?
    python em_debug.py HAZEL_RUNTIME_DIR INDEX_JSON GZ_DIR REPAIR EMULATOR_DIR [ids,...]"""
import copy, gzip, json, sys
from pathlib import Path
rt = Path(sys.argv[1]); sys.path.insert(0, str(rt)); sys.path.insert(0, sys.argv[5])
import rival_emulator as R
index, gz, repair = json.load(open(sys.argv[2])), Path(sys.argv[3]), int(sys.argv[4])
only = {int(x) for x in sys.argv[6].split(",")} if len(sys.argv) > 6 else None
names = "tetsutani_demand,tetsutani_demand_0927,haodou_ledger_0928,leo_pi".split(",")
em = R.RivalEmulator(rt / "engines", names, 12, True, repair)


def check(p, h, me, rival_action, obs):
    actions, privates = [None, None], [None, None]
    actions[me], actions[1 - me] = p["action"], rival_action
    privates[me], privates[1 - me] = p["private"], h.private
    farms, market, after = R.simulate(p["public"], privates, actions, p["configuration"])
    seen = obs.get("farms")
    farm_ok = [not R._differs({k: farms[i][k] for k in R.PUBLIC_FARM}, {k: seen[i][k] for k in R.PUBLIC_FARM})
               for i in range(2)]
    bad = {k: (farms[1 - me][k], seen[1 - me][k]) for k in R.PUBLIC_FARM
           if R._differs(farms[1 - me][k], seen[1 - me][k]) and k != "tiles"}
    inv = {k: (market["inventory"].get(k), obs["market"]["inventory"].get(k)) for k in obs["market"]["inventory"]
           if market["inventory"].get(k) != obs["market"]["inventory"].get(k)}
    return {"agrees": R.agrees(copy.deepcopy(farms), market, obs), "farm_ok": farm_ok, "rival_diff": str(bad)[:300],
            "inv_diff": inv}


for g in index:
    if only and g["id"] not in only:
        continue
    f = gz / f"episode-{g['id']}-replay.json.gz"
    if not f.exists():
        continue
    r = json.loads(gzip.decompress(f.read_bytes()))
    me, cfg = g["seat"], r.get("configuration") or {}
    em.reset(); em.last_step = -1
    for t in range(len(r["steps"]) - 1):
        obs = dict(r["steps"][t][me]["observation"]); obs["player"] = me; obs.setdefault("step", t)
        obs_p = R._plain(obs)
        live = [(h, copy.deepcopy(h.private), copy.deepcopy(h.predicted)) for h in em.alive()]
        p = copy.deepcopy(em.pending)
        em.begin(obs, cfg)
        if live and not em.alive() and t > 0:
            truth = R._plain(r["steps"][t][1 - me].get("action") or {})
            h, priv, pred = max(live, key=lambda x: x[0].matched)
            h.private = priv
            print(g["id"], "lost at", t - 1, h.name, "matched", h.matched, "repairs", h.repairs)
            print("   true action      :", check(p, h, me, truth, obs_p))
            print("   pred farm+true mkt:", check(p, h, me, dict(pred, market=truth.get("market")), obs_p))
            print("   predicted        :", check(p, h, me, pred, obs_p))
            break
        em.finish(r["steps"][t + 1][me].get("action") or {})
