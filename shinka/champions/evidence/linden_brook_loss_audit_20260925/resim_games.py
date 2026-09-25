"""Exact per-seat accounting of Linden Brook's ladder games.

Each Kaggle replay is re-simulated through the pinned engine (kaggle-environments 1.32.7) from
its seed (info.seed) and both seats' recorded actions (the action stored at steps[t+1] is the
one taken from observation t). The engine's unit-commit, hire and land functions are wrapped so
every executed sale / purchase is logged with its seat, step and price. A game counts only if the
re-simulation reproduces both seats' money on every step.

usage: python resim_games.py <games.json> <replay dir> <out.json> [workers]
"""
import collections
import json
import sys
from multiprocessing import Pool

import kaggle_environments.envs.kaggriculture.kaggriculture as K
from kaggle_environments import make

LOG = []
CUR = {"farms": None, "t": None}

_orig_process_market = K._process_market
_orig_commit_unit = K._commit_unit
_orig_do_hire = K._do_hire
_orig_do_buy_land = K._do_buy_land


def _who(farm):
    farms = CUR["farms"]
    return 0 if farm is farms[0] else 1 if farm is farms[1] else -1


def _process_market(state, env):
    CUR["farms"] = state[0].observation.farms
    return _orig_process_market(state, env)


def _commit_unit(op, item, price, farm, private, market, shed_capacity=100):
    ok = _orig_commit_unit(op, item, price, farm, private, market, shed_capacity)
    if ok:
        LOG.append((CUR["t"], _who(farm), op, item, float(price)))
    return ok


def _do_hire(farm, private, board_size, mult=K.FARM_HAND_COST_MULT):
    before = farm["money"]
    _orig_do_hire(farm, private, board_size, mult)
    if farm["money"] != before:
        LOG.append((CUR["t"], _who(farm), "HIRE", None, float(before - farm["money"])))


def _do_buy_land(farm, board_size):
    before = farm["money"]
    _orig_do_buy_land(farm, board_size)
    if farm["money"] != before:
        LOG.append((CUR["t"], _who(farm), "LAND", farm["unlocked_quadrants"][-1], float(before - farm["money"])))


K._process_market = _process_market
K._commit_unit = _commit_unit
K._do_hire = _do_hire
K._do_buy_land = _do_buy_land

SNAP_DAYS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29)


def farm_snapshot(farm):
    plants, animals = collections.Counter(), collections.Counter()
    empty_struct = weeds = 0
    for row in farm["tiles"]:
        for tile in row:
            if not isinstance(tile, dict):
                continue
            kind = tile.get("kind")
            if kind == "PLANT":
                plants[tile.get("crop")] += 1
            elif kind in ("COOP", "PASTURE"):
                if tile.get("animal"):
                    animals[tile["animal"]] += 1
                else:
                    empty_struct += 1
            elif kind == "WEED":
                weeds += 1
    return {"hands": len(farm["hands"]), "quadrants": len(farm["unlocked_quadrants"]),
            "plants": dict(plants), "animals": dict(animals), "empty_struct": empty_struct, "weeds": weeds}


def analyse(job):
    game, path = job
    d = json.load(open(path))
    steps = d["steps"]
    n = len(steps)
    me = game["seat"]
    assert d["info"]["TeamNames"][me] == "Sunshine through fog", (game["id"], d["info"]["TeamNames"])
    LOG.clear()
    env = make("kaggriculture", configuration={"seed": d["info"]["seed"]}, debug=True)
    env.reset(2)
    mismatch = 0
    for t in range(n - 1):
        CUR["t"] = t
        state = env.step([steps[t + 1][s].get("action") or {} for s in (0, 1)])
        env.steps[-2] = None  # keep the list length (the runner's step counter), drop the old state
        got = [state[0].observation.farms[s]["money"] for s in (0, 1)]
        want = [steps[t + 1][0]["observation"]["farms"][s]["money"] for s in (0, 1)]
        mismatch += got != want
    final = [state[s].reward for s in (0, 1)]

    prices = [steps[t][0]["observation"]["market"]["prices"] for t in range(n)]
    seats = {}
    sells_at = [collections.defaultdict(collections.Counter) for _ in (0, 1)]  # seat -> t -> item -> units
    for t, s, op, item, price in LOG:
        if op == "SELL":
            sells_at[s][t][item] += 1
    for s in (0, 1):
        sold = collections.defaultdict(lambda: [0, 0.0])
        rev_day, spend_day, hires_day = [0.0] * 30, [0.0] * 30, [0] * 30
        spend = collections.Counter()
        land = []
        shared = collections.Counter()  # units of an item sold in a step where the other seat also sold it
        after_opp = collections.Counter()  # units sold within 24 steps after the other seat sold the item
        for t, who, op, item, price in LOG:
            if who != s:
                continue
            day = t // 24
            if op == "SELL":
                sold[item][0] += 1
                sold[item][1] += price
                rev_day[day] += price
                if sells_at[1 - s][t].get(item):
                    shared[item] += 1
                if any(sells_at[1 - s][u].get(item) for u in range(max(0, t - 24), t + 1)):
                    after_opp[item] += 1
            else:
                key = {"BUY_SEED": "seed", "BUY_ANIMAL": "animal", "BUY_PRODUCT": "product",
                       "HIRE": "hire", "LAND": "land"}[op]
                spend[key if key in ("hire", "land") else f"{key}:{item}"] += price
                spend_day[day] += price
                if op == "HIRE":
                    hires_day[day] += 1
                if op == "LAND":
                    land.append([t, item, price])
        shed_end = {k: v for k, v in steps[-1][s]["observation"]["private"]["shed"].items() if v}
        seats[s] = {
            "money_by_day": [steps[24 * dd][0]["observation"]["farms"][s]["money"] for dd in range(30)] + [final[s]],
            "sold": {k: [v[0], round(v[1])] for k, v in sold.items()},
            "rev_day": [round(x) for x in rev_day], "spend_day": [round(x) for x in spend_day],
            "spend": {k: round(v) for k, v in spend.items()}, "hires_day": hires_day, "land": land,
            "shared_units": dict(shared), "after_opp_units": dict(after_opp),
            "farm": {dd: farm_snapshot(steps[24 * dd + 12][0]["observation"]["farms"][s]) for dd in SNAP_DAYS},
            "shed_end": shed_end,
            "shed_end_value": round(sum(v * prices[-1].get(k, 0) for k, v in shed_end.items())),
            "opening": [o for t in range(1, 49) for o in (steps[t][s].get("action") or {}).get("market") or []][:12],
        }
    out = dict(game)
    out.update({"seed": d["info"]["seed"], "resim_mismatch_steps": mismatch, "resim_final": final,
                "replay_rewards": d["rewards"], "me": seats[me], "opp": seats[1 - me]})
    return out


if __name__ == "__main__":
    games = json.load(open(sys.argv[1]))
    jobs = [(g, f"{sys.argv[2]}/episode-{g['id']}-replay.json") for g in games]
    with Pool(int(sys.argv[4]) if len(sys.argv) > 4 else 2, maxtasksperchild=4) as pool:
        res = []
        for r in pool.imap(analyse, jobs):
            res.append(r)
            print(r["id"], r["res"], "seed", r["seed"], "mismatch", r["resim_mismatch_steps"], flush=True)
    json.dump(res, open(sys.argv[3], "w"))
