"""Second pass over Linden Brook's ladder games: waste on the farm, the market's state, sell timing.

Every replay is re-simulated exactly as in resim_games.py (seed + both seats' recorded actions; a
game counts only if both seats' money matches the replay on every step), with more of the engine
wrapped:
- unit actions: what every farmer and hand did, and the actions that did nothing (FEED without
  wheat in hand, HARVEST of an empty tile, WATER of a watered plant, DROP into a full shed, whose
  overflow the engine discards);
- the end-of-day refresh: unfed animal-days, escapes, care bonuses paid, production lost to the
  animal's cap (max_held), plants that withered (two days unwatered) with the units they held,
  ongoing crops capped at max_yield, and the shed's end-of-day overflow (discarded);
- plant decay: units lost while a ripe crop is left in the field;
- market orders the engine refused (selling what the shed lacks, buying without money or room);
- every sale with its price and the product's highest price over the 24 steps before it;
- each product's market inventory and price on every step (glut above I0, scarcity below);
- remainingOverageTime of each seat (the time bank beyond the 1 s per step).

usage: python market_audit.py <stats.json> <replay dir> <out.json> [workers]
"""
import collections
import json
import sys
from multiprocessing import Pool

import kaggle_environments.envs.kaggriculture.kaggriculture as K
from kaggle_environments import make

PRODUCTS = K.PRODUCTS
LOG = []  # (t, seat, op, item, price) of every executed market unit
REFUSED = []  # (t, seat, op, item, reason) of every refused unit (it ends that order)
ACC = [collections.Counter(), collections.Counter()]  # per-seat counters
CUR = {"farms": None, "t": None, "seen": [], "calls": collections.Counter()}

_orig = {name: getattr(K, name) for name in (
    "_process_market", "_commit_unit", "_do_hire", "_do_buy_land", "_apply_unit_action",
    "_daily_refresh_animals", "_daily_refresh_plants", "_decay_plants", "_drop_inventories_to_shed",
    "_spawn_weeds")}


def _who(farm):
    farms = CUR["farms"]
    return 0 if farm is farms[0] else 1 if farm is farms[1] else -1


def _in_order(name):
    """The engine calls the per-farm refresh functions for seat 0, then seat 1, once per step."""
    seat = CUR["calls"][name]
    CUR["calls"][name] += 1
    return seat


def _process_market(state, env):
    CUR["farms"] = state[0].observation.farms
    return _orig["_process_market"](state, env)


def _commit_unit(op, item, price, farm, private, market, shed_capacity=100):
    money = farm["money"]
    ok = _orig["_commit_unit"](op, item, price, farm, private, market, shed_capacity)
    seat = _who(farm)
    if ok:
        LOG.append((CUR["t"], seat, op, item, float(price)))
    else:
        reason = ("empty_shed" if op == "SELL" else "no_money" if money < price else "shed_full")
        REFUSED.append((CUR["t"], seat, op, item, reason))
    return ok


def _do_hire(farm, private, board_size, mult=K.FARM_HAND_COST_MULT):
    before = farm["money"]
    _orig["_do_hire"](farm, private, board_size, mult)
    if farm["money"] != before:
        LOG.append((CUR["t"], _who(farm), "HIRE", None, float(before - farm["money"])))


def _do_buy_land(farm, board_size):
    before = farm["money"]
    _orig["_do_buy_land"](farm, board_size)
    if farm["money"] != before:
        LOG.append((CUR["t"], _who(farm), "LAND", farm["unlocked_quadrants"][-1], float(before - farm["money"])))


def _apply_unit_action(farm, private, idx, action, board_size, day, turns_per_day, shed_capacity=100):
    seen = CUR["seen"]  # seat 0's farmer always acts first (a missing action is a PASS)
    seat = next((i for i, f in enumerate(seen) if f is farm), None)
    if seat is None:
        seen.append(farm)
        seat = len(seen) - 1
    c = ACC[seat]
    op = action[0] if isinstance(action, list) and action else None
    pos = K._farmer_position(farm, idx)
    if pos is None:  # no such hand: the engine ignores the action (and must not grow the inventories)
        return _orig["_apply_unit_action"](farm, private, idx, action, board_size, day, turns_per_day, shed_capacity)
    tile = farm["tiles"][pos[1]][pos[0]]
    inv = K._farmer_inventory(private, idx)
    c["unit_steps"] += 1
    if op in (None, "PASS"):
        c["act:PASS"] += 1
    elif op in K.FARMER_MOVES:
        c["act:MOVE"] += 1
    else:
        c[f"act:{op}"] += 1
    animal = tile if isinstance(tile, dict) and "animal" in tile else None
    plant = tile if isinstance(tile, dict) and tile.get("kind") == "PLANT" else None
    if op == "FEED" and animal is not None:
        c["feed_noop:already_fed" if animal["fed_today"] else
          "feed_noop:no_wheat_in_hand" if inv.get("WHEAT", 0) <= 0 else "feed_ok"] += 1
    elif op == "CARE" and animal is not None:
        c["care_noop" if animal["cared_today"] else "care_ok"] += 1
    elif op == "HARVEST":
        units = tile.get("yield_units", 0) if isinstance(tile, dict) else 0
        if units <= 0:
            c["harvest_noop"] += 1
        elif plant is not None:
            cd = K.CROPS[plant["crop"]]
            if day - plant["planted_day"] < cd["first_yield_day"]:
                c["harvest_noop"] += 1
            else:
                c[f"harvests:{plant['crop']}"] += 1
                c[f"harvested:{plant['crop']}"] += units
        elif animal is not None:
            product = K.ANIMALS[animal["animal"]]["product"]
            c[f"harvests:{product}"] += 1
            c[f"harvested:{product}"] += units
    elif op == "WATER":
        c["water_ok" if plant is not None and not plant["watered_today"] else "water_noop"] += 1
    elif op == "COLLECT_FERTILIZER":
        c["collect_fert_ok" if animal is not None and animal["fertilizer_available"] else "collect_fert_noop"] += 1
    elif op == "PLANT":
        if tile is not None:
            c["plant_noop"] += 1
    if op == "DROP":
        if not K._is_shed_adjacent(tuple(pos), board_size):
            c["drop_noop"] += 1
        else:
            held = sum(n for n in inv.values() if n > 0)
            before = sum(private["shed"].values())
            _orig["_apply_unit_action"](farm, private, idx, action, board_size, day, turns_per_day, shed_capacity)
            c["drop_discarded"] += held - (sum(private["shed"].values()) - before)
            return
    return _orig["_apply_unit_action"](farm, private, idx, action, board_size, day, turns_per_day, shed_capacity)


def _daily_refresh_animals(farm, day):
    c = ACC[_in_order("animals")]
    before = {(y, x): dict(t) for y, row in enumerate(farm["tiles"]) for x, t in enumerate(row)
              if isinstance(t, dict) and "animal" in t}
    _orig["_daily_refresh_animals"](farm, day)
    for (y, x), b in before.items():
        name, a = b["animal"], K.ANIMALS[b["animal"]]
        now = farm["tiles"][y][x]
        c[f"animal_days:{name}"] += 1
        c[f"unfed_days:{name}"] += not b["fed_today"]
        c[f"care_days:{name}"] += bool(b["cared_today"] and b["fed_today"])
        if not (isinstance(now, dict) and "animal" in now):
            c[f"escaped:{name}"] += 1
            continue
        since = day + 1 - b["placed_day"] - a["first_yield_day"]
        if since >= 0 and since % a["interval"] == 0:
            bonus = b.get("pending_care_bonus", 0) if b["fed_today"] else 0
            c[f"productions:{name}"] += 1
            c[f"produced:{name}"] += now["yield_units"] - b["yield_units"]
            c[f"care_bonus_paid:{name}"] += bonus
            c[f"lost_at_cap:{name}"] += b["yield_units"] + 1 + bonus - now["yield_units"]
            c[f"care_bonus_forfeited_unfed:{name}"] += 0 if b["fed_today"] else b.get("pending_care_bonus", 0)


def _daily_refresh_plants(farm, current_day, turns_per_day):
    c = ACC[_in_order("plants")]
    before = {(y, x): dict(t) for y, row in enumerate(farm["tiles"]) for x, t in enumerate(row)
              if isinstance(t, dict) and t.get("kind") == "PLANT"}
    _orig["_daily_refresh_plants"](farm, current_day, turns_per_day)
    for (y, x), b in before.items():
        crop, cd = b["crop"], K.CROPS[b["crop"]]
        now = farm["tiles"][y][x]
        c[f"plant_days:{crop}"] += 1
        c[f"unwatered_days:{crop}"] += not b["watered_today"]
        if isinstance(now, dict) and now.get("kind") == "WEED":
            c[f"withered:{crop}"] += 1
            c[f"withered_units:{crop}"] += b["yield_units"]
            continue
        if cd["ongoing"]:
            since = current_day + 1 - b["planted_day"] - cd["first_yield_day"]
            if since >= 0 and since % cd["interval"] == 0 and since // cd["interval"] + 1 <= cd["max_yield"]:
                fert = b["watered_today"] and b.get("fertilized_until_day", -1) >= current_day
                gained = now["yield_units"] - b["yield_units"]
                c[f"produced:{crop}"] += gained
                c[f"lost_at_cap:{crop}"] += (2 if fert else 1) - gained


def _decay_plants(farm, step):
    c = ACC[_in_order("decay")]
    for row in farm["tiles"]:
        for t in row:
            if isinstance(t, dict) and t.get("kind") == "PLANT":
                mls = t["max_lifespan_step"]
                if mls >= 0 and step >= mls and (step - mls) % 2 == 0 and t["yield_units"] > 0:
                    c[f"rotted_units:{t['crop']}"] += 1
    return _orig["_decay_plants"](farm, step)


def _drop_inventories_to_shed(private, capacity):
    c = ACC[_in_order("shed")]
    held = sum(n for inv in private["inventories"] for n in inv.values() if n > 0)
    before = sum(private["shed"].values())
    _orig["_drop_inventories_to_shed"](private, capacity)
    after = sum(private["shed"].values())
    c["eod_discarded"] += held - (after - before)
    c["eod_shed_full_days"] += after >= capacity


def _spawn_weeds(farm, board_size, weed_chance, rng):
    c = ACC[_in_order("weeds")]
    empty = sum(1 for row in farm["tiles"] for t in row if t is None)
    c["empty_tile_days"] += empty
    return _orig["_spawn_weeds"](farm, board_size, weed_chance, rng)


for _name, _fn in (("_process_market", _process_market), ("_commit_unit", _commit_unit), ("_do_hire", _do_hire),
                   ("_do_buy_land", _do_buy_land), ("_apply_unit_action", _apply_unit_action),
                   ("_daily_refresh_animals", _daily_refresh_animals), ("_daily_refresh_plants", _daily_refresh_plants),
                   ("_decay_plants", _decay_plants), ("_drop_inventories_to_shed", _drop_inventories_to_shed),
                   ("_spawn_weeds", _spawn_weeds)):
    setattr(K, _name, _fn)


def analyse(job):
    game, path = job
    d = json.load(open(path))
    steps = d["steps"]
    n = len(steps)
    me = game["seat"]
    assert d["info"]["TeamNames"][me] == "Sunshine through fog", (game["id"], d["info"]["TeamNames"])
    LOG.clear()
    REFUSED.clear()
    for c in ACC:
        c.clear()
    env = make("kaggriculture", configuration={"seed": d["info"]["seed"]}, debug=True)
    env.reset(2)
    mismatch = 0
    for t in range(n - 1):
        CUR["t"], CUR["seen"], CUR["calls"] = t, [], collections.Counter()
        state = env.step([steps[t + 1][s].get("action") or {} for s in (0, 1)])
        env.steps[-2] = None
        got = [state[0].observation.farms[s]["money"] for s in (0, 1)]
        want = [steps[t + 1][0]["observation"]["farms"][s]["money"] for s in (0, 1)]
        mismatch += got != want

    inv = [steps[t][0]["observation"]["market"]["inventory"] for t in range(n)]
    price = [steps[t][0]["observation"]["market"]["prices"] for t in range(n)]
    market = {}
    for p in PRODUCTS:
        i0 = K.MARKET_PARAMS[p]["I0"]
        market[p] = {"glut_steps": sum(inv[t][p] > i0 for t in range(n)),
                     "floor_steps": sum(price[t][p] <= 1 for t in range(n)),
                     "price_by_day": [price[24 * dd][p] for dd in range(30)] + [price[-1][p]],
                     "excess_by_day": [inv[24 * dd][p] - i0 for dd in range(30)] + [inv[-1][p] - i0]}

    seats = {}
    for s in (0, 1):
        sales = [(t, item, pr) for t, who, op, item, pr in LOG if who == s and op == "SELL"]
        by_day = {p: [0] * 30 for p in PRODUCTS}
        units_day = {p: [0] * 30 for p in PRODUCTS}
        timing = collections.defaultdict(lambda: [0, 0.0, 0.0, 0, 0])  # units, revenue, sum(price / peak of the last 24 steps), units at $1, units below base
        for t, item, pr in sales:
            by_day[item][t // 24] += pr
            units_day[item][t // 24] += 1
            peak = max(price[u][item] for u in range(max(0, t - 23), t + 1))
            row = timing[item]
            row[0] += 1
            row[1] += pr
            row[2] += pr / peak if peak > 0 else 1.0
            row[3] += pr <= 1
            row[4] += pr < K.MARKET_PARAMS[item]["base"]
        buys = collections.defaultdict(lambda: [0] * 30)  # "BUY_SEED:STRAWBERRY" -> units per day
        for t, who, op, item, pr in LOG:
            if who == s and op.startswith("BUY"):
                buys[f"{op}:{item}"][t // 24] += 1
        refused = collections.Counter(f"{op}:{item}:{why}" for t, who, op, item, why in REFUSED if who == s)
        overage = [steps[t][s]["observation"].get("remainingOverageTime") for t in range(n)]
        overage = [x for x in overage if isinstance(x, (int, float))]
        seats[s] = {"acc": dict(ACC[s]), "refused": dict(refused),
                    "rev_by_day": {p: [round(x) for x in v] for p, v in by_day.items() if any(v)},
                    "units_by_day": {p: v for p, v in units_day.items() if any(v)},
                    "timing": {p: [r[0], round(r[1]), round(r[2], 3), r[3], r[4]] for p, r in timing.items()},
                    "buys_by_day": dict(buys),
                    "overage_start_end_min": [overage[0], overage[-1], min(overage)] if overage else None}
    return {"id": game["id"], "seat": me, "res": game["res"], "diff": game["diff"], "seed": d["info"]["seed"],
            "resim_mismatch_steps": mismatch, "market": market, "me": seats[me], "opp": seats[1 - me]}


if __name__ == "__main__":
    games = json.load(open(sys.argv[1]))
    jobs = [({k: g[k] for k in ("id", "seat", "res", "diff")}, f"{sys.argv[2]}/episode-{g['id']}-replay.json") for g in games]
    with Pool(int(sys.argv[4]) if len(sys.argv) > 4 else 2, maxtasksperchild=4) as pool:
        res = []
        for r in pool.imap(analyse, jobs):
            res.append(r)
            print(r["id"], r["res"], "mismatch", r["resim_mismatch_steps"], flush=True)
    json.dump(res, open(sys.argv[3], "w"))
