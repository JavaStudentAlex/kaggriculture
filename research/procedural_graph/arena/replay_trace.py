#!/usr/bin/env python3
"""Replay a compact arena trace (arena.py --trace-dir) through an instrumented engine.

    python replay_trace.py <trace.json.gz> [...]

The recorded actions are fed back (the action for observation t is stored at row t+1), so
no agent code runs and a replay takes about 15 s. Writes next to each trace:

- `<trace>.tx.json.gz`: every executed market unit as [step, seat, op, item, price]
  (HIRE and BUY_LAND with their cost), and the town's consumption per step;
- `<trace>.farm.json.gz`: per day and seat, the animals held, those left unfed that day,
  those that escaped at midnight and the empty tiles (the weed draws share the RNG that
  picks each new town shop), plus the unlocked shops.

Exits non-zero unless every replay reproduces the recorded final cash of both seats.
"""
import gzip
import json
import sys

from kaggle_environments import make
import kaggle_environments.envs.kaggriculture.kaggriculture as K

_process, _commit, _town = K._process_market, K._commit_unit, K._town_consume
_hire, _land, _refresh, _eod = K._do_hire, K._do_buy_land, K._daily_refresh_animals, K._end_of_day
log = {}


def _seat(farm):
    return 0 if farm is log["farms"][0] else 1


def process(state, env):
    log["step"] = K.get(state[0].observation, "step", 0)
    log["farms"] = state[0].observation.farms
    return _process(state, env)


def commit(op, item, price, farm, private, market, shed_capacity=100):
    ok = _commit(op, item, price, farm, private, market, shed_capacity)
    if ok:
        log["units"].append([log["step"], _seat(farm), op, item, price])
    return ok


def _paid(op, fn):
    def wrapped(farm, *args):
        before = farm["money"]
        fn(farm, *args)
        if farm["money"] != before:
            log["units"].append([log["step"], _seat(farm), op, None, before - farm["money"]])
    return wrapped


def town(env, state, step):
    inv0 = dict(state[0].observation.market["inventory"])
    _town(env, state, step)
    inv1 = state[0].observation.market["inventory"]
    used = {k: inv0[k] - inv1[k] for k in inv0 if inv0[k] != inv1[k]}
    if used:
        log["town"].append([step, used])


def refresh(farm, day):
    tiles = farm["tiles"]
    before = {(x, y): t["animal"] for y, row in enumerate(tiles) for x, t in enumerate(row)
              if isinstance(t, dict) and "animal" in t}
    unfed = sorted(a for (x, y), a in before.items() if not tiles[y][x]["fed_today"])
    _refresh(farm, day)
    escaped = sorted(a for (x, y), a in before.items() if not (isinstance(tiles[y][x], dict) and "animal" in tiles[y][x]))
    log["seats"].append({"animals": sorted(before.values()), "unfed": unfed, "escaped": escaped})


def end_of_day(state, env, day):
    log["seats"] = []
    empty = [sum(1 for row in f["tiles"] for t in row if t is None) for f in state[0].observation.farms]
    _eod(state, env, day)
    log["days"].append({"day": day, "empty_tiles": empty, "seats": log["seats"],
                        "shops": list(state[0].observation.town["unlocked_shops"])})


K._process_market, K._commit_unit, K._town_consume = process, commit, town
K._do_hire, K._do_buy_land = _paid("HIRE", _hire), _paid("BUY_LAND", _land)
K._daily_refresh_animals, K._end_of_day = refresh, end_of_day


def replay(path):
    trace = json.load(gzip.open(path))
    seed = int(path.split("_seed")[1].split("_")[0])
    log.update(units=[], town=[], days=[])

    def recorded(seat):
        def agent(obs, config):
            t = int(obs["step"]) + 1
            return trace[t]["seats"][seat]["action"] if t < len(trace) else {}
        return agent

    env = make("kaggriculture", configuration={"seed": seed}, debug=False)
    env.run([recorded(0), recorded(1)])
    final = [s.get("reward") for s in env.steps[-1]]
    stem = path[:-len(".json.gz")]
    with gzip.open(stem + ".tx.json.gz", "wt") as fh:
        json.dump({"seed": seed, "final": final, "units": log["units"], "town": log["town"]}, fh,
                  separators=(",", ":"))
    with gzip.open(stem + ".farm.json.gz", "wt") as fh:
        json.dump(log["days"], fh, separators=(",", ":"))
    faithful = final == trace[-1]["money"]
    print(json.dumps({"trace": path.split("/")[-1], "final": final, "faithful": faithful}))
    return faithful


if __name__ == "__main__":
    sys.exit(0 if all([replay(p) for p in sys.argv[1:]]) else 1)
