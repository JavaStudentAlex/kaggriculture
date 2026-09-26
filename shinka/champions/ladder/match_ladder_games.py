#!/usr/bin/env python3
"""Does a ladder-pool bundle play exactly like a recorded ladder opponent?

    python shinka/champions/ladder/match_ladder_games.py BUNDLE_DIR INDEX_JSON REPLAY_DIR [EPISODE ...]

INDEX_JSON is a loss-audit index (fields id, seed, seat = our seat); replays are
REPLAY_DIR/episode-<id>-replay.json. The game is re-run from its seed: our seat replays our
recorded actions (the action stored at steps[t+1] is the one taken from observation t), the
bundle plays the opponent's seat, and its action is compared with the recorded one at every
step, up to the first difference (after which the games diverge). Actions are compared as the
engine reads them: a missing PICKUP/PLACE count is 1, trailing PASS hands are trimmed, and an order
the engine cannot parse (an empty [] included) is kept as a placeholder, because both seats' orders
clear index by index and a placeholder delays the orders after it; trailing placeholders are trimmed.
Prints one JSON line per game.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import time
from pathlib import Path

from kaggle_environments import make


def _unit(a):
    a = list(a) if isinstance(a, (list, tuple)) and a else ["PASS"]
    if a[0] in ("PICKUP", "PLACE") and len(a) == 2:
        a = a + [1]
    return [int(x) if isinstance(x, float) and x == int(x) else x for x in a]


def _order(o):
    if not isinstance(o, (list, tuple)) or not o:
        return None
    if o[0] in ("HIRE", "BUY_LAND"):
        return [o[0]]
    if o[0] in ("BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL") and len(o) >= 3:
        try:
            n = int(o[2])
        except (TypeError, ValueError):
            return None
        return [o[0], o[1], n] if n > 0 else None
    return None


def canon(action):
    action = action or {}
    hands = [_unit(h) for h in (action.get("hands") or [])]
    while hands and hands[-1] == ["PASS"]:
        hands.pop()
    market = [_order(o) for o in (action.get("market") or [])[:10]]
    while market and market[-1] is None:
        market.pop()
    return json.dumps({"farmer": _unit(action.get("farmer")), "hands": hands, "market": market}, sort_keys=True)


def load_agent(bundle: Path):
    spec = importlib.util.spec_from_file_location("_ladder_bundle_main", bundle / "main.py")
    module = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        spec.loader.exec_module(module)
    return module.agent


def match(bundle: Path, game: dict, replay: dict) -> dict:
    agent = load_agent(bundle)  # a fresh module per game: the agents keep per-game globals
    ours, theirs = game["seat"], 1 - game["seat"]
    steps = replay["steps"]
    env = make("kaggriculture", configuration={"seed": replay["info"]["seed"]}, debug=False)
    env.reset(2)
    shared = env._Environment__get_shared_state  # the observation the runner hands an agent
    t0, first = time.time(), None
    for t in range(len(steps) - 1):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            got = agent(shared(theirs).observation, env.configuration)
        want = steps[t + 1][theirs].get("action")
        if canon(got) != canon(want):
            first = {"step": t, "bundle": json.loads(canon(got)), "recorded": json.loads(canon(want))}
            break
        acts = [None, None]
        acts[ours], acts[theirs] = steps[t + 1][ours].get("action"), got
        env.step(acts)
        if env.done:
            break
    equal = first["step"] if first else len(steps) - 1
    return {"bundle": bundle.name, "episode": game["id"], "opponent": game.get("opponent") or game.get("team"),
            "equal_steps": equal, "of": len(steps) - 1, "first_difference": first, "sec": round(time.time() - t0, 1)}


def main():
    bundle, index, replays = Path(sys.argv[1]).resolve(), Path(sys.argv[2]), Path(sys.argv[3])
    wanted = {int(x) for x in sys.argv[4:]}
    for game in json.loads(index.read_text()):
        if wanted and game["id"] not in wanted:
            continue
        replay = json.loads((replays / f"episode-{game['id']}-replay.json").read_text())
        print(json.dumps(match(bundle, game, replay)), flush=True)


if __name__ == "__main__":
    main()
