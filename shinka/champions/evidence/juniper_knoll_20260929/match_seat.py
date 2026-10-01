#!/usr/bin/env python3
"""Where would another version of our agent first have played differently in our recorded ladder games?

    python match_seat.py BUNDLE_DIR INDEX_JSON REPLAY_DIR EPISODE

shinka/champions/ladder/match_ladder_games.py on OUR seat instead of the rival's: the game is re-run from its seed,
the rival's seat replays its recorded actions, the bundle plays our seat, and its action is compared with the one we
really sent at every step, up to the first difference (after which the games diverge). One episode per process: our
bundles keep engine state in module globals, which a second game in the same process would inherit. Replays may be
gzipped (episode-<id>-replay.json.gz). Prints one JSON line.
"""
from __future__ import annotations

import contextlib
import gzip
import importlib.util
import io
import json
import os
import sys
import time
from pathlib import Path

from kaggle_environments import make

sys.path.insert(0, '/home/alex/kagg-evo/repo/shinka/champions/ladder')
from match_ladder_games import canon  # noqa: E402


def load_replay(replays: Path, eid: int) -> dict:
    p = replays / f"episode-{eid}-replay.json"
    if p.exists():
        return json.loads(p.read_text())
    with gzip.open(str(p) + '.gz', 'rt') as f:
        return json.load(f)


def load_agent(bundle: Path):
    sys.path.insert(0, str(bundle))
    spec = importlib.util.spec_from_file_location("_our_bundle_main", bundle / "main.py")
    module = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        spec.loader.exec_module(module)
    return module.agent


def main():
    bundle, index, replays, eid = Path(sys.argv[1]).resolve(), Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4])
    game = next(g for g in json.loads(index.read_text()) if g["id"] == eid)
    replay = load_replay(replays, eid)
    ours, theirs = game["seat"], 1 - game["seat"]
    steps = replay["steps"]
    agent = load_agent(bundle)
    env = make("kaggriculture", configuration={"seed": replay["info"]["seed"]}, debug=False)
    env.reset(2)
    shared = env._Environment__get_shared_state
    t0, first, slow = time.time(), None, 0.0
    for t in range(len(steps) - 1):
        a = time.time()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            got = agent(shared(ours).observation, env.configuration)
        slow = max(slow, time.time() - a)
        want = steps[t + 1][ours].get("action")
        if canon(got) != canon(want):
            first = {"step": t, "bundle": json.loads(canon(got)), "recorded": json.loads(canon(want))}
            break
        acts = [None, None]
        acts[ours], acts[theirs] = got, steps[t + 1][theirs].get("action")
        env.step(acts)
        if env.done:
            break
    equal = first["step"] if first else len(steps) - 1
    print(json.dumps({"bundle": bundle.name, "episode": eid, "team": game.get("team"), "res": game.get("res"),
                      "diff": game.get("diff"), "seat": ours, "equal_steps": equal, "of": len(steps) - 1,
                      "first_difference": first, "sec": round(time.time() - t0, 1), "max_step_sec": round(slow, 3)}),
          flush=True)


if __name__ == "__main__":
    main()
