"""Index the lost and tied ladder games of several submissions and download the replays we do not have yet
(read-only API, 3 at a time).

usage: python fetch_lost.py <replay dir> <index.json> <have ids file> <name>=<submission id> ... [--prev index.json ...]

The index rows have the fields of fetch_new_games.py (orate = the rival's rating before the game) plus `ours`.
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from kaggle.api.kaggle_api_extended import KaggleApi

args = sys.argv[1:]
prev = args[args.index("--prev") + 1:] if "--prev" in args else []
args = args[:args.index("--prev")] if "--prev" in args else args
OUT, INDEX, HAVE, subs = args[0], args[1], args[2], dict(a.split("=") for a in args[3:])
have = {int(l) for l in open(HAVE) if l.strip().isdigit() and len(l.strip()) == 9}
api = KaggleApi()
api.authenticate()
os.makedirs(OUT, exist_ok=True)

games = {}
for p in prev:          # earlier listings keep games Kaggle no longer lists (newest ~200 per submission)
    for g in json.load(open(p)):
        if g.get("ours") in subs:
            games[g["id"]] = g
for name, sub in subs.items():
    sub = int(sub)
    n = 0
    for e in api.competition_list_episodes(sub):
        e = e.to_dict()
        if e["type"] != "EPISODE_TYPE_PUBLIC" or e["state"] != "COMPLETED":
            continue
        me = [a for a in e["agents"] if a.get("submissionId") == sub]
        opp = [a for a in e["agents"] if a.get("submissionId") != sub]
        if len(me) != 1 or len(opp) != 1 or me[0].get("reward") is None or opp[0].get("reward") is None:
            continue
        me, opp = me[0], opp[0]
        diff = me["reward"] - opp["reward"]
        n += 1
        games[e["id"]] = {"id": e["id"], "t": e["createTime"], "seat": me.get("index", 0), "me": me["reward"],
                          "opp": opp["reward"], "diff": diff, "res": "W" if diff > 0 else "L" if diff < 0 else "T",
                          "team": opp["teamName"], "tid": opp["teamId"], "osub": opp["submissionId"],
                          "orate": opp.get("initialScore"), "onew": opp["submissionId"] > sub, "ours": name}
    print(name, sub, "listed", n, flush=True)
games = sorted(games.values(), key=lambda g: g["t"])
json.dump(games, open(INDEX, "w"), indent=1)
lost = [g for g in games if g["res"] in "LT"]
for name in subs:
    mine = [g for g in games if g["ours"] == name]
    print(name, len(mine), "games", {k: sum(g["res"] == k for g in mine) for k in "WLT"},
          "lost/tied new:", sum(1 for g in mine if g["res"] in "LT" and g["id"] not in have))


def fetch(i):
    path = f"{OUT}/episode-{i}-replay.json"
    if os.path.exists(path + ".gz") or (os.path.exists(path) and os.path.getsize(path) > 1e6):
        return "have"
    for attempt in range(5):
        try:
            api.competition_episode_replay(i, OUT)
            json.load(open(path))
            return "ok"
        except Exception as ex:
            if os.path.exists(path):
                os.remove(path)
            print("replay", i, type(ex).__name__, str(ex)[:80], flush=True)
            time.sleep(30 * (attempt + 1))
    return "fail"


todo = [g["id"] for g in sorted(lost, key=lambda g: g["t"], reverse=True) if g["id"] not in have]
print("downloading", len(todo), "replays", flush=True)
with ThreadPoolExecutor(3) as pool:
    res = list(pool.map(fetch, todo))
print("replays", {r: res.count(r) for r in set(res)}, flush=True)
