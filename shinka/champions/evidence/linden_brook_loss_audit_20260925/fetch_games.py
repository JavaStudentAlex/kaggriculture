"""Index and download one submission's ladder games (read-only Kaggle API).

usage: python fetch_games.py <submission id> <replay dir> <index.json>

Lists the submission's episodes, keeps the COMPLETED public ones, identifies our seat by the
exact submission id, records both rewards and the opponent's team and submission, looks up the
opponent submission's current rating (team-submissions, one paced call per team), and downloads
every replay (at most 3 concurrent requests, resumable, back-off on errors). The seed of a game
is `info.seed` inside its replay (resim_games.py copies it into stats.json).
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from kaggle.api.kaggle_api_extended import KaggleApi

SUB, OUT, INDEX = int(sys.argv[1]), sys.argv[2], sys.argv[3]
api = KaggleApi()
api.authenticate()
os.makedirs(OUT, exist_ok=True)

games = []
for e in api.competition_list_episodes(SUB):
    e = e.to_dict()
    if e["type"] != "EPISODE_TYPE_PUBLIC" or e["state"] != "COMPLETED":
        continue
    me = [a for a in e["agents"] if a.get("submissionId") == SUB]
    opp = [a for a in e["agents"] if a.get("submissionId") != SUB]
    if len(me) != 1 or len(opp) != 1 or me[0].get("reward") is None or opp[0].get("reward") is None:
        print("skipped", e["id"], e["agents"])
        continue
    me, opp = me[0], opp[0]
    diff = me["reward"] - opp["reward"]
    games.append({"id": e["id"], "t": e["createTime"], "seat": me.get("index", 0), "me": me["reward"],
                  "opp": opp["reward"], "diff": diff, "res": "W" if diff > 0 else "L" if diff < 0 else "T",
                  "team": opp["teamName"], "tid": opp["teamId"], "osub": opp["submissionId"]})
games.sort(key=lambda g: g["t"])

rating = {}
for tid in sorted({g["tid"] for g in games}):
    for attempt in range(4):
        try:
            for s in api.competition_team_submissions(tid):
                s = s.to_dict()
                score = s.get("publicScore")
                rating[int(s["id"])] = float(score) if score not in (None, "") else None
            break
        except Exception as ex:  # throttling: back off and retry
            print("team-submissions", tid, type(ex).__name__, str(ex)[:80])
            time.sleep(20 * (attempt + 1))
    time.sleep(1)
for g in games:
    g["orate"] = rating.get(g["osub"])  # None: the opponent's submission is no longer active
    g["onew"] = g["osub"] > SUB  # submitted after ours
json.dump(games, open(INDEX, "w"), indent=1)


def fetch(i):
    path = f"{OUT}/episode-{i}-replay.json"
    if os.path.exists(path) and os.path.getsize(path) > 1e6:
        return "have"
    for attempt in range(5):
        try:
            api.competition_episode_replay(i, OUT)
            json.load(open(path))
            return "ok"
        except Exception as ex:
            if os.path.exists(path):
                os.remove(path)
            print("replay", i, type(ex).__name__, str(ex)[:80])
            time.sleep(30 * (attempt + 1))
    return "fail"


with ThreadPoolExecutor(3) as pool:
    res = list(pool.map(fetch, [g["id"] for g in games]))
print(f"{len(games)} public games: W {sum(g['res'] == 'W' for g in games)} L {sum(g['res'] == 'L' for g in games)} "
      f"T {sum(g['res'] == 'T' for g in games)}; replays {dict((r, res.count(r)) for r in set(res))}")
