"""Per-game ratings of our submissions (read-only API): each side's mu and sigma before and after every listed game.

usage: python ratings.py <out.json> <submission id> [<submission id> ...]

Kaggle lists only a submission's newest ~200 episodes. The episode listing carries initialScore / updatedScore
(mu) and initialConfidence / updatedConfidence (sigma) for both agents, i.e. the ratings at the time of the game.
"""
import json
import sys

from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()
out, subs = sys.argv[1], [int(s) for s in sys.argv[2:]]
rows = []
for sub in subs:
    for e in api.competition_list_episodes(sub):
        e = e.to_dict()
        if e["type"] != "EPISODE_TYPE_PUBLIC" or e["state"] != "COMPLETED":
            continue
        me = [a for a in e["agents"] if a.get("submissionId") == sub]
        opp = [a for a in e["agents"] if a.get("submissionId") != sub]
        if len(me) != 1 or len(opp) != 1:
            continue
        me, opp = me[0], opp[0]
        if me.get("reward") is None or opp.get("reward") is None:
            rows.append({"sub": sub, "id": e["id"], "t": e["createTime"], "res": "?", "agents": e["agents"]})
            continue
        diff = me["reward"] - opp["reward"]
        rows.append({"sub": sub, "id": e["id"], "t": e["createTime"], "end": e.get("endTime"), "seat": me.get("index", 0),
                     "res": "W" if diff > 0 else "L" if diff < 0 else "T", "diff": diff, "me": me["reward"],
                     "opp": opp["reward"], "mu0": me.get("initialScore"), "mu1": me.get("updatedScore"),
                     "s0": me.get("initialConfidence"), "s1": me.get("updatedConfidence"),
                     "omu0": opp.get("initialScore"), "omu1": opp.get("updatedScore"),
                     "os0": opp.get("initialConfidence"), "os1": opp.get("updatedConfidence"),
                     "team": opp.get("teamName"), "tid": opp.get("teamId"), "osub": opp.get("submissionId")})
rows.sort(key=lambda r: (r["sub"], r["t"]))
json.dump(rows, open(out, "w"), indent=1)
for sub in subs:
    mine = [r for r in rows if r["sub"] == sub]
    print(sub, len(mine), "games", {k: sum(r["res"] == k for r in mine) for k in "WLT?"})
