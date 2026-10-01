"""Both farms at days 10 and 20 in Aspen Vale's losses (from story_av.py output), for the losses above a margin.

    python farms_av2.py stories_avL2.json [min_loss=2000]
"""
import json
import sys

stories = json.load(open(sys.argv[1]))
min_loss = float(sys.argv[2]) if len(sys.argv) > 2 else 2000
side_keys = None


def gap(g, d):
    by_day = g["gap_by_day"]
    if isinstance(by_day, dict):
        return by_day.get(str(d), by_day.get(d))
    return by_day[d] if d < len(by_day) else None


for gid, g in sorted(stories.items(), key=lambda kv: kv[1]["diff"]):
    if -g["diff"] < min_loss:
        continue
    if side_keys is None:
        side_keys = [k for k in g["ours"] if k not in ("sold", "income", "bought")]
        print("farm keys:", side_keys)
    print(f"\n{gid} {g['team']} ({g['orate']}) seat {g['seat']} margin {g['diff']:+,}; cash gap by day 5/10/15/20/25:",
          [gap(g, d) for d in (5, 10, 15, 20, 25)])
    for k in side_keys:
        print(f"  {k:10s} ours  {json.dumps(g['ours'][k])[:230]}")
        print(f"  {'':10s} rival {json.dumps(g['rival'][k])[:230]}")
