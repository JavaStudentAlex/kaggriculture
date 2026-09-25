"""Town shop timeline of every game: the unlocked shops at the start of each day (0-29) and at the end.

usage: python town_shops.py <index.json> <replay dir> <shops.json>
"""
import json
import sys

out = {}
for g in json.load(open(sys.argv[1])):
    steps = json.load(open(f"{sys.argv[2]}/episode-{g['id']}-replay.json"))["steps"]
    out[g["id"]] = [list(steps[24 * day][0]["observation"]["town"]["unlocked_shops"]) for day in range(30)] + [
        list(steps[-1][0]["observation"]["town"]["unlocked_shops"])]
    del steps
json.dump(out, open(sys.argv[3], "w"))
print(len(out), "games")
