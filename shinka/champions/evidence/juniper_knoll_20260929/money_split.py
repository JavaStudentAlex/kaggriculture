"""Where the money went in recorded games, from flows_av.py output (exact): ours minus the rival's, by product
(sales net of the same product bought back) and by cost (land, hires, seeds and animals; positive = we spent less).

    python money_split.py flows_avL2.json flows_avW2.json [flows_avL.json flows_avW.json]

Prints the table of ../aspen_vale_20260928/README.md ("Where the money went") for each file, then, per game of the
first file, the categories that decided it and the tomato/land picture.
"""
import json
import sys

PRODUCTS = ["TOMATO", "EGG", "CARROT", "STRAWBERRY", "WHEAT", "WOOL", "FERTILIZER", "MELON", "MILK"]


def split(side):
    out = {}
    for p in PRODUCTS:
        sold = side["sell"].get(p, [0, 0.0])[1]
        bought = side["buy"].get(f"BUY_PRODUCT:{p}", [0, 0.0])[1]
        out[p] = sold - bought
    out["land"] = -side["land"]
    out["hires"] = -side["hire"]
    out["seeds, animals"] = -sum(v[1] for k, v in side["buy"].items()
                                 if k.startswith("BUY_SEED:") or k.startswith("BUY_ANIMAL:"))
    return out


def game_diff(g):
    a, b = split(g["ours"]), split(g["rival"])
    return {k: a[k] - b[k] for k in a}


files = sys.argv[1:]
tables = []
for path in files:
    games = json.load(open(path))
    total = {}
    for g in games.values():
        for k, v in game_diff(g).items():
            total[k] = total.get(k, 0.0) + v
    tables.append((path, len(games), total, games))
    mism = sum(g["ours"]["money_mismatch"] + g["rival"]["money_mismatch"] for g in games.values())
    print(f"{path}: {len(games)} games, money mismatches {mism}")

keys = PRODUCTS + ["land", "hires", "seeds, animals"]
header = "| category | " + " | ".join(f"{p.split('/')[-1].replace('flows_', '').replace('.json', '')} ({n}) | per game"
                                        for p, n, _, _ in tables) + " |"
print("\n" + header)
print("|---|" + "---|---|" * len(tables))
order = sorted(keys, key=lambda k: tables[0][2].get(k, 0.0))
for k in order:
    cells = []
    for _, n, total, _ in tables:
        v = total.get(k, 0.0)
        cells.append(f"{v:+,.0f} | {v / max(n, 1):+,.0f}")
    print(f"| {k} | " + " | ".join(cells) + " |")

print("\nper game of", files[0], "(margin, the three largest categories against us, tomatoes and land):")
_, _, _, games = tables[0]
for gid, g in sorted(games.items(), key=lambda kv: kv[1]["diff"]):
    d = game_diff(g)
    worst = sorted(d.items(), key=lambda kv: kv[1])[:3]
    ot = g["ours"]["sell"].get("TOMATO", [0, 0])[0]
    rt = g["rival"]["sell"].get("TOMATO", [0, 0])[0]
    print(f"  {gid} {g['team'][:22]:22s} {g['orate'] or 0:6.0f} seat {g['seat']} {g['diff']:+7d} | "
          + ", ".join(f"{k} {v:+,.0f}" for k, v in worst)
          + f" | tomato units {ot} vs {rt} | land {g['ours']['land']:,.0f} vs {g['rival']['land']:,.0f}")
