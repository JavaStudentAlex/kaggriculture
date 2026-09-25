"""Losses against wins, from stats.json (resim_games.py) and shops.json (town_shops.py).

usage: python aggregate.py <stats.json> <shops.json>
"""
import collections
import json
import statistics as st
import sys

PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]
ANIMALS = ["GOOSE", "COW", "SHEEP"]
SHOPS = {  # kaggle_environments/envs/kaggriculture/kaggriculture.py (1.32.7)
    "BAKERY": ["EGG", "WHEAT"], "PIZZA_SHOP": ["MILK", "TOMATO", "WHEAT"],
    "BRUNCH_SPOT": ["EGG", "WHEAT", "STRAWBERRY"], "YARN_STORE": ["WOOL"],
    "ICE_CREAM_SHOP": ["STRAWBERRY", "MILK", "WHEAT"], "PET_CAFE": ["CARROT"],
    "SMOOTHIE_SHOP": ["STRAWBERRY", "MILK"], "FARMERS_MARKET": ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY"],
}
ABBR = {"BAKERY": "bak", "PIZZA_SHOP": "piz", "BRUNCH_SPOT": "bru", "YARN_STORE": "YARN", "ICE_CREAM_SHOP": "ice",
        "PET_CAFE": "pet", "SMOOTHIE_SHOP": "smo", "FARMERS_MARKET": "fm"}

G = json.load(open(sys.argv[1]))
TOWN = {int(k): v for k, v in json.load(open(sys.argv[2])).items()}
L = [g for g in G if g["res"] == "L"]
W = [g for g in G if g["res"] == "W"]
SIDES = (("me", "ours"), ("opp", "theirs"))


def mean(xs):
    xs = list(xs)
    return st.mean(xs) if xs else float("nan")


def rev(x, p=None):
    return x["sold"].get(p, [0, 0])[1] if p else sum(v[1] for v in x["sold"].values())


def units(x, p):
    return x["sold"].get(p, [0, 0])[0]


def spend(x, prefix=None):
    return sum(v for k, v in x["spend"].items() if prefix is None or k.startswith(prefix))


def animals(x, day, a):
    return x["farm"][str(day)]["animals"].get(a, 0)


def demand(timeline):
    """Units the town's shops draw over the game: one draw every 4 steps, single-product shops x2."""
    dem = collections.Counter()
    for day in range(30):
        for shop in timeline[day]:
            for p in SHOPS[shop]:
                dem[p] += 6 * (2 if len(SHOPS[shop]) == 1 else 1)
    return dem


def yarn_day(timeline):
    return next((day for day in range(31) if "YARN_STORE" in timeline[day]), None)


def route(g):
    return "sheep" if animals(g["me"], 14, "SHEEP") >= 10 else "cows"


def edge(g):
    gaps = {p: rev(g["opp"], p) - rev(g["me"], p) for p in PRODUCTS}
    niche = gaps["EGG"] + gaps["CARROT"] + gaps["TOMATO"]
    if gaps["WOOL"] > 0 and gaps["WOOL"] >= niche:
        return "wool"
    return "egg/carrot/tomato" if niche > 0 else "other"


for g in G:
    g["dem"] = demand(TOWN[g["id"]])
    g["yarn"] = yarn_day(TOWN[g["id"]])

bad = [g["id"] for g in G if g["resim_mismatch_steps"] or g["resim_final"] != g["replay_rewards"]]
print(f"games {len(G)} (W {len(W)}, L {len(L)}); re-simulation mismatches: {bad or 'none'}")

print("\n== totals (mean per game), ours | theirs")
for lab, gs in (("losses", L), ("wins", W)):
    print(f"  {lab:6} final {mean(g['me']['money_by_day'][-1] for g in gs):8,.0f} | {mean(g['opp']['money_by_day'][-1] for g in gs):8,.0f}"
          f"   revenue {mean(rev(g['me']) for g in gs):8,.0f} | {mean(rev(g['opp']) for g in gs):8,.0f}"
          f"   spend {mean(spend(g['me']) for g in gs):7,.0f} | {mean(spend(g['opp']) for g in gs):7,.0f}"
          f"   revenue days 0-14 {mean(sum(g['me']['rev_day'][:15]) for g in gs):7,.0f} | {mean(sum(g['opp']['rev_day'][:15]) for g in gs):7,.0f}"
          f"   days 15-29 {mean(sum(g['me']['rev_day'][15:]) for g in gs):7,.0f} | {mean(sum(g['opp']['rev_day'][15:]) for g in gs):7,.0f}")

print("\n== revenue per product (mean $ per game), ours | theirs; the losses' middle column is that line's share of the mean loss margin")
for p in PRODUCTS:
    lm, lo = mean(rev(g["me"], p) for g in L), mean(rev(g["opp"], p) for g in L)
    wm, wo = mean(rev(g["me"], p) for g in W), mean(rev(g["opp"], p) for g in W)
    print(f"  {p:11} losses {lm:7,.0f} | {lo:7,.0f} {lo - lm:+8,.0f}    wins {wm:7,.0f} | {wo:7,.0f}")
for cat in ("seed:", "animal:", "product:", "hire", "land"):
    lm, lo = mean(spend(g["me"], cat) for g in L), mean(spend(g["opp"], cat) for g in L)
    wm, wo = mean(spend(g["me"], cat) for g in W), mean(spend(g["opp"], cat) for g in W)
    print(f"  spend {cat:9} {lm:7,.0f} | {lo:7,.0f} {lm - lo:+8,.0f}    wins {wm:7,.0f} | {wo:7,.0f}")

print("\n== units sold and mean price per product (per game), ours | theirs")
for p in PRODUCTS:
    cells = []
    for gs in (L, W):
        for side, _ in SIDES:
            u = sum(units(g[side], p) for g in gs)
            cells.append(f"{u / len(gs):4.0f}u @{(sum(rev(g[side], p) for g in gs) / u) if u else float('nan'):6.1f}")
    print(f"  {p:11} losses {cells[0]} | {cells[1]}    wins {cells[2]} | {cells[3]}")

print("\n== net trading (mean per game): wheat revenue - wheat bought - wheat seed; fertilizer revenue - fertilizer bought")
for lab, gs in (("losses", L), ("wins", W)):
    for side, name in SIDES:
        wheat = mean(rev(g[side], "WHEAT") - g[side]["spend"].get("product:WHEAT", 0) - g[side]["spend"].get("seed:WHEAT", 0) for g in gs)
        fert = mean(rev(g[side], "FERTILIZER") - g[side]["spend"].get("product:FERTILIZER", 0) for g in gs)
        print(f"  {lab:6} {name:6} wheat bought {mean(g[side]['spend'].get('product:WHEAT', 0) for g in gs):7,.0f}  net wheat {wheat:7,.0f}  net fertilizer {fert:7,.0f}")

print("\n== cumulative revenue gap (theirs - ours) at the end of day d")
for lab, gs in (("losses", L), ("wins", W)):
    print(f"  {lab:6} " + "  ".join(f"d{d}:{mean(sum(g['opp']['rev_day'][:d + 1]) - sum(g['me']['rev_day'][:d + 1]) for g in gs):+7,.0f}"
                                   for d in (3, 6, 9, 12, 15, 18, 21, 24, 27, 29)))

print("\n== farm at hour 12 of day d (mean), ours | theirs")
for lab, gs in (("losses", L), ("wins", W)):
    for d in (5, 8, 11, 14, 20):
        f = lambda side, key: mean(g[side]["farm"][str(d)][key] for g in gs)
        print(f"  {lab:6} d{d:<2} quadrants {f('me', 'quadrants'):.2f} | {f('opp', 'quadrants'):.2f}  hands {f('me', 'hands'):4.1f} | {f('opp', 'hands'):4.1f}"
              + "".join(f"  {a.lower()} {mean(animals(g['me'], d, a) for g in gs):4.1f} | {mean(animals(g['opp'], d, a) for g in gs):4.1f}" for a in ANIMALS))
for lab, gs in (("losses", L), ("wins", W)):
    for side, name in SIDES:
        c = collections.Counter("/".join(str(animals(g[side], 14, a)) for a in ANIMALS) for g in gs)
        print(f"  {lab:6} {name:6} goose/cow/sheep on day 14: {dict(c.most_common(6))}")
for lab, gs in (("losses", L), ("wins", W)):
    for side, name in SIDES:
        print(f"  {lab:6} {name:6} quadrants at the end: {dict(sorted(collections.Counter(1 + len(g[side]['land']) for g in gs).items()))}")

print("\n== town shop demand over the game (units per game)")
print("  " + "  ".join(f"{p.lower()} {mean(g['dem'][p] for g in L):4.0f} | {mean(g['dem'][p] for g in W):4.0f}" for p in PRODUCTS if p not in ("MELON", "FERTILIZER")) + "   (losses | wins)")
for p in ("STRAWBERRY", "MILK", "WOOL"):
    xs = sorted(G, key=lambda g: g["dem"][p])
    lo, hi = xs[:len(xs) // 2], xs[len(xs) // 2:]
    price = lambda gs: sum(rev(g["me"], p) for g in gs) / max(1, sum(units(g["me"], p) for g in gs))
    print(f"  {p:10} low demand: our revenue {mean(rev(g['me'], p) for g in lo):7,.0f} at {price(lo):5.1f}, lost {sum(g['res'] == 'L' for g in lo)}/{len(lo)}"
          f"   high demand: {mean(rev(g['me'], p) for g in hi):7,.0f} at {price(hi):5.1f}, lost {sum(g['res'] == 'L' for g in hi)}/{len(hi)}")

print("\n== our route (sheep >= 10 on day 14) against the day the yarn store opened")
table = collections.defaultdict(collections.Counter)
for g in G:
    table[(route(g), g["yarn"])][g["res"]] += 1
for (r, y), c in sorted(table.items(), key=lambda kv: (kv[0][0], 99 if kv[0][1] is None else kv[0][1])):
    print(f"  {r:5} yarn store day {str(y):>4}: W {c['W']:2d}  L {c['L']:2d}")

print("\n== where the opponent's revenue edge came from, per loss:", dict(collections.Counter(edge(g) for g in L)))
for k in ("wool", "egg/carrot/tomato"):
    gs = [g for g in L if edge(g) == k]
    print(f"  {k:17} n={len(gs):2d} mean margin {mean(g['diff'] for g in gs):+8,.0f}; sheep on day 14 theirs {mean(animals(g['opp'], 14, 'SHEEP') for g in gs):.1f} "
          f"ours {mean(animals(g['me'], 14, 'SHEEP') for g in gs):.1f}; geese theirs {mean(animals(g['opp'], 14, 'GOOSE') for g in gs):.1f}; "
          f"revenue gaps " + " ".join(f"{p.lower()} {mean(rev(g['opp'], p) - rev(g['me'], p) for g in gs):+,.0f}" for p in ("WOOL", "EGG", "CARROT", "TOMATO", "STRAWBERRY", "MILK")))

print("\n== every loss: seed, opponent (current rating of that submission), margin, yarn-store day, shops by day 12, largest revenue gaps")
for g in sorted(L, key=lambda g: g["t"]):
    gaps = sorted(((rev(g["opp"], p) - rev(g["me"], p), p) for p in PRODUCTS), reverse=True)
    print(f"  {g['id']} {g['t'][11:16]} seat {g['seat']} seed {g['seed']:>10}  {g['team'][:20]:<20} {str(g['orate']):>6}  {g['diff']:+7,}"
          f"  yarn {str(g['yarn']):>4}  {' '.join(ABBR[s] for s in TOWN[g['id']][12]):<17}  {route(g):5} | "
          f"{'/'.join(str(animals(g['opp'], 14, a)) for a in ANIMALS):7}  "
          + ", ".join(f"{p.lower()} {v:+,.0f}" for v, p in gaps[:3] if v > 0))

print("\n== opening market orders of days 0-1 (first 6)")
fmt = lambda x: " ".join("/".join(str(v) for v in o) for o in x["opening"][:6])
print("  ours:", fmt(G[0]["me"]))
for lab, gs in (("losses", L), ("wins", W)):
    c = collections.Counter(fmt(g["opp"]) for g in gs)
    print(f"  {lab}: " + "; ".join(f"{n}x {o}" for o, n in c.most_common(5)))
