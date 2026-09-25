"""Losses against wins from market_audit.json: glutted markets, farm waste, sell timing, time bank.

usage: python market_report.py <market_audit.json> <stats.json> <shops.json>
"""
import collections
import json
import statistics as st
import sys

PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]
BASE = {"WHEAT": 25, "CARROT": 35, "TOMATO": 60, "STRAWBERRY": 120, "MELON": 250, "EGG": 50, "MILK": 160,
        "WOOL": 200, "FERTILIZER": 100}
SHOPS = {  # kaggle_environments/envs/kaggriculture/kaggriculture.py (1.32.7)
    "BAKERY": ["EGG", "WHEAT"], "PIZZA_SHOP": ["MILK", "TOMATO", "WHEAT"],
    "BRUNCH_SPOT": ["EGG", "WHEAT", "STRAWBERRY"], "YARN_STORE": ["WOOL"],
    "ICE_CREAM_SHOP": ["STRAWBERRY", "MILK", "WHEAT"], "PET_CAFE": ["CARROT"],
    "SMOOTHIE_SHOP": ["STRAWBERRY", "MILK"], "FARMERS_MARKET": ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY"],
}
SEED = {"WHEAT": 10, "CARROT": 20, "TOMATO": 50, "STRAWBERRY": 100, "MELON": 80}

A = {g["id"]: g for g in json.load(open(sys.argv[1]))}
S = {g["id"]: g for g in json.load(open(sys.argv[2]))}
TOWN = {int(k): v for k, v in json.load(open(sys.argv[3])).items()}
G = [A[i] for i in S if i in A]
L = [g for g in G if g["res"] == "L"]
W = [g for g in G if g["res"] == "W"]
SIDES = (("me", "ours"), ("opp", "theirs"))


def mean(xs):
    xs = list(xs)
    return st.mean(xs) if xs else float("nan")


def acc(x, key):
    return x["acc"].get(key, 0)


def acc_sum(x, prefix):
    return sum(v for k, v in x["acc"].items() if k.startswith(prefix))


def timing(x, p):
    return x["timing"].get(p, [0, 0, 0.0, 0, 0])


def buyers(g, product, by_day):
    """Shop instances open on `by_day` that buy `product`."""
    return sum(product in SHOPS[s] for s in TOWN[g["id"]][by_day])


bad = [g["id"] for g in G if g["resim_mismatch_steps"]]
print(f"games {len(G)} (W {len(W)}, L {len(L)}); re-simulation mismatches: {bad or 'none'}")

print("\n== sales into a glutted market (mean per game): units sold | at the $1 floor | below the base price | revenue;"
      "\n   glut = share of the game's 719 steps with the market's stock above its neutral level (price below base)")
for p in PRODUCTS:
    for lab, gs in (("losses", L), ("wins", W)):
        cells = []
        for side, _ in SIDES:
            t = [timing(g[side], p) for g in gs]
            cells.append(f"{mean(x[0] for x in t):4.0f}u {mean(x[3] for x in t):4.0f}@$1 {mean(x[4] for x in t):4.0f}<base ${mean(x[1] for x in t):7,.0f}")
        glut = mean(g["market"][p]["glut_steps"] / 719 for g in gs)
        print(f"  {p:10} {lab:6} glut {glut:4.0%}   ours {cells[0]}   theirs {cells[1]}")

print("\n== strawberries and milk against the town: shops buying the product on day 15 (strawberries and milk come in from ~day 12)")
for p, cost_key in (("STRAWBERRY", "BUY_SEED:STRAWBERRY"), ("MILK", "BUY_ANIMAL:COW")):
    for k in (0, 1, 2):
        gs = [g for g in G if min(buyers(g, p, 15), 2) == k]
        if not gs:
            continue
        rows = []
        for side, name in SIDES:
            t = [timing(g[side], p) for g in gs]
            u = sum(x[0] for x in t)
            spent = mean(sum(g[side]["buys_by_day"].get(cost_key, [0])) for g in gs)
            rows.append(f"{name} {mean(x[0] for x in t):4.0f}u at ${(sum(x[1] for x in t) / u) if u else 0:5.1f}, "
                        f"{mean(x[3] for x in t):4.0f} at $1, revenue ${mean(x[1] for x in t):7,.0f}, "
                        f"{'seeds' if p == 'STRAWBERRY' else 'cows'} bought {spent:4.1f}")
        print(f"  {p:10} {('2+' if k == 2 else str(k))} buyer shops: {len(gs):2d} games, lost {sum(g['res'] == 'L' for g in gs):2d}   " + "   ".join(rows))

print("\n== market price at the start of day d (mean), losses | wins")
for p in ("STRAWBERRY", "MILK", "MELON", "FERTILIZER", "WOOL", "WHEAT", "EGG", "CARROT", "TOMATO"):
    print(f"  {p:10} " + "  ".join(f"d{d}:{mean(g['market'][p]['price_by_day'][d] for g in L):4.0f}|{mean(g['market'][p]['price_by_day'][d] for g in W):4.0f}"
                                  for d in (5, 10, 15, 18, 21, 24, 27, 30)))

print("\n== late-game gap: revenue theirs - ours per product (mean per game, losses), days 0-14 | days 15-29")
for p in PRODUCTS:
    early = mean(sum(g["opp"]["rev_by_day"].get(p, [0] * 30)[:15]) - sum(g["me"]["rev_by_day"].get(p, [0] * 30)[:15]) for g in L)
    late = mean(sum(g["opp"]["rev_by_day"].get(p, [0] * 30)[15:]) - sum(g["me"]["rev_by_day"].get(p, [0] * 30)[15:]) for g in L)
    late_w = mean(sum(g["opp"]["rev_by_day"].get(p, [0] * 30)[15:]) - sum(g["me"]["rev_by_day"].get(p, [0] * 30)[15:]) for g in W)
    print(f"  {p:10} losses {early:+8,.0f} | {late:+8,.0f}     (wins, days 15-29: {late_w:+8,.0f})")

print("\n== sell timing: price received / highest price of that product in the 24 steps before the sale (unit-weighted mean)")
for p in PRODUCTS:
    cells = []
    for gs in (L, W):
        for side, _ in SIDES:
            t = [timing(g[side], p) for g in gs]
            u = sum(x[0] for x in t)
            cells.append(f"{(sum(x[2] for x in t) / u) if u else float('nan'):5.2f}")
    print(f"  {p:10} losses ours {cells[0]} theirs {cells[1]}    wins ours {cells[2]} theirs {cells[3]}")

print("\n== farm waste (mean per game), ours | theirs")
rows = [
    ("animal-days", lambda x: acc_sum(x, "animal_days:")),
    ("unfed animal-days", lambda x: acc_sum(x, "unfed_days:")),
    ("animals escaped", lambda x: acc_sum(x, "escaped:")),
    ("care days (cared + fed)", lambda x: acc_sum(x, "care_days:")),
    ("care bonus units paid", lambda x: acc_sum(x, "care_bonus_paid:")),
    ("care bonus forfeited (unfed)", lambda x: acc_sum(x, "care_bonus_forfeited_unfed:")),
    ("animal units lost at max_held", lambda x: sum(acc(x, f"lost_at_cap:{a}") for a in ("COW", "SHEEP", "GOOSE"))),
    ("crop units lost at max_yield", lambda x: sum(acc(x, f"lost_at_cap:{c}") for c in ("TOMATO", "STRAWBERRY"))),
    ("plants withered (unwatered)", lambda x: acc_sum(x, "withered:")),
    ("crop units rotted in the field", lambda x: acc_sum(x, "rotted_units:")),
    ("empty tile-days", lambda x: acc(x, "empty_tile_days")),
    ("shed overflow discarded", lambda x: acc(x, "drop_discarded") + acc(x, "eod_discarded")),
    ("days ending with a full shed", lambda x: acc(x, "eod_shed_full_days")),
    ("unit-steps (farmer + hands)", lambda x: acc(x, "unit_steps")),
    ("  PASS", lambda x: acc(x, "act:PASS")),
    ("  moves", lambda x: acc(x, "act:MOVE")),
    ("  no-op actions", lambda x: sum(acc(x, k) for k in ("harvest_noop", "water_noop", "care_noop", "plant_noop", "drop_noop", "collect_fert_noop"))
     + acc_sum(x, "feed_noop:")),
    ("  FEED with no wheat in hand", lambda x: acc(x, "feed_noop:no_wheat_in_hand")),
]
for name, f in rows:
    print(f"  {name:32} losses {mean(f(g['me']) for g in L):7.1f} | {mean(f(g['opp']) for g in L):7.1f}"
          f"    wins {mean(f(g['me']) for g in W):7.1f} | {mean(f(g['opp']) for g in W):7.1f}")
for lab, gs in (("losses", L), ("wins", W)):
    for side, name in SIDES:
        c = collections.Counter()
        for g in gs:
            c.update(g[side]["refused"])
        print(f"  refused market orders, {lab} {name}: " + ", ".join(f"{k} {v / len(gs):.1f}" for k, v in c.most_common(8)))

print("\n== harvest size (units per harvest), ours | theirs")
for p in PRODUCTS:
    def per(gs, side):
        h = sum(acc(g[side], f"harvests:{p}") for g in gs)
        return (sum(acc(g[side], f"harvested:{p}") for g in gs) / h) if h else float("nan")
    print(f"  {p:10} losses {per(L, 'me'):4.2f} | {per(L, 'opp'):4.2f}    wins {per(W, 'me'):4.2f} | {per(W, 'opp'):4.2f}")

print("\n== time bank (remainingOverageTime, s; 60 at the start): lowest value in the game")
for side, name in SIDES:
    lows = [g[side]["overage_start_end_min"][2] for g in G if g[side]["overage_start_end_min"]]
    print(f"  {name:6} mean {mean(lows):5.1f}, worst {min(lows):5.1f}, games below 50: {sum(x < 50 for x in lows)}, below 30: {sum(x < 30 for x in lows)}")
worst = sorted(G, key=lambda g: g["me"]["overage_start_end_min"][2])[:5]
print("  our five lowest: " + ", ".join(f"{g['id']} ({g['res']}) {g['me']['overage_start_end_min'][2]:.1f}" for g in worst))

print("\n== W-L by the shops buying milk (rows) and strawberries (columns) open on day 15")
tab = collections.defaultdict(collections.Counter)
for g in G:
    tab[min(buyers(g, "MILK", 15), 2)][min(buyers(g, "STRAWBERRY", 15), 2), g["res"]] += 1
print("              strawberry 0  strawberry 1  strawberry 2+")
for r in (0, 1, 2):
    print(f"  milk {('2+' if r == 2 else r):>2}      " + "     ".join(f"{tab[r][c, 'W']:>3} - {tab[r][c, 'L']:<3}" for c in (0, 1, 2)))
print("  by fit = shop instances open on day 15 buying milk + those buying strawberries (each adds 6 units a day):")
for f in range(1, 6):
    gs = [g for g in G if min(buyers(g, "MILK", 15) + buyers(g, "STRAWBERRY", 15), 5) == f]
    if gs:
        print(f"    fit {f}{'+' if f == 5 else ' '} {len(gs):2d} games  W {sum(g['res'] == 'W' for g in gs):2d} L {sum(g['res'] == 'L' for g in gs):2d}"
              f"   final cash ours {mean(S[g['id']]['me']['money_by_day'][-1] for g in gs):8,.0f} theirs {mean(S[g['id']]['opp']['money_by_day'][-1] for g in gs):8,.0f}")

print("\n== our cows by the milk shops open on day 15: milk revenue per cow-day against the feed (1 wheat a day)")
for k in (0, 1, 2):
    gs = [g for g in G if min(buyers(g, "MILK", 15), 2) == k]
    rev = sum(sum(g["me"]["rev_by_day"].get("MILK", [0] * 30)) for g in gs)
    days = sum(acc(g["me"], "animal_days:COW") for g in gs)
    print(f"  milk shops {('2+' if k == 2 else k)}: {len(gs):2d} games, {days / len(gs):4.0f} cow-days a game, milk revenue ${rev / len(gs):7,.0f} = "
          f"${rev / days:5.1f} per cow-day; wheat (the feed) ${mean(mean(g['market']['WHEAT']['price_by_day'][8:30]) for g in gs):4.1f} on days 8-29")

print("\n== opponents' geese (games where they held geese on day 14)")
gs = [g for g in G if S[g["id"]]["opp"]["farm"]["14"]["animals"].get("GOOSE")]
gd = sum(acc(g["opp"], "animal_days:GOOSE") for g in gs)
eggs = [timing(g["opp"], "EGG") for g in gs]
print(f"  {len(gs)} games (our result W {sum(g['res'] == 'W' for g in gs)} L {sum(g['res'] == 'L' for g in gs)}): "
      f"{mean(S[g['id']]['opp']['farm']['14']['animals']['GOOSE'] for g in gs):.1f} geese, {gd / len(gs):.0f} goose-days, "
      f"{sum(x[0] for x in eggs) / len(gs):.0f} eggs at ${sum(x[1] for x in eggs) / max(1, sum(x[0] for x in eggs)):.1f} a game "
      f"= {sum(x[0] for x in eggs) / gd:.2f} eggs and ${sum(x[1] for x in eggs) / gd:.0f} per goose-day")

print("\n== opponents rated >= 1400 now: farm on day 14 and revenue")
for g in sorted(G, key=lambda g: -(S[g["id"]].get("orate") or 0)):
    s = S[g["id"]]
    if (s.get("orate") or 0) < 1400:
        continue
    f14 = s["opp"]["farm"]["14"]
    rev = sorted(s["opp"]["sold"].items(), key=lambda kv: -kv[1][1])
    print(f"  {s['team'][:16]:16} {s['orate']:6.0f} {g['res']} {g['diff']:+8,}  quadrants {1 + len(s['opp']['land'])}, hands {f14['hands']}, "
          f"animals {f14['animals']}, plants {f14['plants']}; spent ${sum(s['opp']['spend'].values()):,}")
    print("      revenue " + ", ".join(f"{p.lower()} {v[1]:,}" for p, v in rev))

print("\n== the opponent family with 33 strawberry and 23-24 wheat plants on day 14 (ours: 38 and 22)")


def family(g):
    plants = S[g["id"]]["opp"]["farm"]["14"]["plants"]
    return plants.get("STRAWBERRY") == 33 and plants.get("WHEAT", 0) in (23, 24)


for name, gs in (("family", [g for g in G if family(g)]), ("everyone else", [g for g in G if not family(g)])):
    rated = [S[g["id"]].get("orate") for g in gs]
    print(f"  {name:13} {len(gs):2d} games  W {sum(g['res'] == 'W' for g in gs):2d} L {sum(g['res'] == 'L' for g in gs):2d}  mean margin {mean(g['diff'] for g in gs):+8,.0f};"
          f" opponents rated >= 1200 now: {sum(1 for r in rated if r and r >= 1200)}; melon revenue exactly $14,389: "
          f"{sum(S[g['id']]['opp']['sold'].get('MELON', [0, 0])[1] == 14389 for g in gs)}")
fam = [g for g in G if family(g)]
herds = collections.Counter("/".join(str(S[g["id"]]["opp"]["farm"]["14"]["animals"].get(a, 0)) for a in ("GOOSE", "COW", "SHEEP")) for g in fam)
print(f"  family herds on day 14 (geese/cows/sheep): {dict(herds.most_common())}")
print("  family games, revenue theirs - ours per game: " + ", ".join(
    f"{p.lower()} {mean(S[g['id']]['opp']['sold'].get(p, [0, 0])[1] - S[g['id']]['me']['sold'].get(p, [0, 0])[1] for g in fam):+,.0f}" for p in PRODUCTS))
