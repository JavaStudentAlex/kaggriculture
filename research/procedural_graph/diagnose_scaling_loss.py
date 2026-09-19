import json
from collections import Counter
from pathlib import Path

replay_path = Path("/home/alex/kaggriculture/research/procedural_graph/ladder_losses/episode-109205361-replay.json")
data = json.loads(replay_path.read_text())
steps = data["steps"]
info = data.get("info", {})

teams = info.get("TeamNames", [])
my_seat = 0 if "Sunshine through fog" in teams[0] else 1
opp_seat = 1 - my_seat

print(f"Episode 109205361: My seat={my_seat} ({teams[my_seat]}), Opp seat={opp_seat} ({teams[opp_seat]})")

# Track total sales by product for each seat
sells = [Counter(), Counter()]
buys = [Counter(), Counter()]

for t in range(1, len(steps)):
    for seat in (0, 1):
        act = steps[t][seat].get("action", {})
        for order in act.get("market", []):
            if not isinstance(order, list) or len(order) < 3:
                continue
            op, prod, qty = order[0], order[1], order[2]
            if op == "SELL":
                sells[seat][prod] += qty
            elif op == "BUY_SEED":
                buys[seat][prod] += qty

print("\n=== Lifetime Seed Buys ===")
print("Hazel Weir buys:   ", dict(buys[my_seat]))
print("Opponent buys:     ", dict(buys[opp_seat]))

print("\n=== Lifetime Market Sells (Total Units Requested) ===")
print("Hazel Weir sells:  ", dict(sells[my_seat]))
print("Opponent sells:    ", dict(sells[opp_seat]))

# Check town shops unlocked
final_obs = steps[-1][0]["observation"]
town = final_obs.get("town", {})
print("\nFinal Unlocked Shops:", town.get("unlocked_shops"))
print("Final Cash: Hazel=$", final_obs["farms"][my_seat]["money"], " | Opponent=$", final_obs["farms"][opp_seat]["money"])
