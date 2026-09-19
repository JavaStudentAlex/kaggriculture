import json
from pathlib import Path

replay_path = Path("/home/alex/kaggriculture/research/procedural_graph/ladder_losses/episode-109205361-replay.json")
data = json.loads(replay_path.read_text())
steps = data["steps"]

# Track cash delta vs units sold to see average price realized
revenue_by_product = [{"WOOL": 0, "MILK": 0, "STRAWBERRY": 0, "MELON": 0, "WHEAT": 0, "FERTILIZER": 0, "CARROT": 0},
                       {"WOOL": 0, "MILK": 0, "STRAWBERRY": 0, "MELON": 0, "WHEAT": 0, "FERTILIZER": 0, "CARROT": 0}]

for t in range(len(steps) - 1):
    cur_step = steps[t]
    nxt_step = steps[t + 1]
    
    # Observe market prices at t
    mkt = cur_step[0]["observation"]["market"]
    prices = mkt.get("prices", {})
    
    # Orders placed from obs t are in steps[t+1]
    for s in (0, 1):
        m0 = cur_step[s]["observation"]["farms"][s]["money"]
        m1 = nxt_step[s]["observation"]["farms"][s]["money"]
        dm = m1 - m0
        
        act = nxt_step[s].get("action", {})
        for order in act.get("market", []):
            if isinstance(order, list) and len(order) >= 3 and order[0] == "SELL":
                p, q = order[1], order[2]
                unit_p = prices.get(p, 0)
                revenue_by_product[s][p] = revenue_by_product[s].get(p, 0) + unit_p * q

print("=== Theoretical Gross Revenue by Product (Orders * Price at Turn) ===")
for p in ["WOOL", "MILK", "STRAWBERRY", "MELON", "WHEAT", "FERTILIZER", "CARROT"]:
    h_rev = revenue_by_product[0].get(p, 0)
    o_rev = revenue_by_product[1].get(p, 0)
    print(f"{p:12s} | Hazel: ${h_rev:8.1f} | Opponent: ${o_rev:8.1f} | Delta: ${o_rev - h_rev:8.1f}")

# Check price trajectory of Milk, Wool, Strawberry over the game
print("\n--- Price trajectory sample ---")
for t in range(0, len(steps), 48):
    p_t = steps[t][0]["observation"]["market"]["prices"]
    print(f"Step {t:3d} (day {t//24:02d}): Wool=${p_t.get('WOOL'):3d}, Milk=${p_t.get('MILK'):3d}, Strawb=${p_t.get('STRAWBERRY'):3d}, Wheat=${p_t.get('WHEAT'):3d}")
