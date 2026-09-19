import json
from pathlib import Path

replay_path = Path("/home/alex/kaggriculture/research/procedural_graph/ladder_losses/episode-109205361-replay.json")
data = json.loads(replay_path.read_text())
steps = data["steps"]

# Track real cash inflows and outflows
# Inflows: delta_m > 0
# Outflows: delta_m < 0
inflows = [0.0, 0.0]
outflows = [0.0, 0.0]
rejected_sells = [0, 0]

for t in range(len(steps) - 1):
    for s in (0, 1):
        m_cur = steps[t][0]["observation"]["farms"][s]["money"]
        m_nxt = steps[t+1][0]["observation"]["farms"][s]["money"]
        dm = m_nxt - m_cur
        if dm > 0:
            inflows[s] += dm
        elif dm < 0:
            outflows[s] += abs(dm)

print(f"Hazel  (Seat 0): Total Inflows=+${inflows[0]:,.1f} | Total Outflows=-${outflows[0]:,.1f} | Net End=${steps[-1][0]['observation']['farms'][0]['money']:,.1f}")
print(f"Opp    (Seat 1): Total Inflows=+${inflows[1]:,.1f} | Total Outflows=-${outflows[1]:,.1f} | Net End=${steps[-1][0]['observation']['farms'][1]['money']:,.1f}")

# Check shed inventory at the end
f0 = steps[-1][0]["observation"]["farms"][0]
f1 = steps[-1][0]["observation"]["farms"][1]
print("\nFinal Farm 0 (Hazel):")
print("  money:", f0["money"])
print("  unlocked_quadrants:", f0["unlocked_quadrants"])
print("  hands count:", len(f0["hands"]))
print("  shed contents:", steps[-1][0]["observation"]["private"]["shed"] if "private" in steps[-1][0]["observation"] else "private in step")
