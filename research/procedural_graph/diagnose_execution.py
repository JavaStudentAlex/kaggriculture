import json
from pathlib import Path

replay_path = Path("/home/alex/kaggriculture/research/procedural_graph/ladder_losses/episode-109205361-replay.json")
data = json.loads(replay_path.read_text())
steps = data["steps"]

# Trace all steps where Hazel placed a SELL order for MILK or WOOL or STRAWBERRY, but money did NOT increase!
unexecuted_sells = 0
executed_sells = 0

for t in range(len(steps) - 1):
    cur_m = steps[t][0]["observation"]["farms"][0]["money"]
    nxt_m = steps[t+1][0]["observation"]["farms"][0]["money"]
    dm = nxt_m - cur_m
    
    act = steps[t+1][0].get("action", {})
    sells = [o for o in act.get("market", []) if isinstance(o, list) and len(o) >= 3 and o[0] == "SELL"]
    if sells:
        if dm > 0:
            executed_sells += 1
        else:
            unexecuted_sells += 1
            if unexecuted_sells <= 8:
                shed = steps[t][0]["observation"].get("private", {}).get("shed", {})
                print(f"Step {t:3d}: Sells requested={sells} | Shed had={shed} | cash_delta=${dm}")

print(f"\nTotal SELL turns: {executed_sells + unexecuted_sells} | Executed (cash > 0): {executed_sells} | Unexecuted (cash == 0): {unexecuted_sells}")
