import json
from pathlib import Path

replay_path = Path("/home/alex/kaggriculture/research/procedural_graph/ladder_losses/episode-109577172-replay.json")
data = json.loads(replay_path.read_text())
steps = data["steps"]

print("=== Day 9 Trace (Steps 216 - 236) ===")
for t in range(216, 237):
    s0 = steps[t][0]
    obs = s0["observation"]
    f0 = obs["farms"][0]
    m = f0.get("money")
    hands = f0.get("hands", [])
    act = s0.get("action", {})
    hour = obs.get("hour")
    day = obs.get("day")
    print(f"t={t:3d} (d{day} h{hour:02d}): cash=${m:7.1f}, hands={len(hands)} | market_act={act.get('market')} | farmer={act.get('farmer')}")
