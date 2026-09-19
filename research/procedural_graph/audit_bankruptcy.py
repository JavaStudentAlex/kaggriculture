import json
from pathlib import Path

replay_path = Path("/home/alex/kaggriculture/research/procedural_graph/ladder_losses/episode-109577172-replay.json")
data = json.loads(replay_path.read_text())
steps = data["steps"]

print("=== Audit around Day 10 Midnight (Steps 235 - 245) ===")
for t in range(235, 245):
    s0 = steps[t][0]
    obs = s0["observation"]
    f0 = obs["farms"][0]
    m = f0.get("money")
    hands = f0.get("hands", [])
    act = s0.get("action", {})
    hour = obs.get("hour")
    day = obs.get("day")
    print(f"t={t:3d} (d{day} h{hour:02d}): cash=${m:6.1f}, n_hands={len(hands)} | market_act={act.get('market')} | farmer={act.get('farmer')}")

print("\n=== Audit around Day 13 Midnight (Steps 305 - 315) ===")
for t in range(305, 316):
    s0 = steps[t][0]
    obs = s0["observation"]
    f0 = obs["farms"][0]
    m = f0.get("money")
    hands = f0.get("hands", [])
    act = s0.get("action", {})
    hour = obs.get("hour")
    day = obs.get("day")
    print(f"t={t:3d} (d{day} h{hour:02d}): cash=${m:6.1f}, n_hands={len(hands)} | market_act={act.get('market')} | farmer={act.get('farmer')}")
