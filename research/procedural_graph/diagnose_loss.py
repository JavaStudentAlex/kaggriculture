import json
from pathlib import Path

replay_path = Path("/home/alex/kaggriculture/research/procedural_graph/ladder_losses/episode-109577172-replay.json")
data = json.loads(replay_path.read_text())
steps = data["steps"]

print(f"Total steps: {len(steps)}")
print("Step 0 farm 0 keys:", list(steps[0][0]["observation"]["farms"][0].keys()))

print("--- Telemetry Sampling ---")
for t in range(0, len(steps), 24):
    obs = steps[t][0]["observation"]
    f0 = obs["farms"][0]
    f1 = obs["farms"][1]
    m0 = f0.get("money", 0)
    m1 = f1.get("money", 0)
    d = obs.get("day", t // 24)
    h = obs.get("hour", t % 24)
    print(f"Step {t:3d} (day {d:2d} hr {h:2d}): S0 cash = ${m0:9.2f} | S1 cash = ${m1:9.2f}")

print("\n--- Last 10 steps ---")
for t in range(710, 720):
    obs = steps[t][0]["observation"]
    f0 = obs["farms"][0]
    f1 = obs["farms"][1]
    r0 = steps[t][0].get("reward")
    r1 = steps[t][1].get("reward")
    act0 = steps[t][0].get("action")
    print(f"Step {t}: S0 cash=${f0.get('money')}, reward={r0}, action={act0}")
    print(f"         S1 cash=${f1.get('money')}, reward={r1}")
