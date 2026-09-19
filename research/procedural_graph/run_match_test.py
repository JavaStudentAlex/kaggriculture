import sys
import time
from pathlib import Path

# Add paths
KAGG_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(KAGG_DIR / "shinka" / "evolution"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import agent_graph
from kaggle_environments import make

print("=== Running 720-step Simulation Game with Procedural Graph Agent ===")
t0 = time.perf_counter()

env = make("kaggriculture", configuration={"randomSeed": 42}, debug=True)

# Run game: agent_graph on Seat 0 vs starter agent on Seat 1
env.run([agent_graph.agent, "random"])

elapsed = time.perf_counter() - t0
state = env.state
s0 = state[0]
s1 = state[1]

print(f"Simulation completed in {elapsed:.2f}s ({elapsed / 720 * 1000:.2f} ms/step)")
print(f"Seat 0 (Procedural Graph Agent) Status: {s0.status} | Final Reward: ${s0.reward:,.2f}")
print(f"Seat 1 (Random Baseline)        Status: {s1.status} | Final Reward: ${s1.reward:,.2f}")

# Check for any errors
if s0.status != "DONE" or (s0.reward is not None and s0.reward <= 0):
    print("TEST FAILED: Unexpected terminal status or zero reward")
    sys.exit(1)
else:
    print("TEST PASSED: Agent successfully completed all 720 turns with zero errors!")
