"""End-to-end: crowned graph -> champion -> played by shard logic -> pool grows."""
import collections
import copy
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path("/home/alex/kaggriculture")
sys.path.insert(0, str(ROOT / "research" / "procedural_graph"))
import champion_pool as cp

stage = Path(tempfile.mkdtemp()) / "pool"
stage.mkdir(parents=True)
real = sorted((ROOT / "shinka" / "champions" / "pool").glob("champ_*.py"))[:3]
for r in real:
    shutil.copy2(r, stage / r.name)
print("seed pool:", len(cp.list_champions(stage)))

graph = json.loads((ROOT / "research" / "procedural_graph" / "policy_graph.json").read_text())
g2 = copy.deepcopy(graph)
g2["name"] = "variant_two"

sizes = []
for i, (g, label) in enumerate([(graph, "A"), (g2, "B"), (graph, "A-again")], 1):
    p, st = cp.induct_graph_champion(g, stage, iteration=i, island="Island-" + label,
                                     win_rate=1.0, cash=2500)
    n = len(cp.list_champions(stage))
    shards = cp.plan_shard_split(n, 5, 2)
    sizes.append((label, st, n, shards))
    print("  iter%d graph=%-8s -> %-9s pool=%d shards=%d" % (i, label, st, n, shards))

assert sizes[-1][1] == "duplicate", "repeat graph must dedupe"
assert sizes[1][2] == 5, "pool should grow to 5"
print("RATCHET OK: pool grew 3 -> 5, duplicate rejected, shard count adapted")

os.environ["KAGG_TASK_DIR"] = str(ROOT)
os.environ["KAGG_HISTORY_DIR"] = str(stage)
for p in [ROOT, ROOT / "shinka" / "evolution",
          ROOT / "shinka" / "champions" / "dependencies" / "mohui_v66"]:
    sys.path.insert(0, str(p))
import evaluate
from kaggle_environments import make

champs = evaluate.list_champions()
print("evaluate.list_champions sees:", len(champs))
assert len(champs) == 5

buckets = [cp.assign_champions(champs, i, 3) for i in range(3)]
flat = [c.name for b in buckets for c in b]
covers = sorted(flat) == sorted(c.name for c in champs)
print("split across 3 notebooks:", [len(b) for b in buckets], "covers all:", covers)
assert covers

new_champ = [c for c in champs if c.name.startswith("champ_evo_")][0]
print("playing new champion:", new_champ.name)
a0 = evaluate.load_agent(str(new_champ), "newchamp")
a1 = evaluate.load_agent(str(real[0]), "oldchamp")
env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": 303})
env.run([a0, a1])
last = env.steps[-1]
c = collections.Counter(pl.get("status") for s in env.steps for pl in s)
print("statuses:", dict(c), "steps:", len(env.steps))
print("new=%.2f old=%.2f" % (float(last[0].get("reward") or 0), float(last[1].get("reward") or 0)))
assert c.get("ERROR", 0) == 0, "champion must not error"
print("INTEGRATION OK")
