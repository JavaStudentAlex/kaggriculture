"""Final mutation run on FULL graphs (predictor + KAD on), 2026-09-30 evening. Seed = candidate E (Sorrel Meadow).
Gauntlet (335 games): the lost/tied ladder games of our last five submissions that a small change can flip (margin
>= -$2,000: 243), the won replays of the 179-game set with margin <= $1,000 (72, so a change cannot win by giving
wins away), and 20 games against Ice and Fire 09-30. Queue: three hand-picked predictor/engine edits."""
import json, shutil
from pathlib import Path
K = Path("/home/alex/kagg-evo")
OUT = K / "final-0930"
PG = K / "lit0930e/repo/research/procedural_graph"
margins = {}
for d in (K / "lost5/bundles", K / "replays_all0930"):
    for f in d.glob("replay_*/SOURCE.json"):
        s = json.loads(f.read_text()); margins[s["name"]] = s.get("recorded_margin_for_us") or 0
full = json.load(open(OUT / "plan.json"))
plan, ice = [], 0
for p in full:
    if p["set"] == "replay":
        m = margins[p["tag"]]
        if (m <= 0 and m >= -2000) or (0 < m <= 1000):
            plan.append(p)
    elif p["tag"] == "leo_icefire_0930" and ice < 20:
        plan.append(p); ice += 1
json.dump(plan, open(OUT / "plan_full.json", "w"), indent=0)
shutil.copy(K / "candidate_b/candE.json", OUT / "seed_graph_full.json")
BLIND = ["MILK", "WOOL", "STRAWBERRY", "EGG", "MELON"]
queue = [
    {"island": "Island-KAD-Sell", "edit": {"engine_parameters": {"_OR2_SN_K": 1}},
     "rationale": "Ice and Fire's author switched this on in the notebook run of 2026-09-30 15:31 (leo_pi with _OR2_SN_K 1): "
                  "the engine tracks the rival's stock from market movements and, while the rival holds MILK/STRAWBERRY/WOOL/"
                  "MELON/EGG, sells now the units its route tape plans for the next 24 steps (queued by hand 2026-09-30)."},
    {"island": "Island-KAD-Oracle", "edit": {"channels": {"rival_counter": True},
                                             "counters": {"other_opening": {"_OG_SCORE": 0.4, "_OG_ITEMS": BLIND}}},
     "rationale": "Use the predictor more where the rival emulator is blind: rivals whose opening differs from ours before "
                  "step 92 (other_opening) run no engine we emulate and include our biggest losses; for them only, the "
                  "guard front-runs at score 0.4 instead of 0.5 and also on EGG and MELON (queued by hand 2026-09-30)."},
    {"island": "Island-Oracle", "edit": {"parameters": {"_OG_ITEMS": BLIND}},
     "rationale": "The guard front-runs predicted dumps of MILK, WOOL and STRAWBERRY only; EGG and MELON are also sold in "
                  "batches by the engine family most rivals run (queued by hand 2026-09-30)."},
]
json.dump(queue, open(OUT / "queue_full.json", "w"), indent=1)
note = """

## Final run, 2026-09-30 evening (seed: candidate E = Elm Crossing + wider emulator repair + engine leo_icefire_0930)

- The gauntlet is the ladder games our last five submissions (Aspen Vale, Juniper Knoll, Hawthorn Dale, Laurel Field,
  Elm Crossing) lost or tied by at most $2,000, replayed against the rival's recorded moves, plus close wins (a change
  must not give them away) and 20 games against Ice and Fire 09-30. Half of all our losses are by less than $500:
  they are decided by sale timing, market-slot order and small quantities, not by the farm plan.
- Most rivals near our rating run the public 0927 engine or copies of it (Harvest Ledger = 0927 + a mirror detector
  that lengthens its sale look-ahead when our farm mirrors it; leo_pi / Ice and Fire = 0927 + _CA_MARGIN -22, Ice and
  Fire also _OR2_SN_K 1, which front-runs the rival's stock). The rival emulator predicts those rivals' orders and
  the race puts our sells ahead; the losses by more than $2,000 come from rivals with other openings (a third
  quadrant, geese, 18-28 wheat by day 10), which no parameter of ours has fixed.
- The predictor (oracle_guard) and KAD (kad_copilot) are required; they are used narrowly (guard: MILK/WOOL/
  STRAWBERRY at score 0.5 from step 256; KAD: WATER jobs for idle hands from step 144). Where the emulator is blind
  (rival class other_opening) the predictor is our only view of the rival.
"""
kn = PG / "evolution_knowledge_final.md"
kn.write_text((PG / "evolution_knowledge_ladder.md").read_text() + note)
from collections import Counter
print(len(plan), "games:", Counter(p["set"] for p in plan), Counter(p.get("result") for p in plan if p["set"] == "replay"))
