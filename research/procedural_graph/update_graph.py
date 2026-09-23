import json

with open("/home/alex/kaggriculture/research/procedural_graph/policy_graph.json") as f:
    graph = json.load(f)

new_nodes = [
    {
      "id": "keiz_opening_scalp",
      "name": "Keiz Opening Wheat Scalp",
      "category": "market",
      "description": "On step 0, execute an opening purchase of exactly one unit of WHEAT. On step 1, sell exactly one unit of WHEAT. This scalp attempts to capture opening volatility safely and must take precedence over other early orders."
    },
    {
      "id": "tiered_shed_pressure_valve",
      "name": "Tiered Shed Pressure Valve",
      "category": "capacity",
      "description": "When shed utilization exceeds 90% (and more aggressively at 95%), relax normal price floors (to 0.15 or 0.05 of base price) to forcefully evacuate inventory. Protect a minimal feed reserve, but ensure sufficient space is freed up immediately."
    },
    {
      "id": "fertilizer_monetization",
      "name": "Fertilizer Monetization",
      "category": "revenue",
      "description": "Fertilizer has no town shop demand. When holding 3 or more units of FERTILIZER and its price is at least 45% of base, monetize the surplus immediately in small batches (e.g., 5 units), retaining exactly 1 unit to protect ongoing crop fertilization."
    },
    {
      "id": "early_endgame_liquidation",
      "name": "Predictive Early-Endgame Liquidation",
      "category": "liquidation",
      "description": "During steps 680 to 711, preemptively liquidate goods the opponent is forecast to dump (using a 24-turn horizon). Liquidate high-value assets (MELON, TOMATO, FERTILIZER, WOOL) while their market prices remain reasonably intact, ignoring ordinary shop reservations."
    },
    {
      "id": "final_turn_liquidation",
      "name": "Final Turn Forced Liquidation",
      "category": "liquidation",
      "description": "In the absolute final two turns (steps 718 and 719), execute 100% full liquidation of every item in the shed. Propose up to 10 sell orders matching total holdings, entirely ignoring price floors and reservations, to maximize final cash."
    }
]

new_edges = [
    {"source": "keiz_opening_scalp", "target": "order_scheduler", "description": "Prioritize step 0/1 WHEAT trades"},
    {"source": "tiered_shed_pressure_valve", "target": "order_scheduler", "description": "Propose high-priority shed clearance orders"},
    {"source": "fertilizer_monetization", "target": "order_scheduler", "description": "Propose fertilizer monetization orders"},
    {"source": "early_endgame_liquidation", "target": "order_scheduler", "description": "Propose preemptive liquidation orders"},
    {"source": "final_turn_liquidation", "target": "order_scheduler", "description": "Propose absolute final forced liquidation"}
]

graph["nodes"].extend(new_nodes)
graph["edges"].extend(new_edges)

with open("/home/alex/kaggriculture/research/procedural_graph/policy_graph.json", "w") as f:
    json.dump(graph, f, indent=2)
