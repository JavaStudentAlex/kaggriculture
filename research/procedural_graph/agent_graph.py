"""Procedural Graph Agent for Kaggriculture.

Drop-in replacement for Hazel Weir (56246758) that executes turns via the
compiled Procedural Graph (policy_graph.json) and ProceduralGraphEngine.

Eliminates the ladder failure modes identified in live replay audits:
- Midnight payroll defaults (episode 109577172)
- Order slot contention and shed harvest overflows (episode 109205361)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Ensure access to shinka/evolution codebase and oracle
EVO_DIR = Path(__file__).resolve().parent.parent.parent / "shinka" / "evolution"
if str(EVO_DIR) not in sys.path:
    sys.path.insert(0, str(EVO_DIR))

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import initial as hazel
from graph_engine import ProceduralGraphEngine

_ENGINE: ProceduralGraphEngine | None = None


def get_engine() -> ProceduralGraphEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = ProceduralGraphEngine(CURRENT_DIR / "policy_graph.json")
    return _ENGINE


def agent(obs: dict[str, Any], configuration: dict[str, Any] | None = None) -> dict[str, Any]:
    """Public entry point for Kaggle environments."""
    player_idx = int(obs.get("player", 0) or 0)
    base = hazel._backbone_action(obs, configuration)

    farms = obs.get("farms", []) or []
    farm = farms[player_idx] if player_idx < len(farms) else {}
    n_hands = len(farm.get("hands") or [])

    forecast = hazel._oracle_observe(obs, configuration)
    engine = get_engine()

    try:
        st = hazel.farm_state(obs, player_idx, forecast)
        guards = engine.evaluate_guards(st, obs)

        # Baseline actions
        farmer_act = hazel.evolve_farmer_action(obs, player_idx, base.get("farmer"), st)
        hands_act = hazel.evolve_hand_actions(obs, player_idx, base.get("hands"), st)
        base_market = hazel.evolve_market_orders(obs, player_idx, base.get("market"), st)

        # Segment orders by procedural graph node
        orders_by_node: dict[str, list[list[Any]]] = {
            "wage_defense": [],
            "shed_headroom": [],
            "town_shop_preempt": [],
            "oracle_frontrun": [],
            "farm_execution": []
        }

        # 1. Node: Wage Defense Orders (Priority 1)
        if guards["wage_defense"]:
            # If in payroll deficit near midnight, force emergency liquidity orders to front
            for o in base_market:
                if isinstance(o, (list, tuple)) and len(o) >= 3 and o[0] == "SELL":
                    orders_by_node["wage_defense"].append(list(o))

        # 2. Node: Shed Headroom (Priority 2)
        if guards["shed_headroom"]:
            for o in base_market:
                if isinstance(o, (list, tuple)) and len(o) >= 3 and o[0] == "SELL" and o[1] in ("WHEAT", "FERTILIZER", "STRAWBERRY"):
                    if o not in orders_by_node["wage_defense"]:
                        orders_by_node["shed_headroom"].append(list(o))

        # 3. Node: Town Shop Preempt (Priority 3)
        if guards["town_shop_preempt"]:
            for o in base_market:
                if isinstance(o, (list, tuple)) and len(o) >= 3 and o[1] in ("CARROT", "TOMATO", "EGG"):
                    orders_by_node["town_shop_preempt"].append(list(o))

        # 4. Node: Oracle Frontrun (Priority 4)
        if guards["oracle_frontrun"]:
            for o in base_market:
                if isinstance(o, (list, tuple)) and len(o) >= 3 and o[0] == "SELL" and o[1] in ("WOOL", "MILK"):
                    orders_by_node["oracle_frontrun"].append(list(o))

        # 5. Routine orders
        for o in base_market:
            orders_by_node["farm_execution"].append(list(o))

        # Topologically arbitrate up to 10 orders
        arbitrated_market = engine.arbitrate_market_orders(orders_by_node, max_orders=10)

        evolved = {
            "farmer": farmer_act,
            "hands": hands_act,
            "market": arbitrated_market
        }
        evolved = hazel.policy_sanitize(evolved, base, st)
    except Exception:
        evolved = base

    final = hazel._hard_sanitize(evolved, base, n_hands)
    hazel._oracle_record(final)
    return final
