"""Kaggle-Compatible Agent Integrating Jev System One Actor with the Procedural Graph Engine.

Combines Hazel Weir's proven backbone with:
1. Jev System One non-autoregressive strategic decisions (calibrated branch choice,
   emergency lock, expansion freeze).
2. Procedural Graph dynamic priority arbitration.
3. Resilient fallback to deterministic ProceduralGraphEngine if Jev is unreachable.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

EVO_DIR = Path(__file__).resolve().parent.parent.parent / "shinka" / "evolution"
if str(EVO_DIR) not in sys.path:
    sys.path.insert(0, str(EVO_DIR))

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import initial as hazel
from graph_engine import ProceduralGraphEngine
from jev_graph_actor import JevGraphActor

_ACTOR: Optional[JevGraphActor] = None
_FALLBACK_ENGINE: Optional[ProceduralGraphEngine] = None


def get_actor(graph_path: Optional[Path] = None, query_interval: int = 3) -> JevGraphActor:
    global _ACTOR
    target_path = graph_path or (CURRENT_DIR / "policy_graph.json")
    if _ACTOR is None:
        _ACTOR = JevGraphActor(target_path, query_interval=query_interval)
    elif graph_path is not None and _ACTOR.graph != target_path:
        _ACTOR.set_graph(target_path)
    return _ACTOR


def get_fallback_engine(graph_path: Optional[Path] = None) -> ProceduralGraphEngine:
    global _FALLBACK_ENGINE
    target_path = graph_path or (CURRENT_DIR / "policy_graph.json")
    if _FALLBACK_ENGINE is None:
        _FALLBACK_ENGINE = ProceduralGraphEngine(target_path)
    return _FALLBACK_ENGINE


def agent(obs: Dict[str, Any], configuration: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Public entry point invoked by Kaggle environments on each game step."""
    player_idx = int(obs.get("player", 0) or 0)
    base = hazel._backbone_action(obs, configuration)

    farms = obs.get("farms", []) or []
    farm = farms[player_idx] if player_idx < len(farms) else {}
    n_hands = len(farm.get("hands") or [])

    forecast = hazel._oracle_observe(obs, configuration)
    engine = get_fallback_engine()

    try:
        st = hazel.farm_state(obs, player_idx, forecast)
        guards = engine.evaluate_guards(st, obs)

        # Query Jev for dynamic graph branch selection and safety flags
        jev_decision: Optional[Dict[str, Any]] = None
        try:
            actor = get_actor()
            jev_decision = actor.query_decision(obs, player_idx, st, forecast)
        except Exception as exc:
            print(f"[agent_jev_graph ERROR at step {obs.get('step')}]: {type(exc).__name__}: {exc}", file=sys.stderr)
            jev_decision = None

        farmer_act = hazel.evolve_farmer_action(obs, player_idx, base.get("farmer"), st)
        hands_act = hazel.evolve_hand_actions(obs, player_idx, base.get("hands"), st)
        base_market = hazel.evolve_market_orders(obs, player_idx, base.get("market"), st)

        # Filter market orders based on Jev safety nouls
        if jev_decision is not None:
            filtered_market: List[Any] = []
            for order in base_market:
                if not isinstance(order, (list, tuple)) or len(order) < 1:
                    continue
                action_type = order[0]
                item_name = order[1] if len(order) > 1 else ""

                # If Jev activates emergency lock AND we have payroll deficit, cancel non-essential buy orders
                if jev_decision.get("emergency_lock") and action_type in ("BUY_SEED", "BUY_ANIMAL", "EXPAND_LAND"):
                    if st.get("money", 0.0) < (len(st.get("hands", [])) * 120 + 100):
                        continue
                # If Jev activates expansion freeze (Day >= 13), cancel land expansion & slow crops
                if jev_decision.get("expansion_freeze"):
                    if action_type in ("EXPAND_LAND",) or (action_type == "BUY_SEED" and item_name == "STRAWBERRY"):
                        continue
                filtered_market.append(order)
            base_market = filtered_market

        # Partition orders by procedural graph nodes
        orders_by_node: Dict[str, List[List[Any]]] = {
            "wage_defense": [],
            "terminal_liquidation": [],
            "shed_headroom": [],
            "town_shop_preempt": [],
            "oracle_frontrun": [],
            "farm_execution": []
        }

        step = int(obs.get("step", 0) or 0)

        # 1. Wage Defense orders (Priority 1)
        if (jev_decision and jev_decision.get("selected_branch") == "wage_defense") or guards.get("wage_defense", False):
            for o in base_market:
                if isinstance(o, (list, tuple)) and len(o) >= 3 and o[0] == "SELL":
                    orders_by_node["wage_defense"].append(list(o))

        # 2. Terminal Liquidation orders (Priority 2, active at step >= 700)
        if (jev_decision and jev_decision.get("selected_branch") == "terminal_liquidation") or step >= 700:
            for o in base_market:
                if isinstance(o, (list, tuple)) and len(o) >= 3 and o[0] == "SELL":
                    orders_by_node["terminal_liquidation"].append(list(o))

        # 3. Shed Headroom orders
        if (jev_decision and jev_decision.get("selected_branch") == "shed_headroom") or guards.get("shed_headroom", False):
            for o in base_market:
                if isinstance(o, (list, tuple)) and len(o) >= 3 and o[0] == "SELL" and o[1] in ("WHEAT", "FERTILIZER", "STRAWBERRY"):
                    if o not in orders_by_node["wage_defense"] and o not in orders_by_node["terminal_liquidation"]:
                        orders_by_node["shed_headroom"].append(list(o))

        # 4. Town Shop Preempt orders
        if (jev_decision and jev_decision.get("selected_branch") == "town_shop_preempt") or guards.get("town_shop_preempt", False):
            for o in base_market:
                if isinstance(o, (list, tuple)) and len(o) >= 3 and o[1] in ("CARROT", "TOMATO", "EGG"):
                    orders_by_node["town_shop_preempt"].append(list(o))

        # 5. Oracle Frontrun orders
        if (jev_decision and jev_decision.get("selected_branch") == "oracle_frontrun") or guards.get("oracle_frontrun", False):
            for o in base_market:
                if isinstance(o, (list, tuple)) and len(o) >= 3 and o[0] == "SELL" and o[1] in ("WOOL", "MILK"):
                    orders_by_node["oracle_frontrun"].append(list(o))

        # 6. Routine Farm execution orders (all baseline buy and sell orders)
        for o in base_market:
            orders_by_node["farm_execution"].append(list(o))


        # Arbitrate orders using Jev's dynamic ranking if available, else static priority
        if jev_decision and "ranked_branches" in jev_decision:
            arbitrated_market: List[List[Any]] = []
            added_order_ids: set[int] = set()
            slots_left = 10

            for branch in jev_decision["ranked_branches"]:
                for o in orders_by_node.get(branch, []):
                    oid = id(o)
                    if oid not in added_order_ids and slots_left > 0:
                        arbitrated_market.append(o)
                        added_order_ids.add(oid)
                        slots_left -= 1

            # Fill remaining slots with routine execution orders
            for o in orders_by_node.get("farm_execution", []):
                oid = id(o)
                if oid not in added_order_ids and slots_left > 0:
                    arbitrated_market.append(o)
                    added_order_ids.add(oid)
                    slots_left -= 1
        else:
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
