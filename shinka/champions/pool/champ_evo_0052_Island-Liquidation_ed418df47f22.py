"""Auto-generated graph champion - DO NOT EDIT BY HAND.

Crowned by the distributed island evolution.
    source iteration : 52
    source island    : Island-Liquidation
    win rate         : 0.833
    mean cash        : 1769.67
    graph sha256     : ed418df47f22f6d8135394b23b40886fc4918d64801d83bfc69b6aa23da816a5

This file is a frozen *opponent*. It is intentionally deterministic: it never
contacts the LLM proxy, so its strength cannot drift between generations.

It also keeps its engine in this module's own namespace instead of reusing
`agent_graph` / `agent_jev_graph`, whose module-level engine globals are shared
with the candidate under test inside the same worker process.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

_HERE = Path(__file__).resolve()
# pool/ -> champions/ -> shinka/ -> <repo root>
_ROOT = _HERE.parent.parent.parent.parent
for _p in (
    _ROOT / "shinka" / "evolution",
    _ROOT / "research" / "procedural_graph",
    _ROOT / "shinka" / "champions" / "dependencies" / "mohui_v66",
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import initial as hazel
from graph_engine import ProceduralGraphEngine

# The graph is inlined so this champion is independent of policy_graph.json,
# which the evolution keeps rewriting.
GRAPH_JSON = r"""{
 "derived_state": {
  "available_cash": "observed cash minus existing committed debits in current batch; excludes speculative sale proceeds",
  "can_refill_needed": "not terminal_active and (worker watering can charges == 0 or (worker watering can charges < 2 and worker is adjacent to well))",
  "emergency_lock": "(hour >= 18 and payroll_shortfall > 0) or (day == 0 and hour >= 18 and available_cash < midnight_payroll_due)",
  "expansion_freeze": "day >= 13 or terminal_active",
  "midnight_payroll_due": "hands_count * 120 + 100, reconciled against authoritative payroll field",
  "payroll_buffer_target": "1.2 * midnight_payroll_due",
  "payroll_shortfall": "max(0, midnight_payroll_due - available_cash)",
  "phase_hand_cap": "3 for day <= 2; 6 for 3 <= day <= 6; 9 for 7 <= day <= 12; no hiring thereafter",
  "phase_plot_cap": "8 for day <= 2; 16 for 3 <= day <= 6; 26 for 7 <= day <= 12; no purchases thereafter",
  "projected_shed_use": "observed shed use plus feasible scheduled deposits minus acknowledged removals or guaranteed earlier same-tick removals",
  "remaining_orders": "10 minus the number of validated orders already selected for this tick",
  "shop_reserves_enabled": "not terminal_active and not emergency_lock",
  "terminal_active": "step >= 696 and step < 720, unless authoritative match timing specifies a different final window",
  "terminal_zeroing": "step >= 712",
  "worker_deposit_urgency": "terminal_active ? worker_pack_count > 0 : worker_pack_count >= 8"
 },
 "description": "Eliminates terminal inventory waste and order choke: unfreezes all shop reserves at step 696, halts futile crop hydration during the terminal window to prioritize harvest-and-deposit, drops worker deposit threshold to any positive pack content, and enforces a prioritized multi-step shed inventory zeroing engine.",
 "edges": [
  {
   "attributes": {
    "condition": "required_state_valid == false",
    "guidance": "Suspend execution and request authoritative state. Do not fabricate game state or assume zero payroll liability.",
    "pitfalls": "Coercing null values to zero; issuing blind growth orders during observation gaps."
   },
   "priority": 1,
   "relation": "TRIGGERS",
   "source": "state_audit",
   "target": "observation_recovery"
  },
  {
   "attributes": {
    "condition": "true",
    "guidance": "Log recovery state and yield tick safely.",
    "pitfalls": "Submitting unverified production orders while state is invalid."
   },
   "priority": 1,
   "relation": "LEADS_TO",
   "source": "observation_recovery",
   "target": "execution_feedback"
  },
  {
   "attributes": {
    "condition": "required_state_valid == true",
    "guidance": "Begin planning pipeline. If terminal_active is true, strip reserve locks, cancel watering proposals, and queue continuous shed liquidation.",
    "pitfalls": "Holding shop reserves past step 696; spending action orders on watering crops that cannot mature before step 720."
   },
   "priority": 2,
   "relation": "LEADS_TO",
   "source": "state_audit",
   "target": "terminal_liquidation"
  },
  {
   "attributes": {
    "condition": "true",
    "guidance": "Enforce payroll defense. Ensure cash remains >= midnight_payroll_due. On Days 1+, lock spending at hour >= 18 if available cash < midnight payroll.",
    "pitfalls": "Allowing cash to drop below midnight payroll; assuming unharvested crops can be sold on Day 0."
   },
   "priority": 1,
   "relation": "LEADS_TO",
   "source": "terminal_liquidation",
   "target": "wage_defense"
  },
  {
   "attributes": {
    "condition": "true",
    "guidance": "Calculate available shed capacity accounting for proposed payroll sales and scheduled deposits.",
    "pitfalls": "Double-counting shed relief sales or allowing pack overflow."
   },
   "priority": 1,
   "relation": "LEADS_TO",
   "source": "wage_defense",
   "target": "shed_headroom"
  },
  {
   "attributes": {
    "condition": "true",
    "guidance": "Propose deliveries for premium shop orders. Only retain 12-16 unit reserves if shop_reserves_enabled is true; unfreeze completely during terminal window or payroll emergencies.",
    "pitfalls": "Refusing town shop fulfillment to maintain reserves during terminal fire sale."
   },
   "priority": 1,
   "relation": "LEADS_TO",
   "source": "shed_headroom",
   "target": "town_shop_preempt"
  },
  {
   "attributes": {
    "condition": "true",
    "guidance": "Sell vulnerable Wool/Milk ahead of predicted market gluts.",
    "pitfalls": "Holding inventory into a forecasted price collapse."
   },
   "priority": 1,
   "relation": "LEADS_TO",
   "source": "town_shop_preempt",
   "target": "oracle_frontrun"
  },
  {
   "attributes": {
    "condition": "true",
    "guidance": "Propose growth only if day < 13, hour < 18, emergency_lock is false, terminal_active is false, and total post-purchase cash >= midnight_payroll_due + 60.",
    "pitfalls": "Purchasing seeds or land after Day 13 or during the terminal window."
   },
   "priority": 1,
   "relation": "LEADS_TO",
   "source": "oracle_frontrun",
   "target": "capital_compounding"
  },
  {
   "attributes": {
    "condition": "true",
    "guidance": "Schedule worker assignments. If terminal_active is true, force all workers with non-empty packs to deposit at shed, and assign remaining workers only to harvest ripe crops. In normal play, prioritize can refill -> strawberry hydration -> harvest -> deposit.",
    "pitfalls": "Refilling watering cans or watering plots at step >= 696; leaving harvested crops in worker packs at step 720."
   },
   "priority": 1,
   "relation": "LEADS_TO",
   "source": "capital_compounding",
   "target": "closed_loop_hydration_pipeline"
  },
  {
   "attributes": {
    "condition": "true",
    "guidance": "Pass sequential proposals into the unified 10-order scheduler.",
    "pitfalls": "Exceeding 10 total orders or assigning multiple actions to one worker."
   },
   "priority": 1,
   "relation": "LEADS_TO",
   "source": "closed_loop_hydration_pipeline",
   "target": "action_scheduler"
  },
  {
   "attributes": {
    "condition": "step <= 1",
    "guidance": "Prioritize step 0 buy and step 1 sell of WHEAT.",
    "pitfalls": "Continuing scalp after step 1."
   },
   "priority": 1,
   "relation": "FEEDS_INTO",
   "source": "keiz_opening_scalp",
   "target": "action_scheduler"
  },
  {
   "attributes": {
    "condition": "projected_shed_use >= 90",
    "guidance": "Submit high-priority shed clearance orders with relaxed price floors.",
    "pitfalls": "Allowing shed to reach 100% capacity and stall worker harvesting."
   },
   "priority": 2,
   "relation": "FEEDS_INTO",
   "source": "tiered_shed_pressure_valve",
   "target": "action_scheduler"
  },
  {
   "attributes": {
    "condition": "fertilizer_count >= 3 or (terminal_active and fertilizer_count > 0)",
    "guidance": "Monetize surplus fertilizer when price >= 0.45 base; liquidate 100% of fertilizer during terminal_active.",
    "pitfalls": "Holding unsold fertilizer into match completion."
   },
   "priority": 3,
   "relation": "FEEDS_INTO",
   "source": "fertilizer_monetization",
   "target": "action_scheduler"
  },
  {
   "attributes": {
    "condition": "step >= 680 and step <= 711",
    "guidance": "Propose preemptive liquidation of high-value goods before opponent endgame dumps.",
    "pitfalls": "Holding shop reserves through the endgame collapse."
   },
   "priority": 2,
   "relation": "FEEDS_INTO",
   "source": "early_endgame_liquidation",
   "target": "action_scheduler"
  },
  {
   "attributes": {
    "condition": "step >= 712",
    "guidance": "Submit unconditional 100% sell orders for all remaining shed inventory across steps 712-719.",
    "pitfalls": "Waiting until step 719 to sell bulky shed goods, exceeding the 10-order limit."
   },
   "priority": 1,
   "relation": "FEEDS_INTO",
   "source": "final_turn_liquidation",
   "target": "action_scheduler"
  },
  {
   "attributes": {
    "condition": "true",
    "guidance": "Submit validated batch of at most 10 orders; capture engine response and update execution telemetry.",
    "pitfalls": "Submitting more than 10 orders; ignoring rejected order feedback."
   },
   "priority": 1,
   "relation": "LEADS_TO",
   "source": "action_scheduler",
   "target": "execution_feedback"
  },
  {
   "attributes": {
    "condition": "match_active == true",
    "guidance": "Advance to next tick and re-audit authoritative state.",
    "pitfalls": "Preserving stale proposals across game ticks."
   },
   "priority": 1,
   "relation": "LEADS_TO",
   "source": "execution_feedback",
   "target": "state_audit"
  }
 ],
 "execution_contract": {
  "action_limit": 10,
  "dependency_semantics": "Use same-tick dependencies only when engine execution order is documented and guaranteed; otherwise wait for acknowledgment and re-observation.",
  "entry_node": "state_audit",
  "feedback": "Re-audit after each engine response. Never infer successful sales, hiring, deposits, or watering from submission alone.",
  "planning_semantics": "Nodes propose individual legal actions with prerequisites, quantity, estimated cash effect, capacity effect, deadline, and purpose. Proposals are evaluated cumulatively against remaining cash and order limits.",
  "submission_semantics": "Count all trades, purchases, movements, refills, deposits, and worker commands against the engine's actual order rules. Do not assume multi-step operations execute in a single command.",
  "traversal": "Evaluate edges in ascending priority. State audit selects exactly one validation branch. On the valid-state path, visit planning nodes sequentially; specialized liquidation/scalping nodes feed directly into action_scheduler. Only action_scheduler submits engine orders.",
  "unknown_values": "Null is unknown, never zero or false. Derive missing fields only from authoritative observations and documented engine timing.",
  "worker_limit": 9
 },
 "name": "terminal_fire_sale_and_zero_inventory_clearing_graph",
 "nodes": [
  {
   "category": "audit",
   "description": "Validate time, cash, worker count, worker positions, packs, watering-can charges, plot soil moisture levels, crop maturity stages, shed capacity, and action limits. Reconcile prior orders. Record missing required fields explicitly rather than hallucinating state.",
   "id": "state_audit",
   "name": "Authoritative State and Telemetry Audit"
  },
  {
   "category": "recovery",
   "description": "Request or re-read authoritative state through supported observation mechanisms. Suspend unsupported trades, hiring, planting, and expansion during telemetry blackout; execute only verified safe hold actions.",
   "id": "observation_recovery",
   "name": "Observation Recovery and Safe Hold"
  },
  {
   "category": "liquidation",
   "description": "During steps 696-720, permanently release all shop and input reserves, suppress crop watering and seed purchases, route workers carrying any inventory to deposit immediately, and propose recurring shed liquidation orders down to 0 units.",
   "id": "terminal_liquidation",
   "name": "Deadline-Aware Terminal Planner"
  },
  {
   "category": "defense",
   "description": "Enforce absolute payroll solvency. Always maintain cash >= midnight_payroll_due (hands * 120 + 100). On any day at hour >= 18 with payroll_shortfall > 0, lock discretionary spending and propose minimal executable liquidation needed to close the gap.",
   "id": "wage_defense",
   "name": "Actual-Liability Payroll Defense"
  },
  {
   "category": "capacity",
   "description": "Maintain capacity for incoming worker deposits. Initiate relief when projected utilization exceeds 88%, selling vulnerable surplus commodities or low-opportunity excess inputs before workers deposit.",
   "id": "shed_headroom",
   "name": "Deposit-Aware Capacity Planner"
  },
  {
   "category": "revenue",
   "description": "Deliver eligible inventory to unlocked town shops paying 2.0x-3.5x premiums. Maintain 12-16 reserved units per commodity only while shop_reserves_enabled is true; during terminal_active or payroll shortfalls, unfreeze all units.",
   "id": "town_shop_preempt",
   "name": "Demand-Aware Premium Deliveries"
  },
  {
   "category": "market",
   "description": "When forecasts predict an imminent opponent supply glut, liquidate held Wool and Milk 4-8 steps ahead before prices collapse. Prefer shop sales if available; otherwise execute market orders.",
   "id": "oracle_frontrun",
   "name": "Wool and Milk Glut Front-Run"
  },
  {
   "category": "growth",
   "description": "During Days 0-12 before hour 18, propose hiring, land, and strawberry seeds as an atomic bundle where cash remaining after all proposed purchases exceeds midnight_payroll_due + 60. Obey phase hand/plot caps; halt all hiring, land, and strawberry seed buying on Day 13.",
   "id": "capital_compounding",
   "name": "Post-Cost Affordable Phased Growth"
  },
  {
   "category": "execution",
   "description": "Execute deterministic closed-loop worker routing: 1) In terminal window (step >= 696), disable watering/refilling, prioritize harvesting mature crops and immediately depositing packs with count > 0; 2) In non-terminal steps, refill cans when empty, water plots before moisture decay, harvest ripe crops, and deposit when pack >= 80%.",
   "id": "closed_loop_hydration_pipeline",
   "name": "Closed-Loop Strawberry Hydration and Zero-Waste Harvest Planner"
  },
  {
   "category": "market",
   "description": "On step 0, buy exactly 1 unit of WHEAT. On step 1, sell exactly 1 unit of WHEAT to capture opening spread.",
   "id": "keiz_opening_scalp",
   "name": "Keiz Opening Wheat Scalp"
  },
  {
   "category": "capacity",
   "description": "When shed utilization exceeds 90% (and aggressively at 95%), relax price floors down to 0.15 or 0.05 of base price to immediately evacuate bulky inventory and prevent worker deposit chokes.",
   "id": "tiered_shed_pressure_valve",
   "name": "Tiered Shed Pressure Valve"
  },
  {
   "category": "revenue",
   "description": "When holding 3+ units of FERTILIZER and price is >= 45% of base, sell excess units in small batches, retaining 1 unit for crop fertilization (or 0 units during terminal_active).",
   "id": "fertilizer_monetization",
   "name": "Fertilizer Monetization"
  },
  {
   "category": "liquidation",
   "description": "During steps 680 to 711, preemptively liquidate high-value assets (MELON, TOMATO, FERTILIZER, WOOL) forecast to suffer terminal opponent dumping while market prices remain high.",
   "id": "early_endgame_liquidation",
   "name": "Predictive Early-Endgame Liquidation"
  },
  {
   "category": "liquidation",
   "description": "Across steps 712 to 719, submit 100% unconditional sell orders for all items in the shed regardless of price floors to guarantee zero inventory remains at step 720.",
   "id": "final_turn_liquidation",
   "name": "Final Horizon Inventory Zeroing"
  },
  {
   "category": "execution",
   "description": "Merge all candidate proposals from sequential planners and specialized modules into one strictly valid batch of at most 10 orders. Prioritize terminal liquidation, payroll solvency, deposit clearance, harvest cycles, shop premiums, and affordable growth without duplicate worker assignments or overselling.",
   "id": "action_scheduler",
   "name": "Shared Ten-Order Scheduler and Legality Gate"
  },
  {
   "category": "audit",
   "description": "Record submitted orders, engine execution responses, updated cash balances, and worker states. Reconcile rejections before the next planning cycle.",
   "id": "execution_feedback",
   "name": "Acknowledgment and Decision Trace"
  }
 ],
 "version": "3.5.0"
}"""
GRAPH_FINGERPRINT = "ed418df47f22f6d8135394b23b40886fc4918d64801d83bfc69b6aa23da816a5"

_ENGINE = None


def _engine() -> ProceduralGraphEngine:
    """Private engine bound to this champion's own inlined graph."""
    global _ENGINE
    if _ENGINE is None:
        tmp = Path(__file__).with_suffix(".graph.json")
        try:
            if not tmp.exists():
                tmp.write_text(GRAPH_JSON, encoding="utf-8")
            _ENGINE = ProceduralGraphEngine(tmp)
        except OSError:
            # Read-only filesystem (Kaggle input dirs are read-only): fall back
            # to a temp copy so the champion still plays.
            import tempfile
            fd, name = tempfile.mkstemp(suffix=".graph.json")
            with open(fd, "w", encoding="utf-8") as fh:
                fh.write(GRAPH_JSON)
            _ENGINE = ProceduralGraphEngine(Path(name))
    return _ENGINE


def agent(obs: dict, configuration: dict | None = None) -> dict:
    """Public entry point invoked by kaggle_environments each step."""
    player_idx = int(obs.get("player", 0) or 0)
    base = hazel._backbone_action(obs, configuration)

    farms = obs.get("farms", []) or []
    farm = farms[player_idx] if player_idx < len(farms) else {}
    n_hands = len(farm.get("hands") or [])

    forecast = hazel._oracle_observe(obs, configuration)

    try:
        engine = _engine()
        st = hazel.farm_state(obs, player_idx, forecast)
        guards = engine.evaluate_guards(st, obs)

        farmer_act = hazel.evolve_farmer_action(obs, player_idx, base.get("farmer"), st)
        hands_act = hazel.evolve_hand_actions(obs, player_idx, base.get("hands"), st)
        base_market = hazel.evolve_market_orders(obs, player_idx, base.get("market"), st)

        orders_by_node: dict[str, list[list[Any]]] = {
            "wage_defense": [],
            "terminal_liquidation": [],
            "shed_headroom": [],
            "town_shop_preempt": [],
            "oracle_frontrun": [],
            "farm_execution": [],
        }

        step = int(obs.get("step", 0) or 0)

        if guards.get("wage_defense", False):
            for o in base_market:
                if isinstance(o, (list, tuple)) and len(o) >= 3 and o[0] == "SELL":
                    orders_by_node["wage_defense"].append(list(o))

        if step >= 700:
            for o in base_market:
                if isinstance(o, (list, tuple)) and len(o) >= 3 and o[0] == "SELL":
                    if list(o) not in orders_by_node["wage_defense"]:
                        orders_by_node["terminal_liquidation"].append(list(o))

        if guards.get("shed_headroom", False):
            for o in base_market:
                if (isinstance(o, (list, tuple)) and len(o) >= 3 and o[0] == "SELL"
                        and o[1] in ("WHEAT", "FERTILIZER", "STRAWBERRY")):
                    if list(o) not in orders_by_node["wage_defense"]:
                        orders_by_node["shed_headroom"].append(list(o))

        if guards.get("town_shop_preempt", False):
            for o in base_market:
                if isinstance(o, (list, tuple)) and len(o) >= 3 and o[1] in ("CARROT", "TOMATO", "EGG"):
                    orders_by_node["town_shop_preempt"].append(list(o))

        if guards.get("oracle_frontrun", False):
            for o in base_market:
                if (isinstance(o, (list, tuple)) and len(o) >= 3 and o[0] == "SELL"
                        and o[1] in ("WOOL", "MILK")):
                    orders_by_node["oracle_frontrun"].append(list(o))

        for o in base_market:
            orders_by_node["farm_execution"].append(list(o))

        arbitrated = engine.arbitrate_market_orders(orders_by_node, max_orders=10)

        evolved = {"farmer": farmer_act, "hands": hands_act, "market": arbitrated}
        evolved = hazel.policy_sanitize(evolved, base, st)
    except Exception:
        evolved = base

    final = hazel._hard_sanitize(evolved, base, n_hands)
    hazel._oracle_record(final)
    return final
