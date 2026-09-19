"""Jev-Powered Procedural Graph Actor for Kaggriculture.

Bridges TypeSafe AI's Jev System One non-autoregressive model with the
Procedural Decision Graph. Injects the graph's active nodes, outgoing edges,
procedural guidance, and pitfall warnings directly into Jev's Choice criteria.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

# Load environment variable if not already set
env_path = Path(__file__).resolve().parent.parent.parent / ".env"
if env_path.exists():
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[len("export "):]
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            if k.strip() not in os.environ:
                os.environ[k.strip()] = v.strip().strip("'\"")

from typesafe_sdk import Choice, Noul, Score, TypeSafeClient
from kaggriculture_domain_knowledge import build_jev_knowledge_summary


class JevGraphActor:
    """System One actor that routes game state through the Procedural Graph via Jev."""

    def __init__(
        self,
        graph_data: Dict[str, Any] | Path,
        api_key: Optional[str] = None,
        query_interval: int = 6  # Query Jev every 6 hours by default (or on emergency)
    ):
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY")
        if not self.api_key:
            raise ValueError("TYPESAFE_API_KEY must be set in environment or .env")
        self.client = TypeSafeClient(api_key=self.api_key)

        if isinstance(graph_data, Path):
            with open(graph_data, encoding="utf-8") as f:
                self.graph = json.load(f)
        else:
            self.graph = graph_data

        self.query_interval = max(1, query_interval)
        self.nodes = {n["id"]: n for n in self.graph.get("nodes", [])}
        self.edges = self.graph.get("edges", [])
        self.active_node_id = "state_audit"

        self.strategy_summary = build_jev_knowledge_summary()
        self.decision_trace: List[Dict[str, Any]] = []
        self._cached_decision: Optional[Dict[str, Any]] = None
        self._last_decision_step: int = -1

    def set_graph(self, graph_data: Dict[str, Any] | Path) -> None:
        """Updates the internal procedural graph (e.g. after mutation)."""
        if isinstance(graph_data, Path):
            with open(graph_data, encoding="utf-8") as f:
                self.graph = json.load(f)
        else:
            self.graph = graph_data
        self.nodes = {n["id"]: n for n in self.graph.get("nodes", [])}
        self.edges = self.graph.get("edges", [])
        self._cached_decision = None
        self._last_decision_step = -1

    def get_candidate_branches(self, current_node: str = "state_audit") -> Dict[str, Dict[str, Any]]:
        """Extracts outgoing edges from current node with guidance, conditions, and pitfalls."""
        candidates: Dict[str, Dict[str, Any]] = {}
        for edge in self.edges:
            if edge.get("source") == current_node:
                target = edge.get("target")
                attrs = edge.get("attributes", {})
                candidates[target] = {
                    "target": target,
                    "priority": edge.get("priority", 99),
                    "condition": attrs.get("condition", "True"),
                    "guidance": attrs.get("guidance", ""),
                    "pitfalls": attrs.get("pitfalls", "")
                }
        # Fallback if no outgoing edges found
        if not candidates:
            for node_id in self.nodes:
                if node_id != current_node:
                    candidates[node_id] = {
                        "target": node_id,
                        "priority": 99,
                        "condition": "True",
                        "guidance": self.nodes[node_id].get("description", ""),
                        "pitfalls": ""
                    }
        return candidates

    def should_query_jev(self, step: int, hour: int, cash: float, payroll_due: float, shed_used: int) -> bool:
        """Determines whether to execute a fresh Jev query or use cached decision."""
        if self._cached_decision is None:
            return True
        # Periodic cadence check
        if step - self._last_decision_step >= self.query_interval:
            return True
        # Critical event interrupts:
        # 1. Approaching midnight payroll deficit (hour >= 18)
        if hour >= 18 and cash < payroll_due and self._cached_decision.get("selected_branch") != "wage_defense":
            return True
        # 2. Shed in immediate danger of overflowing (> 85 items)
        if shed_used >= 85 and self._cached_decision.get("selected_branch") != "shed_headroom":
            return True
        return False

    def query_decision(
        self,
        obs: Dict[str, Any],
        player_idx: int,
        farm_state: Dict[str, Any],
        forecast: Any = None
    ) -> Dict[str, Any]:
        """Queries Jev to evaluate candidate procedural graph branches."""
        step = int(obs.get("step", 0) or 0)
        day = int(obs.get("day", 0) or 0)
        hour = int(obs.get("hour", 0) or 0)

        farms = obs.get("farms", []) or []
        my_farm = farms[player_idx] if player_idx < len(farms) else {}
        cash = float(my_farm.get("money", 0.0) or 0.0)
        hands = my_farm.get("hands", []) or []
        n_hands = len(hands)

        private = obs.get("private", {}) or {}
        shed = private.get("shed", {}) or {}
        shed_used = sum(int(v or 0) for v in shed.values())

        # Midnight payroll calculation
        midnight_payroll_due = float(n_hands * 120 + 100)
        net_wage_surplus = cash - midnight_payroll_due

        # Fast path check
        if not self.should_query_jev(step, hour, cash, midnight_payroll_due, shed_used):
            return self._cached_decision  # type: ignore

        # Extract candidate branches from current graph node
        candidates = self.get_candidate_branches(self.active_node_id)

        # Formulate criteria for Jev Choice question directly from graph edge attributes
        choice_criteria: Dict[str, str] = {}
        for target_id, edge_info in candidates.items():
            node_name = self.nodes.get(target_id, {}).get("name", target_id)
            desc = (
                f"[{node_name}] Directive: {edge_info['guidance']} | "
                f"Trigger Rule: {edge_info['condition']} | "
                f"Pitfalls: {edge_info['pitfalls']}"
            )
            choice_criteria[target_id] = desc[:400]  # Keep criteria punchy

        # Unlocked town shops
        unlocked_shops = []
        for shop in obs.get("town_shops", []) or []:
            if shop.get("unlocked"):
                unlocked_shops.append(shop.get("name", "SHOP"))

        # Oracle glut risk
        oracle_risk = 0.0
        if forecast is not None and hasattr(forecast, "get"):
            oracle_risk = float(forecast.get("glut_risk", 0.0) or 0.0)

        state_payload = {
            "step": step,
            "day": day,
            "hour": hour,
            "cash": round(cash, 1),
            "hands_count": n_hands,
            "midnight_payroll_due": midnight_payroll_due,
            "net_wage_surplus": round(net_wage_surplus, 1),
            "shed_capacity_used": shed_used,
            "unlocked_town_shops": unlocked_shops,
            "oracle_opponent_glut_risk": round(oracle_risk, 2),
            "competitive_doctrine": self.strategy_summary,
            "current_graph_node": self.active_node_id
        }

        questions = {
            "branch_selection": Choice(
                instructions=(
                    "Evaluate the current game state against the Procedural Graph. "
                    "Which graph branch must be activated and given top execution priority?"
                ),
                criteria=choice_criteria
            ),
            "emergency_lock": Noul(
                instructions=(
                    "Is the farm facing an immediate insolvency or overflow crisis where net available cash is strictly "
                    "below midnight payroll liability (net_wage_surplus < 0) or shed capacity is full (shed_capacity_used >= 95)?"
                )
            ),
            "expansion_freeze": Noul(
                instructions=(
                    "Is the game in the late phase (day >= 13) where all new land expansion and strawberry seed purchases "
                    "must be permanently halted to focus 100% on harvesting?"
                )
            ),
            "liquidation_urgency": Score(
                instructions="Rate the urgency of liquidating shed commodities right now.",
                criteria=["LOW_HOLD", "MEDIUM_PACED", "HIGH_URGENT_CLEAR"]
            )
        }

        t0 = time.perf_counter()
        response = self.client.system_one(state=state_payload, questions=questions)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        branch_res = response.choices["branch_selection"]
        emerg_res = response.nouls["emergency_lock"]
        freeze_res = response.nouls["expansion_freeze"]
        urgency_res = response.scores["liquidation_urgency"]

        selected_branch = branch_res.choice
        # Order branches by Jev's calibrated probabilities
        ranked_branches = sorted(
            branch_res.probabilities.keys(),
            key=lambda k: branch_res.probabilities.get(k, 0.0),
            reverse=True
        )

        decision = {
            "step": step,
            "day": day,
            "hour": hour,
            "latency_ms": round(latency_ms, 2),
            "selected_branch": selected_branch,
            "confidence": round(branch_res.confidence or 0.0, 3),
            "probabilities": branch_res.probabilities,
            "ranked_branches": ranked_branches,
            "emergency_lock": bool(emerg_res.noul >= 0.50),
            "emergency_prob": round(emerg_res.noul or 0.0, 3),
            "expansion_freeze": bool(freeze_res.noul >= 0.50),
            "expansion_freeze_prob": round(freeze_res.noul or 0.0, 3),
            "liquidation_urgency": round(urgency_res.score or 0.0, 3),
            "state_snapshot": {
                "cash": cash,
                "hands": n_hands,
                "shed_used": shed_used,
                "payroll_due": midnight_payroll_due
            }
        }

        self.decision_trace.append(decision)
        self._cached_decision = decision
        self._last_decision_step = step
        return decision
