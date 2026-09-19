"""Jev-Powered System One Strategic Actor for Kaggriculture.

Queries TypeSafe AI's Jev model with structured game state, real-world Kaggle
strategy rules, and typed question schemas (Choice, Noul, Score).
Translates Jev's calibrated decisions into procedural graph priorities and actions.
"""
from __future__ import annotations

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


class JevSystemOneActor:
    """System One live strategic actor powered by TypeSafe AI's Jev model."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY")
        if not self.api_key:
            raise ValueError("TYPESAFE_API_KEY must be set in environment or .env")
        self.client = TypeSafeClient(api_key=self.api_key)
        self.decision_history: List[Dict[str, Any]] = []
        self.strategy_summary = build_jev_knowledge_summary()
        self._last_decision_step = -1
        self._cached_decision: Optional[Dict[str, Any]] = None

    def query_action_policy(
        self,
        obs: Dict[str, Any],
        player_idx: int,
        farm_state: Dict[str, Any],
        forecast: Any = None
    ) -> Dict[str, Any]:
        """Queries Jev for high-level tactical decisions and calibrated confidence."""
        step = int(obs.get("step", 0) or 0)
        day = int(obs.get("day", 0) or 0)
        hour = int(obs.get("hour", 0) or 0)

        # Extract farm telemetry
        farms = obs.get("farms", []) or []
        my_farm = farms[player_idx] if player_idx < len(farms) else {}
        cash = float(my_farm.get("money", 0.0) or 0.0)
        hands = my_farm.get("hands", []) or []
        n_hands = len(hands)

        private = obs.get("private", {}) or {}
        shed = private.get("shed", {}) or {}
        shed_used = sum(int(v or 0) for v in shed.values())

        # Unlocked town shops
        unlocked_shops = []
        shop_orders_needed: Dict[str, int] = {}
        for shop in obs.get("town_shops", []) or []:
            if shop.get("unlocked"):
                unlocked_shops.append(shop.get("name", "SHOP"))
                for prod, qty in (shop.get("demands") or {}).items():
                    shop_orders_needed[prod] = shop_orders_needed.get(prod, 0) + int(qty)

        # Payroll calculation
        midnight_payroll_due = float(n_hands * 120 + 100)
        net_wage_surplus = cash - midnight_payroll_due

        # Oracle glut risk
        oracle_risk = 0.0
        if forecast is not None and hasattr(forecast, "get"):
            oracle_risk = float(forecast.get("glut_risk", 0.0) or 0.0)

        # Formulate structured state for Jev
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
            "shop_demand_summary": shop_orders_needed,
            "oracle_opponent_glut_risk": round(oracle_risk, 2),
            "competitive_doctrine": self.strategy_summary
        }

        # Questions schema using Jev primitives
        questions = {
            "strategic_posture": Choice(
                instructions="Determine the current highest-priority strategic posture for the farm.",
                criteria={
                    "WAGE_DEFENSE": "Hour >= 18 and net cash < midnight payroll liability. Must liquidate to avoid hand firings.",
                    "TOWN_SHOP_ARBITRAGE": "Active unlocked town shop contracts paying 2.0x-3.5x premium on demanded goods.",
                    "CAPITAL_COMPOUNDING": "Early game (Day <= 12) with safe payroll reserve; invest cash in hands and plots.",
                    "ORACLE_FRONTRUN": "Predicted opponent glut on wool or milk; preemptively dump inventory.",
                    "HARVEST_VELOCITY": "Standard midday execution focusing on crop harvesting and shed clearance."
                }
            ),
            "liquidation_mode": Choice(
                instructions="How should the shed inventory be liquidated?",
                criteria={
                    "PROTECT_SHOP_BUFFER": "Hold demanded shop crops in shed buffer; only sell uncontracted commodities.",
                    "CLEAR_SHED_SURPLUS": "Shed fullness is high (>=70%); sell surplus commodities to prevent harvest drops.",
                    "EMERGENCY_DUMP": "Immediate payroll crisis; dump fertilizer and bulk crops to secure cash.",
                    "HOLD_FOR_PEAK": "Hold luxury goods for higher price cycles."
                }
            ),
            "labor_focus": Choice(
                instructions="What is the primary allocation focus for farm workers?",
                criteria={
                    "WATERING_RUSH": "Prioritize drying soil to maintain 3-day recurring Strawberry regrowth cycles.",
                    "HARVEST_AND_REFILL": "Prioritize picking ripe crops and transit to well for immediate can refills.",
                    "LAND_EXPANSION": "Till and plant new plots before Day 13 cutoff.",
                    "BALANCED_ROUTINE": "Standard balanced worker distribution."
                }
            ),
            "expansion_freeze": Noul(
                instructions="Should the farm enforce a hard freeze on buying new plots and Strawberry seeds (e.g. Day >= 13)?"
            ),
            "liquidation_urgency": Score(
                instructions="Rate the urgency of liquidating shed commodities right now.",
                criteria=["LOW_HOLD", "MEDIUM_PACED", "HIGH_URGENT_CLEAR"]
            )
        }

        t0 = time.perf_counter()
        response = self.client.system_one(state=state_payload, questions=questions)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        posture_res = response.choices["strategic_posture"]
        liq_res = response.choices["liquidation_mode"]
        labor_res = response.choices["labor_focus"]
        freeze_res = response.nouls["expansion_freeze"]
        urgency_res = response.scores["liquidation_urgency"]

        decision_data = {
            "step": step,
            "day": day,
            "hour": hour,
            "latency_ms": round(latency_ms, 2),
            "state_snapshot": {
                "cash": cash,
                "hands": n_hands,
                "shed_used": shed_used,
                "payroll_due": midnight_payroll_due
            },
            "posture": posture_res.choice,
            "posture_confidence": round(posture_res.confidence or 0.0, 3),
            "posture_probs": posture_res.probabilities,
            "liquidation_mode": liq_res.choice,
            "liquidation_probs": liq_res.probabilities,
            "labor_focus": labor_res.choice,
            "labor_probs": labor_res.probabilities,
            "expansion_freeze_prob": round(freeze_res.noul or 0.0, 3),
            "liquidation_urgency_score": round(urgency_res.score or 0.0, 3)
        }

        self.decision_history.append(decision_data)
        self._last_decision_step = step
        self._cached_decision = decision_data
        return decision_data
