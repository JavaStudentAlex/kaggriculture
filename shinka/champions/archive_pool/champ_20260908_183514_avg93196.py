"""
Grandmaster Seed Champion for Kaggriculture DRQ Evolution.
Backbone: Mohui v66 meta closed-loop controller.
Policy Layer: Keiz Step 0-1 Opening Wheat Scalp + Dynamic Town-Shop Scarcity Harvesting + Liquidity Guard.
"""
from __future__ import annotations
import sys
import os
from pathlib import Path

# Resolve the preserved Mohui dependency closure relative to this file
MOHUI_DIR = str(Path(__file__).resolve().parent.parent / "dependencies" / "mohui_v66")
if MOHUI_DIR not in sys.path:
    sys.path.insert(0, MOHUI_DIR)

import candidate_v66_meta_closed_loop as _mohui

# =================== EVOLVE-BLOCK-START ===================

# Policy hyperparameters and baseline prices
_BASE_PRICE = {
    "MELON": 250, "STRAWBERRY": 120, "MILK": 160, "WOOL": 200,
    "FERTILIZER": 100, "WHEAT": 25, "CARROT": 35, "TOMATO": 60, "EGG": 50
}

_OPENING_BUY_WHEAT_QTY = 35
_OPENING_SELL_WHEAT_QTY = 30
_WAGE_RESERVE_FLOOR = 1000.0  # Safeguards midnight worker rehire fees
_GLIDE_ITEMS = ("MELON", "WOOL", "STRAWBERRY", "TOMATO", "MILK", "EGG")
_LIQUIDATION_ITEMS = ("MELON", "WOOL", "STRAWBERRY", "TOMATO", "MILK", "EGG")


def evolve_market_orders(obs, player_idx: int, base_orders: list) -> list:
    """
    Evolvable market policy layer.
    Applies Keiz Opening Wheat Scalp (steps 0-1), midnight wage floor protection (hours 20-23 & steps 220-245),
    dynamic inventory-scaled price-cliff glide path (steps 480-600), and proportional endgame liquidation (step 680+).
    """
    step = int(obs.get("step") if obs.get("step") is not None else (24 * obs.get("day", 0) + obs.get("hour", 0)))
    hour = int(obs.get("hour", 0) or 0)
    farms = obs.get("farms", []) or []
    my_farm = farms[player_idx] if player_idx < len(farms) else {}
    my_money = float(my_farm.get("money", 0.0) or 0.0)
    shed = (obs.get("private") or {}).get("shed", {}) or {}
    market = obs.get("market") or {}
    prices = market.get("prices") or {}

    mkt = list(base_orders or [])

    # 1. Keiz Opening Wheat Scalp (Spikes price on Step 0, locks in profit on Step 1)
    if step == 0:
        return [["BUY_PRODUCT", "WHEAT", _OPENING_BUY_WHEAT_QTY]]
    if step == 1:
        mkt = [["SELL", "WHEAT", _OPENING_SELL_WHEAT_QTY]] + [
            o for o in mkt if not (isinstance(o, (list, tuple)) and len(o) >= 2 and o[0] in ("BUY_PRODUCT", "BUY") and o[1] == "WHEAT")
        ]
        return mkt[:10]

    # 2. Midnight Liquidity & Wage Floor Guard
    # Prevents worker famine by ensuring sufficient cash remains for midnight rehire
    if hour >= 20 or (220 <= step <= 245):
        filtered_orders = []
        for o in mkt:
            if isinstance(o, (list, tuple)) and len(o) >= 3 and o[0] in ("BUY_PRODUCT", "BUY"):
                cost = float(prices.get(str(o[1]), _BASE_PRICE.get(str(o[1]), 50))) * int(o[2] or 0)
                if my_money - cost < _WAGE_RESERVE_FLOOR:
                    continue  # Protect wage floor
                my_money -= cost
            filtered_orders.append(o)
        mkt = filtered_orders

    # 3. Dynamic Inventory-Scaled Commodity Price-Cliff Glide Path (Steps 480-600)
    # Tapers high-value perishables steadily ahead of opponent dump cliff
    if 480 <= step < 600 and (step % 4 == 0):
        for item in _GLIDE_ITEMS:
            if len(mkt) >= 10:
                break
            held = int(shed.get(item, 0) or 0)
            if held > 0:
                already = sum(
                    int(o[2] or 0) for o in mkt
                    if isinstance(o, (list, tuple)) and len(o) >= 3 and o[0] == "SELL" and o[1] == item
                )
                batch_max = max(4, held // 3)
                qty = min(held - already, batch_max)
                if qty > 0:
                    mkt.append(["SELL", item, qty])

    # 4. Endgame Proportional Liquidation (Step 680+: aggressively monetize leftover produce)
    if step >= 680 and (step % 2 == 0):
        for item in _LIQUIDATION_ITEMS:
            if len(mkt) >= 10:
                break
            held = int(shed.get(item, 0) or 0)
            if held > 0:
                already = sum(
                    int(o[2] or 0) for o in mkt
                    if isinstance(o, (list, tuple)) and len(o) >= 3 and o[0] == "SELL" and o[1] == item
                )
                batch_max = held if step >= 712 else max(6, held // 2)
                qty = min(held - already, batch_max)
                if qty > 0:
                    mkt.append(["SELL", item, qty])

    return mkt[:10]

# =================== EVOLVE-BLOCK-END =====================


def agent(obs, configuration=None):
    """Top-level agent called by Kaggle simulation runner and Shinka evaluator."""
    player_idx = int(obs.get("player", 0) or 0)
    action = _mohui.kaggle_agent_v66_meta_closed_loop(obs, configuration)
    if action is None:
        action = {"farmer": ["PASS"], "hands": [], "market": []}

    base_market = list(action.get("market", []) or [])
    action["market"] = evolve_market_orders(obs, player_idx, base_market)
    return action