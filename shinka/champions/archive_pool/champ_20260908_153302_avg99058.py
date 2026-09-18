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

# Verified town shop consumption recipes from engine 1.32.7
_SHOP_DEMANDS = {
    "SMOOTHIE_SHOP": ("STRAWBERRY", "MILK"),
    "ICE_CREAM_SHOP": ("STRAWBERRY", "MILK", "WHEAT"),
    "PIZZA_SHOP": ("TOMATO", "MILK", "WHEAT"),
    "YARN_STORE": ("WOOL",),
    "BAKERY": ("EGG", "WHEAT"),
    "BRUNCH_SPOT": ("EGG", "WHEAT", "STRAWBERRY"),
    "PET_CAFE": ("CARROT",),
    "FARMERS_MARKET": ("CARROT", "TOMATO", "STRAWBERRY", "MELON"),
}

_BASE_PRICE = {
    "MELON": 250, "STRAWBERRY": 120, "MILK": 160, "WOOL": 200,
    "FERTILIZER": 100, "WHEAT": 25, "CARROT": 35, "TOMATO": 60, "EGG": 50
}

# Configurable policy hyperparameters
_OPENING_BUY_WHEAT_QTY = 35
_OPENING_SELL_WHEAT_QTY = 30
_SHOP_SELL_BATCH_MAX = 4
_PRICE_THRESHOLD_RATIO = 0.85
_WAGE_RESERVE_FLOOR = 800.0  # Safeguards midnight rehire fees (Day 10 Step 240)

# Animal feed and minimum operational shed reserves: never sell below these quantities
_ITEM_MIN_RESERVE = {
    "WHEAT": 22,       # Essential livestock feed for cows/chickens
    "CARROT": 12,      # Rabbit feed reserve
    "MILK": 4,         # Processing / recipe buffer
    "EGG": 4,          # Recipe buffer
    "STRAWBERRY": 0,   # High-value cash crop
    "TOMATO": 0,       # High-value cash crop
    "MELON": 0,        # High-value cash crop
    "WOOL": 0,         # High-value commodity
}


def evolve_market_orders(obs, player_idx: int, base_orders: list) -> list:
    """
    Evolvable market policy layer.
    Transforms base orders by applying opening scalps, liquidity reserve guards,
    feed-safe shop-demand selling, and late-game glide path tapers.
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
    elif step == 1:
        mkt = [["SELL", "WHEAT", _OPENING_SELL_WHEAT_QTY]] + [
            o for o in mkt if not (isinstance(o, (list, tuple)) and len(o) >= 2 and o[0] in ("BUY_PRODUCT", "BUY") and o[1] == "WHEAT")
        ]
        return mkt[:10]

    # 2. Midnight Liquidity & Wage Floor Guard
    # If cash is tight approaching midnight rehire (hours 20-23), append SELL orders to raise payroll
    if hour >= 20 and my_money < _WAGE_RESERVE_FLOOR and len(mkt) < 10:
        for cash_item in ("MELON", "WOOL", "STRAWBERRY", "TOMATO", "MILK", "EGG"):
            if len(mkt) >= 10:
                break
            held = int(shed.get(cash_item, 0) or 0)
            reserve = _ITEM_MIN_RESERVE.get(cash_item, 0)
            if held > reserve:
                already = sum(o[2] for o in mkt if isinstance(o, (list, tuple)) and len(o) >= 3 and o[0] == "SELL" and o[1] == cash_item)
                qty = min(held - reserve - already, _SHOP_SELL_BATCH_MAX)
                if qty > 0:
                    mkt.append(["SELL", cash_item, qty])
                    my_money += qty * float(prices.get(cash_item, _BASE_PRICE.get(cash_item, 50)))
                    if my_money >= _WAGE_RESERVE_FLOOR:
                        break

    # Critical midnight rehire fee brake (hour 23 only, if cash < 300)
    if hour == 23 and my_money < 300.0:
        safe_orders = []
        for o in mkt:
            if isinstance(o, (list, tuple)) and len(o) >= 3 and str(o[0]).startswith("BUY"):
                cost = float(prices.get(str(o[1]), _BASE_PRICE.get(str(o[1]), 50))) * int(o[2] or 0)
                if my_money - cost < 200.0:
                    continue
                my_money -= cost
            safe_orders.append(o)
        mkt = safe_orders

    # 3. Commodity Price-Cliff Glide Path (Steps 460-580: taper inventory before opponent crash)
    if 460 <= step < 580 and (step % 3 == 0) and len(mkt) < 10:
        for item in ("MELON", "WOOL", "STRAWBERRY"):
            if len(mkt) >= 10:
                break
            held = int(shed.get(item, 0) or 0)
            reserve = _ITEM_MIN_RESERVE.get(item, 0)
            if held > reserve:
                already = sum(o[2] for o in mkt if isinstance(o, (list, tuple)) and len(o) >= 3 and o[0] == "SELL" and o[1] == item)
                qty = min(held - reserve - already, _SHOP_SELL_BATCH_MAX)
                if qty > 0:
                    mkt.append(["SELL", item, qty])

    # 4. Feed-Safe Dynamic Town-Shop Demand Harvesting
    if 144 <= step < 680 and (step % 4 == 0) and len(mkt) < 10:
        town = obs.get("town", {}) or {}
        shops = town.get("unlocked_shops", []) or []
        demanded = set()
        for shop in shops:
            for item in _SHOP_DEMANDS.get(shop, ()):
                demanded.add(item)

        for item in ("MELON", "WOOL", "STRAWBERRY", "TOMATO", "MILK", "EGG", "CARROT", "WHEAT"):
            if len(mkt) >= 10:
                break
            if item in demanded:
                held = int(shed.get(item, 0) or 0)
                reserve = _ITEM_MIN_RESERVE.get(item, 0)
                sellable = held - reserve
                p = float(prices.get(item, 0) or 0)
                base = _BASE_PRICE.get(item, 100)
                if sellable > 0 and p >= _PRICE_THRESHOLD_RATIO * base:
                    already = sum(
                        o[2] for o in mkt
                        if isinstance(o, (list, tuple)) and len(o) >= 3 and o[0] == "SELL" and o[1] == item
                    )
                    qty = min(sellable - already, _SHOP_SELL_BATCH_MAX)
                    if qty > 0 and len(mkt) < 10:
                        mkt.append(["SELL", item, qty])

    # 5. Endgame Liquidation Glide Path (Step 680+: monetize leftover perishables)
    if step >= 680 and (step % 2 == 0) and len(mkt) < 10:
        for item in ("MELON", "WOOL", "STRAWBERRY", "TOMATO", "MILK"):
            if len(mkt) >= 10:
                break
            held = int(shed.get(item, 0) or 0)
            if held > 0:
                already = sum(o[2] for o in mkt if isinstance(o, (list, tuple)) and len(o) >= 3 and o[0] == "SELL" and o[1] == item)
                qty = min(held - already, _SHOP_SELL_BATCH_MAX)
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