"""Exact Kaggriculture market mechanics, re-derived from the pinned engine.

Everything here is closed-form: given public state it is computed, never learned.
Constants are imported from the installed engine so they cannot drift from it.
"""

from __future__ import annotations

from collections import Counter

from kaggle_environments.envs.kaggriculture.kaggriculture import (
    ANIMALS,
    CROPS,
    MARKET_PARAMS,
    PRODUCTS,
    SHOPS,
    TOWN_CENTER_PRODUCTS,
    market_price,
)

# Re-exported engine constants (features.py / extract.py import them from here so
# the engine is touched in exactly one module).
__all__ = [
    "ANIMALS",
    "ANIMAL_NAMES",
    "BUYABLE_PRODUCTS",
    "CENTER_INTERVAL",
    "CROPS",
    "CROP_NAMES",
    "MARKET_PARAMS",
    "PRODUCTS",
    "SHED_CAPACITY",
    "SHOPS",
    "SHOP_INTERVAL",
    "SHOP_NAMES",
    "TURNS_PER_DAY",
    "config_intervals",
    "market_price",
    "parse_market_orders",
    "town_draw",
]

# Engine defaults; overridden per-episode from `configuration` where present.
SHOP_INTERVAL = 4
CENTER_INTERVAL = 24
SHED_CAPACITY = 100
TURNS_PER_DAY = 24

# Only these two are buyable back out of the market, so for every other product
# an inventory increase can only have come from a SELL (engine:599).
BUYABLE_PRODUCTS = ("WHEAT", "FERTILIZER")

SHOP_NAMES = sorted(SHOPS)
ANIMAL_NAMES = sorted(ANIMALS)
CROP_NAMES = sorted(CROPS)


def town_draw(step, unlocked_shops, shop_interval=SHOP_INTERVAL, center_interval=CENTER_INTERVAL):
    """Units the town removes from market inventory at `step` (engine:728-750).

    Shops are drawn with replacement, so `unlocked_shops` may repeat a name and
    each instance consumes independently. Single-product shops pull double.
    """
    drawn = Counter()
    if step % shop_interval == 0:
        for shop_name in unlocked_shops or ():
            products = SHOPS.get(shop_name)
            if not products:
                continue
            multiplier = 2 if len(products) == 1 else 1
            for item in products:
                drawn[item] += multiplier
    if step % center_interval == 0:
        for item in TOWN_CENTER_PRODUCTS:
            drawn[item] += 1
    return drawn


def parse_market_orders(action):
    """Requested SELL/BUY units per product from one agent's action.

    Mirrors engine `_parse_order`, including the `maxMarketOrdersPerTurn` cut,
    which the caller applies. Requested is NOT executed: agents commonly submit
    a large sentinel meaning "sell whatever is in the shed".
    """
    sells, buys = Counter(), Counter()
    market = (action or {}).get("market") or []
    for order in market:
        if not isinstance(order, list) or len(order) < 3:
            continue
        op, item = order[0], order[1]
        try:
            n = int(order[2])
        except (TypeError, ValueError):
            continue
        if n <= 0:
            continue
        if op == "SELL" and item in PRODUCTS:
            sells[item] += n
        elif op == "BUY_PRODUCT" and item in BUYABLE_PRODUCTS:
            buys[item] += n
    return sells, buys


def price_curve(item, inventory, params=None):
    """Closed-form price. Thin alias so callers never re-implement it."""
    return market_price(item, inventory, params)


def price_after(item, inventory, delta, params=None):
    """Price if `delta` net units were added to this product's inventory."""
    return market_price(item, inventory + delta, params)


def config_intervals(configuration):
    """Resolved town intervals and shed capacity for one episode."""
    cfg = configuration or {}
    return {
        "shop_interval": int(cfg.get("townShopSellInterval", SHOP_INTERVAL) or SHOP_INTERVAL),
        "center_interval": int(cfg.get("townCenterSellInterval", CENTER_INTERVAL) or CENTER_INTERVAL),
        "shed_capacity": int(cfg.get("shedCapacity", SHED_CAPACITY) or SHED_CAPACITY),
        "turns_per_day": int(cfg.get("turnsPerDay", TURNS_PER_DAY) or TURNS_PER_DAY),
    }
