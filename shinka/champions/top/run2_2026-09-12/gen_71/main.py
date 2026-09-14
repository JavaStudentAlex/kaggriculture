"""
Grandmaster Seed Champion for Kaggriculture DRQ Evolution.

Backbone: Mohui v66 meta closed-loop controller (read-only, outside EVOLVE-BLOCK).
Oracle:   the frozen TinyTimeMixer opponent order-flow checkpoint in
          shinka/evolution/checkpoint/ (the newest promoted daily fine-tune; its
          README says which), served by kagg_oracle.py (read-only, outside
          EVOLVE-BLOCK). Every turn it forecasts how many units of each
          product the opponent will put on the market at each of the next 96 turns;
          the forecast reaches the policy as st["oracle"].
Policy Layer: fully evolvable override of ALL THREE action channels
(farmer / hands / market), seeded with fixes for the three measured loss modes
and with oracle-driven selling levers.

The backbone always produces a schema-valid base action. The evolvable layer
rewrites it. `_sanitize` (outside the block) guarantees the final action is
well-formed, so a malformed mutation degrades to the backbone instead of crashing.
"""
from __future__ import annotations
import os
import sys
import time
from pathlib import Path

# Mohui backbone. Shinka COPIES this file into shinka_results/gen_N/main.py, so a
# purely __file__-relative path breaks the moment it is evaluated. Resolve through
# an env override first, then a set of known locations, and take the first that
# actually exists.
def _find_mohui():
    cands = []
    env = os.environ.get("KAGG_MOHUI_DIR")
    if env:
        cands.append(Path(env))
    here = Path(__file__).resolve()
    for base in (here.parent.parent, here.parent.parent.parent, Path.home() / "kaggriculture" / "shinka"):
        cands.append(base / "champions" / "dependencies" / "mohui_v66")
    for c in cands:
        if (c / "candidate_v66_meta_closed_loop.py").is_file():
            return str(c)
    return str(cands[-1])

MOHUI_DIR = _find_mohui()
if MOHUI_DIR not in sys.path:
    sys.path.insert(0, MOHUI_DIR)

import candidate_v66_meta_closed_loop as _mohui


# Opponent order-flow oracle (kagg_oracle.py next to this file in the repo; the
# launcher exports KAGG_ORACLE_SRC because shinka evaluates a COPY of this file).
def _find_oracle_src():
    cands = []
    env = os.environ.get("KAGG_ORACLE_SRC")
    if env:
        cands.append(Path(env))
    here = Path(__file__).resolve()
    cands += [here.parent, here.parent.parent / "shinka" / "evolution",
              Path.home() / "kaggriculture" / "shinka" / "evolution"]
    for c in cands:
        if (c / "kagg_oracle.py").is_file():
            return str(c)
    return str(cands[-1])

ORACLE_SRC = _find_oracle_src()
if ORACLE_SRC not in sys.path:
    sys.path.insert(0, ORACLE_SRC)

ORACLE_DEVICE = "off"
ORACLE_STATS = {"forecasts": 0, "ms_per_forecast": 0.0, "errors": 0, "last_error": ""}
_TRACKER = None
try:
    import kagg_oracle as _oracle

    # KAGG_ORACLE_EAGER=0 (set by the evaluator's parent while it probes that the
    # file imports) skips the model load; every worker process loads it once.
    if os.environ.get("KAGG_ORACLE_EAGER", "1") != "0":
        _MODEL = _oracle.get_model()
        _TRACKER = _oracle.OpponentTracker(_MODEL)
        ORACLE_DEVICE = str(_MODEL.device)
except Exception as _exc:  # no torch / no checkpoint: play without forecasts
    ORACLE_STATS["errors"] += 1
    ORACLE_STATS["last_error"] = f"{type(_exc).__name__}: {_exc}"[:200]


def _oracle_observe(obs, configuration):
    """Feed this turn to the tracker; returns the forecast dict or None."""
    global _TRACKER
    if _TRACKER is None and os.environ.get("KAGG_ORACLE_EAGER", "1") == "0" and ORACLE_STATS["errors"] == 0:
        try:  # lazily loaded when the parent process merely probed the import
            _TRACKER = _oracle.OpponentTracker(_oracle.get_model())
        except Exception as exc:
            ORACLE_STATS["errors"] += 1
            ORACLE_STATS["last_error"] = f"{type(exc).__name__}: {exc}"[:200]
            return None
    if _TRACKER is None:
        return None
    try:
        t0 = time.perf_counter()
        forecast = _TRACKER.observe(obs, configuration)
        if forecast is not None:
            n = ORACLE_STATS["forecasts"]
            ms = (time.perf_counter() - t0) * 1000.0
            ORACLE_STATS["ms_per_forecast"] = (ORACLE_STATS["ms_per_forecast"] * n + ms) / (n + 1)
            ORACLE_STATS["forecasts"] = n + 1
        return forecast
    except Exception as exc:
        ORACLE_STATS["errors"] += 1
        ORACLE_STATS["last_error"] = f"{type(exc).__name__}: {exc}"[:200]
        return None


def _oracle_record(action):
    if _TRACKER is not None:
        try:
            _TRACKER.record_action(action)
        except Exception:
            pass

# =================== EVOLVE-BLOCK-START ===================

# ---------------------------------------------------------------- reference data
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
    "FERTILIZER": 100, "WHEAT": 25, "CARROT": 35, "TOMATO": 60, "EGG": 50,
}

# Yield per tile per day. Goose dominates; the audited agent never bought one.
_ANIMAL_YIELD = {"GOOSE": 1.00, "COW": 0.50, "SHEEP": 0.33}

# ---------------------------------------------------------------- hyperparameters
_WAGE_RESERVE_FLOOR = 600.0    # proven: midnight rehire guard, narrow window only

# LOSS MODE 1+2 (cash starvation -> failed livestock buys -> 0% win rate in 9/110 games)
# BOTH MEASURED HARMFUL AS WRITTEN, both OFF. These are the naive implementations;
# the underlying problems are real, so fix the mechanism rather than the flag.
#   _ENABLE_CASH_FLOOR   -> 0% win rate, avg cash $16,589 vs baseline $73,019.
#       An always-on floor blocks the day-0/1 establishment spend. Every strong
#       agent runs down to <$70 on day 1; the winners differ in day 2-10 RECOVERY,
#       not in the day-1 floor. A working version must be time- or state-scoped.
#   _ENABLE_ANIMAL_TOPUP -> 0% win rate, avg cash $707 (!). Buying is not owning:
#       animals land in the SHED and do nothing until a hand PICKUPs and PLACEs
#       them on a PASTURE. This lever buys and never places, so it burns cash AND
#       occupies scarce shed capacity. A working version must drive the hands too,
#       which the widened EVOLVE-BLOCK now makes possible.
_ENABLE_CASH_FLOOR = False
_ENABLE_ANIMAL_TOPUP = False
_CASH_FLOOR_EARLY = 700.0      # hold this much through _CASH_FLOOR_UNTIL_DAY
_CASH_FLOOR_UNTIL_DAY = 8
_CASH_FLOOR_LATE = 250.0
_ANIMAL_TARGET = 17            # opponents that beat us reach 17; we stall at 5-14
_ANIMAL_TOPUP_MIN_CASH = 1500.0
_ANIMAL_TOPUP_LAST_DAY = 26    # audited agent stopped buying on day 13
_ANIMAL_PREFERENCE = ("GOOSE", "COW", "SHEEP")

# LOSS MODE 3 (revenue concentration -> price collapse in mirror matches, 26% WR)
_PRICE_THRESHOLD_RATIO = 0.85
_SHOP_SELL_BATCH_MAX = 4
_MIN_HELD_FOR_SHOP_SALE = 2
_SELL_PRIORITY = ("STRAWBERRY", "MILK", "WOOL", "TOMATO", "CARROT", "EGG", "WHEAT", "MELON")

# SHED OVERFLOW (measured): every deposit path discards overflow UNCONDITIONALLY
_SHED_CAPACITY = 100
_ENABLE_SHED_PRESSURE = True
_SHED_PRESSURE_AT = 80          # start dumping once the shed is this full
_SHED_PRESSURE_BATCH = 10       # max units per order while under pressure
_SHED_PRESSURE_PRICE_RATIO = 0.35   # accept a worse price to avoid destruction

# Plant survival: plants die at consecutive_unwatered >= 2.
# Strictly idle-only override: only workers outputting PASS are reassigned maintenance.
_ENABLE_HAND_RESCUE = True
_WATER_RESCUE_THRESHOLD = 1     # rescue a tile once it has missed this many days
_FEED_RESCUE_THRESHOLD = 1      # animals escape at consecutive_unfed >= 2

# Opening scalp
_OPENING_BUY_WHEAT_QTY = 35
_OPENING_SELL_WHEAT_QTY = 30

# ORACLE LEVERS. st["oracle"] is the opponent order-flow forecast (None until turn 512)
_ENABLE_ORACLE_PRIORITY = True   # order shop sales by predicted opponent supply (least first)
_ENABLE_ORACLE_FRONTRUN = True   # sell ahead of a predicted opponent dump
_ENABLE_ORACLE_HOLD = False      # holding cancelled needed sales due to false positives; keep False
_ORACLE_FRONTRUN_SCORE = 0.30    # fallback front-run threshold
_ORACLE_FRONTRUN_BATCH = 6       # fallback batch size
_ORACLE_FRONTRUN_PRICE_RATIO = 0.60   # fallback price ratio
_ORACLE_FRONTRUN_MIN_HELD = 1
_ORACLE_HOLD_SCORE = 0.50
_ORACLE_PRIORITY_HORIZON = "units_24"  # which window ranks the products

# Item-specific front-running thresholds: lower for high-value/slow-yield, higher for bulk
_ORACLE_FRONTRUN_THRESHOLDS = {
    "WOOL": 0.22,         # base 200, sheep yield 0.33/day; crash to 18 is catastrophic
    "MILK": 0.24,         # base 160, cow yield 0.50/day; mirror crash to 1 destroys revenue
    "FERTILIZER": 0.24,   # base 100, AUC 0.77; slow animal byproduct
    "STRAWBERRY": 0.26,   # base 120, ongoing crop; mirror crash 180 -> 1
    "EGG": 0.28,          # base 50, goose yield 1.00/day; AUC 0.80
    "TOMATO": 0.30,       # base 60; drifts to 247 if unsold, require conviction before dump
    "CARROT": 0.32,       # base 35, one-time crop; AUC 0.81
    "WHEAT": 0.38,        # base 25, bulk feed; high certainty to protect animal feed reserve
    "MELON": 1.00,        # AUC 0.51 is pure noise; excluded
}

# Item-specific price ratio floors: accept deeper discount on high-margin goods to avoid floor collapse
_ORACLE_FRONTRUN_PRICE_RATIOS = {
    "WOOL": 0.45,
    "MILK": 0.45,
    "STRAWBERRY": 0.50,
    "FERTILIZER": 0.50,
    "EGG": 0.55,
    "TOMATO": 0.65,
    "CARROT": 0.60,
    "WHEAT": 0.65,
}

# Item-specific batch sizes: tailored to typical yield and shed footprint
_ORACLE_FRONTRUN_BATCHES = {
    "WOOL": 4,
    "MILK": 8,
    "STRAWBERRY": 8,
    "FERTILIZER": 4,
    "TOMATO": 6,
    "EGG": 6,
    "CARROT": 6,
    "WHEAT": 4,
}

# Item-specific cadence bypass thresholds for town shop demand
_ORACLE_BYPASS_THRESHOLDS = {
    "WOOL": 0.28,
    "MILK": 0.30,
    "STRAWBERRY": 0.32,
    "FERTILIZER": 0.35,
    "EGG": 0.36,
    "TOMATO": 0.38,
    "CARROT": 0.40,
    "WHEAT": 0.45,
}

# Item-specific early-endgame score_24 liquidation thresholds (steps 650..711)
_ORACLE_24_THRESHOLDS = {
    "WOOL": 0.16,
    "MILK": 0.18,
    "FERTILIZER": 0.18,
    "STRAWBERRY": 0.20,
    "EGG": 0.22,
    "TOMATO": 0.22,
    "CARROT": 0.24,
    "WHEAT": 0.28,
}


# ---------------------------------------------------------------- derived state
def farm_state(obs, player_idx, oracle=None):
    """Per-turn derived features. Extend freely; everything here is evolvable.
    `oracle` is the opponent order-flow forecast dict (or None); see the ORACLE
    LEVERS comment for its keys."""
    farms = obs.get("farms", []) or []
    farm = farms[player_idx] if player_idx < len(farms) else {}
    tiles = farm.get("tiles", []) or []

    plants, animals, weeds, empties, locked = [], [], [], [], 0
    thirsty, hungry = [], []
    for r, row in enumerate(tiles):
        for c, cell in enumerate(row or []):
            if cell is None:
                empties.append((r, c)); continue
            if isinstance(cell, str):
                locked += 1; continue
            kind = cell.get("kind")
            if kind == "PLANT":
                plants.append((r, c, cell))
                if int(cell.get("consecutive_unwatered", 0) or 0) >= _WATER_RESCUE_THRESHOLD \
                        and not cell.get("watered_today"):
                    thirsty.append((r, c, cell))
            elif kind == "PASTURE":
                if cell.get("animal"):
                    animals.append((r, c, cell))
                    if int(cell.get("consecutive_unfed", 0) or 0) >= _FEED_RESCUE_THRESHOLD \
                            and not cell.get("fed_today"):
                        hungry.append((r, c, cell))
                else:
                    empties.append((r, c))
            elif kind == "WEED":
                weeds.append((r, c))

    day = int(obs.get("day", 0) or 0)
    hour = int(obs.get("hour", 0) or 0)
    _priv = obs.get("private") or {}
    _shed = _priv.get("shed", {}) or {}
    _shed_used = sum(v for v in _shed.values() if isinstance(v, (int, float)))

    # Count inventory units currently held in private worker bags (farmer + hired hands)
    _bag_units = 0
    for k, v in _priv.items():
        if k == "shed":
            continue
        if isinstance(v, dict):
            _bag_units += sum(int(cnt or 0) for cnt in v.values() if isinstance(cnt, (int, float)))
        elif isinstance(v, list):
            for sub in v:
                if isinstance(sub, dict):
                    _bag_units += sum(int(cnt or 0) for cnt in sub.values() if isinstance(cnt, (int, float)))
                elif isinstance(sub, (list, tuple)) and len(sub) >= 2 and isinstance(sub[1], (int, float)):
                    _bag_units += int(sub[1] or 0)

    # Detect ripe / near-harvest crops
    ripe_plants = 0
    for _, _, cell in plants:
        if cell.get("ripe") or cell.get("harvestable") or cell.get("stage") in (2, 3, "RIPE", "ripe"):
            ripe_plants += 1
        elif (cell.get("age", 0) or 0) >= 4 and cell.get("watered_today"):
            ripe_plants += 1

    # Effective shed load accounts for private bags and imminent harvest
    effective_shed_load = _shed_used + _bag_units + (ripe_plants if hour >= 18 else (ripe_plants + 1) // 2)

    return {
        "day": day,
        "hour": hour,
        "money": float(farm.get("money", 0.0) or 0.0),
        "farmer": farm.get("farmer") or [0, 0],
        "hands": farm.get("hands") or [],
        "tiles": tiles,
        "plants": plants, "animals": animals, "weeds": weeds,
        "empties": empties, "locked": locked,
        "thirsty": thirsty, "hungry": hungry,
        "n_animals": len(animals), "n_plants": len(plants),
        "cash_floor": _CASH_FLOOR_EARLY if day <= _CASH_FLOOR_UNTIL_DAY else _CASH_FLOOR_LATE,
        "shed": _shed, "shed_used": _shed_used, "shed_cap": _SHED_CAPACITY,
        "shed_room": max(0, _SHED_CAPACITY - _shed_used),
        "shed_wheat_avail": int(_shed.get("WHEAT", 0) or 0),
        "bag_units": _bag_units,
        "ripe_plants": ripe_plants,
        "effective_shed_load": effective_shed_load,
        "prices": (obs.get("market") or {}).get("prices") or {},
        "shops": (obs.get("town") or {}).get("unlocked_shops", []) or [],
        "oracle": oracle if isinstance(oracle, dict) else None,
        "_claimed_tiles": set(),
    }


def _oracle_scores(st, key):
    """Per-product dict from the forecast, or {} when there is no forecast yet."""
    fc = st.get("oracle")
    if not fc:
        return {}
    vals = fc.get(key)
    return vals if isinstance(vals, dict) else {}


# ---------------------------------------------------------------- market channel
def evolve_market_orders(obs, player_idx, base_orders, st):
    """
    Evolvable market policy. Max 10 orders per turn.

    NOTE: this is the proven champion policy, unchanged. The cash-floor and
    livestock-top-up levers below are declared but DISABLED (_ENABLE_* = False)
    because a naive always-on cash floor starves the day-0/1 establishment spend
    (every strong agent, ours and the opponents', runs down to <$70 on day 1;
    what separates them is the day 2-10 recovery rate, not the day-1 floor).
    Enabling and shaping them is a job for evolution, not a baked-in assumption.
    """
    step = int(obs.get("step") if obs.get("step") is not None else (24 * st["day"] + st["hour"]))
    hour = st["hour"]
    money, prices, shed = st["money"], st["prices"], st["shed"]
    mkt = list(base_orders or [])

    # 1. Keiz opening wheat scalp
    if step == 0:
        return [["BUY_PRODUCT", "WHEAT", _OPENING_BUY_WHEAT_QTY]]
    elif step == 1:
        mkt = [["SELL", "WHEAT", _OPENING_SELL_WHEAT_QTY]] + [
            o for o in mkt
            if not (isinstance(o, (list, tuple)) and len(o) >= 2
                    and o[0] == "BUY_PRODUCT" and o[1] == "WHEAT")
        ]
        return mkt[:10]

    # 2. Midnight liquidity & wage floor guard (narrow window only, day > 1 to preserve day 0/1 setup)
    if (hour >= 20 or (220 <= step <= 245)) and st["day"] > 1:
        n_hands = len(st.get("hands", []))
        wage_reserve = max(250.0, n_hands * 120.0)
        filtered = []
        for o in mkt:
            if isinstance(o, (list, tuple)) and len(o) >= 3 and o[0] == "BUY_PRODUCT":
                cost = float(prices.get(str(o[1]), _BASE_PRICE.get(str(o[1]), 50))) * int(o[2] or 0)
                if money - cost < wage_reserve:
                    continue
                money -= cost
            filtered.append(o)
        mkt = filtered

    # 2b. Cancel non-paying investments in the last 2 days (days 28-29: cannot mature before turn 720)
    if st["day"] >= 28:
        mkt = [
            o for o in mkt
            if not (isinstance(o, (list, tuple)) and len(o) >= 2 and o[0] in ("BUY_SEED", "BUY_ANIMAL", "BUY_LAND"))
        ]

    # 3. OPTIONAL cash-floor guard (loss modes 1+2) - disabled by default.
    if _ENABLE_CASH_FLOOR:
        floor = _CASH_FLOOR_EARLY if st["day"] <= _CASH_FLOOR_UNTIL_DAY else _CASH_FLOOR_LATE
        kept, projected = [], money
        for o in mkt:
            if isinstance(o, (list, tuple)) and len(o) >= 3 and o[0] in ("BUY_PRODUCT", "BUY_SEED", "BUY_LAND"):
                unit = float(prices.get(str(o[1]), _BASE_PRICE.get(str(o[1]), 50)) or 50)
                cost = unit * int(o[2] or 0)
                if projected - cost < floor:
                    continue
                projected -= cost
            kept.append(o)
        mkt = kept

    # 4. OPTIONAL state-driven livestock top-up (loss mode 2) - disabled by default.
    if _ENABLE_ANIMAL_TOPUP and st["n_animals"] < _ANIMAL_TARGET and len(mkt) < 10:
        if money >= _ANIMAL_TOPUP_MIN_CASH and st["day"] <= _ANIMAL_TOPUP_LAST_DAY:
            already = sum(
                int(o[2] or 0) for o in mkt
                if isinstance(o, (list, tuple)) and len(o) >= 3 and o[0] == "BUY_ANIMAL"
            )
            want = _ANIMAL_TARGET - st["n_animals"] - already
            if want > 0:
                mkt.append(["BUY_ANIMAL", _ANIMAL_PREFERENCE[0], max(1, min(want, 2))])

    # Dynamic feed reserve: steps down in endgame when no future days need feeding
    feed_reserve = 0 if step >= 696 else (max(2, st["n_animals"]) if step >= 672 else max(4, st["n_animals"] * 2))

    # 4b. HARVEST-AWARE DYNAMIC SHED PRESSURE VALVE:
    #     Dynamically scales activation threshold based on worker bag occupancy, ripe crops,
    #     and imminent midnight sweep to preempt silent 100/100 harvest deletion.
    #     Prioritizes venting non-demanded surplus (Fertilizer, excess Wheat) and low-margin goods
    #     to preserve high-value town shop inventory while restoring safe shed headroom.
    shed_used = st["shed_used"]
    bag_units = st.get("bag_units", 0)
    ripe_plants = st.get("ripe_plants", 0)
    effective_load = st.get("effective_shed_load", shed_used)

    # Dynamic pressure trigger: lower when imminent harvest or full bags threaten overflow
    pressure_trigger = _SHED_PRESSURE_AT
    if bag_units >= 8 or ripe_plants >= 8:
        pressure_trigger = 68
    elif bag_units >= 4 or ripe_plants >= 4:
        pressure_trigger = 74
    if hour >= 20 and (shed_used + bag_units) >= 72:
        pressure_trigger = min(pressure_trigger, 70)

    is_under_pressure = _ENABLE_SHED_PRESSURE and (shed_used >= pressure_trigger or effective_load >= 80)
    if is_under_pressure and len(mkt) < 10:
        p_ratio_floor = 0.05 if shed_used >= 95 else (0.15 if shed_used >= 90 else _SHED_PRESSURE_PRICE_RATIO)
        batch_limit = 10 if shed_used >= 90 else _SHED_PRESSURE_BATCH

        demanded_town = set()
        for shop in st.get("shops", []):
            demanded_town.update(_SHOP_DEMANDS.get(shop, ()))

        # Group shed holdings into non-demanded surplus (vent first) vs demanded goods (cheapest first)
        tier1_cands = []  # Fertilizer, excess Wheat, un-demanded goods
        tier2_cands = []  # Demanded town goods, sorted by base price ascending
        for item, q in shed.items():
            qty_held = int(q or 0)
            if qty_held <= 0:
                continue
            already = sum(
                int(o[2] or 0) for o in mkt
                if isinstance(o, (list, tuple)) and len(o) >= 3
                and o[0] == "SELL" and o[1] == item
            )
            reserve = feed_reserve if item == "WHEAT" else 0
            avail = qty_held - already - reserve
            if avail <= 0:
                continue

            base = _BASE_PRICE.get(item, 100)
            is_demanded = (item in demanded_town)
            if (not is_demanded) or item == "FERTILIZER" or (item == "WHEAT" and avail > 2):
                tier1_cands.append((avail, item, base))
            else:
                tier2_cands.append((base, item, avail))

        # Tier 1: highest available volume first
        tier1_cands.sort(key=lambda x: -x[0])
        # Tier 2: lowest base price first (protect high-value Strawberry, Milk, Wool, Melon)
        tier2_cands.sort(key=lambda x: x[0])

        ordered_holdings = [(it, av, bs) for av, it, bs in tier1_cands] + [(it, av, bs) for bs, it, av in tier2_cands]

        target_shed_level = 68
        needed_vent = max(batch_limit, effective_load - target_shed_level)

        for item, avail, base in ordered_holdings:
            if len(mkt) >= 10 or needed_vent <= 0:
                break
            price = float(prices.get(item, 0) or 0)
            if price < p_ratio_floor * base and price < 1.0:
                continue
            qty = min(avail, batch_limit, needed_vent)
            if qty > 0:
                mkt.append(["SELL", item, qty])
                needed_vent -= qty
                shed_used -= qty

    # 4c. ORACLE FRONT-RUN: Sell ahead of predicted opponent dump before price crashes.
    #     Uses item-specific trigger scores, price ratio floors, batch sizes, and value-at-risk ranking.
    #     MELON AUC is 0.51 (coin flip / pure noise), so MELON is excluded.
    #     WHEAT feed reserve is protected so front-running never starves animals.
    sell_ahead = _oracle_scores(st, "score_4")
    if _ENABLE_ORACLE_FRONTRUN and sell_ahead and len(mkt) < 10:
        cands = [
            (it, float(sc or 0.0)) for it, sc in sell_ahead.items()
            if it != "MELON" and float(sc or 0.0) >= _ORACLE_FRONTRUN_THRESHOLDS.get(it, _ORACLE_FRONTRUN_SCORE)
        ]
        # Sort by value at risk (score * base_price) to prioritize saving high-margin assets first
        cands.sort(key=lambda kv: -(kv[1] * _BASE_PRICE.get(kv[0], 50)))

        for item, score in cands:
            if len(mkt) >= 10:
                break
            held = int(shed.get(item, 0) or 0)
            price = float(prices.get(item, 0) or 0)
            base = _BASE_PRICE.get(item, 100)
            reserve = feed_reserve if item == "WHEAT" else 0
            min_p_ratio = _ORACLE_FRONTRUN_PRICE_RATIOS.get(item, _ORACLE_FRONTRUN_PRICE_RATIO)
            if (held - reserve) < _ORACLE_FRONTRUN_MIN_HELD or price < min_p_ratio * base:
                continue
            already = sum(
                int(o[2] or 0) for o in mkt
                if isinstance(o, (list, tuple)) and len(o) >= 3 and o[0] == "SELL" and o[1] == item
            )
            base_batch = _ORACLE_FRONTRUN_BATCHES.get(item, _ORACLE_FRONTRUN_BATCH)
            batch = base_batch + 2 if score >= 0.45 else base_batch
            qty = min(held - reserve - already, batch)
            if qty > 0:
                mkt.append(["SELL", item, qty])

    # 5. Town-shop demand harvesting & Oracle-Driven Dynamic Fertilizer Monetization.
    #    Runs through step 711 to capture full endgame revenue from town shops.
    #    Opponent supply ranks goods to avoid mirror dumps.
    oracle_cadence_bypass = False
    fert_dump_imminent = False
    fert_dump_mild = False
    sc4_fert = 0.0
    sc24_fert = 0.0

    if step >= 512:
        _score_4_bypass = _oracle_scores(st, "score_4")
        _score_24_bypass = _oracle_scores(st, "score_24")
        if _score_4_bypass:
            sc4_fert = float(_score_4_bypass.get("FERTILIZER", 0.0) or 0.0)
        if _score_24_bypass:
            sc24_fert = float(_score_24_bypass.get("FERTILIZER", 0.0) or 0.0)

        # Monitor oracle forecast for fertilizer dump risk
        fert_held_check = int(shed.get("FERTILIZER", 0) or 0)
        if (sc4_fert >= 0.24 or sc24_fert >= 0.22) and fert_held_check >= 2:
            fert_dump_imminent = True
            oracle_cadence_bypass = True
        elif (sc4_fert >= 0.18 or sc24_fert >= 0.15) and fert_held_check >= 3:
            fert_dump_mild = True

        if _ENABLE_ORACLE_PRIORITY and not oracle_cadence_bypass and _score_4_bypass:
            _demanded_bypass = set()
            for shop in st["shops"]:
                _demanded_bypass.update(_SHOP_DEMANDS.get(shop, ()))
            for _it_b in _demanded_bypass:
                if _it_b != "MELON" and float(_score_4_bypass.get(_it_b, 0.0)) >= _ORACLE_BYPASS_THRESHOLDS.get(_it_b, 0.40):
                    oracle_cadence_bypass = True
                    break

    # Adaptive cadence: faster when shed filling, endgame, or oracle bypass
    cadence = oracle_cadence_bypass or (step % 4 == 0) or (shed_used >= 60 and step % 2 == 0) or (step >= 672 and step % 2 == 0)
    if 144 <= step < 712 and cadence and len(mkt) < 10:
        demanded = set()
        for shop in st["shops"]:
            demanded.update(_SHOP_DEMANDS.get(shop, ()))
        priority = list(_SELL_PRIORITY)
        opp_supply = _oracle_scores(st, _ORACLE_PRIORITY_HORIZON)
        opp_score_24 = _oracle_scores(st, "score_24")
        if _ENABLE_ORACLE_PRIORITY and opp_supply:
            # MELON prediction is noise (AUC 0.51); treat its opp_supply as 0 to prevent noise penalty
            priority.sort(key=lambda it: (0.0 if it == "MELON" else float(opp_supply.get(it, 0.0)), _SELL_PRIORITY.index(it)))

        for item in priority:
            if len(mkt) >= 10:
                break
            if item in demanded:
                held = int(shed.get(item, 0) or 0)
                reserve = feed_reserve if item == "WHEAT" else 0
                avail = held - reserve
                min_req = 1 if item == "MELON" else _MIN_HELD_FOR_SHOP_SALE
                if avail < min_req:
                    continue
                p = float(prices.get(item, 0) or 0)
                base = _BASE_PRICE.get(item, 100)
                # Lower threshold when oracle bypass is active (front-run is time-critical)
                p_thresh = 0.70 * base if oracle_cadence_bypass else _PRICE_THRESHOLD_RATIO * base
                if p >= p_thresh:
                    already = sum(
                        int(o[2] or 0) for o in mkt
                        if isinstance(o, (list, tuple)) and len(o) >= 3
                        and o[0] == "SELL" and o[1] == item
                    )

                    # DYNAMIC BATCH SIZING: Front-run shop demand if opponent is predicted to supply,
                    # else restrict batch size to 2-3 to act as a monopoly and preserve high prices.
                    batch_max = _SHOP_SELL_BATCH_MAX
                    if _ENABLE_ORACLE_PRIORITY and step >= 512 and item != "MELON":
                        sc24 = float(opp_score_24.get(item, 0.0)) if opp_score_24 else 0.0
                        supp = float(opp_supply.get(item, 0.0)) if opp_supply else 0.0
                        if sc24 >= 0.30 or supp >= 1.2:
                            batch_max = 8
                        elif sc24 >= 0.15 or supp >= 0.6:
                            batch_max = 6
                        else:
                            batch_max = 3
                    else:
                        batch_max = 3 if item != "MELON" else _SHOP_SELL_BATCH_MAX

                    qty = min(avail - already, batch_max)
                    if qty > 0 and len(mkt) < 10:
                        mkt.append(["SELL", item, qty])
                        shed_used -= qty

        # Oracle-Driven Dynamic Fertilizer Monetization
        fert_held = int(shed.get("FERTILIZER", 0) or 0)
        already_fert = sum(
            int(o[2] or 0) for o in mkt
            if isinstance(o, (list, tuple)) and len(o) >= 3 and o[0] == "SELL" and o[1] == "FERTILIZER"
        )
        avail_fert = fert_held - already_fert
        if avail_fert >= 2 and len(mkt) < 10:
            p_fert = float(prices.get("FERTILIZER", 0) or 0)
            base_fert = 100.0

            if fert_dump_imminent:
                p_thresh_fert = 0.48 * base_fert
                batch_fert = 8 if (sc4_fert >= 0.35 or sc24_fert >= 0.30) else 6
                fert_reserve = 0 if step >= 672 else 1
            elif fert_dump_mild:
                p_thresh_fert = 0.60 * base_fert
                batch_fert = 5
                fert_reserve = 1
            elif shed_used >= 70 or st.get("effective_shed_load", shed_used) >= 75:
                p_thresh_fert = 0.58 * base_fert
                batch_fert = 5
                fert_reserve = 1
            else:
                p_thresh_fert = 0.70 * base_fert
                batch_fert = 4 if fert_held >= 7 else 3
                fert_reserve = 1 if step >= 672 else 2

            sellable_fert = max(0, avail_fert - fert_reserve)
            if p_fert >= p_thresh_fert and sellable_fert > 0:
                qty_fert = min(sellable_fert, batch_fert)
                if qty_fert > 0:
                    mkt.append(["SELL", "FERTILIZER", qty_fert])
                    shed_used -= qty_fert

    # 5b. PREDICTIVE EARLY-ENDGAME LIQUIDATION (steps 680..711):
    #     Preemptively liquidates goods the opponent is forecast to dump (score_24, P=0.93)
    #     and high-value assets with Dynamic Time-Decaying Price Floors, locking in
    #     optimal profits before mirror collapse and preventing Day-29 fire sales.
    if 680 <= step < 712 and len(mkt) < 10:
        opp_dumps_24 = _oracle_scores(st, "score_24")
        step_progress = min(1.0, max(0.0, (step - 680) / 32.0))

        # 1. Oracle 24-turn dump front-running with time-decaying price protection
        if opp_dumps_24 and len(mkt) < 10:
            cands_24 = [
                (it, float(sc or 0.0)) for it, sc in opp_dumps_24.items()
                if it != "MELON" and float(sc or 0.0) >= _ORACLE_24_THRESHOLDS.get(it, 0.20)
            ]
            cands_24.sort(key=lambda kv: -(kv[1] * _BASE_PRICE.get(kv[0], 50)))
            for item, score in cands_24:
                if len(mkt) >= 10:
                    break
                held = int(shed.get(item, 0) or 0)
                reserve = feed_reserve if item == "WHEAT" else 0
                already = sum(
                    int(o[2] or 0) for o in mkt
                    if isinstance(o, (list, tuple)) and len(o) >= 3
                    and o[0] == "SELL" and o[1] == item
                )
                avail = held - reserve - already
                if avail <= 0:
                    continue
                p = float(prices.get(item, 0) or 0)
                base = _BASE_PRICE.get(item, 100)
                decayed_floor = 0.52 * (1.0 - step_progress * 0.25)
                if score >= 0.35:
                    decayed_floor -= 0.05
                if p >= max(0.36, decayed_floor) * base:
                    batch = 8 if (score >= 0.35 or step >= 700) else 6
                    qty = min(avail, batch)
                    if qty > 0:
                        mkt.append(["SELL", item, qty])

        # 2. High-value asset pre-liquidation with Dynamic Time-Decaying Price Floors
        #    Progressively scales down price floor over steps 680..711 to capture top partial profits.
        #    Accelerates liquidation if opponent is forecast to dump this item (P=0.93).
        if len(mkt) < 10:
            high_val_targets = [
                ("MELON", 0.65, 5, 1),
                ("TOMATO", 0.75, 4, 1),
                ("FERTILIZER", 0.50, 6, 2),
                ("WOOL", 0.60, 4, 1),
                ("STRAWBERRY", 0.62, 5, 2),
                ("MILK", 0.60, 5, 2),
                ("EGG", 0.58, 5, 2),
                ("CARROT", 0.58, 5, 2),
            ]
            for item, min_p_ratio, max_b, min_h in high_val_targets:
                if len(mkt) >= 10:
                    break
                held = int(shed.get(item, 0) or 0)
                already = sum(
                    int(o[2] or 0) for o in mkt
                    if isinstance(o, (list, tuple)) and len(o) >= 3
                    and o[0] == "SELL" and o[1] == item
                )
                avail = held - already
                if avail < min_h:
                    continue
                p = float(prices.get(item, 0) or 0)
                base = _BASE_PRICE.get(item, 100)

                # Smoothly decay price floor over 32 endgame steps
                decay_factor = 1.0 - (step_progress * 0.30)
                sc24 = float(opp_dumps_24.get(item, 0.0) or 0.0) if opp_dumps_24 else 0.0
                opp_discount = 0.06 if sc24 >= 0.20 else 0.0
                current_p_ratio = max(0.38, (min_p_ratio * decay_factor) - opp_discount)

                if p >= current_p_ratio * base:
                    batch = max_b + 2 if (step >= 700 and avail > max_b) else max_b
                    qty = min(avail, batch)
                    if qty > 0:
                        mkt.append(["SELL", item, qty])

        # 3. Phased day-29 orderly liquidation (step >= 696)
        if step >= 696 and len(mkt) < 10:
            cands_day29 = sorted(
                ((int(q or 0), it) for it, q in shed.items() if int(q or 0) > 0),
                key=lambda x: -(x[0] * float(prices.get(x[1], 0) or 0))
            )
            d29_progress = min(1.0, max(0.0, (step - 696) / 16.0))
            for qty_held, it in cands_day29:
                if len(mkt) >= 10:
                    break
                p = float(prices.get(it, 0) or 0)
                base = _BASE_PRICE.get(it, 100)
                floor_ratio = max(0.20, 0.35 * (1.0 - d29_progress * 0.35))
                if p < floor_ratio * base and p < 1.0:
                    continue
                already = sum(
                    int(o[2] or 0) for o in mkt
                    if isinstance(o, (list, tuple)) and len(o) >= 3
                    and o[0] == "SELL" and o[1] == it
                )
                avail = qty_held - already
                if avail <= 0:
                    continue
                batch = 8 if step >= 704 else 6
                qty = min(avail, batch)
                if qty > 0:
                    mkt.append(["SELL", it, qty])

    # 6. Endgame liquidation (final 8 steps of day 29): convert all remaining shed inventory to cash.
    if step >= 712 and len(mkt) < 10:
        liquidate_cands = sorted(
            ((int(q or 0), it) for it, q in shed.items() if int(q or 0) > 0),
            key=lambda x: -(x[0] * float(prices.get(x[1], 0) or 0))
        )
        for qty_held, it in liquidate_cands:
            if len(mkt) >= 10:
                break
            p = float(prices.get(it, 0) or 0)
            if p < 1.0:
                continue
            already = sum(
                int(o[2] or 0) for o in mkt
                if isinstance(o, (list, tuple)) and len(o) >= 3 and o[0] == "SELL" and o[1] == it
            )
            qty = min(qty_held - already, 10)
            if qty > 0:
                mkt.append(["SELL", it, qty])

    return mkt[:10]

# ---------------------------------------------------------------- field coordinator
def _try_tile_maintenance(r, c, worker_idx, obs, st):
    """
    Unified field task coordinator for idle workers (farmer or hired hands).
    Evaluates maintenance opportunities on tile (r, c):
    1. DIG weeds.
    2. WATER thirsty plants (urgent rescue if missed >= 1 day, or opportunistic yield boost).
    3. FEED animals:
       - Urgent rescue: consecutive_unfed >= 1.
       - Proactive Feed Conversion: when shed_used >= 75 or effective_load >= 78 or late in day (hour >= 18),
         proactively feeds unfed animals even if consecutive_unfed == 0.
         This converts excess WHEAT into animal buffers, freeing shed capacity safely and boosting yields!
    Decrements st['shed_wheat_avail'], st['shed_used'], and st['effective_shed_load'] in-place when shed
    wheat is consumed so downstream market orders immediately see the relieved shed headroom.
    """
    tiles = st.get("tiles") or []
    if not (0 <= r < len(tiles) and 0 <= c < len(tiles[r])):
        return None
    cell = tiles[r][c]
    if not isinstance(cell, dict):
        return None

    claimed = st.setdefault("_claimed_tiles", set())
    if (r, c) in claimed:
        return None

    kind = cell.get("kind")
    if kind == "WEED":
        claimed.add((r, c))
        return ["DIG"]

    elif kind == "PLANT":
        unwatered = int(cell.get("consecutive_unwatered", 0) or 0)
        watered = cell.get("watered_today", False)
        crop = str(cell.get("crop") or cell.get("plant_type") or "").upper()
        if not watered:
            # Urgent plant rescue
            if unwatered >= _WATER_RESCUE_THRESHOLD:
                claimed.add((r, c))
                cell["watered_today"] = True
                return ["WATER"]
            # Opportunistic yield boost for one-time crops or evening lock-in
            elif crop in ("WHEAT", "CARROT", "MELON") or st["hour"] >= 16:
                claimed.add((r, c))
                cell["watered_today"] = True
                return ["WATER"]

    elif kind == "PASTURE" and cell.get("animal"):
        unfed = int(cell.get("consecutive_unfed", 0) or 0)
        fed = cell.get("fed_today", False)
        # Animals do not escape on Day 29 (no post-game midnight resolution); preserve wheat for cash
        if not fed and st.get("day", 0) < 29:
            # Determine if wheat is available in worker bag or shed
            priv = obs.get("private") or {}
            has_bag_wheat = False
            if worker_idx == -1:
                f_inv = priv.get("farmer") or {}
                if isinstance(f_inv, dict) and int(f_inv.get("WHEAT", 0) or 0) > 0:
                    has_bag_wheat = True
            elif 0 <= worker_idx:
                h_invs = priv.get("hands") or []
                if isinstance(h_invs, list) and worker_idx < len(h_invs) and isinstance(h_invs[worker_idx], dict):
                    if int(h_invs[worker_idx].get("WHEAT", 0) or 0) > 0:
                        has_bag_wheat = True

            has_shed_wheat = (st.get("shed_wheat_avail", 0) > 0)
            if has_bag_wheat or has_shed_wheat:
                # 1. Starvation rescue (urgent)
                if unfed >= _FEED_RESCUE_THRESHOLD:
                    if not has_bag_wheat:
                        st["shed_wheat_avail"] = max(0, st["shed_wheat_avail"] - 1)
                        st["shed_used"] = max(0, st["shed_used"] - 1)
                        st["effective_shed_load"] = max(0, st["effective_shed_load"] - 1)
                        if "WHEAT" in st["shed"]:
                            st["shed"]["WHEAT"] = max(0, int(st["shed"]["WHEAT"] or 0) - 1)
                    cell["fed_today"] = True
                    claimed.add((r, c))
                    return ["FEED"]

                # 2. Proactive feed conversion: converts wheat into animal buffer under shed congestion or evening
                elif (st["shed_used"] >= 75 or st["effective_shed_load"] >= 78 or st["hour"] >= 18):
                    if not has_bag_wheat:
                        st["shed_wheat_avail"] = max(0, st["shed_wheat_avail"] - 1)
                        st["shed_used"] = max(0, st["shed_used"] - 1)
                        st["effective_shed_load"] = max(0, st["effective_shed_load"] - 1)
                        if "WHEAT" in st["shed"]:
                            st["shed"]["WHEAT"] = max(0, int(st["shed"]["WHEAT"] or 0) - 1)
                    cell["fed_today"] = True
                    claimed.add((r, c))
                    return ["FEED"]

    return None


# ---------------------------------------------------------------- farmer channel
def evolve_farmer_action(obs, player_idx, base_farmer, st):
    """Evolvable single-farmer action. Return [OP, ...args].
    Strictly idle-only maintenance: when farmer outputs PASS, safely execute
    on-tile maintenance via the Field Task Coordinator.
    """
    base = list(base_farmer or ["PASS"])
    farmer_pos = st.get("farmer")
    if not (farmer_pos and isinstance(farmer_pos, (list, tuple)) and len(farmer_pos) >= 2):
        return base

    r, c = int(farmer_pos[0]), int(farmer_pos[1])
    if base[0] != "PASS":
        # Farmer is actively executing a route; mark tile as claimed
        st.setdefault("_claimed_tiles", set()).add((r, c))
        return base

    try:
        op = _try_tile_maintenance(r, c, -1, obs, st)
        if op:
            return op
    except Exception:
        pass
    return base


# ---------------------------------------------------------------- hands channel
def evolve_hand_actions(obs, player_idx, base_hands, st):
    """
    Evolvable per-hand actions, one entry per hired hand.
    Strictly idle-only maintenance: hands outputting PASS are coordinated by the
    Field Task Coordinator for weed digging, crop watering, and proactive feed conversion.
    """
    hands = st["hands"]
    out = [list(a) if isinstance(a, (list, tuple)) else ["PASS"] for a in (base_hands or [])]
    while len(out) < len(hands):
        out.append(["PASS"])

    if _ENABLE_HAND_RESCUE:
        claimed = st.setdefault("_claimed_tiles", set())
        # First register active non-PASS operations to prevent collisions
        for i, pos in enumerate(hands):
            if i < len(out) and isinstance(pos, (list, tuple)) and len(pos) >= 2:
                if out[i] and out[i][0] != "PASS":
                    claimed.add((int(pos[0]), int(pos[1])))

        # Coordinate maintenance for idle hands
        for i, pos in enumerate(hands):
            if i >= len(out) or not isinstance(pos, (list, tuple)) or len(pos) < 2:
                continue
            verb = out[i][0] if out[i] else "PASS"
            if verb != "PASS":
                continue
            r, c = int(pos[0]), int(pos[1])
            try:
                op = _try_tile_maintenance(r, c, i, obs, st)
                if op:
                    out[i] = op
            except Exception:
                pass
    return out

# =================== EVOLVE-BLOCK-END =====================


# Derived from every op observed in decoded ladder replays, NOT from the env spec
# description -- that description omits DROP, and silently rewriting DROP to PASS
# blocks hand carrying capacity and costs ~3% of final score.
_FARMER_OPS = {
    "NORTH", "SOUTH", "EAST", "WEST", "PASS", "PICKUP", "PLANT", "WATER", "HARVEST",
    "FERTILIZE", "BUILD_COOP", "BUILD_PASTURE", "DIG", "PLACE", "FEED", "DROP",
    "COLLECT_FERTILIZER", "CARE",
}
_MARKET_OPS = {"BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL", "HIRE", "BUY_LAND"}
_MAX_MARKET_ORDERS = 10


def _clean_op(entry, allowed):
    if isinstance(entry, str):
        entry = [entry]
    if not isinstance(entry, (list, tuple)) or not entry:
        return None
    op = str(entry[0])
    if op not in allowed:
        return None
    return [op] + [a for a in list(entry)[1:]]


def _sanitize(action, base, n_hands):
    """Guarantee a schema-valid action. Any malformed channel falls back to `base`."""
    if not isinstance(action, dict):
        return base
    out = {}

    farmer = _clean_op(action.get("farmer"), _FARMER_OPS)
    out["farmer"] = farmer or _clean_op(base.get("farmer"), _FARMER_OPS) or ["PASS"]

    hands = action.get("hands")
    if not isinstance(hands, (list, tuple)):
        hands = base.get("hands") or []
    cleaned = [(_clean_op(h, _FARMER_OPS) or ["PASS"]) for h in hands]
    out["hands"] = cleaned[:n_hands] + [["PASS"]] * max(0, n_hands - len(cleaned))

    market = action.get("market")
    if not isinstance(market, (list, tuple)):
        market = base.get("market") or []
    out["market"] = [m for m in (_clean_op(o, _MARKET_OPS) for o in market) if m][:_MAX_MARKET_ORDERS]
    return out


def agent(obs, configuration=None):
    """Top-level agent called by the Kaggle runner and the Shinka evaluator."""
    player_idx = int(obs.get("player", 0) or 0)
    try:
        base = _mohui.kaggle_agent_v66_meta_closed_loop(obs, configuration)
    except Exception:
        base = None
    if not isinstance(base, dict):
        base = {"farmer": ["PASS"], "hands": [], "market": []}

    farms = obs.get("farms", []) or []
    farm = farms[player_idx] if player_idx < len(farms) else {}
    n_hands = len(farm.get("hands") or [])

    forecast = _oracle_observe(obs, configuration)
    try:
        st = farm_state(obs, player_idx, forecast)
        evolved = {
            "farmer": evolve_farmer_action(obs, player_idx, base.get("farmer"), st),
            "hands": evolve_hand_actions(obs, player_idx, base.get("hands"), st),
            "market": evolve_market_orders(obs, player_idx, base.get("market"), st),
        }
    except Exception:
        evolved = base
    final = _sanitize(evolved, base, n_hands)
    _oracle_record(final)  # what we really submitted: the tracker's market accounting needs it
    return final