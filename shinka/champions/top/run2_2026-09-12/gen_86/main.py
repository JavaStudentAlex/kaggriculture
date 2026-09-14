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

_ANIMAL_YIELD = {"GOOSE": 1.00, "COW": 0.50, "SHEEP": 0.33}

# All sellable products for iteration
_ALL_PRODUCTS = ("STRAWBERRY", "MILK", "WOOL", "TOMATO", "CARROT", "EGG", "WHEAT", "MELON", "FERTILIZER")

# ---------------------------------------------------------------- hyperparameters
_ENABLE_CASH_FLOOR = False
_ENABLE_ANIMAL_TOPUP = False
_CASH_FLOOR_EARLY = 700.0
_CASH_FLOOR_UNTIL_DAY = 8
_CASH_FLOOR_LATE = 250.0
_ANIMAL_TARGET = 17
_ANIMAL_TOPUP_MIN_CASH = 1500.0
_ANIMAL_TOPUP_LAST_DAY = 26
_ANIMAL_PREFERENCE = ("GOOSE", "COW", "SHEEP")

_PRICE_THRESHOLD_RATIO = 0.85
_SHOP_SELL_BATCH_MAX = 4
_MIN_HELD_FOR_SHOP_SALE = 2
_SELL_PRIORITY = ("STRAWBERRY", "MILK", "WOOL", "TOMATO", "CARROT", "EGG", "WHEAT", "MELON")

_SHED_CAPACITY = 100
_ENABLE_SHED_PRESSURE = True
_SHED_PRESSURE_AT = 80
_SHED_PRESSURE_BATCH = 10
_SHED_PRESSURE_PRICE_RATIO = 0.35

_ENABLE_HAND_RESCUE = True
_WATER_RESCUE_THRESHOLD = 1
_FEED_RESCUE_THRESHOLD = 1

_OPENING_BUY_WHEAT_QTY = 35
_OPENING_SELL_WHEAT_QTY = 30

_ENABLE_ORACLE_PRIORITY = True
_ENABLE_ORACLE_FRONTRUN = True
_ENABLE_ORACLE_HOLD = False
_ORACLE_FRONTRUN_SCORE = 0.30
_ORACLE_FRONTRUN_BATCH = 6
_ORACLE_FRONTRUN_PRICE_RATIO = 0.60
_ORACLE_FRONTRUN_MIN_HELD = 1
_ORACLE_HOLD_SCORE = 0.50
_ORACLE_PRIORITY_HORIZON = "units_24"

# NEW: Premium selling threshold for non-demanded items
_PREMIUM_SELL_RATIO = 1.30       # sell non-demanded items when price >= 1.3x base
_PREMIUM_SELL_BATCH = 3          # small batches to avoid self-crashing


# ---------------------------------------------------------------- shared helpers
def _already_selling(mkt, item):
    """Count units of `item` already scheduled for sale in current order list."""
    total = 0
    for o in mkt:
        if isinstance(o, (list, tuple)) and len(o) >= 3 and o[0] == "SELL" and o[1] == item:
            total += int(o[2] or 0)
    return total


def _add_sell(mkt, item, max_qty, shed, reserve=0):
    """Add a sell order if possible. Returns qty scheduled (may be 0)."""
    if len(mkt) >= 10 or max_qty <= 0:
        return 0
    held = int(shed.get(item, 0) or 0)
    already = _already_selling(mkt, item)
    avail = held - already - reserve
    qty = min(max_qty, avail)
    if qty > 0:
        mkt.append(["SELL", item, qty])
    return max(0, qty)


# ---------------------------------------------------------------- derived state
def farm_state(obs, player_idx, oracle=None):
    """Per-turn derived features."""
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
    _shed = (obs.get("private") or {}).get("shed", {}) or {}
    _shed_used = sum(v for v in _shed.values() if isinstance(v, (int, float)))
    return {
        "day": day,
        "hour": int(obs.get("hour", 0) or 0),
        "step": int(obs.get("step") if obs.get("step") is not None else (24 * day + int(obs.get("hour", 0) or 0))),
        "money": float(farm.get("money", 0.0) or 0.0),
        "farmer": farm.get("farmer") or [0, 0],
        "hands": farm.get("hands") or [],
        "tiles": tiles,
        "plants": plants, "animals": animals, "weeds": weeds,
        "empties": empties, "locked": locked,
        "thirsty": thirsty, "hungry": hungry,
        "n_animals": len(animals), "n_plants": len(plants),
        "shed": _shed, "shed_used": _shed_used, "shed_cap": _SHED_CAPACITY,
        "shed_room": max(0, _SHED_CAPACITY - _shed_used),
        "prices": (obs.get("market") or {}).get("prices") or {},
        "shops": (obs.get("town") or {}).get("unlocked_shops", []) or [],
        "oracle": oracle if isinstance(oracle, dict) else None,
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
    Phased market pipeline. Max 10 orders per turn.

    Phase 0: Opening wheat scalp (steps 0-1)
    Phase 1: Base order filtering (wage guard, investment cutoff)
    Phase 2: Shed pressure valve (prevent overflow)
    Phase 3: Oracle front-run (sell before opponent dumps)
    Phase 4: Demand-driven selling (shop-demanded items)
    Phase 5: Premium selling (non-demanded items at elevated prices)
    Phase 6: Fertilizer monetization
    Phase 7: Endgame pre-liquidation (steps 660-711)
    Phase 8: Final liquidation (steps 712-720)
    """
    step = st["step"]
    hour = st["hour"]
    day = st["day"]
    money, prices, shed = st["money"], st["prices"], st["shed"]
    shed_used = st["shed_used"]
    mkt = list(base_orders or [])

    # ---- Phase 0: Opening wheat scalp ----
    if step == 0:
        return [["BUY_PRODUCT", "WHEAT", _OPENING_BUY_WHEAT_QTY]]
    elif step == 1:
        mkt = [["SELL", "WHEAT", _OPENING_SELL_WHEAT_QTY]] + [
            o for o in mkt
            if not (isinstance(o, (list, tuple)) and len(o) >= 2
                    and o[0] == "BUY_PRODUCT" and o[1] == "WHEAT")
        ]
        return mkt[:10]

    # ---- Phase 1: Base order filtering ----

    # 1a. Midnight liquidity & wage floor guard
    if (hour >= 20 or (220 <= step <= 245)) and day > 1:
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

    # 1b. Investment cutoff: graduated by type (animals/land day 25, seeds day 27, products day 29)
    filtered_inv = []
    for o in mkt:
        if isinstance(o, (list, tuple)) and len(o) >= 2:
            op = o[0]
            if op in ("BUY_ANIMAL", "BUY_LAND") and day >= 25:
                continue
            if op == "BUY_SEED" and day >= 27:
                continue
            if op == "BUY_PRODUCT" and step >= 696:
                continue
        filtered_inv.append(o)
    mkt = filtered_inv

    # 1c. Optional cash-floor guard (disabled)
    if _ENABLE_CASH_FLOOR:
        floor = _CASH_FLOOR_EARLY if day <= _CASH_FLOOR_UNTIL_DAY else _CASH_FLOOR_LATE
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

    # 1d. Optional animal top-up (disabled)
    if _ENABLE_ANIMAL_TOPUP and st["n_animals"] < _ANIMAL_TARGET and len(mkt) < 10:
        if money >= _ANIMAL_TOPUP_MIN_CASH and day <= _ANIMAL_TOPUP_LAST_DAY:
            already = sum(
                int(o[2] or 0) for o in mkt
                if isinstance(o, (list, tuple)) and len(o) >= 3 and o[0] == "BUY_ANIMAL"
            )
            want = _ANIMAL_TARGET - st["n_animals"] - already
            if want > 0:
                mkt.append(["BUY_ANIMAL", _ANIMAL_PREFERENCE[0], max(1, min(want, 2))])

    # Dynamic feed reserve: protect wheat for animal feeding
    feed_reserve = 0 if step >= 696 else (max(2, st["n_animals"]) if step >= 672 else max(4, st["n_animals"] * 2))

    def _item_reserve(item):
        if item == "WHEAT":
            return feed_reserve
        return 0

    # ---- Phase 2: Shed pressure valve ----
    # Pre-midnight flush: lower trigger at hour >= 21 to leave room for midnight sweep
    shed_trigger = 60 if hour >= 21 else _SHED_PRESSURE_AT
    if _ENABLE_SHED_PRESSURE and shed_used >= shed_trigger and len(mkt) < 10:
        if shed_used >= 95:
            p_ratio_floor, batch_limit = 0.05, 10
        elif shed_used >= 90:
            p_ratio_floor, batch_limit = 0.15, 10
        elif shed_used >= _SHED_PRESSURE_AT:
            p_ratio_floor, batch_limit = _SHED_PRESSURE_PRICE_RATIO, _SHED_PRESSURE_BATCH
        else:  # 60-79 range (pre-midnight flush)
            p_ratio_floor, batch_limit = 0.40, 6

        holdings = sorted(
            ((int(q or 0), it) for it, q in shed.items() if int(q or 0) > 0),
            reverse=True)
        for qty_held, item in holdings:
            if len(mkt) >= 10:
                break
            price = float(prices.get(item, 0) or 0)
            base = _BASE_PRICE.get(item, 100)
            if price < p_ratio_floor * base and price < 1.0:
                continue
            _add_sell(mkt, item, batch_limit, shed, reserve=_item_reserve(item))

    # ---- Phase 3: Oracle front-run ----
    # Sell ahead of predicted opponent dump before price crashes.
    # MELON excluded (AUC 0.51 = noise). Severity-scaled batches and price thresholds.
    sell_ahead = _oracle_scores(st, "score_4")
    if _ENABLE_ORACLE_FRONTRUN and sell_ahead and len(mkt) < 10:
        cands = [
            (it, float(sc or 0.0)) for it, sc in sell_ahead.items()
            if it != "MELON" and float(sc or 0.0) >= _ORACLE_FRONTRUN_SCORE
        ]
        # Sort by expected revenue impact (score * base price)
        cands.sort(key=lambda kv: -(kv[1] * _BASE_PRICE.get(kv[0], 50)))

        for item, score in cands:
            if len(mkt) >= 10:
                break
            held = int(shed.get(item, 0) or 0)
            price = float(prices.get(item, 0) or 0)
            base = _BASE_PRICE.get(item, 100)
            reserve = _item_reserve(item)
            if item == "FERTILIZER" and step < 672:
                reserve = max(reserve, 1)

            # Severity-scaled thresholds
            if score >= 0.60:
                p_ratio_req, batch = 0.50, 10
            elif score >= 0.45:
                p_ratio_req, batch = 0.55, 8
            else:
                p_ratio_req, batch = _ORACLE_FRONTRUN_PRICE_RATIO, _ORACLE_FRONTRUN_BATCH

            if (held - reserve) < _ORACLE_FRONTRUN_MIN_HELD or price < p_ratio_req * base:
                continue
            _add_sell(mkt, item, batch, shed, reserve=reserve)

    # ---- Oracle cadence bypass detection ----
    oracle_cadence_bypass = False
    _score_4_cache = {}
    if _ENABLE_ORACLE_PRIORITY and step >= 512:
        _score_4_cache = _oracle_scores(st, "score_4")
        if _score_4_cache:
            demanded_bypass = set()
            for shop in st["shops"]:
                demanded_bypass.update(_SHOP_DEMANDS.get(shop, ()))
            for it_b in demanded_bypass:
                if it_b != "MELON" and float(_score_4_cache.get(it_b, 0.0) or 0.0) >= 0.40:
                    oracle_cadence_bypass = True
                    break
            if not oracle_cadence_bypass:
                f_sc4 = float(_score_4_cache.get("FERTILIZER", 0.0) or 0.0)
                f_sc24 = float(_oracle_scores(st, "score_24").get("FERTILIZER", 0.0) or 0.0)
                f_held = int(shed.get("FERTILIZER", 0) or 0)
                if f_held >= 3 and (f_sc4 >= 0.30 or f_sc24 >= 0.25):
                    oracle_cadence_bypass = True

    # ---- Cadence: when to run Phases 4-6 ----
    # Every turn from step 660 onward (endgame), otherwise adaptive cadence
    if step >= 660:
        cadence = True
    else:
        cadence = (oracle_cadence_bypass or
                   (step % 4 == 0) or
                   (shed_used >= 60 and step % 2 == 0))

    if 144 <= step < 712 and cadence and len(mkt) < 10:
        # Build demanded set from unlocked shops
        demanded = set()
        for shop in st["shops"]:
            demanded.update(_SHOP_DEMANDS.get(shop, ()))

        # Oracle-sorted priority (least opponent supply first)
        priority = list(_SELL_PRIORITY)
        opp_supply = _oracle_scores(st, _ORACLE_PRIORITY_HORIZON)
        if _ENABLE_ORACLE_PRIORITY and opp_supply:
            priority.sort(key=lambda it: (
                0.0 if it == "MELON" else float(opp_supply.get(it, 0.0)),
                _SELL_PRIORITY.index(it)
            ))

        # ---- Phase 4: Demand-driven selling ----
        for item in priority:
            if len(mkt) >= 10:
                break
            if item not in demanded:
                continue
            held = int(shed.get(item, 0) or 0)
            reserve = _item_reserve(item)
            avail = held - reserve - _already_selling(mkt, item)
            min_req = 1 if item == "MELON" else _MIN_HELD_FOR_SHOP_SALE
            if avail < min_req:
                continue
            p = float(prices.get(item, 0) or 0)
            base = _BASE_PRICE.get(item, 100)

            # Severity-scaled thresholds based on oracle
            item_sc4 = float(_score_4_cache.get(item, 0.0) or 0.0) if _score_4_cache else 0.0
            if item_sc4 >= 0.60:
                p_thresh, batch_max = 0.55 * base, 6
            elif item_sc4 >= 0.40 or oracle_cadence_bypass:
                p_thresh, batch_max = 0.70 * base, _SHOP_SELL_BATCH_MAX
            else:
                p_thresh, batch_max = _PRICE_THRESHOLD_RATIO * base, _SHOP_SELL_BATCH_MAX

            if p >= p_thresh:
                _add_sell(mkt, item, batch_max, shed, reserve=reserve)

        # ---- Phase 5: Premium selling of non-demanded items ----
        # Captures price drift in items nobody is selling (e.g., TOMATO 60→247).
        # Small batches to avoid self-crashing the price.
        if len(mkt) < 10:
            # Sort by revenue potential: price * available qty, descending
            premium_cands = []
            for item in _ALL_PRODUCTS:
                if item in demanded or item == "WHEAT":  # wheat reserved for feed; demanded handled above
                    continue
                p = float(prices.get(item, 0) or 0)
                base = _BASE_PRICE.get(item, 100)
                if p < _PREMIUM_SELL_RATIO * base:
                    continue
                held = int(shed.get(item, 0) or 0)
                reserve = _item_reserve(item)
                avail = held - reserve - _already_selling(mkt, item)
                if avail < 1:
                    continue
                premium_cands.append((item, p * avail, p, avail))

            premium_cands.sort(key=lambda x: -x[1])
            for item, _, p, avail in premium_cands:
                if len(mkt) >= 10:
                    break
                batch = min(_PREMIUM_SELL_BATCH, avail)
                # In endgame, sell more aggressively
                if step >= 600:
                    batch = min(5, avail)
                _add_sell(mkt, item, batch, shed, reserve=_item_reserve(item))

        # ---- Phase 6: Fertilizer monetization ----
        fert_held = int(shed.get("FERTILIZER", 0) or 0)
        fert_min_trigger = 1 if day >= 27 else 3
        if fert_held >= fert_min_trigger and len(mkt) < 10:
            p_fert = float(prices.get("FERTILIZER", 0) or 0)
            avail_fert = fert_held - _already_selling(mkt, "FERTILIZER")
            if avail_fert > 0:
                f_sc4 = float(_score_4_cache.get("FERTILIZER", 0.0) or 0.0) if _score_4_cache else 0.0
                f_sc24 = float(_oracle_scores(st, "score_24").get("FERTILIZER", 0.0) or 0.0) if step >= 512 else 0.0
                dump_imminent = (f_sc4 >= 0.30) or (f_sc24 >= 0.25)

                if dump_imminent:
                    fert_reserve = 0 if step >= 648 else 1
                    p_thresh = 40.0 if f_sc4 >= 0.50 else 48.0
                    batch_max = 10 if (f_sc4 >= 0.40 or f_sc24 >= 0.35) else 8
                elif day >= 27:
                    # Late endgame: no crops can mature, liquidate all fertilizer
                    fert_reserve = 0
                    p_thresh = 42.0 if step >= 672 else 50.0
                    batch_max = 10
                else:
                    fert_reserve = 1 if step >= 600 else 2
                    p_thresh = 65.0 if shed_used >= 60 else 70.0
                    batch_max = 5 if fert_held >= 7 else 3

                if p_fert >= p_thresh and avail_fert > fert_reserve:
                    qty = min(avail_fert - fert_reserve, batch_max)
                    if qty > 0:
                        _add_sell(mkt, "FERTILIZER", qty, shed, reserve=0)

    # ---- Phase 7: Endgame pre-liquidation (steps 660-711) ----
    if 660 <= step < 712 and len(mkt) < 10:
        opp_dumps_24 = _oracle_scores(st, "score_24")

        # 7a. Oracle 24-turn dump front-running (P=0.93 for score_24 >= 0.20)
        if opp_dumps_24 and len(mkt) < 10:
            cands_24 = [
                (it, float(sc or 0.0)) for it, sc in opp_dumps_24.items()
                if it != "MELON" and float(sc or 0.0) >= 0.20
            ]
            cands_24.sort(key=lambda kv: -(kv[1] * _BASE_PRICE.get(kv[0], 50)))
            decay_dump = 1.0 - ((step - 660) / 52.0) * 0.20
            for item, score in cands_24:
                if len(mkt) >= 10:
                    break
                p = float(prices.get(item, 0) or 0)
                base = _BASE_PRICE.get(item, 100)
                min_p = max(0.35 * base, (0.42 if score >= 0.50 else 0.50) * decay_dump * base)
                if p < min_p:
                    continue
                batch = 10 if score >= 0.50 else (8 if score >= 0.35 else 6)
                _add_sell(mkt, item, batch, shed, reserve=_item_reserve(item))

        # 7b. High-value asset pre-liquidation with time-decaying price floors
        if len(mkt) < 10:
            decay = 1.0 - ((step - 660) / 52.0) * 0.25
            high_val_targets = [
                ("MELON", 0.65, 5, 1),
                ("WOOL", 0.60, 4, 1),
                ("TOMATO", 0.75, 4, 1),
                ("MILK", 0.70, 4, 1),
                ("STRAWBERRY", 0.70, 4, 1),
                ("FERTILIZER", 0.45, 8, 1),
                ("EGG", 0.65, 4, 1),
                ("CARROT", 0.65, 4, 1),
            ]
            for item, min_p_ratio, max_b, min_h in high_val_targets:
                if len(mkt) >= 10:
                    break
                held = int(shed.get(item, 0) or 0)
                already = _already_selling(mkt, item)
                avail = held - already
                if avail < min_h:
                    continue
                p = float(prices.get(item, 0) or 0)
                base = _BASE_PRICE.get(item, 100)
                effective_ratio = max(0.35, min_p_ratio * decay)
                if p >= effective_ratio * base:
                    batch = max_b + (2 if step >= 696 else 0)
                    _add_sell(mkt, item, batch, shed, reserve=0)

        # 7c. Phased day-29 orderly liquidation (step >= 696)
        if step >= 696 and len(mkt) < 10:
            cands_day29 = sorted(
                ((int(q or 0), it) for it, q in shed.items() if int(q or 0) > 0),
                key=lambda x: -(x[0] * float(prices.get(x[1], 0) or 0))
            )
            for qty_held, it in cands_day29:
                if len(mkt) >= 10:
                    break
                p = float(prices.get(it, 0) or 0)
                base = _BASE_PRICE.get(it, 100)
                if p < 0.35 * base and p < 1.0:
                    continue
                _add_sell(mkt, it, 6, shed, reserve=0)

    # ---- Phase 8: Final liquidation (steps 712-720) ----
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
            _add_sell(mkt, it, 10, shed, reserve=0)

    return mkt[:10]


# ---------------------------------------------------------------- farmer channel
def evolve_farmer_action(obs, player_idx, base_farmer, st):
    """Evolvable single-farmer action.
    When farmer is idle (PASS):
    1. Act on current tile if it needs maintenance
    2. Otherwise move toward the highest-value emergency
    """
    base = list(base_farmer or ["PASS"])
    if base[0] != "PASS" or not st.get("tiles") or not st.get("farmer"):
        return base

    try:
        r, c = int(st["farmer"][0]), int(st["farmer"][1])
        tiles = st["tiles"]
        if 0 <= r < len(tiles) and 0 <= c < len(tiles[r]):
            cell = tiles[r][c]
            if isinstance(cell, dict):
                kind = cell.get("kind")
                if kind == "WEED":
                    st["_farmer_rescue"] = (r, c, "DIG")
                    return ["DIG"]
                elif kind == "PLANT":
                    if int(cell.get("consecutive_unwatered", 0) or 0) >= _WATER_RESCUE_THRESHOLD \
                            and not cell.get("watered_today"):
                        st["_farmer_rescue"] = (r, c, "WATER")
                        return ["WATER"]
                elif kind == "PASTURE" and cell.get("animal"):
                    if int(cell.get("consecutive_unfed", 0) or 0) >= _FEED_RESCUE_THRESHOLD \
                            and not cell.get("fed_today"):
                        st["_farmer_rescue"] = (r, c, "FEED")
                        return ["FEED"]

        # Current tile is fine; move toward highest-value emergency
        if _ENABLE_HAND_RESCUE:
            emergencies = []
            for er, ec, cell in st.get("hungry", []):
                an = cell.get("animal")
                val = 1000 if an == "GOOSE" else (500 if an == "COW" else 330)
                emergencies.append((er, ec, val))
            for er, ec, cell in st.get("thirsty", []):
                crop = cell.get("crop") or cell.get("seed")
                bprice = _BASE_PRICE.get(crop, 50)
                emergencies.append((er, ec, bprice))
            for er, ec in st.get("weeds", []):
                emergencies.append((er, ec, 10))

            if emergencies:
                emergencies.sort(key=lambda x: (-x[2], abs(r - x[0]) + abs(c - x[1])))
                er, ec = emergencies[0][0], emergencies[0][1]
                st["_farmer_rescue"] = (er, ec, "MOVE")
                if r < er: return ["SOUTH"]
                elif r > er: return ["NORTH"]
                elif c < ec: return ["EAST"]
                elif c > ec: return ["WEST"]
    except Exception:
        pass
    return base


# ---------------------------------------------------------------- hands channel
def evolve_hand_actions(obs, player_idx, base_hands, st):
    """Evolvable per-hand actions.
    Rescue policy: idle hands act on emergencies (water/feed/dig),
    move toward unhandled emergencies, or do preventive maintenance.
    """
    hands = st["hands"]
    out = [list(a) if isinstance(a, (list, tuple)) else ["PASS"] for a in (base_hands or [])]
    while len(out) < len(hands):
        out.append(["PASS"])

    if not _ENABLE_HAND_RESCUE:
        return out

    tiles = st.get("tiles") or []

    # Build emergency list sorted by asset value
    emergencies = []
    for r, c, cell in st["hungry"]:
        an = cell.get("animal")
        val = 1000 if an == "GOOSE" else (500 if an == "COW" else 330)
        emergencies.append((r, c, "FEED", val))
    for r, c, cell in st["thirsty"]:
        crop = cell.get("crop") or cell.get("seed")
        base = _BASE_PRICE.get(crop, 50)
        emergencies.append((r, c, "WATER", base))
    for r, c in st.get("weeds", []):
        emergencies.append((r, c, "DIG", 10))

    emergencies.sort(key=lambda x: -x[3])

    claimed = set()
    f_res = st.get("_farmer_rescue")
    # Only block tile if farmer is actively acting (not just moving)
    if f_res and len(f_res) >= 3 and f_res[2] != "MOVE":
        claimed.add((f_res[0], f_res[1]))

    idle_hands = []
    for i, pos in enumerate(hands):
        if i >= len(out) or not isinstance(pos, (list, tuple)) or len(pos) < 2:
            continue
        verb = out[i][0] if out[i] else "PASS"
        if verb != "PASS":
            continue
        key = (int(pos[0]), int(pos[1]))

        # Check if standing on an emergency
        acted = False
        for er, ec, eact, evalue in emergencies:
            if key == (er, ec) and key not in claimed:
                out[i] = [eact]
                claimed.add(key)
                acted = True
                break
        if not acted:
            idle_hands.append((i, key))

    # Route idle hands toward unhandled emergencies
    unhandled = [e for e in emergencies if (e[0], e[1]) not in claimed]
    for er, ec, eact, evalue in unhandled:
        if not idle_hands:
            break
        best_idx = -1
        best_dist = 999
        for idx, (i, (hr, hc)) in enumerate(idle_hands):
            dist = abs(hr - er) + abs(hc - ec)
            if dist < best_dist:
                best_dist = dist
                best_idx = idx

        if best_idx != -1:
            i, (hr, hc) = idle_hands.pop(best_idx)
            claimed.add((er, ec))
            if hr < er: out[i] = ["SOUTH"]
            elif hr > er: out[i] = ["NORTH"]
            elif hc < ec: out[i] = ["EAST"]
            elif hc > ec: out[i] = ["WEST"]

    # Remaining idle hands: preventive maintenance on current tile
    shed_wheat = int(st.get("shed", {}).get("WHEAT", 0) or 0)
    for i, (hr, hc) in idle_hands:
        if 0 <= hr < len(tiles) and 0 <= hc < len(tiles[hr]):
            cell = tiles[hr][hc]
            if isinstance(cell, dict):
                ck = cell.get("kind")
                if ck == "PLANT" and not cell.get("watered_today"):
                    out[i] = ["WATER"]
                    claimed.add((hr, hc))
                elif ck == "PASTURE" and cell.get("animal") and not cell.get("fed_today") and shed_wheat > 0:
                    out[i] = ["FEED"]
                    shed_wheat -= 1
                    claimed.add((hr, hc))

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