"""
Grandmaster Seed Champion for Kaggriculture DRQ Evolution.

Run-3 seed (2026-09-14): run 2's crowned gen 33 (dynamic shop batch sizing, "Orchard
Tide" on Kaggle) with the oracle levers switched on from turn 256 instead of 512.

Backbone: Mohui v66 meta closed-loop controller (read-only, outside EVOLVE-BLOCK).
Oracle:   the frozen TinyTimeMixer opponent order-flow checkpoint in
          shinka/evolution/checkpoint/ (the newest promoted daily fine-tune; its
          README says which; context 256 turns since 2026-09-14), served by
          kagg_oracle.py (read-only, outside EVOLVE-BLOCK). From turn 256 (day 10
          h16) it forecasts every turn how many units of each product the opponent
          will put on the market at each of the next 96 turns; the forecast
          reaches the policy as st["oracle"] (None before turn 256).
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

# ---------------------------------------------------------------- hyperparameters
# Base: Program 3 (0.76 combined, 72% WR) — simplicity proven best.
# Animal topup and complex routing DISABLED: measured net negative.
_ENABLE_CASH_FLOOR = False
_ENABLE_ANIMAL_TOPUP = False
_CASH_FLOOR_EARLY = 700.0
_CASH_FLOOR_UNTIL_DAY = 8
_CASH_FLOOR_LATE = 250.0

# Revenue concentration / price collapse
_PRICE_THRESHOLD_RATIO = 0.85
_SHOP_SELL_BATCH_MAX = 4
_MIN_HELD_FOR_SHOP_SALE = 2
_SELL_PRIORITY = ("STRAWBERRY", "MILK", "WOOL", "TOMATO", "CARROT", "EGG", "WHEAT", "MELON")

# Premium sell for non-demanded goods (oracle-aware front-running)
_PREMIUM_SELL_RATIO = 0.80
_PREMIUM_SELL_BATCH = 3

# Shed overflow
_SHED_CAPACITY = 100
_ENABLE_SHED_PRESSURE = True
_SHED_PRESSURE_AT = 80
_SHED_PRESSURE_BATCH = 10
_SHED_PRESSURE_PRICE_RATIO = 0.35

# Plant/animal survival
_ENABLE_HAND_RESCUE = True
_WATER_RESCUE_THRESHOLD = 1
_FEED_RESCUE_THRESHOLD = 1

# Opening scalp
_OPENING_BUY_WHEAT_QTY = 35
_OPENING_SELL_WHEAT_QTY = 30

# Oracle levers
_ORACLE_FROM_STEP = 256
_ENABLE_ORACLE_PRIORITY = True
_ENABLE_ORACLE_FRONTRUN = True
_ENABLE_ORACLE_HOLD = False
_ORACLE_FRONTRUN_SCORE = 0.30
_ORACLE_FRONTRUN_BATCH = 6
_ORACLE_FRONTRUN_PRICE_RATIO = 0.60
_ORACLE_FRONTRUN_MIN_HELD = 1
_ORACLE_HOLD_SCORE = 0.50
_ORACLE_PRIORITY_HORIZON = "units_24"

# Seed substitution (oracle compliance)
_ENABLE_SEED_SUBSTITUTION = True
_SEED_SUB_THRESHOLD = 1.5


# ---------------------------------------------------------------- derived state
def farm_state(obs, player_idx, oracle=None):
    """Per-turn derived features."""
    farms = obs.get("farms", []) or []
    farm = farms[player_idx] if player_idx < len(farms) else {}
    tiles = farm.get("tiles", []) or []

    day = int(obs.get("day", 0) or 0)
    hour = int(obs.get("hour", 0) or 0)

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
                    unfed = int(cell.get("consecutive_unfed", 0) or 0)
                    # Enhanced hungry detection: late-day proactive feeding (from Program 3)
                    if (unfed >= _FEED_RESCUE_THRESHOLD or (hour >= 18 and unfed >= 0)) \
                            and not cell.get("fed_today"):
                        hungry.append((r, c, cell))
                else:
                    empties.append((r, c))
            elif kind == "WEED":
                weeds.append((r, c))

    _shed = (obs.get("private") or {}).get("shed", {}) or {}
    _shed_used = sum(v for v in _shed.values() if isinstance(v, (int, float)))
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
    """Evolvable market policy. Max 10 orders per turn."""
    step = int(obs.get("step") if obs.get("step") is not None else (24 * st["day"] + st["hour"]))
    hour = st["hour"]
    money, prices, shed = st["money"], st["prices"], st["shed"]
    mkt = list(base_orders or [])

    # 1. Opening wheat scalp
    if step == 0:
        return [["BUY_PRODUCT", "WHEAT", _OPENING_BUY_WHEAT_QTY]]
    elif step == 1:
        mkt = [["SELL", "WHEAT", _OPENING_SELL_WHEAT_QTY]] + [
            o for o in mkt
            if not (isinstance(o, (list, tuple)) and len(o) >= 2
                    and o[0] == "BUY_PRODUCT" and o[1] == "WHEAT")
        ]
        return mkt[:10]

    # 1b. Oracle-Guided Seed Substitution (oracle compliance)
    if _ENABLE_SEED_SUBSTITUTION and step >= _ORACLE_FROM_STEP:
        units_96 = _oracle_scores(st, "units_96")
        if units_96:
            new_mkt = []
            for o in mkt:
                if isinstance(o, (list, tuple)) and len(o) >= 3 and o[0] == "BUY_SEED":
                    crop = str(o[1])
                    qty = int(o[2] or 0)
                    if crop not in ("WHEAT", "MELON") and float(units_96.get(crop, 0.0)) >= _SEED_SUB_THRESHOLD:
                        alts = {
                            "CARROT": ["TOMATO", "STRAWBERRY"],
                            "TOMATO": ["CARROT", "STRAWBERRY"],
                            "STRAWBERRY": ["TOMATO", "CARROT"]
                        }
                        best_alt = crop
                        best_score = float(units_96.get(crop, 0.0))
                        target_score = best_score - 0.5
                        for alt in alts.get(crop, []):
                            alt_score = float(units_96.get(alt, 0.0))
                            if alt_score < target_score:
                                best_alt = alt
                                target_score = alt_score
                        if best_alt != crop:
                            p_crop = float(prices.get(crop, _BASE_PRICE.get(crop, 50)))
                            p_alt = float(prices.get(best_alt, _BASE_PRICE.get(best_alt, 50)))
                            if qty > 0:
                                new_qty = max(1, int((qty * p_crop) / p_alt))
                                if new_qty > qty:
                                    avail_space = len(st.get("empties", [])) + len(st.get("weeds", []))
                                    new_qty = min(new_qty, max(qty, avail_space))
                            else:
                                new_qty = 0
                            o = ["BUY_SEED", best_alt, new_qty]
                new_mkt.append(o)
            mkt = new_mkt

    # 2. Midnight wage guard (enhanced: urgent wheat reserve from Program 3)
    if (hour >= 20 or (220 <= step <= 245)) and st["day"] > 1:
        n_hands = len(st.get("hands", []))
        wage_reserve = max(250.0, n_hands * 120.0)
        urgent_wage_reserve = max(20.0, n_hands * 25.0)
        wheat_held = int(shed.get("WHEAT", 0) or 0)
        filtered = []
        for o in mkt:
            if isinstance(o, (list, tuple)) and len(o) >= 3 and o[0] == "BUY_PRODUCT":
                cost = float(prices.get(str(o[1]), _BASE_PRICE.get(str(o[1]), 50))) * int(o[2] or 0)
                reserve = urgent_wage_reserve if (str(o[1]) == "WHEAT" and wheat_held < st["n_animals"]) else wage_reserve
                if money - cost < reserve:
                    continue
                money -= cost
            filtered.append(o)
        mkt = filtered

    # 2b. Enhanced day 28-29 cancellation (from Program 3: also cancels HIRE, BUY_PRODUCT)
    if st["day"] >= 28:
        cancel_ops = {"BUY_SEED", "BUY_ANIMAL", "BUY_LAND", "HIRE"}
        wheat_held = int(shed.get("WHEAT", 0) or 0)
        if st["day"] >= 29 or wheat_held >= st["n_animals"]:
            cancel_ops.add("BUY_PRODUCT")
        mkt = [
            o for o in mkt
            if not (isinstance(o, (list, tuple)) and len(o) >= 2 and o[0] in cancel_ops)
        ]

    # 3. Optional cash-floor guard (disabled)
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

    # Dynamic feed reserve: steps down in endgame
    feed_reserve = 0 if step >= 696 else (max(2, st["n_animals"]) if step >= 672 else max(4, st["n_animals"] * 2))

    # 4b. Tiered shed pressure valve
    shed_used = st["shed_used"]
    if _ENABLE_SHED_PRESSURE and shed_used >= _SHED_PRESSURE_AT and len(mkt) < 10:
        p_ratio_floor = 0.05 if shed_used >= 95 else (0.15 if shed_used >= 90 else _SHED_PRESSURE_PRICE_RATIO)
        batch_limit = 10 if shed_used >= 90 else _SHED_PRESSURE_BATCH
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
            already = sum(
                int(o[2] or 0) for o in mkt
                if isinstance(o, (list, tuple)) and len(o) >= 3
                and o[0] == "SELL" and o[1] == item
            )
            reserve = feed_reserve if item == "WHEAT" else 0
            avail = qty_held - already - reserve
            qty = min(avail, batch_limit)
            if qty > 0:
                mkt.append(["SELL", item, qty])

    # 4c. Oracle front-run (seat-asymmetric, confidence-scaled batches)
    sell_ahead = _oracle_scores(st, "score_4")
    frontrun_score_threshold = 0.25 if player_idx == 0 else _ORACLE_FRONTRUN_SCORE
    if _ENABLE_ORACLE_FRONTRUN and sell_ahead and len(mkt) < 10:
        cands = [
            (it, float(sc or 0.0)) for it, sc in sell_ahead.items()
            if it != "MELON" and float(sc or 0.0) >= frontrun_score_threshold
        ]
        cands.sort(key=lambda kv: -kv[1])
        for item, score in cands:
            if len(mkt) >= 10:
                break
            held = int(shed.get(item, 0) or 0)
            price = float(prices.get(item, 0) or 0)
            base = _BASE_PRICE.get(item, 100)
            reserve = feed_reserve if item == "WHEAT" else 0
            if (held - reserve) < _ORACLE_FRONTRUN_MIN_HELD or price < _ORACLE_FRONTRUN_PRICE_RATIO * base:
                continue
            already = sum(
                int(o[2] or 0) for o in mkt
                if isinstance(o, (list, tuple)) and len(o) >= 3 and o[0] == "SELL" and o[1] == item
            )
            # Confidence-scaled batch sizing
            batch = 10 if score >= 0.50 else (8 if score >= 0.40 else _ORACLE_FRONTRUN_BATCH)
            qty = min(held - reserve - already, batch)
            if qty > 0:
                mkt.append(["SELL", item, qty])

    # 5. Town-shop demand harvesting with oracle cadence bypass
    oracle_cadence_bypass = False
    if _ENABLE_ORACLE_PRIORITY and step >= _ORACLE_FROM_STEP:
        _score_4_bypass = _oracle_scores(st, "score_4")
        if _score_4_bypass:
            _demanded_bypass = set()
            for shop in st["shops"]:
                _demanded_bypass.update(_SHOP_DEMANDS.get(shop, ()))
            bypass_thresh = 0.35 if player_idx == 0 else 0.40
            for _it_b in _demanded_bypass:
                if _it_b != "MELON" and float(_score_4_bypass.get(_it_b, 0.0)) >= bypass_thresh:
                    oracle_cadence_bypass = True
                    break

    # Simple cadence (proven best in Program 3: 77% seat1 WR)
    base_cadence = (step % 4 == 0)
    cadence = oracle_cadence_bypass or base_cadence or (shed_used >= 60 and step % 2 == 0) or (step >= 672 and step % 2 == 0)

    if 144 <= step < 712 and cadence and len(mkt) < 10:
        demanded = set()
        for shop in st["shops"]:
            demanded.update(_SHOP_DEMANDS.get(shop, ()))
        priority = list(_SELL_PRIORITY)
        opp_supply = _oracle_scores(st, _ORACLE_PRIORITY_HORIZON)
        opp_score_24 = _oracle_scores(st, "score_24")
        if _ENABLE_ORACLE_PRIORITY and opp_supply:
            priority.sort(key=lambda it: (0.0 if it == "MELON" else float(opp_supply.get(it, 0.0)), _SELL_PRIORITY.index(it)))

        for item in priority:
            if len(mkt) >= 10:
                break

            is_demanded = item in demanded
            p = float(prices.get(item, 0) or 0)
            base = _BASE_PRICE.get(item, 100)

            # Unified handling: demanded and non-demanded premium items
            if is_demanded:
                p_thresh = 0.70 * base if oracle_cadence_bypass else _PRICE_THRESHOLD_RATIO * base
                # Non-demanded: skip WHEAT/FERTILIZER, sell at fair price
                # Oracle-aware: FRONT-RUN opponent dumps (sell before their volume crashes price)
                if item == "WHEAT" or item == "FERTILIZER":
                    continue
                if opp_score_24 and item != "MELON":
                    sc = float(opp_score_24.get(item, 0.0))
                    if sc >= 0.30:
                        p_thresh = 0.65 * base  # Front-run: sell before their dump crashes price
                    else:
                        p_thresh = _PREMIUM_SELL_RATIO * base
                else:
                    p_thresh = _PREMIUM_SELL_RATIO * base

            if p < p_thresh:
                continue

            held = int(shed.get(item, 0) or 0)
            reserve = feed_reserve if item == "WHEAT" else 0
            avail = held - reserve
            min_req = 1 if item in ("MELON", "TOMATO", "EGG") else _MIN_HELD_FOR_SHOP_SALE
            if avail < min_req:
                continue

            already = sum(
                int(o[2] or 0) for o in mkt
                if isinstance(o, (list, tuple)) and len(o) >= 3
                and o[0] == "SELL" and o[1] == item
            )

            if is_demanded:
                # Dynamic batch sizing
                batch_max = _SHOP_SELL_BATCH_MAX
                if _ENABLE_ORACLE_PRIORITY and step >= _ORACLE_FROM_STEP and item != "MELON":
                    sc24 = float(opp_score_24.get(item, 0.0)) if opp_score_24 else 0.0
                    supp = float(opp_supply.get(item, 0.0)) if opp_supply else 0.0
                    if sc24 >= 0.30 or supp >= 1.2:
                        batch_max = 8
                    elif sc24 >= 0.15 or supp >= 0.6:
                        batch_max = 6
                    else:
                        batch_max = 3
                else:
                    batch_max = 2 if item == "MELON" else 3
            else:
                # Non-demanded: increase batch when front-running opponent dump
                if opp_score_24 and item != "MELON":
                    sc24_nd = float(opp_score_24.get(item, 0.0))
                    batch_max = 5 if sc24_nd >= 0.30 else _PREMIUM_SELL_BATCH
                else:
                    batch_max = _PREMIUM_SELL_BATCH

            qty = min(avail - already, batch_max)
            if qty > 0 and len(mkt) < 10:
                mkt.append(["SELL", item, qty])

        # Fertilizer monetization (farm-state aware reserve)
        fert_held = int(shed.get("FERTILIZER", 0) or 0)
        fert_res_crop = 4 if (st["n_plants"] >= 8 and step < 700) else 2
        if fert_held >= (fert_res_crop + 2) and len(mkt) < 10:
            p_fert = float(prices.get("FERTILIZER", 0) or 0)
            if p_fert >= 0.70 * 100:
                already_fert = sum(
                    int(o[2] or 0) for o in mkt
                    if isinstance(o, (list, tuple)) and len(o) >= 3 and o[0] == "SELL" and o[1] == "FERTILIZER"
                )
                qty_fert = min(fert_held - fert_res_crop - already_fert, 3)
                if qty_fert > 0:
                    mkt.append(["SELL", "FERTILIZER", qty_fert])

    # 5b. Oracle-triggered early endgame liquidation (seat-aware timing)
    opp_dumps_24 = _oracle_scores(st, "score_24")
    early_liq_start = 624 if player_idx == 0 else 640
    early_dump_thresh = 0.20 if player_idx == 0 else 0.25
    early_liquidation_active = (680 <= step < 712)
    if not early_liquidation_active and early_liq_start <= step < 680 and opp_dumps_24:
        trigger_items = ["MELON", "TOMATO", "WOOL"]
        if st["n_plants"] < 8:
            trigger_items.append("FERTILIZER")
        for item in trigger_items:
            if float(opp_dumps_24.get(item, 0.0)) >= early_dump_thresh:
                early_liquidation_active = True
                break

    if early_liquidation_active and len(mkt) < 10:
        # Oracle 24-turn dump front-running
        if opp_dumps_24 and len(mkt) < 10:
            cands_24 = [
                (it, float(sc or 0.0)) for it, sc in opp_dumps_24.items()
                if it != "MELON" and float(sc or 0.0) >= early_dump_thresh
            ]
            cands_24.sort(key=lambda kv: -kv[1])
            for item, score in cands_24:
                if len(mkt) >= 10:
                    break
                held = int(shed.get(item, 0) or 0)
                fert_res = 4 if (item == "FERTILIZER" and st["n_plants"] >= 8 and step < 700) else 0
                reserve = feed_reserve if item == "WHEAT" else fert_res
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
                if p >= 0.52 * base:
                    batch = 8 if score >= 0.35 else 6
                    qty = min(avail, batch)
                    if qty > 0:
                        mkt.append(["SELL", item, qty])

        # High-value pre-liquidation (farm-state aware)
        if len(mkt) < 10:
            fert_active = (st["n_plants"] >= 8 and step < 700)
            fert_p_ratio = 0.80 if fert_active else 0.50
            fert_batch = 2 if fert_active else 6
            fert_min_h = 4 if fert_active else 2
            high_val_targets = [
                ("MELON", 0.65, 5, 1),
                ("TOMATO", 0.75, 4, 1),
                ("FERTILIZER", fert_p_ratio, fert_batch, fert_min_h),
                ("WOOL", 0.60, 4, 1),
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
                if p >= min_p_ratio * base:
                    sellable = (avail - min_h) if (item == "FERTILIZER" and fert_active) else avail
                    qty = min(sellable, max_b)
                    if qty > 0:
                        mkt.append(["SELL", item, qty])

        # Day-29 orderly liquidation (step >= 696)
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
                already = sum(
                    int(o[2] or 0) for o in mkt
                    if isinstance(o, (list, tuple)) and len(o) >= 3
                    and o[0] == "SELL" and o[1] == it
                )
                avail = qty_held - already
                qty = min(avail, 6)
                if qty > 0:
                    mkt.append(["SELL", it, qty])

    # 6. Enhanced endgame liquidation (from Program 3: replaces mkt entirely at step >= 712)
    if step >= 712:
        liquidate_cands = sorted(
            ((int(q or 0), it) for it, q in shed.items() if int(q or 0) > 0),
            key=lambda x: -(x[0] * float(prices.get(x[1], 0) or 0))
        )
        mkt_liq = []
        already_sold = {}
        for qty_held, it in liquidate_cands:
            if len(mkt_liq) >= 10:
                break
            p = float(prices.get(it, 0) or 0)
            if p < 1.0:
                continue
            sold = already_sold.get(it, 0)
            rem = qty_held - sold
            if rem > 0:
                take = min(rem, 10)
                mkt_liq.append(["SELL", it, take])
                already_sold[it] = sold + take
        if mkt_liq:
            return mkt_liq[:10]

    return mkt[:10]


# ---------------------------------------------------------------- farmer channel
def evolve_farmer_action(obs, player_idx, base_farmer, st):
    """Farmer action: on-tile rescue + opportunistic DROP."""
    base = list(base_farmer or ["PASS"])
    if base[0] == "PASS" and st.get("tiles") and st.get("farmer"):
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
                        unfed = int(cell.get("consecutive_unfed", 0) or 0)
                        if (unfed >= _FEED_RESCUE_THRESHOLD or (st["hour"] >= 18 and unfed >= 0)) \
                                and not cell.get("fed_today"):
                            st["_farmer_rescue"] = (r, c, "FEED")
                            return ["FEED"]

            # Opportunistic DROP: deposit harvest into shed if adjacent
            if st.get("shed_room", 0) >= 2:
                priv = obs.get("private") or {}
                f_inv = priv.get("farmer") or {}
                if isinstance(f_inv, dict) and any(int(v or 0) > 0 for v in f_inv.values() if isinstance(v, (int, float))):
                    shed_tiles = [(rr, cc) for rr, row in enumerate(tiles) for cc, cell_x in enumerate(row or []) if isinstance(cell_x, dict) and cell_x.get("kind") == "SHED"]
                    if any(abs(r - sr) + abs(c - sc) == 1 for sr, sc in shed_tiles):
                        return ["DROP"]
        except Exception:
            pass
    return base


# ---------------------------------------------------------------- hands channel
def evolve_hand_actions(obs, player_idx, base_hands, st):
    """Simple on-tile hand rescue (from Program 3 — proven best, no complex routing)."""
    hands = st["hands"]
    out = [list(a) if isinstance(a, (list, tuple)) else ["PASS"] for a in (base_hands or [])]
    while len(out) < len(hands):
        out.append(["PASS"])

    if _ENABLE_HAND_RESCUE:
        thirsty = {(r, c) for r, c, _ in st["thirsty"]}
        hungry = {(r, c) for r, c, _ in st["hungry"]}
        weeds = {(r, c) for r, c in st.get("weeds", [])}
        farmer_claimed = set()
        f_res = st.get("_farmer_rescue")
        if f_res:
            farmer_claimed.add((f_res[0], f_res[1]))

        for i, pos in enumerate(hands):
            if i >= len(out) or not isinstance(pos, (list, tuple)) or len(pos) < 2:
                continue
            verb = out[i][0] if out[i] else "PASS"
            # STRICT IDLE-ONLY: never override active backbone operations
            if verb != "PASS":
                continue
            key = (int(pos[0]), int(pos[1]))
            if key in farmer_claimed:
                continue
            # Priority: hungry animals (prevent escape) > weeds > thirsty plants
            if key in hungry:
                out[i] = ["FEED"]
                farmer_claimed.add(key)
            elif key in weeds:
                out[i] = ["DIG"]
                farmer_claimed.add(key)
            elif key in thirsty:
                out[i] = ["WATER"]
                farmer_claimed.add(key)
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