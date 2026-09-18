"""
Kaggriculture DRQ seed program - FULLY EVOLVABLE agent.

Seed: **Hazel Weir** (shinka/champions/top/patch_2026-09-14/hazel_weir/main.py), the
last champion added (2026-09-15, Kaggle submission 56246758). Run-3 gen 198 "Copper
Weir" plus the shed-overflow patch: 87.7% (526W-74L) vs the 15-pool, 516-44 on fresh
tournament seeds, avg cash $81,317.

WHAT IS EVOLVABLE
-----------------
Everything that decides anything. The EVOLVE-BLOCK spans the WHOLE agent:
  * reference data (`_SHOP_DEMANDS`, `_BASE_PRICE`)          - retune / extend freely
  * every hyperparameter                                     - retune freely
  * `farm_state`      - the derived-state/feature builder     - add or drop features
  * `evolve_market_orders` / `evolve_farmer_action` / `evolve_hand_actions`
                      - the three action channels             - rewrite freely
  * `policy_sanitize` - the policy's own validity pass        - rewrite freely
  * `decide`          - the top-level arbitration: whether to call the backbone at
                        all, how to blend it with the evolved channels, ordering,
                        early exits, per-seat or per-phase strategy switching
You may add, delete, rename, merge or split ANY function inside the block, introduce
new state, new helpers and new control flow. Nothing inside the block is sacred.

WHAT IS IMMUTABLE (outside the block)
-------------------------------------
Only the plumbing that makes the run survivable and gradeable:
  * the Mohui v66 backbone import and the oracle glue
  * `_hard_sanitize` - the final schema guarantee
  * `agent(obs, configuration)` - the public entry point the Kaggle runner and the
    evaluator call. Its name and signature MUST NOT change.
A malformed mutation degrades to the backbone action instead of crashing the game.

THE OPPONENT ORDER-FLOW ORACLE
------------------------------
`shinka/evolution/checkpoint/` (TinyTimeMixer, context 256, byte-identical copy of the
newest promoted daily fine-tune) is served by `kagg_oracle.py`. From turn 256 (day 10
h16) it forecasts, for each of the 9 products and each of the next 96 turns, how many
units the OPPONENT will put on the market. It reaches the policy as `st["oracle"]`,
which is `None` before turn 256. Keys:
    score_1 / score_4 / score_24 / score_96 : {product: max log1p score in window}
    units_1 / units_4 / units_24 / units_96 : {product: summed predicted units}
    units (96x9 array), log (96x9 array), products (list), step (int)
"""
from __future__ import annotations
import os
import sys
import time
from pathlib import Path


# ------------------------------------------------------------------ IMMUTABLE
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
    for base in (here.parent.parent, here.parent.parent.parent,
                 Path.home() / "kaggriculture" / "shinka"):
        cands.append(base / "champions" / "dependencies" / "mohui_v66")
    for c in cands:
        if (c / "candidate_v66_meta_closed_loop.py").is_file():
            return str(c)
    return str(cands[-1])


MOHUI_DIR = _find_mohui()
if MOHUI_DIR not in sys.path:
    sys.path.insert(0, MOHUI_DIR)

import candidate_v66_meta_closed_loop as _mohui  # noqa: E402


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
    if _TRACKER is None and os.environ.get("KAGG_ORACLE_EAGER", "1") == "0" \
            and ORACLE_STATS["errors"] == 0:
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


def _backbone_action(obs, configuration):
    """The Mohui v66 base action. Always schema-valid, never raises."""
    try:
        base = _mohui.kaggle_agent_v66_meta_closed_loop(obs, configuration)
    except Exception:
        base = None
    if not isinstance(base, dict):
        base = {"farmer": ["PASS"], "hands": [], "market": []}
    return base


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
# Kept False: naive static cash floors disrupt correct early establishment spend.
_ENABLE_CASH_FLOOR = False
_ENABLE_ANIMAL_TOPUP = False
_CASH_FLOOR_EARLY = 700.0
_CASH_FLOOR_UNTIL_DAY = 8
_CASH_FLOOR_LATE = 250.0
_ANIMAL_TARGET = 17
_ANIMAL_TOPUP_MIN_CASH = 1500.0
_ANIMAL_TOPUP_LAST_DAY = 26
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
# PRE-DROP HEADROOM (measured 2026-09-14, seeds 12316/12720): the end-of-day drop pushes the
# carried harvest into the shed and deletes the overflow; on high-output worlds 20+ units of
# parked fertilizer cost 4 wool + 6 wheat + 1 strawberry. From _HEADROOM_FROM_HOUR on, sell the
# cheapest stock until shed + carried fits, keeping _HEADROOM_MARGIN units spare.
_HEADROOM_FROM_HOUR = 20
_HEADROOM_MARGIN = 2
# An order evicted by the value-weighted preemption is re-issued on the following turns
# instead of being dropped (the backbone's daily fertilizer dump was being cancelled for
# good at the hour-0 re-hire step, which is how the fertilizer piled up).
_DEFERRED_TTL = 24
_deferred_sells = {}   # player_idx -> {item: [qty, step_deferred]}

# Plant survival: plants die at consecutive_unwatered >= 2.
# Strictly idle-only override: only workers outputting PASS are reassigned maintenance.
_ENABLE_HAND_RESCUE = True
_WATER_RESCUE_THRESHOLD = 1     # rescue a tile once it has missed this many days
_FEED_RESCUE_THRESHOLD = 1      # animals escape at consecutive_unfed >= 2

# Opening scalp
_OPENING_BUY_WHEAT_QTY = 35
_OPENING_SELL_WHEAT_QTY = 30

# ORACLE LEVERS. st["oracle"] is the opponent order-flow forecast: None until turn 256 (day 10 h16).
_ORACLE_FROM_STEP = 256          # first turn with a forecast (the checkpoint's context length)
_ENABLE_ORACLE_PRIORITY = True   # order shop sales by predicted opponent supply (least first)
_ENABLE_ORACLE_FRONTRUN = True   # sell ahead of a predicted opponent dump
_ENABLE_ORACLE_HOLD = False      # holding cancelled needed sales due to false positives; keep False
_ORACLE_FRONTRUN_SCORE = 0.30    # score_4 at/above which the opponent is "about to dump" (P~0.67)
_ORACLE_FRONTRUN_BATCH = 6       # base units per front-run order
_ORACLE_FRONTRUN_PRICE_RATIO = 0.60   # still needs a sane price: never front-run into a crashed market
_ORACLE_FRONTRUN_MIN_HELD = 1
_ORACLE_HOLD_SCORE = 0.50
_ORACLE_PRIORITY_HORIZON = "units_24"  # which window ranks the products


# ---------------------------------------------------------------- derived state
def farm_state(obs, player_idx, oracle=None):
    """Per-turn derived features. Extend freely; everything here is evolvable.
    `oracle` is the opponent order-flow forecast dict (or None); see the ORACLE
    LEVERS comment for its keys."""
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
                    if (unfed >= _FEED_RESCUE_THRESHOLD or (hour >= 18 and unfed >= 0)) \
                            and not cell.get("fed_today"):
                        hungry.append((r, c, cell))
                else:
                    empties.append((r, c))
            elif kind == "WEED":
                weeds.append((r, c))
    _shed = (obs.get("private") or {}).get("shed", {}) or {}
    _shed_used = sum(v for v in _shed.values() if isinstance(v, (int, float)))
    # Units the farmer and hands carry: they land in the shed at the latest with the
    # end-of-day drop, which discards whatever does not fit (engine: overflow deleted).
    _carried = 0
    for _inv in (obs.get("private") or {}).get("inventories", []) or []:
        if isinstance(_inv, dict):
            _carried += sum(int(v or 0) for v in _inv.values() if isinstance(v, (int, float)))
    return {
        "day": day,
        "hour": int(obs.get("hour", 0) or 0),
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
        "carried": _carried,
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
    Evolvable market policy. Max 10 orders per turn.
    """
    step = int(obs.get("step") if obs.get("step") is not None else (24 * st["day"] + st["hour"]))
    hour = st["hour"]
    money, prices, shed = st["money"], st["prices"], st["shed"]
    mkt = list(base_orders or [])

    def _already_selling(orders, it):
        return sum(
            int(o[2] or 0) for o in orders
            if isinstance(o, (list, tuple)) and len(o) >= 3 and o[0] == "SELL" and o[1] == it
        )

    def _order_val(o):
        if not (isinstance(o, (list, tuple)) and len(o) >= 3 and o[0] == "SELL"):
            return 0.0
        it_name = str(o[1])
        q_val = float(o[2] or 0)
        p_val = float(prices.get(it_name, 0) or 0)
        if p_val <= 0:
            p_val = float(_BASE_PRICE.get(it_name, 50))
        return q_val * p_val

    # Value-weighted order preemption: consolidates existing sells, appends if < 10,
    # or evicts lowest-value SELL order when 10-order limit is reached and candidate is worth more.
    protected_items = set()
    deferred = _deferred_sells.setdefault(player_idx, {})

    def _add_sell(orders, it, q, protect=False):
        if q <= 0:
            return False
        if protect:
            protected_items.add(it)
        for i, o in enumerate(orders):
            if isinstance(o, (list, tuple)) and len(o) >= 3 and o[0] == "SELL" and o[1] == it:
                orders[i] = ["SELL", it, int(o[2] or 0) + int(q)]
                return True
        if len(orders) < 10:
            orders.append(["SELL", it, int(q)])
            return True

        p_new = float(prices.get(it, 0) or 0)
        if p_new <= 0:
            p_new = float(_BASE_PRICE.get(it, 50))
        new_val = float(q) * p_new

        min_val = None
        min_idx = -1
        for i, o in enumerate(orders):
            if isinstance(o, (list, tuple)) and len(o) >= 3 and o[0] == "SELL" and o[1] not in protected_items:
                v = _order_val(o)
                if min_val is None or v < min_val:
                    min_val = v
                    min_idx = i

        if min_idx >= 0 and min_val is not None and new_val > min_val * 1.1:
            ev = orders[min_idx]
            ev_it, ev_q = str(ev[1]), int(ev[2] or 0)
            if ev_q > 0:
                prev = deferred.get(ev_it)
                deferred[ev_it] = [max(ev_q, prev[0] if prev else 0), step]
            orders[min_idx] = ["SELL", it, int(q)]
            return True
        return False

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

    # 1b. Re-issue sells evicted on earlier turns while the stock is still there.
    if step <= 1:
        deferred.clear()
    for d_it in list(deferred.keys()):
        d_q, d_step = deferred[d_it]
        held_d = int(shed.get(d_it, 0) or 0)
        q_d = min(int(d_q), held_d - _already_selling(mkt, d_it))
        if q_d <= 0 or step - d_step > _DEFERRED_TTL:
            del deferred[d_it]
            continue
        if _add_sell(mkt, d_it, q_d):
            deferred.pop(d_it, None)

    # 2. Midnight liquidity & wage floor guard (ensures hired hands are never fired at midnight)
    if (hour >= 18 or (216 <= step <= 245)) and st["day"] > 1:
        n_hands = len(st.get("hands", []))
        wage_reserve = max(_WAGE_RESERVE_FLOOR, n_hands * 120.0 + 100.0)
        urgent_wage_reserve = max(350.0, n_hands * 120.0)
        wheat_held = int(shed.get("WHEAT", 0) or 0)
        filtered = []
        for o in mkt:
            if isinstance(o, (list, tuple)) and len(o) >= 2:
                op = str(o[0])
                if op in ("BUY_PRODUCT", "BUY_SEED", "BUY_ANIMAL", "BUY_LAND"):
                    it_name = str(o[1]) if len(o) >= 3 else "WHEAT"
                    qty = int(o[2] or 1) if len(o) >= 3 else 1
                    cost = float(prices.get(it_name, _BASE_PRICE.get(it_name, 50))) * qty
                    reserve = urgent_wage_reserve if (it_name == "WHEAT" and wheat_held < st["n_animals"]) else wage_reserve
                    if money - cost < reserve:
                        continue
                    money -= cost
            filtered.append(o)
        mkt = filtered

        # Emergency liquidity: if cash is below wage reserve near midnight, sell surplus non-feed stock
        if hour >= 21 and money < wage_reserve:
            deficit = wage_reserve - money
            for em_it in ("STRAWBERRY", "FERTILIZER", "MELON", "CARROT", "TOMATO", "EGG", "MILK", "WOOL"):
                held = int(shed.get(em_it, 0) or 0)
                already = _already_selling(mkt, em_it)
                avail = held - already
                if avail > 0:
                    p = float(prices.get(em_it, 0) or 0)
                    if p >= 5.0:
                        qty = min(avail, max(1, int(deficit / p) + 1))
                        if _add_sell(mkt, em_it, qty, protect=True):
                            money += qty * p
                            deficit -= qty * p
                            if deficit <= 0:
                                break

    # 2b. Cancel non-paying investments in the last 2 days (days 28-29: cannot mature before turn 720)
    if st["day"] >= 28:
        cancel_ops = {"BUY_SEED", "BUY_ANIMAL", "BUY_LAND", "HIRE"}
        wheat_held = int(shed.get("WHEAT", 0) or 0)
        if st["day"] >= 29 or wheat_held >= st["n_animals"]:
            cancel_ops.add("BUY_PRODUCT")
        mkt = [
            o for o in mkt
            if not (isinstance(o, (list, tuple)) and len(o) >= 2 and o[0] in cancel_ops)
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

    # 4a. Town Shop Demand Arbitrage (diversify into high-margin shop crops)
    unlocked_shops = obs.get("town", {}).get("unlocked_shops", []) or []
    if unlocked_shops and 3 <= st["day"] <= 24 and len(mkt) < 10 and money >= 1200.0:
        shop_needs = set()
        for sh in unlocked_shops:
            shop_needs.update(_SHOP_DEMANDS.get(sh, ()))
        seeds_held = obs.get("private", {}).get("seeds", {}) or {}
        if "CARROT" in shop_needs and int(seeds_held.get("CARROT", 0) or 0) < 2 and int(shed.get("CARROT", 0) or 0) < 6:
            if not any(isinstance(o, (list, tuple)) and len(o) >= 3 and o[0] == "BUY_SEED" and o[1] == "CARROT" for o in mkt):
                mkt.append(["BUY_SEED", "CARROT", 2])
        if len(mkt) < 10 and "TOMATO" in shop_needs and int(seeds_held.get("TOMATO", 0) or 0) < 2 and int(shed.get("TOMATO", 0) or 0) < 6:
            if not any(isinstance(o, (list, tuple)) and len(o) >= 3 and o[0] == "BUY_SEED" and o[1] == "TOMATO" for o in mkt):
                mkt.append(["BUY_SEED", "TOMATO", 2])

    # Dynamic feed reserve: steps down in endgame when no future days need feeding.
    # Day 28 keeps one pickup batch (6) of slack on top of one unit per animal: the hands
    # collect feed 6 at a time, so a reserve of exactly n_animals leaves the last feeding
    # hand empty-handed (measured: 3 animals unfed on day 28 -> 4 milk + 3 wool short on day 29).
    feed_reserve = 0 if step >= 696 else (max(2, st["n_animals"] + 6) if step >= 672 else max(4, st["n_animals"] * 2))

    # 4b. TIERED SHED PRESSURE VALVE: Prevents silent 100/100 harvest deletion while protecting animal feed
    shed_used = st["shed_used"]
    if _ENABLE_SHED_PRESSURE and shed_used >= _SHED_PRESSURE_AT:
        p_ratio_floor = 0.05 if shed_used >= 95 else (0.15 if shed_used >= 90 else _SHED_PRESSURE_PRICE_RATIO)
        batch_limit = 10 if shed_used >= 90 else _SHED_PRESSURE_BATCH

        holdings = sorted(
            ((int(q or 0), it) for it, q in shed.items() if int(q or 0) > 0),
            reverse=True)
        for qty_held, item in holdings:
            price = float(prices.get(item, 0) or 0)
            base = _BASE_PRICE.get(item, 100)
            # Floor-price dampener: avoid giving product away for <= $3 unless under extreme shed overflow risk
            if price <= 3.0 and shed_used < 95:
                continue
            if price < p_ratio_floor * base and price < 1.0:
                continue
            already = _already_selling(mkt, item)
            reserve = feed_reserve if item == "WHEAT" else 0
            avail = qty_held - already - reserve
            qty = min(avail, batch_limit)
            if qty > 0:
                _add_sell(mkt, item, qty)

    # 4b'. PRE-DROP HEADROOM: what the hands carry lands in the shed at the end-of-day drop and
    #      the overflow is deleted. From hour 20, sell the cheapest stock (feed reserve kept)
    #      until shed + carried fits; these orders are protected from value-weighted eviction.
    if hour >= _HEADROOM_FROM_HOUR and step < 712:
        need = shed_used + int(st.get("carried", 0) or 0) - (_SHED_CAPACITY - _HEADROOM_MARGIN)
        if need > 0:
            cheapest = sorted(
                ((float(prices.get(it, 0) or 0), int(q or 0), it) for it, q in shed.items() if int(q or 0) > 0),
                key=lambda x: (x[0], -x[1]))
            for price_h, qty_held, item in cheapest:
                if need <= 0:
                    break
                if price_h < 1.0:
                    continue
                reserve = feed_reserve if item == "WHEAT" else 0
                avail = qty_held - _already_selling(mkt, item) - reserve
                qty = min(avail, need)
                if qty > 0 and _add_sell(mkt, item, qty, protect=True):
                    need -= qty

    # 4c. ORACLE FRONT-RUN: Sell ahead of predicted opponent dump before price crashes.
    #     MELON AUC is excluded from aggressive dumping due to low base rate.
    #     WHEAT feed reserve is protected so front-running never starves animals.
    #     1 unit of FERTILIZER is reserved to protect ongoing crop fertilization.
    sell_ahead = _oracle_scores(st, "score_4")
    if _ENABLE_ORACLE_FRONTRUN and sell_ahead:
        cands = [
            (it, float(sc or 0.0)) for it, sc in sell_ahead.items()
            if it != "MELON" and float(sc or 0.0) >= _ORACLE_FRONTRUN_SCORE
        ]
        cands.sort(key=lambda kv: -kv[1])

        demanded_shops = set()
        for shop in st.get("shops", []):
            demanded_shops.update(_SHOP_DEMANDS.get(shop, ()))

        for item, score in cands:
            held = int(shed.get(item, 0) or 0)
            price = float(prices.get(item, 0) or 0)
            base = _BASE_PRICE.get(item, 100)
            reserve = feed_reserve if item == "WHEAT" else (1 if item == "FERTILIZER" else 0)
            # Keep an unbuffered WOOL price guard, but allow earlier exits before forecast supply.
            min_ratio = 0.70 if (item == "WOOL" and item not in demanded_shops) else _ORACLE_FRONTRUN_PRICE_RATIO
            if (held - reserve) < _ORACLE_FRONTRUN_MIN_HELD or price < min_ratio * base:
                continue
            already = _already_selling(mkt, item)
            avail_fr = held - reserve - already
            opp_supp_4c = _oracle_scores(st, _ORACLE_PRIORITY_HORIZON)
            opp_sc24_4c = _oracle_scores(st, "score_24")
            supp_4c = float(opp_supp_4c.get(item, 0.0) or 0.0) if opp_supp_4c else 0.0
            sc24_4c = float(opp_sc24_4c.get(item, 0.0) or 0.0) if opp_sc24_4c else 0.0
            if sc24_4c >= 0.45 or supp_4c >= 2.0 or score >= 0.60:
                batch = avail_fr
            elif score >= 0.45:
                batch = 8
            else:
                batch = _ORACLE_FRONTRUN_BATCH
            qty = min(avail_fr, batch)
            if qty > 0:
                _add_sell(mkt, item, qty)

    # 4d. Pre-step 540 Livestock Glut Protection:
    # WOOL and MILK crash from $200+ to $1.00 around steps 480-540 in live ladder play.
    # Orderly phased sales from step 480 before the price collapses:
    if 480 <= step < 680:
        for it in ("WOOL", "MILK"):
            held = int(shed.get(it, 0) or 0)
            already = _already_selling(mkt, it)
            avail = held - already
            if avail >= 2:
                p = float(prices.get(it, 0) or 0)
                if p >= 15.0:
                    _add_sell(mkt, it, min(avail, 6))

    # 5. Town-shop demand harvesting.
    #    Runs through step 711 to capture full endgame revenue from town shops.
    #    Opponent supply ranks goods to avoid mirror dumps.
    #    ORACLE CADENCE BYPASS: when oracle predicts imminent opponent dump for a demanded
    #    item (score_4 >= 0.40), bypass the 4-turn cadence to front-run before price crash.
    oracle_cadence_bypass = False
    if _ENABLE_ORACLE_PRIORITY and step >= _ORACLE_FROM_STEP:
        _score_4_bypass = _oracle_scores(st, "score_4")
        if _score_4_bypass:
            # Dynamic bypass threshold: aggressive when shed is filling, selective when low
            if shed_used >= 70:
                _bypass_thresh = 0.25
            elif shed_used < 40:
                _bypass_thresh = 0.50
            else:
                _bypass_thresh = 0.40
            _demanded_bypass = set()
            for shop in st["shops"]:
                _demanded_bypass.update(_SHOP_DEMANDS.get(shop, ()))
            for _it_b in _demanded_bypass:
                if _it_b != "MELON" and float(_score_4_bypass.get(_it_b, 0.0)) >= _bypass_thresh:
                    oracle_cadence_bypass = True
                    break

    # Adaptive cadence: faster when shed filling, endgame, or oracle bypass
    cadence = oracle_cadence_bypass or (step % 4 == 0) or (shed_used >= 60 and step % 2 == 0) or (step >= 672 and step % 2 == 0)
    if 144 <= step < 712 and cadence:
        demanded = set()
        for shop in st["shops"]:
            demanded.update(_SHOP_DEMANDS.get(shop, ()))
        priority = list(_SELL_PRIORITY)
        opp_supply = _oracle_scores(st, _ORACLE_PRIORITY_HORIZON)
        opp_score_24 = _oracle_scores(st, "score_24")
        if _ENABLE_ORACLE_PRIORITY and opp_supply:
            priority.sort(key=lambda it: (0.0 if it == "MELON" else float(opp_supply.get(it, 0.0)), _SELL_PRIORITY.index(it)))

        for item in priority:
            is_demanded = item in demanded
            p = float(prices.get(item, 0) or 0)
            base = _BASE_PRICE.get(item, 100)
            if is_demanded:
                p_thresh = 0.70 * base if oracle_cadence_bypass else _PRICE_THRESHOLD_RATIO * base
            else:
                p_thresh = 0.80 * base if item in ("MELON", "TOMATO") else _PRICE_THRESHOLD_RATIO * base

            # Oracle premium shield: relax price protection as unsold stock builds.
            # Count only stock not already committed to another sell order.
            # The ordinary product floor bounds the concession; batches stay small.
            if not is_demanded and item in ("MELON", "WOOL") and (_ORACLE_FROM_STEP <= step < 640):
                supp_val = float(opp_supply.get(item, 0.0) or 0.0)
                sc24_val = float(opp_score_24.get(item, 0.0) or 0.0)
                if supp_val >= 1.5 or sc24_val >= 0.35:
                    premium_held = max(0, int(shed.get(item, 0) or 0))
                    premium_committed = _already_selling(mkt, item)
                    premium_backlog = max(0, premium_held - premium_committed)
                    shield_ratio = max(0.80, 0.95 - 0.015 * premium_backlog)
                    p_thresh = max(p_thresh, shield_ratio * base)

            if is_demanded or (p >= p_thresh):
                held = int(shed.get(item, 0) or 0)
                reserve = feed_reserve if item == "WHEAT" else 0
                avail = held - reserve
                min_req = 1 if item in ("MELON", "TOMATO", "EGG") else _MIN_HELD_FOR_SHOP_SALE
                if avail < min_req:
                    continue
                if p >= p_thresh:
                    already = _already_selling(mkt, item)

                    # Size additional sales from uncommitted, feed-adjusted stock.
                    # Quiet forecasts are weak evidence, not a guarantee of monopoly:
                    # expand batches only at intact quotes, with bounded own-price impact.
                    remaining = max(0, avail - already)
                    batch_max = _SHOP_SELL_BATCH_MAX
                    if _ENABLE_ORACLE_PRIORITY and step >= _ORACLE_FROM_STEP and item != "MELON":
                        sc24 = float(opp_score_24.get(item, 0.0) or 0.0)
                        supp = float(opp_supply.get(item, 0.0) or 0.0)
                        quiet_known = item in opp_score_24 and item in opp_supply
                        if is_demanded:
                            if sc24 >= 0.45 or supp >= 2.0:
                                batch_max = remaining
                            elif sc24 >= 0.30 or supp >= 1.2:
                                batch_max = 8
                            elif sc24 >= 0.15 or supp >= 0.6:
                                batch_max = 6
                            else:
                                batch_max = 3
                                if quiet_known and p >= base:
                                    batch_max = min(6, max(3, remaining // 2))
                        else:
                            if sc24 >= 0.45 or supp >= 2.0:
                                batch_max = remaining
                            else:
                                batch_max = 2
                                if quiet_known and sc24 < 0.10 and supp < 0.6:
                                    batch_max = 3
                                    if p >= base:
                                        batch_max = min(4, max(3, remaining // 2))
                    else:
                        batch_max = 2 if item == "MELON" else (3 if not is_demanded else _SHOP_SELL_BATCH_MAX)

                    qty = min(remaining, batch_max)
                    if qty > 0:
                        _add_sell(mkt, item, qty)

        # Fertilizer has no town-shop demand: monetize surplus before it accumulates,
        # retaining one unit rather than waiting indefinitely for a premium quote.
        fert_held = int(shed.get("FERTILIZER", 0) or 0)
        if fert_held >= 3:
            p_fert = float(prices.get("FERTILIZER", 0) or 0)
            if p_fert >= 0.45 * 100:
                already_fert = _already_selling(mkt, "FERTILIZER")
                qty_fert = min(fert_held - 1 - already_fert, 5)
                if qty_fert > 0:
                    _add_sell(mkt, "FERTILIZER", qty_fert)

    # 5b. PREDICTIVE EARLY-ENDGAME LIQUIDATION:
    #     Preemptively liquidates goods the opponent is forecast to dump (score_24, P=0.93)
    #     and high-value items before mirror collapse.
    early_liq_active = (680 <= step < 712)
    opp_dumps_24 = _oracle_scores(st, "score_24") if step >= _ORACLE_FROM_STEP else {}
    if not early_liq_active and (640 <= step < 680) and opp_dumps_24:
        if any(float(opp_dumps_24.get(it, 0.0) or 0.0) >= 0.30 for it in ("WOOL", "MILK", "TOMATO", "STRAWBERRY", "EGG")):
            early_liq_active = True

    if early_liq_active:
        # Oracle 24-turn dump front-running (P = 0.93 for score_24 >= 0.20)
        if opp_dumps_24:
            cands_24 = [
                (it, float(sc or 0.0)) for it, sc in opp_dumps_24.items()
                if it != "MELON" and float(sc or 0.0) >= (0.20 if step >= 680 else 0.30)
            ]
            cands_24.sort(key=lambda kv: -kv[1])
            for item, score in cands_24:
                held = int(shed.get(item, 0) or 0)
                reserve = feed_reserve if item == "WHEAT" else (1 if item == "FERTILIZER" else 0)
                already = _already_selling(mkt, item)
                avail = held - reserve - already
                if avail <= 0:
                    continue
                p = float(prices.get(item, 0) or 0)
                base = _BASE_PRICE.get(item, 100)
                p_floor_ratio = 0.52 if step >= 680 else 0.65
                if p >= p_floor_ratio * base:
                    batch = avail if score >= 0.45 else (8 if score >= 0.35 else 6)
                    qty = min(avail, batch)
                    if qty > 0:
                        _add_sell(mkt, item, qty)

        # High-value asset pre-liquidation (step >= 680: true endgame window only)
        if step >= 680:
            high_val_targets = [
                ("MELON", 0.65, 5, 1),
                ("TOMATO", 0.75, 4, 1),
                ("FERTILIZER", 0.50, 6, 2),
                ("WOOL", 0.60, 4, 1),
            ]
            for item, min_p_ratio, max_b, min_h in high_val_targets:
                held = int(shed.get(item, 0) or 0)
                already = _already_selling(mkt, item)
                avail = held - already
                if avail < min_h:
                    continue
                p = float(prices.get(item, 0) or 0)
                base = _BASE_PRICE.get(item, 100)
                if p >= min_p_ratio * base:
                    qty = min(avail, max_b)
                    if qty > 0:
                        _add_sell(mkt, item, qty)

        # Phased day-29 orderly liquidation (step >= 693, earlier start with batch 7)
        if step >= 693:
            cands_day29 = sorted(
                ((int(q or 0), it) for it, q in shed.items() if int(q or 0) > 0),
                key=lambda x: -(x[0] * float(prices.get(x[1], 0) or 0))
            )
            for qty_held, it in cands_day29:
                p = float(prices.get(it, 0) or 0)
                base = _BASE_PRICE.get(it, 100)
                if p < 0.35 * base and p < 1.0:
                    continue
                already = _already_selling(mkt, it)
                avail = qty_held - already
                qty = min(avail, 7)
                if qty > 0:
                    _add_sell(mkt, it, qty)

    # 6. Endgame liquidation (final 8 steps of day 29): convert all remaining shed inventory to cash.
    if step >= 712:
        liquidate_cands = sorted(
            ((int(q or 0), it) for it, q in shed.items() if int(q or 0) > 0),
            key=lambda x: -(x[0] * float(prices.get(x[1], 0) or 0))
        )

        # Final 2 turns (718, 719): 100% full liquidation of every item in shed (1 order per item, max 9 items <= 10)
        if step >= 718:
            mkt_liq = []
            for qty_held, it in liquidate_cands:
                if len(mkt_liq) >= 10:
                    break
                p = float(prices.get(it, 0) or 0)
                if p >= 1.0 and qty_held > 0:
                    mkt_liq.append(["SELL", it, qty_held])
            if mkt_liq:
                return mkt_liq[:10]
        else:
            # Steps 712..717: balanced multi-product liquidation
            mkt_liq = []
            already_sold = {}
            for qty_held, it in liquidate_cands:
                if len(mkt_liq) >= 10:
                    break
                p = float(prices.get(it, 0) or 0)
                if p < 1.0:
                    continue
                take = min(qty_held, 10)
                if take > 0:
                    mkt_liq.append(["SELL", it, take])
                    already_sold[it] = take
            if len(mkt_liq) < 10:
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
    """Evolvable single-farmer action. Return [OP, ...args].
    When farmer is idle (PASS), safely execute on-tile maintenance:
    water thirsty plants, feed hungry animals, dig weeds.
    """
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
                            wheat_held = int(st.get("shed", {}).get("WHEAT", 0) or 0)
                            if wheat_held > 0:
                                st["_farmer_rescue"] = (r, c, "FEED")
                                return ["FEED"]
        except Exception:
            pass
    return base


# ---------------------------------------------------------------- hands channel
def evolve_hand_actions(obs, player_idx, base_hands, st):
    """
    Evolvable per-hand actions, one entry per hired hand.
    Strictly idle-only: only workers outputting PASS are reassigned on-tile maintenance.
    """
    hands = st["hands"]
    out = [list(a) if isinstance(a, (list, tuple)) else ["PASS"] for a in (base_hands or [])]
    while len(out) < len(hands):
        out.append(["PASS"])

    if _ENABLE_HAND_RESCUE:
        tiles = st.get("tiles") or []

        def _get_price(cell):
            if not isinstance(cell, dict): return 0
            if "crop" in cell: return _BASE_PRICE.get(cell["crop"], 0)
            if "animal" in cell:
                a = cell["animal"]
                if a == "SHEEP": return _BASE_PRICE.get("WOOL", 200)
                if a == "COW": return _BASE_PRICE.get("MILK", 160)
                if a == "GOOSE": return _BASE_PRICE.get("EGG", 50)
            return 0

        thirsty = sorted(st["thirsty"], key=lambda x: (-int(x[2].get("consecutive_unwatered", 0) or 0), -_get_price(x[2])))
        hungry = sorted(st["hungry"], key=lambda x: (-int(x[2].get("consecutive_unfed", 0) or 0), -_get_price(x[2])))
        weeds = st.get("weeds", [])

        farmer_claimed = set()
        f_res = st.get("_farmer_rescue")
        if f_res:
            farmer_claimed.add((f_res[0], f_res[1]))

        wheat_avail = int(st.get("shed", {}).get("WHEAT", 0) or 0)
        if f_res and f_res[2] == "FEED":
            wheat_avail -= 1

        idle_hands = {}
        for i, pos in enumerate(hands):
            if i >= len(out) or not isinstance(pos, (list, tuple)) or len(pos) < 2:
                continue
            verb = out[i][0] if out[i] else "PASS"
            if verb == "PASS":
                idle_hands[(int(pos[0]), int(pos[1]))] = i

        for r, c in weeds:
            key = (r, c)
            if key in idle_hands and key not in farmer_claimed:
                out[idle_hands[key]] = ["DIG"]
                farmer_claimed.add(key)

        for r, c, cell in hungry:
            key = (r, c)
            if key in idle_hands and key not in farmer_claimed:
                if wheat_avail > 0:
                    out[idle_hands[key]] = ["FEED"]
                    farmer_claimed.add(key)
                    wheat_avail -= 1

        for r, c, cell in thirsty:
            key = (r, c)
            if key in idle_hands and key not in farmer_claimed:
                out[idle_hands[key]] = ["WATER"]
                farmer_claimed.add(key)
    return out


# Policy-side cap on market orders per turn (the engine's hard limit is 10; the
# immutable hard sanitizer enforces it regardless of what this is set to).
_MAX_MARKET_ORDERS = 10


# ---------------------------------------------------------------- arbitration
# The top-level decision. THIS IS EVOLVABLE: change how (or whether) the backbone
# is consulted, how its channels are blended with the evolved ones, the order the
# channels are computed in, and any phase/seat-specific switching.
def decide(obs, player_idx, base, st):
    """Return the action dict {farmer, hands, market} for this turn.

    `base` is the Mohui backbone's action, `st` the derived state from farm_state.
    Seeded with the champion's behaviour: override all three channels.
    """
    return {
        "farmer": evolve_farmer_action(obs, player_idx, base.get("farmer"), st),
        "hands": evolve_hand_actions(obs, player_idx, base.get("hands"), st),
        "market": evolve_market_orders(obs, player_idx, base.get("market"), st),
    }


# The policy's OWN validity pass, applied before the immutable hard sanitizer.
# Evolvable: tighten, loosen, reorder or drop orders by your own rules.
def policy_sanitize(action, base, st):
    """Policy-level cleanup. Must return a dict; the hard sanitizer runs after."""
    if not isinstance(action, dict):
        return base
    market = action.get("market")
    if isinstance(market, (list, tuple)):
        action["market"] = list(market)[:_MAX_MARKET_ORDERS]
    return action

# =================== EVOLVE-BLOCK-END =====================

# ------------------------------------------------------------------ IMMUTABLE
# Derived from every op observed in decoded ladder replays, NOT from the env spec
# description -- that description omits DROP, and silently rewriting DROP to PASS
# blocks hand carrying capacity and costs ~3% of final score.
_HARD_FARMER_OPS = {
    "NORTH", "SOUTH", "EAST", "WEST", "PASS", "PICKUP", "PLANT", "WATER", "HARVEST",
    "FERTILIZE", "BUILD_COOP", "BUILD_PASTURE", "DIG", "PLACE", "FEED", "DROP",
    "COLLECT_FERTILIZER", "CARE",
}
_HARD_MARKET_OPS = {"BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL", "HIRE", "BUY_LAND"}
_HARD_MAX_MARKET_ORDERS = 10


def _hard_clean_op(entry, allowed):
    if isinstance(entry, str):
        entry = [entry]
    if not isinstance(entry, (list, tuple)) or not entry:
        return None
    op = str(entry[0])
    if op not in allowed:
        return None
    return [op] + [a for a in list(entry)[1:]]


def _hard_sanitize(action, base, n_hands):
    """Guarantee a schema-valid action. Any malformed channel falls back to `base`."""
    if not isinstance(action, dict):
        return base
    out = {}

    farmer = _hard_clean_op(action.get("farmer"), _HARD_FARMER_OPS)
    out["farmer"] = farmer or _hard_clean_op(base.get("farmer"), _HARD_FARMER_OPS) or ["PASS"]

    hands = action.get("hands")
    if not isinstance(hands, (list, tuple)):
        hands = base.get("hands") or []
    cleaned = [(_hard_clean_op(h, _HARD_FARMER_OPS) or ["PASS"]) for h in hands]
    out["hands"] = cleaned[:n_hands] + [["PASS"]] * max(0, n_hands - len(cleaned))

    market = action.get("market")
    if not isinstance(market, (list, tuple)):
        market = base.get("market") or []
    out["market"] = [m for m in (_hard_clean_op(o, _HARD_MARKET_OPS) for o in market)
                     if m][:_HARD_MAX_MARKET_ORDERS]
    return out


def agent(obs, configuration=None):
    """Public entry point. Called by the Kaggle runner and the Shinka evaluator.

    IMMUTABLE: the name and signature must not change. Every failure mode inside
    the evolvable block degrades to the backbone action instead of crashing.
    """
    player_idx = int(obs.get("player", 0) or 0)
    base = _backbone_action(obs, configuration)

    farms = obs.get("farms", []) or []
    farm = farms[player_idx] if player_idx < len(farms) else {}
    n_hands = len(farm.get("hands") or [])

    forecast = _oracle_observe(obs, configuration)
    try:
        st = farm_state(obs, player_idx, forecast)
        evolved = decide(obs, player_idx, base, st)
        evolved = policy_sanitize(evolved, base, st)
    except Exception:
        evolved = base
    final = _hard_sanitize(evolved, base, n_hands)
    _oracle_record(final)  # what we really submitted: the tracker's market accounting needs it
    return final
