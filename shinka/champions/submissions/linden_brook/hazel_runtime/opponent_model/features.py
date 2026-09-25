"""Deployment-legal features for predicting opponent market supply.

Hard rule enforced here: every feature is derivable from what a live agent sees
at `obs` time -- shared public state, plus *our own* private shed. The
opponent's `private` block exists in replays but is a LABEL source only; it must
never reach a feature, or the model trains on information it will not have.
"""

from __future__ import annotations

from collections import Counter, deque

from mechanics import (
    ANIMAL_NAMES,
    ANIMALS,
    CROP_NAMES,
    MARKET_PARAMS,
    PRODUCTS,
    SHOP_NAMES,
    town_draw,
)

# Product each animal yields, so pending animal stock maps onto market products.
ANIMAL_PRODUCT = {name: ANIMALS[name]["product"] for name in ANIMAL_NAMES}
RECENT_WINDOWS = (4, 24)


class OpponentHistory:
    """Rolling record of opponent supply, rebuilt from public accounting only."""

    def __init__(self, windows=RECENT_WINDOWS):
        self.windows = windows
        self.cumulative = Counter()
        self.last_sell_step = {}
        self._recent = deque(maxlen=max(windows))

    def update(self, step, opp_sells):
        self._recent.append((step, Counter(opp_sells)))
        for item, units in opp_sells.items():
            if units > 0:
                self.cumulative[item] += units
                self.last_sell_step[item] = step

    def window_sum(self, item, window):
        return sum(c.get(item, 0) for _, c in list(self._recent)[-window:])

    def turns_since(self, step, item, default=999):
        last = self.last_sell_step.get(item)
        return default if last is None else step - last


def _farm_summary(farm):
    """Public-only summary of one farm's tiles."""
    plants_by_crop = Counter()
    pending_by_product = Counter()
    ripe_plants = 0
    unwatered = 0
    animals_by_type = Counter()
    weeds = 0

    for row in farm.get("tiles") or ():
        for tile in row:
            if not isinstance(tile, dict):
                continue
            kind = tile.get("kind")
            if kind == "PLANT":
                crop = tile.get("crop")
                plants_by_crop[crop] += 1
                units = int(tile.get("yield_units") or 0)
                if units > 0:
                    pending_by_product[crop] += units
                    ripe_plants += 1
                if int(tile.get("consecutive_unwatered") or 0) > 0:
                    unwatered += 1
            elif kind in ("COOP", "PASTURE"):
                animal = tile.get("animal")
                animals_by_type[animal] += 1
                units = int(tile.get("yield_units") or 0)
                if units > 0:
                    pending_by_product[ANIMAL_PRODUCT.get(animal)] += units
            elif kind == "WEED":
                weeds += 1

    return {
        "plants_by_crop": plants_by_crop,
        "animals_by_type": animals_by_type,
        "pending_by_product": pending_by_product,
        "ripe_plants": ripe_plants,
        "unwatered": unwatered,
        "weeds": weeds,
    }


def build_features(obs, me, cfg, history, shed_capacity=100):
    """Feature dict for seat `me` at this observation. Public state + own shed."""
    farms = obs["farms"]
    opp = 1 - me
    my_farm, opp_farm = farms[me], farms[opp]
    market = obs["market"]
    inventory, prices = market["inventory"], market["prices"]
    step = int(obs.get("step") or 0)
    shops = (obs.get("town") or {}).get("unlocked_shops") or []

    f = {
        "step": step,
        "day": int(obs.get("day") or 0),
        "hour": int(obs.get("hour") or 0),
        "my_money": float(my_farm.get("money") or 0.0),
        "opp_money": float(opp_farm.get("money") or 0.0),
        "money_ratio": float(my_farm.get("money") or 0.0) / max(1.0, float(opp_farm.get("money") or 0.0)),
        "opp_hands": len(opp_farm.get("hands") or ()),
        "my_hands": len(my_farm.get("hands") or ()),
        "opp_hires_today": int(opp_farm.get("hires_today") or 0),
        "opp_quadrants": len(opp_farm.get("unlocked_quadrants") or ()),
    }

    # Our own shed is legal to use; the opponent's is not.
    my_shed = (obs.get("private") or {}).get("shed") or {}
    my_shed_total = sum(int(v or 0) for v in my_shed.values())
    f["my_shed_total"] = my_shed_total
    f["my_shed_room"] = max(0, shed_capacity - my_shed_total)

    opp_sum = _farm_summary(opp_farm)
    my_sum = _farm_summary(my_farm)
    f["opp_ripe_plants"] = opp_sum["ripe_plants"]
    f["opp_unwatered"] = opp_sum["unwatered"]
    f["opp_weeds"] = opp_sum["weeds"]
    f["opp_pending_total"] = sum(opp_sum["pending_by_product"].values())
    f["my_pending_total"] = sum(my_sum["pending_by_product"].values())

    for crop in CROP_NAMES:
        f[f"opp_plants_{crop}"] = opp_sum["plants_by_crop"].get(crop, 0)
    for animal in ANIMAL_NAMES:
        f[f"opp_animals_{animal}"] = opp_sum["animals_by_type"].get(animal, 0)
    for shop in SHOP_NAMES:
        f[f"shop_{shop}"] = shops.count(shop)

    draw_now = town_draw(step, shops, cfg["shop_interval"], cfg["center_interval"])
    draw_next = Counter()
    for ahead in range(1, cfg["turns_per_day"] + 1):
        for item, units in town_draw(step + ahead, shops, cfg["shop_interval"], cfg["center_interval"]).items():
            draw_next[item] += units

    for p in PRODUCTS:
        base = MARKET_PARAMS[p]["base"]
        inv = int(inventory.get(p, 0))
        price = int(prices.get(p, base))
        f[f"inv_{p}"] = inv
        f[f"invdev_{p}"] = inv - MARKET_PARAMS[p]["I0"]
        f[f"price_{p}"] = price
        f[f"pxbase_{p}"] = price / base
        f[f"myshed_{p}"] = int(my_shed.get(p, 0) or 0)
        f[f"opppend_{p}"] = opp_sum["pending_by_product"].get(p, 0)
        f[f"towndraw_now_{p}"] = draw_now.get(p, 0)
        f[f"towndraw_24_{p}"] = draw_next.get(p, 0)
        f[f"oppcum_{p}"] = history.cumulative.get(p, 0)
        f[f"oppsince_{p}"] = history.turns_since(step, p)
        for w in RECENT_WINDOWS:
            f[f"opplast{w}_{p}"] = history.window_sum(p, w)

    return f


def feature_names(cfg_turns_per_day=24):
    """Stable ordered feature names, for building dense matrices."""
    probe_history = OpponentHistory()
    obs = _blank_obs()
    cfg = {"shop_interval": 4, "center_interval": 24, "turns_per_day": cfg_turns_per_day}
    return sorted(build_features(obs, 0, cfg, probe_history))


def _blank_obs():
    empty_farm = {"money": 0.0, "tiles": [], "hands": [], "hires_today": 0, "unlocked_quadrants": []}
    return {
        "farms": [dict(empty_farm), dict(empty_farm)],
        "market": {
            "inventory": {p: MARKET_PARAMS[p]["I0"] for p in PRODUCTS},
            "prices": {p: MARKET_PARAMS[p]["base"] for p in PRODUCTS},
        },
        "town": {"unlocked_shops": []},
        "private": {"shed": {}},
        "step": 0,
        "day": 0,
        "hour": 0,
        "player": 0,
    }
