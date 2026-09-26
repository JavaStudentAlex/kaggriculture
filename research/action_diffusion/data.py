"""Leak-free Kaggriculture action-diffusion data (NumPy + standard library only).

Replay row t is (observation[t], action[t+1]); a window ending its context at t
predicts A[t:t+horizon]. No rewards, opponent actions/private, or future states
are features. Fixed schema targets the official 10x10, 24-turn-day engine.

Source contract inspected: kaggriculture engine 1.32.7 _apply_unit_action,
_parse_order, _new_plant, _new_animal and real 2026-08-28/09-18 replays.
999 and 1000 are ordinary oversized numeric requests, NOT engine ALL tokens.
Quantities >100 share bin 101 and decode to 101 (lossy for seeds/large holdings).
Negative requests become 0 (engine nonpositive no-op), numeric strings/int(float)
follow engine parsing. PICKUP/PLACE omitted quantity defaults to 1. No category
is reserved for MASK here: models append one extra category to each field.
A command the engine skips without effect (e.g. a literal "INVALID" order, or
BUY_PRODUCT of a product the market does not sell) is an empty slot at its index,
counted as skipped_*; before, one such command dropped the whole episode.

Records carry each seat's team, the final cash of both seats, the result (win 1, loss 0,
draw 0.5: more final cash wins) and the day (the YYYY-MM-DD in the source's file name, for
recency weighting); for a Kaggle daily zip, add_ratings() also rates every team of the day
and every seat from the zip's manifest.csv game ratings.

Each worker loads one JSON episode at a time (prepare(workers=N) runs N of them).
Outputs are per-seat compressed NPZ; WindowDataset caches a bounded number/size of
uncompressed episodes per process.
"""
from __future__ import annotations

import argparse
from array import array
from bisect import bisect_right
from collections import Counter, OrderedDict
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import time
import zipfile

import numpy as np

try:   # 3-5x faster than json on 30 MB replays; same values whenever both parse
    import orjson
except ImportError:
    orjson = None

HORIZON = 24
CONTEXT = 64
MAX_HANDS = 32
MAX_MARKET_ORDERS = 10
COMMAND_SLOTS = 43
ACTION_DIM = COMMAND_SLOTS * 3
BOARD_SIZE = 10

# Explicit order is part of the serialized, versioned contract. Never derive IDs
# from sets, dict traversal, observed frequency, or a training split.
UNIT_OPS = ("NORTH", "SOUTH", "EAST", "WEST", "PASS", "PICKUP", "DROP",
            "PLANT", "WATER", "HARVEST", "FERTILIZE", "BUILD_COOP",
            "BUILD_PASTURE", "DIG", "PLACE", "FEED", "COLLECT_FERTILIZER", "CARE")
MARKET_OPS = ("BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL", "HIRE", "BUY_LAND")
OPERATIONS = ("NONE",) + UNIT_OPS + MARKET_OPS
CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
ANIMALS = ("GOOSE", "COW", "SHEEP")
PRODUCTS = CROPS + ("EGG", "MILK", "WOOL", "FERTILIZER")
STORAGE_ITEMS = PRODUCTS + ANIMALS
ITEMS = ("NONE",) + STORAGE_ITEMS
SHOPS = ("BAKERY", "PIZZA_SHOP", "BRUNCH_SPOT", "YARN_STORE",
         "ICE_CREAM_SHOP", "PET_CAFE", "SMOOTHIE_SHOP", "FARMERS_MARKET")
QUADRANTS = ("NW", "NE", "SW", "SE")
OP_TO_ID = {s: i for i, s in enumerate(OPERATIONS)}
ITEM_TO_ID = {s: i for i, s in enumerate(ITEMS)}
QUANTITY_BINS = 102
field_sizes = [len(OPERATIONS), len(ITEMS), QUANTITY_BINS] * COMMAND_SLOTS
FIELD_SIZES = field_sizes
ARGUMENTS = {
    "PLANT": CROPS, "PICKUP": STORAGE_ITEMS, "PLACE": STORAGE_ITEMS,
    "BUY_SEED": CROPS, "BUY_PRODUCT": ("WHEAT", "FERTILIZER"),
    "BUY_ANIMAL": ANIMALS, "SELL": PRODUCTS,
}
QUANTITY_OPS = frozenset(("PICKUP", "PLACE", "BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL"))
CODEC_CONFIG = {
    "version": 1, "operations": list(OPERATIONS), "items": list(ITEMS),
    "field_sizes": field_sizes, "command_slots": COMMAND_SLOTS,
    "slots": {"farmer": [0, 1], "hands": [1, 33], "market": [33, 43]},
    "arguments": {k: list(v) for k, v in ARGUMENTS.items()},
    "quantity_operations": sorted(QUANTITY_OPS),
    "quantity": {"bins": 102, "exact": [0, 100], "overflow_bin": 101,
                 "overflow_decodes_to": 101, "negative_decodes_to": 0,
                 "conversion": "engine int(value); omitted PICKUP/PLACE => 1",
                 "sentinels": "999/1000 are numeric requests; ALL is unsupported"},
    "mask": "model adds category field_sizes[i]; not a data category",
    "limits": "Reject >32 hand commands/observed hands; retain first 10 market orders (engine cap), validate even discarded commands; never cap feature magnitudes.",
    "skipped": "a command engine 1.32.7 skips without effect (unknown or misplaced operation, an item the operation cannot take, a market order without an integer quantity, a non-dict action) is the empty slot at its index, counted as skipped_*; only what the engine would execute but the codec cannot hold rejects the episode",
    "none": "omit farmer/market NONE; hand NONE => PASS to preserve live hand indices",
}


class DataError(ValueError):
    """A counted episode-level rejection: the replay holds something the engine would act on
    that the codec cannot represent. Commands the engine skips are not errors (_skipped)."""
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def _fail(code, message):
    raise DataError(code, message)


def _quantity(value, stats):
    try:
        q = int(value)
    except (ValueError, TypeError, OverflowError):
        _fail("unsupported_quantity", f"Cannot parse engine numeric quantity {value!r}")
    stats["quantity_fields"] += 1
    if isinstance(value, str):
        stats["quantity_numeric_strings"] += 1
    if isinstance(value, (float, np.floating)) and value != q:
        stats["quantity_fractional_truncated"] += 1
    if q > 100:
        stats["quantity_overflow_101"] += 1
    if q < 0:
        stats["quantity_negative_to_zero"] += 1
    stats["quantity_max_requested"] = max(stats["quantity_max_requested"], q)
    return min(101, max(0, q))


def _skipped(code, stats):
    """A command the engine skips without any effect, encoded as the empty slot it amounts to
    and counted as skipped_<code>. The slot keeps its index, so later market orders keep their
    place in the engine's order-by-order lockstep."""
    stats["skipped_" + code] += 1
    return (0, 0, 0)


def _encode_command(command, market, stats):
    if command is None or command == []:
        return (0, 0, 0)
    if not isinstance(command, (list, tuple)) or not 1 <= len(command) <= 3:
        _fail("malformed_command", repr(command))
    op = command[0]
    if not isinstance(op, str):
        _fail("unsupported_operation", repr(op))
    # Engine 1.32.7 skips an unknown operation, and one in the wrong list (a unit command among
    # the market orders or the reverse): _apply_unit_action falls through, _parse_order -> None.
    if op not in OP_TO_ID:
        return _skipped("unsupported_operation", stats)
    if op != "NONE" and op not in (MARKET_OPS if market else UNIT_OPS):
        return _skipped("unsupported_operation_slot", stats)
    item_id = 0
    if op in ARGUMENTS:
        # It also skips an item the operation cannot take: PLANT of a non-crop, BUY_PRODUCT of
        # anything but WHEAT/FERTILIZER (the market aborts the order), PICKUP/PLACE of a non-item.
        if len(command) < 2 or not isinstance(command[1], str) or command[1] not in ARGUMENTS[op]:
            return _skipped("unsupported_item", stats)
        item_id = ITEM_TO_ID[command[1]]
    elif len(command) > 1 and (not isinstance(command[1], str) or command[1] not in ITEM_TO_ID):
        stats["ignored_arguments"] += 1   # operations without an item never read one
    quantity = 0
    if op in QUANTITY_OPS:
        if len(command) < 3:
            if market:   # _parse_order skips an order without a quantity
                return _skipped("missing_quantity", stats)
            stats["quantity_default_one"] += 1
        elif market:
            try:
                int(command[2])
            except (ValueError, TypeError):   # _parse_order skips it too
                return _skipped("unsupported_quantity", stats)
        quantity = _quantity(command[2] if len(command) > 2 else 1, stats)
    elif len(command) > 2:
        stats["ignored_quantity_fields"] += 1
    return OP_TO_ID[op], item_id, quantity


def encode_action(action, *, stats=None):
    """Return int64[129], the action as the engine executes it: commands it skips are empty slots
    (counted as skipped_*); anything it would execute but the codec cannot hold raises DataError."""
    stats = Counter() if stats is None else stats
    if not isinstance(action, Mapping):   # the engine plays a non-dict action as {}
        stats["skipped_malformed_action"] += 1
        action = {}
    hands, market = action.get("hands", []), action.get("market", [])
    if not isinstance(hands, (list, tuple)):   # engine: no hand commands
        stats["skipped_malformed_action"] += 1
        hands = []
    if not isinstance(market, (list, tuple)):   # engine: no market orders
        stats["skipped_malformed_action"] += 1
        market = []
    if len(hands) > MAX_HANDS:
        _fail("too_many_hands", str(len(hands)))
    out = np.zeros((COMMAND_SLOTS, 3), dtype=np.int64)
    out[0] = _encode_command(action.get("farmer"), False, stats)
    for i, command in enumerate(hands):
        out[i + 1] = _encode_command(command, False, stats)
    for i, command in enumerate(market):
        encoded = _encode_command(command, True, stats)
        if i < MAX_MARKET_ORDERS:
            out[33 + i] = encoded
    stats["market_orders_truncated"] += max(0, len(market) - MAX_MARKET_ORDERS)
    stats["actions"] += 1
    return out.reshape(ACTION_DIM)


def decode_action(array, hand_count):
    """Decode requests, not a legality solver; filter incompatible sampled fields.

    NONE hand slots become PASS, never compacted (which would retarget commands).
    Refuse MASK/out-of-range categories. Overflow quantity is the representative
    101, not an invented ALL operation or a claim to recover original amounts.
    """
    if not isinstance(hand_count, (int, np.integer)) or not 0 <= hand_count <= MAX_HANDS:
        raise ValueError("hand_count must be an integer in [0,32]")
    a = np.asarray(array)
    if a.shape != (ACTION_DIM,) or not np.issubdtype(a.dtype, np.integer):
        raise ValueError("action must be integer array[129]")
    if np.any(a < 0) or np.any(a >= np.asarray(field_sizes)):
        raise ValueError("action contains out-of-range or MASK categories")
    a = a.reshape(COMMAND_SLOTS, 3)

    def command(row, market=False):
        op, item, q = OPERATIONS[int(row[0])], ITEMS[int(row[1])], int(row[2])
        if op == "NONE" or op not in (MARKET_OPS if market else UNIT_OPS):
            return None
        result = [op]
        if op in ARGUMENTS:
            if item not in ARGUMENTS[op]:
                return None
            result.append(item)
        if op in QUANTITY_OPS:
            result.append(q)
        return result

    result = {"hands": [command(a[1 + i]) or ["PASS"] for i in range(hand_count)],
              "market": [c for row in a[33:] if (c := command(row, True)) is not None]}
    farmer = command(a[0])
    if farmer is not None:
        result["farmer"] = farmer
    return result


def active_field_mask(array):
    """Optional bool[...,129] mask; all operation fields remain supervised."""
    a = np.asarray(array).reshape(*np.asarray(array).shape[:-1], COMMAND_SLOTS, 3)
    m = np.zeros(a.shape, dtype=bool)
    m[..., 0] = True
    m[..., 1] = np.isin(a[..., 0], [OP_TO_ID[o] for o in ARGUMENTS])
    m[..., 2] = np.isin(a[..., 0], [OP_TO_ID[o] for o in QUANTITY_OPS])
    return m.reshape(np.asarray(array).shape)


TILE_KINDS = ("EMPTY", "LOCKED", "WEED", "PLANT", "COOP", "PASTURE")
TILE_SCALARS = (
    ("watered_today", "identity"), ("consecutive_unwatered", "log"),
    ("fertilized_until_day", "day"), ("max_lifespan_step", "step"),
    ("planted_day", "day"), ("yield_units", "log"),
    ("cared_today", "identity"), ("consecutive_unfed", "log"),
    ("fed_today", "identity"), ("fertilizer_available", "identity"),
    ("pending_care_bonus", "log"), ("placed_day", "day"),
)
TILE_FEATURE_NAMES = (tuple("kind." + k for k in TILE_KINDS)
                      + tuple("crop." + k for k in CROPS)
                      + tuple("animal." + k for k in ANIMALS)
                      + tuple(k for k, _ in TILE_SCALARS))
TILE_DIM = len(TILE_FEATURE_NAMES)
SCALES = {"identity": 1.0, "day": 30.0, "step": 720.0, "hour": 24.0,
          "position": 9.0, "log": math.log1p(100.0)}


def _feature_layout():
    fields = []
    def add(name, transform="log"):
        fields.append({"name": name, "transform": transform, "divisor": SCALES[transform]})
    for k, transform in (("seat", "identity"), ("step", "step"), ("day", "day"), ("hour", "hour")):
        add(k, transform)
    for shop in SHOPS:
        add("town.count." + shop)
    for item in PRODUCTS:
        add("market.price." + item)
        add("market.inventory." + item)
    for side in ("own", "opponent"):
        for k in ("money", "hires_today", "hand_count"):
            add(side + "." + k)
        for quadrant in QUADRANTS:
            add(side + ".unlocked." + quadrant, "identity")
        for unit in range(MAX_HANDS + 1):
            for k, transform in (("exists", "identity"), ("x", "position"), ("y", "position")):
                add(f"{side}.unit.{unit}.{k}", transform)
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                for k in TILE_FEATURE_NAMES:
                    add(f"{side}.tile.{y}.{x}.{k}", dict(TILE_SCALARS).get(k, "identity"))
    for unit in range(MAX_HANDS + 1):
        for item in STORAGE_ITEMS:
            add(f"own.inventory.{unit}.{item}")
    for item in CROPS:
        add("own.seeds." + item)
    for item in STORAGE_ITEMS:
        add("own.shed." + item)
    return fields


FEATURE_FIELDS = _feature_layout()
FEATURE_NAMES = tuple(f["name"] for f in FEATURE_FIELDS)
FEATURE_DIM = len(FEATURE_NAMES)
FEATURE_SPEC = {
    "version": 1, "feature_dim": FEATURE_DIM, "fields": FEATURE_FIELDS,
    "magnitude_transform": "sign(x)*log1p(abs(x))/log(101); no data-fitted scaler",
    "missing": "missing numeric scalar => 0; absent tile categorical => all-zero except EMPTY kind",
    "privacy": "public farms/market/town/day/hour and own private only; explicit whitelist",
    "alignment": "X[t]=obs[t], A[t]=steps[t+1][seat].action; loop supplies step",
    "board_size": BOARD_SIZE, "max_hands": MAX_HANDS,
    "farm_order": ["own", "opponent"], "tile_order": "row-major y,x",
}
FEATURES_SPEC = FEATURE_SPEC
_DIVISORS = np.asarray([f["divisor"] for f in FEATURE_FIELDS], dtype=np.float32)
_LOG_FIELDS = np.asarray([f["transform"] == "log" for f in FEATURE_FIELDS])
_PUBLIC_KEYS = ("farms", "market", "town", "day", "hour")


def merge_observation(public_obs, seat_obs, seat, step):
    """Reconstruct an agent view, never copying seat0 private into seat1.

    Public values in seat0 are authoritative; a seat-local public value is only
    a fallback when seat0 omits that key. Unknown top-level data is discarded.
    """
    obs = {k: public_obs[k] if k in public_obs else seat_obs.get(k) for k in _PUBLIC_KEYS}
    if "private" not in seat_obs or not isinstance(seat_obs["private"], Mapping):
        _fail("missing_private", f"seat={seat}, step={step}")
    obs.update(private=seat_obs["private"], player=seat, step=step)
    return obs


_F32_MAX = float(np.finfo(np.float32).max)


def _number(value):
    try:
        n = float(value)
    except (ValueError, TypeError, OverflowError):
        _fail("invalid_observation_number", repr(value))
    # n != n is NaN; +-inf exceed the float32 range: same test as isfinite and abs <= max
    if n != n or n > _F32_MAX or n < -_F32_MAX:
        _fail("invalid_observation_number", repr(value))
    return n


# Precomputed tile blocks: the tile loop is ~90% of the encoding time (200 tiles/observation).
_TILE_ONEHOT = {kind: tuple(float(kind == k) for k in TILE_KINDS) for kind in TILE_KINDS}
_CROP_ONEHOT = {None: (0.0,) * len(CROPS), **{c: tuple(float(c == k) for k in CROPS) for c in CROPS}}
_ANIMAL_ONEHOT = {None: (0.0,) * len(ANIMALS), **{a: tuple(float(a == k) for k in ANIMALS) for a in ANIMALS}}
_SCALAR_KEYS = tuple(k for k, _ in TILE_SCALARS)
_STORAGE_SET = frozenset(STORAGE_ITEMS)
_NO_ITEMS = (0.0,) * len(STORAGE_ITEMS)
_PLAIN_TILE = {kind: _TILE_ONEHOT[kind] + (0.0,) * (len(CROPS) + len(ANIMALS) + len(TILE_SCALARS))
               for kind in TILE_KINDS}


def _numbers(mapping, keys):
    """[_number(mapping.get(k, 0)) for k in keys], checked in bulk: a NaN makes the sum NaN, and
    without one, min/max see every infinity or float32 overflow."""
    get = mapping.get
    try:
        numbers = [float(get(k, 0)) for k in keys]
    except (ValueError, TypeError, OverflowError):
        numbers = None
    if numbers is not None:
        total = sum(numbers)
        if total == total and max(numbers) <= _F32_MAX and min(numbers) >= -_F32_MAX:
            return numbers
    return [_number(get(k, 0)) for k in keys]   # raises on the first bad value, as before


def _encode_tile(tile, values):
    """Append one tile's TILE_DIM values; the checks and errors of the per-value loop it replaced."""
    if tile is None:
        values.extend(_PLAIN_TILE["EMPTY"])
        return
    if isinstance(tile, str):
        block = _PLAIN_TILE.get(tile)
        if block is None:
            _fail("unsupported_tile", str(tile))
        values.extend(block)
        return
    if type(tile) is not dict and not isinstance(tile, Mapping):   # dict first: the ABC check is slow
        _fail("unsupported_tile", repr(tile))
    kind = str(tile.get("kind"))
    if kind not in _TILE_ONEHOT:
        _fail("unsupported_tile", kind)
    crop, animal = tile.get("crop"), tile.get("animal")
    if crop is not None and crop not in CROPS:
        _fail("unsupported_item", str(crop))
    if animal is not None and animal not in ANIMALS:
        _fail("unsupported_item", str(animal))
    values.extend(_TILE_ONEHOT[kind])
    values.extend(_CROP_ONEHOT[crop])
    values.extend(_ANIMAL_ONEHOT[animal])
    values.extend(_numbers(tile, _SCALAR_KEYS))


def encode_observation(obs, seat):
    """Fixed float32[FEATURE_DIM] agent-visible observation; no engine import.

    `obs` must be the requested agent view (use merge_observation for replays).
    Other private/oracle/debug keys, even inside farm/tile mappings, are ignored.
    Unknown public tile types fail closed instead of silently erasing state.
    """
    if seat not in (0, 1) or obs.get("player", seat) != seat:
        _fail("seat_mismatch", str(seat))
    if not isinstance(obs.get("private"), Mapping):
        _fail("missing_private", str(seat))
    farms = obs.get("farms")
    if not isinstance(farms, (list, tuple)) or len(farms) != 2:
        _fail("missing_public", "Expected both public farms")
    if "step" not in obs:
        _fail("missing_step", "Replay caller must supply loop index")
    values = [seat, _number(obs["step"]), _number(obs.get("day", 0)), _number(obs.get("hour", 0))]
    shops = Counter((obs.get("town") or {}).get("unlocked_shops", []))
    if set(shops) - set(SHOPS):
        _fail("unsupported_shop", str(set(shops) - set(SHOPS)))
    values.extend(shops[k] for k in SHOPS)
    market = obs.get("market")
    if not isinstance(market, Mapping):
        _fail("missing_public", "market")
    for item in PRODUCTS:
        for key in ("prices", "inventory"):
            values.append(_number((market.get(key) or {}).get(item, 0)))
    for s in (seat, 1 - seat):
        farm = farms[s]
        hands = farm.get("hands", [])
        if len(hands) > MAX_HANDS:
            _fail("too_many_hands", f"observed {len(hands)}")
        values.extend((_number(farm.get("money", 0)), _number(farm.get("hires_today", 0)), len(hands)))
        quadrants = farm.get("unlocked_quadrants", [])
        values.extend(float(q in quadrants) for q in QUADRANTS)
        positions = [farm.get("farmer")] + list(hands)
        for i in range(MAX_HANDS + 1):
            if i < len(positions) and positions[i] is not None:
                pos = positions[i]
                if len(pos) != 2:
                    _fail("invalid_position", repr(pos))
                values.extend((1, _number(pos[0]), _number(pos[1])))
            else:
                values.extend((0, 0, 0))
        tiles = farm.get("tiles", [])
        if len(tiles) != BOARD_SIZE or any(len(row) != BOARD_SIZE for row in tiles):
            _fail("unsupported_board_size", "Expected 10x10")
        for row in tiles:
            for tile in row:
                _encode_tile(tile, values)
    private = obs["private"]
    inventories = private.get("inventories", [])
    if len(inventories) > MAX_HANDS + 1:
        _fail("too_many_hands", f"{len(inventories)} inventories")
    stores = list(inventories) + [private.get("seeds", {}), private.get("shed", {})]
    for store in stores:
        if not isinstance(store, Mapping) or set(store) - _STORAGE_SET:
            _fail("unsupported_item", f"private store keys: {store!r}")
    for i in range(MAX_HANDS + 1):
        inv = inventories[i] if i < len(inventories) else {}
        values.extend(_numbers(inv, STORAGE_ITEMS) if inv else _NO_ITEMS)
    values.extend(_numbers(private.get("seeds", {}), CROPS))
    values.extend(_numbers(private.get("shed", {}), STORAGE_ITEMS))
    # via float64: the same float32 rounding as np.asarray(values, float32), a third of the time
    out = np.frombuffer(array("d", values), dtype=np.float64).astype(np.float32)
    assert out.shape == (FEATURE_DIM,)
    v = out[_LOG_FIELDS]
    out[_LOG_FIELDS] = np.sign(v) * np.log1p(np.abs(v))
    out /= _DIVISORS
    return out


def episode_arrays(doc, *, stats=None):
    """Build both seats transactionally; reject the whole episode on any error."""
    cfg = doc.get("configuration") or {}
    for key, expected in (("boardSize", 10), ("turnsPerDay", 24), ("maxMarketOrdersPerTurn", 10)):
        if cfg.get(key, expected) != expected:
            _fail("unsupported_configuration", f"{key}={cfg[key]}, expected {expected}")
    steps = doc.get("steps")
    if not isinstance(steps, list) or len(steps) < 2:
        _fail("not_episode", "Need at least two replay steps")
    stats = Counter() if stats is None else stats
    result = []
    for seat in (0, 1):
        X = np.empty((len(steps) - 1, FEATURE_DIM), dtype=np.float32)
        A = np.empty((len(steps) - 1, ACTION_DIM), dtype=np.int64)
        for t in range(len(steps) - 1):
            cur, nxt = steps[t], steps[t + 1]
            if len(cur) != 2 or len(nxt) != 2:
                _fail("malformed_episode", f"step {t} lacks two seats")
            public = cur[0].get("observation") or {}
            own = cur[seat].get("observation") or {}
            X[t] = encode_observation(merge_observation(public, own, seat, t), seat)
            A[t] = encode_action(nxt[seat].get("action"), stats=stats)
        result.append((X, A))
    return result


def episode_id(doc, member):
    """Kaggle EpisodeId first; UUID fallback; otherwise JSON basename."""
    value = (doc.get("info") or {}).get("EpisodeId") or doc.get("id") or Path(member).stem
    return str(value)


def split_for_episode(identifier, val_fraction=0.1):
    if not 0 <= val_fraction <= 1:
        raise ValueError("val_fraction must be in [0,1]")
    h = int.from_bytes(hashlib.sha256(str(identifier).encode("utf-8")).digest()[:8], "big")
    return "val" if h < int(val_fraction * (1 << 64)) else "train"


def _loads(raw):
    if orjson is not None:
        try:
            return orjson.loads(raw)
        except orjson.JSONDecodeError:   # e.g. NaN literals or non-UTF-8: json decides
            pass
    return json.loads(raw)


def _source_members(source):
    """The JSON member names of a source, in the order prepare() takes them."""
    if source.is_dir():
        return [p.relative_to(source).as_posix() for p in sorted(source.rglob("*.json"))]
    if source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as z:
            return sorted(n for n in z.namelist() if n.lower().endswith(".json"))
    if source.suffix.lower() == ".json":
        return [source.name]
    raise ValueError(f"Unsupported replay input: {source}")


_ARCHIVES = {}   # per process: archives stay open between the members a worker reads


def _read_member(source, member):
    if source.is_dir():
        return (source / member).read_bytes()
    if source.suffix.lower() == ".zip":
        if source not in _ARCHIVES:
            _ARCHIVES[source] = zipfile.ZipFile(source)
        return _ARCHIVES[source].read(member)
    return source.read_bytes()


def _merge_stats(destination, source):
    for key, value in source.items():
        if key == "quantity_max_requested":
            destination[key] = max(destination[key], value)
        else:
            destination[key] += value


def _extract(task):
    """Encode one replay member into `<final path>.part<index>` files; prepare() renames or drops
    them. Returns the identifier (None if parsing failed before it), records, or an error."""
    index, source, member, output, val_fraction = task
    source, output = Path(source), Path(output)
    result = {"identifier": None}
    written = []
    try:
        try:
            raw = _read_member(source, member)
        except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
            return {"identifier": None, "code": "source_error", "message": str(exc)[:300]}
        doc = _loads(raw)
        del raw
        if not isinstance(doc, Mapping):
            _fail("not_episode", member)
        identifier = result["identifier"] = episode_id(doc, member)
        info = doc.get("info") or {}
        teams = info.get("TeamNames") or [a.get("Name") if isinstance(a, Mapping) else None
                                          for a in info.get("Agents") or []]
        final = (doc.get("steps") or [[]])[-1]   # the last step's reward is each seat's final cash
        cash = [final[s].get("reward") if s < len(final) and isinstance(final[s], Mapping) else None for s in (0, 1)]
        known = all(isinstance(c, (int, float)) for c in cash)
        win = [(1.0 if cash[s] > cash[1 - s] else 0.0 if cash[s] < cash[1 - s] else 0.5) if known else None
               for s in (0, 1)]
        day = re.search(r"\d{4}-\d{2}-\d{2}", source.name)
        stats = Counter()
        arrays = episode_arrays(doc, stats=stats)
        del doc
        split = split_for_episode(identifier, val_fraction)
        # Include original ID visibly and a hash to prevent sanitized-name collisions.
        safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", identifier)[:100]
        suffix = hashlib.sha256(identifier.encode()).hexdigest()[:12]
        records = []
        for seat, (X, A) in enumerate(arrays):
            rel = Path(split) / f"episode_{safe_id}_{suffix}_seat{seat}.npz"
            part = output / f"{rel}.part{index}"
            written.append(part)
            with open(part, "wb") as fh:   # a file object: numpy would append .npz to the name
                np.savez_compressed(fh, X=X, A=A, episode_id=np.asarray(identifier), seat=np.int64(seat))
            records.append({"path": rel.as_posix(), "episode_id": identifier, "seat": seat, "split": split,
                            "length": len(X), "source": str(source), "member": member,
                            "team": teams[seat] if seat < len(teams) else None,
                            "cash": cash[seat], "opponent_cash": cash[1 - seat], "win": win[seat],
                            "day": day.group(0) if day else None})
        result.update(records=records, split=split, rows=sum(len(x) for x, _ in arrays), stats=stats,
                      parts=written)
    except (DataError, ValueError, TypeError, KeyError, IndexError, OverflowError) as exc:
        for path in written:
            path.unlink(missing_ok=True)
        result.update(code=getattr(exc, "code", "malformed_episode"), message=str(exc)[:300])
    return result


def _run_tasks(tasks, workers, progress):
    """_extract over tasks, results in task order; `workers` processes (1: in this process)."""
    start = time.monotonic()
    if workers <= 1:
        results = map(_extract, tasks)
    else:
        pool = ProcessPoolExecutor(max_workers=workers)
        results = pool.map(_extract, tasks, chunksize=1)
    try:
        for done, result in enumerate(results, 1):
            if progress and (done % 50 == 0 or done == len(tasks)):
                progress(done, len(tasks), time.monotonic() - start)
            yield result
    finally:
        if workers > 1:
            pool.shutdown(cancel_futures=True)


def _day_scores(source):
    """{episode_id: (sum_score, min_score)} from a Kaggle daily zip's manifest.csv ({} without one)."""
    source = Path(source)
    if source.suffix.lower() != ".zip" or not source.exists():
        return {}
    with zipfile.ZipFile(source) as z:
        if "manifest.csv" not in z.namelist():
            return {}
        with z.open("manifest.csv") as fh:
            rows = list(csv.DictReader(io.TextIOWrapper(fh, "utf-8")))
    scores = {}
    for row in rows:
        try:
            scores[row["episode_id"]] = (float(row["sum_score"]), float(row["min_score"]))
        except (KeyError, TypeError, ValueError):
            continue
    return scores


def add_ratings(manifest):
    """Rate each team of a source (one Kaggle day) and each seat, from the games' ratings.

    A daily zip's manifest.csv gives both players' ratings of a game as sum_score and min_score,
    not which seat had which. Team ratings solve sum_score = r[team0] + r[team1] over the day's
    games by least squares (weakly pulled to the team's mean game average, so every team stays
    defined); in each game the lower-rated team's seat takes min_score, the other the rest.
    Adds rating, team_rating and team_rank (1 = the day's best) to the file records and a
    `teams` table to each source; sources without scores are left as they are.

    Final cash is not comparable across games (both seats of a game end with similar cash: the
    game sets the level), so a seat's result is its margin, (own - opponent cash) / their mean,
    and expected_margin is what its rating edge predicts, fitted over the day's games
    (`margin_fit` of the source). A seat played its game well when margin > expected_margin.
    """
    by_source = {}
    for record in manifest["files"]:
        pair = by_source.setdefault(record["source"], {}).setdefault(record["episode_id"], [None, None])
        pair[record["seat"]] = record
    for coverage in manifest["sources"]:
        scores = _day_scores(coverage["source"])
        games = [(pair, scores[eid]) for eid, pair in by_source.get(coverage["source"], {}).items()
                 if eid in scores and None not in pair and all(r.get("team") for r in pair)]
        if not games:
            continue
        teams = sorted({r["team"] for pair, _ in games for r in pair})
        ix = {team: i for i, team in enumerate(teams)}
        seats = Counter(r["team"] for pair, _ in games for r in pair)
        design = np.zeros((len(games) + len(teams), len(teams)))
        target = np.zeros(len(games) + len(teams))
        mean_game = np.zeros(len(teams))
        for g, (pair, (total, _)) in enumerate(games):
            for r in pair:
                design[g, ix[r["team"]]] += 1
                mean_game[ix[r["team"]]] += total / 2 / seats[r["team"]]
            target[g] = total
        prior = 1e-3
        design[len(games):] = prior * np.eye(len(teams))
        target[len(games):] = prior * mean_game
        rating = np.linalg.lstsq(design, target, rcond=None)[0]
        rank = {team: k + 1 for k, team in enumerate(sorted(teams, key=lambda t: -rating[ix[t]]))}
        gaps, margins = [], []
        for pair, (total, low) in games:
            r0, r1 = (rating[ix[r["team"]]] for r in pair)
            split = (low, total - low) if r0 < r1 else (total - low, low) if r0 > r1 else (total / 2, total / 2)
            for r, own in zip(pair, split):
                r.update(rating=round(own, 1), team_rating=round(float(rating[ix[r["team"]]]), 1),
                         team_rank=rank[r["team"]])
            cash = [r.get("cash") for r in pair]
            if all(isinstance(c, (int, float)) for c in cash) and cash[0] + cash[1] > 0:
                margin = (cash[0] - cash[1]) / ((cash[0] + cash[1]) / 2)
                pair[0]["margin"], pair[1]["margin"] = round(margin, 5), round(-margin, 5)
                gaps.append(split[0] - split[1])
                margins.append(margin)
        residual = design[:len(games)] @ rating - target[:len(games)]
        coverage["rating_fit_rms"] = round(float(np.sqrt(np.mean(residual ** 2))), 1)
        if gaps and any(gaps):
            # The margin a seat's rating edge predicts: margin = per_point * (own - opponent rating),
            # fitted over the day's games through the origin (the game is symmetric).
            gaps, margins = np.asarray(gaps), np.asarray(margins)
            per_point = float(gaps @ margins / (gaps @ gaps))
            explained = 1 - float(((margins - per_point * gaps) ** 2).sum() / max((margins ** 2).sum(), 1e-12))
            coverage["margin_fit"] = {"per_point": per_point, "r2": round(explained, 4), "games": len(gaps)}
            for pair, _ in games:
                if "margin" in pair[0]:
                    edge = pair[0]["rating"] - pair[1]["rating"]
                    pair[0]["expected_margin"] = round(per_point * edge, 5)
                    pair[1]["expected_margin"] = round(-per_point * edge, 5)
        coverage["teams"] = [{"team": team, "rating": round(float(rating[ix[team]]), 1), "rank": rank[team],
                              "seats": seats[team]} for team in sorted(teams, key=rank.get)]


def prepare(inputs, output, max_episodes_per_source=None, *, val_fraction=0.1, workers=1, progress=None):
    """Stream ZIPs/JSON directories to a NEW output directory and return manifest.

    Limit counts candidate JSON members, including rejected/duplicate episodes,
    so broken/duplicate archives cannot evade a bounded extraction budget.
    Existing managed outputs are refused, rather than silently mixing schemas.
    `workers` processes encode episodes in parallel; the manifest, duplicates (first
    occurrence in input order wins) and error counts are the same for any number. A member
    an archive cannot read counts as one source_error and the source goes on.
    `progress(done, total, seconds)` is called every 50 episodes.
    """
    if max_episodes_per_source is not None and max_episodes_per_source < 0:
        raise ValueError("max_episodes_per_source must be nonnegative or None")
    split_for_episode("validate", val_fraction)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use an empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        (output / split).mkdir(exist_ok=True)
    manifest = {
        "version": 1, "alignment": FEATURE_SPEC["alignment"],
        "feature_dim": FEATURE_DIM, "field_sizes": field_sizes,
        "codec": CODEC_CONFIG, "features_file": "features.json",
        "split": {"algorithm": "sha256(str(episode_id).utf8), first 8 bytes big-endian", "val_fraction": val_fraction},
        "counts": {s: {"episodes": 0, "seats": 0, "rows": 0} for s in ("train", "val")},
        "errors": {}, "error_examples": [], "quantization": {},
        "duplicates": 0, "sources": [], "files": [],
        "max_episodes_per_source": max_episodes_per_source,
    }
    errors, quantization = Counter(), Counter()

    def error(coverage, source_errors, code, example):
        errors[code] += 1
        source_errors[code] += 1
        if len(manifest["error_examples"]) < 30:
            manifest["error_examples"].append(example)

    tasks, owners = [], []   # owners[i]: (coverage, source_errors) of task i's source
    for raw_source in inputs:
        source = Path(raw_source)
        coverage = {"source": str(source), "candidates": 0, "accepted": 0, "duplicates": 0,
                    "errors": {}, "train": 0, "val": 0}
        source_errors = Counter()
        manifest["sources"].append(coverage)
        try:
            members = _source_members(source)
        except (OSError, zipfile.BadZipFile, ValueError, RuntimeError) as exc:
            error(coverage, source_errors, "source_error",
                  {"source": str(source), "code": "source_error", "message": str(exc)[:300]})
            members = []
        coverage["errors"] = source_errors   # made a plain dict below
        for member in members[:max_episodes_per_source]:
            coverage["candidates"] += 1
            tasks.append((len(tasks), str(source), member, str(output), val_fraction))
            owners.append((coverage, source_errors))
    seen = set()
    results = _run_tasks(tasks, workers, progress)
    for task, result in zip(tasks, results):
        (_, source, member, _, _), (coverage, source_errors) = task, owners[task[0]]
        identifier = result["identifier"]
        if identifier is not None and identifier in seen:   # first occurrence in input order wins
            coverage["duplicates"] += 1
            manifest["duplicates"] += 1
            for part in result.get("parts", []):
                part.unlink(missing_ok=True)
            continue
        if identifier is not None:
            seen.add(identifier)
        if "code" in result:
            error(coverage, source_errors, result["code"],
                  {"source": source, "member": member, "code": result["code"], "message": result["message"]})
            continue
        for part, record in zip(result["parts"], result["records"]):
            os.replace(part, output / record["path"])
        split = result["split"]
        manifest["files"].extend(result["records"])
        manifest["counts"][split]["episodes"] += 1
        manifest["counts"][split]["seats"] += 2
        manifest["counts"][split]["rows"] += result["rows"]
        coverage["accepted"] += 1
        coverage[split] += 1
        _merge_stats(quantization, result["stats"])
    results.close()   # shuts the worker pool down
    for coverage in manifest["sources"]:
        coverage["errors"] = dict(coverage["errors"])
    manifest["errors"], manifest["quantization"] = dict(errors), dict(quantization)
    add_ratings(manifest)
    for name, obj in (("codec.json", CODEC_CONFIG), ("features.json", FEATURE_SPEC), ("manifest.json", manifest)):
        (output / name).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    return manifest


class WindowDataset:
    """Torch DataLoader-compatible map dataset, without importing torch.

    Anchor t contains context X[max(0,t-context+1):t+1] and labels A[t:t+horizon].
    Starts at 0, then every stride; includes partial tails, never crosses a file.
    NumPy arrays are collatable by PyTorch. Metadata stays in manifest/files.
    LRU is reset after fork/pickle; cache bounds are per worker, not global.

    Replay step t is hour t % 24, and A[t] is the action the agent executes. A stride that
    shares a factor with 24 anchors windows at a few hours only (8: hours 0/8/16; 24: hour 0),
    so the default 7 is coprime with 24 and covers every hour.
    """
    def __init__(self, directory, split, context=CONTEXT, horizon=HORIZON, stride=7,
                 *, cache_size=2, cache_bytes=128 * 1024 * 1024):
        if split not in ("train", "val"):
            raise ValueError("split must be train or val")
        if min(context, horizon, stride) <= 0 or cache_size < 0 or cache_bytes < 0:
            raise ValueError("positive window lengths/stride and nonnegative cache limits required")
        self.directory = Path(directory)
        self.context, self.horizon, self.stride = int(context), int(horizon), int(stride)
        self.cache_size, self.cache_bytes = cache_size, cache_bytes
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        if self.manifest["feature_dim"] != FEATURE_DIM or self.manifest["field_sizes"] != field_sizes:
            raise ValueError("Dataset schema differs from this data module")
        self.feature_dim, self.field_sizes = FEATURE_DIM, list(field_sizes)
        self.select([f for f in self.manifest["files"] if f["split"] == split and f["length"] > 0])
        self._cache = OrderedDict()
        self._cached_bytes = 0
        self._pid = os.getpid()

    def select(self, files):
        """Keep only these manifest records (e.g. an evenly spaced subset of validation files)."""
        self.files = list(files)
        self._ends = []
        total = 0
        for f in self.files:
            total += len(self.anchors(f["length"]))
            self._ends.append(total)

    def anchors(self, length):
        return range(0, length, self.stride)

    def __len__(self):
        return self._ends[-1] if self._ends else 0

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(_cache=OrderedDict(), _cached_bytes=0, _pid=None)
        return state

    def read(self, index):
        """(X, A) of file `index`, uncached."""
        record = self.files[index]
        with np.load(self.directory / record["path"], allow_pickle=False) as z:
            X, A = z["X"], z["A"]
        if X.shape != (record["length"], FEATURE_DIM) or A.shape != (len(X), ACTION_DIM):
            raise ValueError(f"Corrupt shard shapes: {record['path']}")
        return X, A

    def _load(self, index):
        if self._pid != os.getpid():
            self._cache.clear()
            self._cached_bytes = 0
            self._pid = os.getpid()
        if index in self._cache:
            self._cache.move_to_end(index)
            return self._cache[index]
        X, A = self.read(index)
        size = X.nbytes + A.nbytes
        if self.cache_size and size <= self.cache_bytes:
            while self._cache and (len(self._cache) >= self.cache_size or self._cached_bytes + size > self.cache_bytes):
                _, (old_x, old_a) = self._cache.popitem(last=False)
                self._cached_bytes -= old_x.nbytes + old_a.nbytes
            self._cache[index] = (X, A)
            self._cached_bytes += size
        return X, A

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        file_index = bisect_right(self._ends, index)
        start_index = self._ends[file_index - 1] if file_index else 0
        t = (index - start_index) * self.stride
        X, A = self._load(file_index)
        item = self.window(X, A, t)
        item["file"] = np.int64(file_index)   # index into self.files, e.g. for per-seat weights
        return item

    def window(self, X, A, t):
        """The sample anchored at step t of one file's arrays; `anchor` is t (hour t % 24)."""
        history_start = max(0, t - self.context + 1)
        n_history = t - history_start + 1
        n_actions = min(self.horizon, len(A) - t)
        context = np.zeros((self.context, FEATURE_DIM), dtype=np.float32)
        context_mask = np.zeros(self.context, dtype=bool)
        actions = np.zeros((self.horizon, ACTION_DIM), dtype=np.int64)
        action_mask = np.zeros(self.horizon, dtype=bool)
        context[-n_history:] = X[history_start:t + 1]
        context_mask[-n_history:] = True
        actions[:n_actions] = A[t:t + n_actions]
        action_mask[:n_actions] = True
        return {"context": context, "context_mask": context_mask,
                "actions": actions, "action_mask": action_mask, "anchor": np.int64(t)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", "--out", required=True, type=Path)
    parser.add_argument("--max-episodes-per-source", type=int, default=None)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=1, help="parallel encoding processes")
    args = parser.parse_args()
    manifest = prepare(args.inputs, args.output, args.max_episodes_per_source, val_fraction=args.val_fraction,
                       workers=args.workers,
                       progress=lambda done, total, s: print(f"prepared {done}/{total} in {s:.0f}s", flush=True))
    print(json.dumps({k: manifest[k] for k in ("counts", "errors", "quantization", "sources")}, indent=2))


if __name__ == "__main__":
    main()
