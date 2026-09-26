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

Only one JSON episode is loaded at a time. Outputs are per-seat compressed NPZ;
WindowDataset caches a bounded number/size of uncompressed episodes per process.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter, OrderedDict
from collections.abc import Mapping
import hashlib
import json
import math
import os
from pathlib import Path
import re
import zipfile

import numpy as np

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
    "none": "omit farmer/market NONE; hand NONE => PASS to preserve live hand indices",
}


class DataError(ValueError):
    """A counted episode-level schema rejection, never silent relabeling."""
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


def _encode_command(command, market, stats):
    if command is None or command == []:
        return (0, 0, 0)
    if not isinstance(command, (list, tuple)) or not 1 <= len(command) <= 3:
        _fail("malformed_command", repr(command))
    op = command[0]
    if not isinstance(op, str) or op not in OP_TO_ID:
        _fail("unsupported_operation", repr(op))
    if op != "NONE" and op not in (MARKET_OPS if market else UNIT_OPS):
        _fail("unsupported_operation_slot", f"{op}: market={market}")
    # Validate supplied arguments even when the engine ignores them.
    if len(command) > 1 and (not isinstance(command[1], str) or command[1] not in ITEM_TO_ID):
        _fail("unsupported_item", repr(command[1]))
    item_id = 0
    if op in ARGUMENTS:
        if len(command) < 2 or command[1] not in ARGUMENTS[op]:
            _fail("unsupported_item", f"{op}: {command[1:]!r}")
        item_id = ITEM_TO_ID[command[1]]
    quantity = 0
    if op in QUANTITY_OPS:
        if len(command) < 3:
            if op not in ("PICKUP", "PLACE"):
                _fail("malformed_command", f"Missing market quantity: {command!r}")
            stats["quantity_default_one"] += 1
        quantity = _quantity(command[2] if len(command) > 2 else 1, stats)
    elif len(command) > 2:
        stats["ignored_quantity_fields"] += 1
    return OP_TO_ID[op], item_id, quantity


def encode_action(action, *, stats=None):
    """Return int64[129]; unknown ops/items raise DataError for episode rejection."""
    stats = Counter() if stats is None else stats
    if not isinstance(action, Mapping):
        _fail("malformed_action", f"Expected action dict, got {type(action).__name__}")
    hands, market = action.get("hands", []), action.get("market", [])
    if not isinstance(hands, (list, tuple)) or not isinstance(market, (list, tuple)):
        _fail("malformed_action", "hands and market must be lists")
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


def _number(value):
    try:
        n = float(value)
    except (ValueError, TypeError, OverflowError):
        _fail("invalid_observation_number", repr(value))
    if not math.isfinite(n) or abs(n) > np.finfo(np.float32).max:
        _fail("invalid_observation_number", repr(value))
    return n


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
                tile_dict: Mapping = {}
                if tile is None:
                    kind = "EMPTY"
                elif isinstance(tile, str):
                    kind = tile
                elif isinstance(tile, Mapping):
                    kind = str(tile.get("kind"))
                    tile_dict = tile
                else:
                    _fail("unsupported_tile", repr(tile))
                if kind not in TILE_KINDS:
                    _fail("unsupported_tile", str(kind))
                if tile_dict.get("crop") is not None and tile_dict["crop"] not in CROPS:
                    _fail("unsupported_item", str(tile_dict["crop"]))
                if tile_dict.get("animal") is not None and tile_dict["animal"] not in ANIMALS:
                    _fail("unsupported_item", str(tile_dict["animal"]))
                values.extend(float(kind == k) for k in TILE_KINDS)
                values.extend(float(tile_dict.get("crop") == k) for k in CROPS)
                values.extend(float(tile_dict.get("animal") == k) for k in ANIMALS)
                values.extend(_number(tile_dict.get(k, 0)) for k, _ in TILE_SCALARS)
    private = obs["private"]
    inventories = private.get("inventories", [])
    if len(inventories) > MAX_HANDS + 1:
        _fail("too_many_hands", f"{len(inventories)} inventories")
    stores = list(inventories) + [private.get("seeds", {}), private.get("shed", {})]
    for store in stores:
        if not isinstance(store, Mapping) or set(store) - set(STORAGE_ITEMS):
            _fail("unsupported_item", f"private store keys: {store!r}")
    for i in range(MAX_HANDS + 1):
        inv = inventories[i] if i < len(inventories) else {}
        values.extend(_number(inv.get(k, 0)) for k in STORAGE_ITEMS)
    values.extend(_number(private.get("seeds", {}).get(k, 0)) for k in CROPS)
    values.extend(_number(private.get("shed", {}).get(k, 0)) for k in STORAGE_ITEMS)
    out = np.asarray(values, dtype=np.float32)
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


def _source_members(source):
    """Yield (name, bytes) without extracting archives or retaining other episodes."""
    if source.is_dir():
        for p in sorted(source.rglob("*.json")):
            yield p.relative_to(source).as_posix(), p.read_bytes()
    elif source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as z:
            for name in sorted(n for n in z.namelist() if n.lower().endswith(".json")):
                yield name, z.read(name)
    elif source.suffix.lower() == ".json":
        yield source.name, source.read_bytes()
    else:
        raise ValueError(f"Unsupported replay input: {source}")


def _merge_stats(destination, source):
    for key, value in source.items():
        if key == "quantity_max_requested":
            destination[key] = max(destination[key], value)
        else:
            destination[key] += value


def prepare(inputs, output, max_episodes_per_source=None, *, val_fraction=0.1):
    """Stream ZIPs/JSON directories to a NEW output directory and return manifest.

    Limit counts candidate JSON members, including rejected/duplicate episodes,
    so broken/duplicate archives cannot evade a bounded extraction budget.
    Existing managed outputs are refused, rather than silently mixing schemas.
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
    seen, errors, quantization = set(), Counter(), Counter()
    for raw_source in inputs:
        source = Path(raw_source)
        coverage = {"source": str(source), "candidates": 0, "accepted": 0, "duplicates": 0,
                    "errors": {}, "train": 0, "val": 0}
        source_errors = Counter()
        iterator = iter(_source_members(source))
        try:
            while max_episodes_per_source is None or coverage["candidates"] < max_episodes_per_source:
                try:
                    member, raw = next(iterator)
                except StopIteration:
                    break
                coverage["candidates"] += 1
                written = []
                try:
                    doc = json.loads(raw)
                    del raw
                    if not isinstance(doc, Mapping):
                        _fail("not_episode", member)
                    identifier = episode_id(doc, member)
                    if identifier in seen:
                        coverage["duplicates"] += 1
                        manifest["duplicates"] += 1
                        continue
                    seen.add(identifier)
                    episode_stats = Counter()
                    arrays = episode_arrays(doc, stats=episode_stats)
                    split = split_for_episode(identifier, val_fraction)
                    # Include original ID visibly and a hash to prevent sanitized-name collisions.
                    safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", identifier)[:100]
                    suffix = hashlib.sha256(identifier.encode()).hexdigest()[:12]
                    records = []
                    for seat, (X, A) in enumerate(arrays):
                        rel = Path(split) / f"episode_{safe_id}_{suffix}_seat{seat}.npz"
                        path = output / rel
                        written.append(path)
                        np.savez_compressed(path, X=X, A=A, episode_id=np.asarray(identifier), seat=np.int64(seat))
                        records.append({"path": rel.as_posix(), "episode_id": identifier,
                                        "seat": seat, "split": split, "length": len(X), "source": str(source), "member": member})
                    manifest["files"].extend(records)
                    manifest["counts"][split]["episodes"] += 1
                    manifest["counts"][split]["seats"] += 2
                    manifest["counts"][split]["rows"] += sum(len(x) for x, _ in arrays)
                    coverage["accepted"] += 1
                    coverage[split] += 1
                    _merge_stats(quantization, episode_stats)
                    del arrays, doc
                except (DataError, ValueError, TypeError, KeyError, IndexError, OverflowError) as exc:
                    for path in written:
                        path.unlink(missing_ok=True)
                    code = getattr(exc, "code", "malformed_episode")
                    errors[code] += 1
                    source_errors[code] += 1
                    if len(manifest["error_examples"]) < 30:
                        manifest["error_examples"].append({"source": str(source), "member": member,
                                                           "code": code, "message": str(exc)[:300]})
        except (OSError, zipfile.BadZipFile, ValueError, RuntimeError) as exc:
            errors["source_error"] += 1
            source_errors["source_error"] += 1
            if len(manifest["error_examples"]) < 30:
                manifest["error_examples"].append({"source": str(source), "code": "source_error", "message": str(exc)[:300]})
        finally:
            iterator.close()
        coverage["errors"] = dict(source_errors)
        manifest["sources"].append(coverage)
    manifest["errors"], manifest["quantization"] = dict(errors), dict(quantization)
    for name, obj in (("codec.json", CODEC_CONFIG), ("features.json", FEATURE_SPEC), ("manifest.json", manifest)):
        (output / name).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    return manifest


class WindowDataset:
    """Torch DataLoader-compatible map dataset, without importing torch.

    Anchor t contains context X[max(0,t-context+1):t+1] and labels A[t:t+horizon].
    Starts at 0, then every stride; includes partial tails, never crosses a file.
    NumPy arrays are collatable by PyTorch. Metadata stays in manifest/files.
    LRU is reset after fork/pickle; cache bounds are per worker, not global.
    """
    def __init__(self, directory, split, context=CONTEXT, horizon=HORIZON, stride=8,
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
        self.files = [f for f in self.manifest["files"] if f["split"] == split and f["length"] > 0]
        self._ends = []
        total = 0
        for f in self.files:
            total += (f["length"] + self.stride - 1) // self.stride
            self._ends.append(total)
        self._cache = OrderedDict()
        self._cached_bytes = 0
        self._pid = os.getpid()

    def __len__(self):
        return self._ends[-1] if self._ends else 0

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(_cache=OrderedDict(), _cached_bytes=0, _pid=None)
        return state

    def _load(self, index):
        if self._pid != os.getpid():
            self._cache.clear()
            self._cached_bytes = 0
            self._pid = os.getpid()
        if index in self._cache:
            self._cache.move_to_end(index)
            return self._cache[index]
        record = self.files[index]
        with np.load(self.directory / record["path"], allow_pickle=False) as z:
            X, A = z["X"], z["A"]
        if X.shape != (record["length"], FEATURE_DIM) or A.shape != (len(X), ACTION_DIM):
            raise ValueError(f"Corrupt shard shapes: {record['path']}")
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
                "actions": actions, "action_mask": action_mask}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", "--out", required=True, type=Path)
    parser.add_argument("--max-episodes-per-source", type=int, default=None)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    args = parser.parse_args()
    manifest = prepare(args.inputs, args.output, args.max_episodes_per_source, val_fraction=args.val_fraction)
    print(json.dumps({k: manifest[k] for k in ("counts", "errors", "quantization", "sources")}, indent=2))


if __name__ == "__main__":
    main()
