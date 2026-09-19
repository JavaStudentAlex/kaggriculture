"""Opponent order-flow oracle: the frozen TTM checkpoint served to a live agent.

Fixed infrastructure, NOT an evolution target. It wraps the fine-tuned
TinyTimeMixer in `shinka/evolution/checkpoint/` (a byte-identical copy of the
promoted `models/ttm_v3_h96_ft_<newest day>/` -- its README and labels.json say
which; weights untouched) and rebuilds, turn by turn, exactly the input the
model was trained on (`research/opponent_model/{features,extract}.py`):

  * 141 public features per turn (`features.build_features`), standardised with
    the checkpoint's own `scaler.npz`;
  * 9 channels of opponent supply per product, log1p-compressed, rebuilt from
    market-inventory accounting the way `extract.py` labels replays:
    total(t) = inv[t+1] - inv[t] + town_draw(t), attributed between the seats
    by who placed orders. Two attribution rules exist (`ALIGNMENT`):
      - "legacy": what every checkpoint so far (v3 and its daily fine-tunes)
        was trained on. extract.py reads the orders stored at replay index t, but
        Kaggle stores at index t the action taken FROM observation t-1, so the
        flow of step t is attributed by the orders of step t-1. The live agent
        reproduces that rule (own orders are known; the opponent's activity at
        t-1 is inferred from the flow it caused), so the checkpoint sees the
        input distribution it was fitted to.
      - "next_action": the correct rule (orders of step t explain the flow of
        step t); for checkpoints trained on shards extracted with that fix.
    The rule is read from `<model dir>/labels.json` (`{"alignment": ...}`) and
    defaults to "legacy" when the file is absent; KAGG_ORACLE_ALIGNMENT overrides.

At step t the model sees rows t-C .. t-1, C being the checkpoint's context
length (`config.json`: 512 turns for v3 and its fine-tunes = forecasts from day
21; a 240-turn checkpoint forecasts from day 10), and returns, for every
product, the opponent's predicted supply at steps t .. t+95 (log1p units).

Process model: one model per worker process, loaded once and cached across the
re-imports the evaluator does per game (`sys` registry). Device policy
(`KAGG_ORACLE_DEVICE`, default `auto`): spread worker processes over the visible
GPUs -- pid-based round robin, skipping a GPU with less than 1 GiB free -- and
fall back to CPU if CUDA is missing or the load fails. Inference is ~5 ms on an
A100 and ~80 ms single-threaded on CPU, so the fallback keeps a game playable
but slow.

Backend (`KAGG_ORACLE_BACKEND`, default `auto`): `torch` is the checkpoint run
through tsfm_public; `numpy` is the same network reimplemented in
`kagg_ttm_numpy.py` (no torch, no transformers: ~150 ms per forecast on one
core, but the agent file imports in well under a second, which is what Kaggle's
sandbox needs -- see that module's docstring). `auto` takes torch when it
imports and numpy otherwise, so a missing torch no longer switches the oracle
off silently.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

DEFAULT_CONTEXT = 512  # only for a tracker without a model; a model brings its own
HORIZON = 96
MAX_STEPS = 720
# KAGG_ORACLE_MIN_CONTEXT = rows of real history required before the model is
# asked. Default = the checkpoint's full context, i.e. exactly the training
# regime. Lower values left-pad the context with zeros (= feature means); see
# check_oracle.py padding for how much that costs at each origin (a lot).
ALIGNMENTS = ("legacy", "next_action")

_REGISTRY_KEY = "_kagg_oracle_registry"


# ----------------------------------------------------------------- path resolution
def _first_dir(env_name, candidates, marker):
    """First existing directory among env override + candidates containing `marker`."""
    cands = []
    env = os.environ.get(env_name)
    if env:
        cands.append(Path(env))
    cands.extend(candidates)
    for c in cands:
        if (Path(c) / marker).exists():
            return str(Path(c).resolve())
    return str(cands[-1]) if cands else ""


def _repo_root_candidates():
    here = Path(__file__).resolve()
    out = []
    env = os.environ.get("KAGG_TASK_DIR")
    if env:
        out.append(Path(env))
    # shinka/evolution/kagg_oracle.py -> repo ; shinka_results/kagg_oracle.py -> repo
    out += [here.parent.parent.parent, here.parent.parent, Path.home() / "kaggriculture"]
    return out


# The checkpoint lives next to this file (a byte-identical copy of the newest
# promoted models/ttm_v3_h96_ft_<day>/); KAGG_TTM_DIR points elsewhere only to try another one.
MODEL_DIR = os.environ.get("KAGG_TTM_DIR") or str(Path(__file__).resolve().parent / "checkpoint")
FEATURE_SRC = _first_dir(
    "KAGG_OPP_MODEL_SRC",
    [r / "research" / "opponent_model" for r in _repo_root_candidates()],
    "features.py",
)
if FEATURE_SRC and FEATURE_SRC not in sys.path:
    sys.path.insert(0, FEATURE_SRC)

from features import OpponentHistory, build_features, feature_names  # noqa: E402
from mechanics import ANIMALS, BUYABLE_PRODUCTS, CROPS, PRODUCTS, config_intervals, market_price, town_draw  # noqa: E402

try:  # the pinned engine: costs our own orders incur, and the shed-access rule for DROP/PICKUP
    from kaggle_environments.envs.kaggriculture.kaggriculture import (
        LAND_ORDER,
        LAND_PRICES,
        _hire_cost,
        _is_shed_adjacent,
    )
except Exception:  # pragma: no cover - engine layout drift; fall back to 1.32.7 values
    LAND_ORDER, LAND_PRICES = ["NE", "SW", "SE"], [1000, 2000, 4000]

    def _hire_cost(n_already_today, mult=1):
        a, b = 1, 1
        for _ in range(n_already_today):
            a, b = b, a + b
        return mult * a

    def _is_shed_adjacent(pos, board_size):
        return tuple(pos) in {(1, 0), (0, 1)}

FEATURE_NAMES = feature_names()
N_FEATURES = len(FEATURE_NAMES)
N_TARGETS = len(PRODUCTS)
N_CHANNELS = N_FEATURES + N_TARGETS
PRODUCT_INDEX = {p: i for i, p in enumerate(PRODUCTS)}


# ----------------------------------------------------------------- model singleton
class _Model:
    """The checkpoint on one device, shared by every agent instance in this process."""

    def __init__(self, model_dir, device_spec):
        import torch  # deferred: the evaluator's parent process never needs it

        torch.set_num_threads(1)
        from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction

        self.torch = torch
        self.model_dir = str(model_dir)
        with np.load(Path(model_dir) / "scaler.npz") as sc:
            self.mean = sc["mean"].astype(np.float32)
            self.std = sc["std"].astype(np.float32)
        if self.mean.shape[0] != N_FEATURES:
            raise RuntimeError(f"scaler has {self.mean.shape[0]} features, live features have {N_FEATURES}")

        self.alignment = _model_alignment(self.model_dir)
        model = TinyTimeMixerForPrediction.from_pretrained(self.model_dir).eval()
        if int(model.config.num_input_channels) != N_CHANNELS:
            raise RuntimeError(
                f"checkpoint expects {model.config.num_input_channels} channels, live layout has {N_CHANNELS}"
            )
        if int(model.config.prediction_length) != HORIZON:
            raise RuntimeError(f"checkpoint forecasts {model.config.prediction_length} steps, the policy expects {HORIZON}")
        self.context = int(model.config.context_length)  # turns of history per forecast
        self.device = self._pick_device(torch, device_spec)
        try:
            self.model = model.to(self.device)
            self._warm()
        except Exception as exc:  # CUDA context/MPS limits etc. -> keep playing on CPU
            print(f"[kagg_oracle] {self.device} unusable ({type(exc).__name__}: {exc}); using cpu", file=sys.stderr)
            self.device = torch.device("cpu")
            self.model = model.to(self.device)
            self._warm()

    @staticmethod
    def _pick_device(torch, spec):
        spec = (spec or "auto").strip().lower()
        if spec != "auto":
            return torch.device(spec)
        if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            return torch.device("cpu")
        n = torch.cuda.device_count()
        # Query free VRAM across all available GPUs to prioritize GPUs with lowest memory usage
        devices_by_free = []
        for idx in range(n):
            try:
                free, _total = torch.cuda.mem_get_info(idx)
                devices_by_free.append((free, idx))
            except Exception:
                continue
        # Gather all GPUs with >= 6 GB free headroom (lowest memory usage)
        safe_gpus = [idx for free, idx in devices_by_free if free >= (6 << 30)]
        if safe_gpus:
            # Distribute workers evenly across the low-memory-usage GPUs based on PID
            chosen_idx = safe_gpus[os.getpid() % len(safe_gpus)]
            return torch.device(f"cuda:{chosen_idx}")
        for free, idx in devices_by_free:
            if free >= (2 << 30):
                return torch.device(f"cuda:{idx}")
        return torch.device("cpu")

    def _warm(self):
        torch = self.torch
        with torch.no_grad():
            x = torch.zeros(1, self.context, N_CHANNELS, device=self.device)
            self.model(past_values=x)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def predict(self, block):
        """block: (context, N_CHANNELS) float32 -> (HORIZON, N_TARGETS) log1p units."""
        torch = self.torch
        with torch.no_grad():
            x = torch.from_numpy(np.ascontiguousarray(block, dtype=np.float32)).unsqueeze(0).to(self.device)
            out = self.model(past_values=x).prediction_outputs
        return out[0].float().cpu().numpy()


class _NumpyModel:
    """`_Model`'s interface (context, mean/std, device, alignment, predict) on the
    numpy reimplementation of the checkpoint; nothing beyond numpy is imported."""

    def __init__(self, model_dir):
        from kagg_ttm_numpy import TTMNumpy

        self.model_dir = str(model_dir)
        with np.load(Path(model_dir) / "scaler.npz") as sc:
            self.mean = sc["mean"].astype(np.float32)
            self.std = sc["std"].astype(np.float32)
        if self.mean.shape[0] != N_FEATURES:
            raise RuntimeError(f"scaler has {self.mean.shape[0]} features, live features have {N_FEATURES}")
        self.alignment = _model_alignment(self.model_dir)
        self.net = TTMNumpy(self.model_dir)
        if self.net.n_channels != N_CHANNELS:
            raise RuntimeError(f"checkpoint expects {self.net.n_channels} channels, live layout has {N_CHANNELS}")
        if self.net.horizon != HORIZON:
            raise RuntimeError(f"checkpoint forecasts {self.net.horizon} steps, the policy expects {HORIZON}")
        self.context = int(self.net.context)
        self.device = "cpu"
        self.backend = "numpy"
        self.net.predict(np.zeros((self.context, N_CHANNELS), dtype=np.float32))  # warm

    def predict(self, block):
        """block: (context, N_CHANNELS) float32 -> (HORIZON, N_TARGETS) log1p units."""
        return self.net.predict(block)


def _build_model(model_dir, device_spec):
    backend = os.environ.get("KAGG_ORACLE_BACKEND", "auto").strip().lower()
    if backend == "numpy":
        return _NumpyModel(model_dir)
    if backend == "torch":
        model = _Model(model_dir, device_spec)
        model.backend = "torch"
        return model
    if backend != "auto":
        raise RuntimeError(f"unknown KAGG_ORACLE_BACKEND {backend!r} (auto|torch|numpy)")
    try:
        import torch  # noqa: F401
        import tsfm_public.models.tinytimemixer  # noqa: F401
    except Exception as exc:
        print(f"[kagg_oracle] torch/tsfm unavailable ({type(exc).__name__}: {exc}); numpy backend", file=sys.stderr)
        return _NumpyModel(model_dir)
    model = _Model(model_dir, device_spec)
    model.backend = "torch"
    return model


def _model_alignment(model_dir):
    env = os.environ.get("KAGG_ORACLE_ALIGNMENT")
    if env:
        value = env
    else:
        value = "legacy"
        marker = Path(model_dir) / "labels.json"
        if marker.exists():
            try:
                import json

                value = str(json.loads(marker.read_text()).get("alignment", value))
            except (OSError, ValueError):
                pass
    if value not in ALIGNMENTS:
        raise RuntimeError(f"unknown label alignment {value!r}; expected one of {ALIGNMENTS}")
    return value


def get_model(model_dir=None, device=None):
    """Process-wide cached model. Survives the evaluator re-importing the agent per game."""
    model_dir = str(model_dir or MODEL_DIR)
    device = device or os.environ.get("KAGG_ORACLE_DEVICE", "auto")
    registry = getattr(sys, _REGISTRY_KEY, None)
    if registry is None:
        registry = {}
        setattr(sys, _REGISTRY_KEY, registry)
    key = (model_dir, device, os.environ.get("KAGG_ORACLE_BACKEND", "auto"))
    if key not in registry:
        registry[key] = _build_model(model_dir, device)
    return registry[key]


# ----------------------------------------------------------------- per-game tracker
def _market_orders(action):
    market = (action or {}).get("market") if isinstance(action, dict) else None
    return market if isinstance(market, (list, tuple)) else []


def _requested(orders):
    """Requested SELL (+) / BUY_PRODUCT (-) units per product, engine parsing rules."""
    req = {}
    for order in orders:
        if not isinstance(order, (list, tuple)) or len(order) < 3:
            continue
        op, item = order[0], order[1]
        try:
            n = int(order[2])
        except (TypeError, ValueError):
            continue
        if n <= 0:
            continue
        if op == "SELL" and item in PRODUCT_INDEX:
            req[item] = req.get(item, 0) + n
        elif op == "BUY_PRODUCT" and item in BUYABLE_PRODUCTS:
            req[item] = req.get(item, 0) - n
    return req


class _Turn:
    """What we knew when we acted at one step, kept until its flow is observed."""

    __slots__ = (
        "step", "inventory", "shops", "shed", "money", "hires_today", "quadrants",
        "positions", "inventories", "orders", "units", "opp_flow",
    )

    def __init__(self, step, inventory, shops, shed, money, hires_today=0, quadrants=1, positions=(), inventories=()):
        self.step = step
        self.inventory = inventory
        self.shops = shops
        self.shed = shed
        self.money = money
        self.hires_today = hires_today
        self.quadrants = quadrants
        self.positions = list(positions)  # [farmer, *hands]
        self.inventories = list(inventories)  # [farmer bag, *hand bags]
        self.orders = []  # market orders we submitted
        self.units = []  # [farmer op, *hand ops] we submitted
        self.opp_flow = None  # net units the opponent moved at this step (filled next turn)


class OpponentTracker:
    """Per-game, per-seat rebuild of the training rows, plus the model's forecast.

    Call `observe(obs, configuration)` once per turn BEFORE deciding, then
    `record_action(final_action)` with what was actually submitted, so later
    turns can attribute the market flow between the two seats.
    """

    def __init__(self, model=None, min_context=None, alignment=None):
        self.model = model
        self.context = int(model.context if model is not None else DEFAULT_CONTEXT)
        if min_context is None:
            min_context = os.environ.get("KAGG_ORACLE_MIN_CONTEXT", self.context)
        self.min_context = int(min_context)
        self.alignment = alignment or (model.alignment if model is not None else "legacy")
        if self.alignment not in ALIGNMENTS:
            raise ValueError(f"alignment must be one of {ALIGNMENTS}")
        self.reset()

    def reset(self):
        self.history = OpponentHistory()
        self.rows = np.zeros((MAX_STEPS + HORIZON, N_CHANNELS), dtype=np.float32)
        self.supply = np.zeros((MAX_STEPS + HORIZON, N_TARGETS), dtype=np.int32)
        self.n_complete = 0  # rows [0, n_complete) have both features and supply
        self.step = -1
        self.turns = {}  # step -> _Turn, last few only
        self.pred = None  # (HORIZON, N_TARGETS) log1p units for steps step .. step+95
        self.pred_step = -1
        self.cfg = None

    # -- accounting --------------------------------------------------------------
    def _our_executed(self, turn):
        """Net units WE added to market inventory at `turn` (sells - buys).

        Replays the engine's order processing on our own orders alone -- list
        order, unit by unit, with the cash and shed we had -- so a sell early in
        the list funds a buy later in it and a buy stops when the shed is full or
        the money runs out. The opponent's concurrent trades (they move prices)
        and drops our hands made during the step are not visible; both are small.
        """
        flow = {}
        cash = float(turn.money)
        held = self._shed_at_market(turn)
        shed_total = int(sum(held.values()))
        inventory = dict(turn.inventory)
        hires, quadrants = int(turn.hires_today), int(turn.quadrants)
        cap, hire_mult = self.cfg["shed_capacity"], self.cfg.get("hire_mult", 1)
        for order in turn.orders:
            if not isinstance(order, (list, tuple)) or not order:
                continue
            op = order[0]
            if op == "HIRE":
                cost = _hire_cost(hires, hire_mult)
                if cash >= cost:
                    cash -= cost
                    hires += 1
                continue
            if op == "BUY_LAND":
                extra = quadrants - 1
                if extra < len(LAND_ORDER) and cash >= LAND_PRICES[extra]:
                    cash -= LAND_PRICES[extra]
                    quadrants += 1
                continue
            if len(order) < 3:
                continue
            item = order[1]
            try:
                n = int(order[2])
            except (TypeError, ValueError):
                continue
            if n <= 0:
                continue
            for _ in range(n):
                if op == "SELL" and item in PRODUCT_INDEX:
                    if held.get(item, 0) <= 0:
                        break
                    price = market_price(item, int(inventory.get(item, 0)))
                    held[item] -= 1
                    shed_total -= 1
                    cash += price
                    if price > 1:  # sales at the floor do not add supply
                        inventory[item] = int(inventory.get(item, 0)) + 1
                        flow[item] = flow.get(item, 0) + 1
                elif op == "BUY_PRODUCT" and item in BUYABLE_PRODUCTS:
                    price = market_price(item, int(inventory.get(item, 0)) - 1)
                    if cash < price or shed_total >= cap:
                        break
                    cash -= price
                    shed_total += 1
                    held[item] = held.get(item, 0) + 1
                    inventory[item] = int(inventory.get(item, 0)) - 1
                    flow[item] = flow.get(item, 0) - 1
                elif op == "BUY_SEED" and item in CROPS:
                    price = CROPS[item]["seed"]
                    if cash < price:
                        break
                    cash -= price
                elif op == "BUY_ANIMAL" and item in ANIMALS:
                    price = ANIMALS[item]["cost"]
                    if cash < price or shed_total >= cap:
                        break
                    cash -= price
                    shed_total += 1
                else:
                    break
        return flow

    def _shed_at_market(self, turn):
        """Our shed when the market runs: the observed shed plus what our own
        DROP / PICKUP / PLACE ops moved during the step (engine order: farmer,
        then each hand; the market runs after all unit actions)."""
        shed = {k: int(v or 0) for k, v in turn.shed.items()}
        cap = self.cfg["shed_capacity"]
        board = self.cfg.get("board_size", 10)
        bags = [dict(b) for b in turn.inventories]
        for idx, op in enumerate(turn.units):
            if not op or idx >= len(turn.positions):
                continue
            pos = turn.positions[idx]
            if not isinstance(pos, (list, tuple)) or len(pos) < 2:
                continue
            try:
                adjacent = _is_shed_adjacent((int(pos[0]), int(pos[1])), board)
            except Exception:
                adjacent = False
            if not adjacent:
                continue
            while len(bags) <= idx:
                bags.append({})
            bag = bags[idx]
            verb = op[0]
            if verb == "DROP":
                for item, n in list(bag.items()):
                    n = int(n or 0)
                    if n > 0:
                        take = min(n, max(0, cap - sum(shed.values())))
                        if take > 0:
                            shed[item] = shed.get(item, 0) + take
                    del bag[item]
            elif verb == "PICKUP" and len(op) >= 2:
                item = op[1]
                try:
                    n = int(op[2]) if len(op) >= 3 else 1
                except (TypeError, ValueError):
                    continue
                n = min(n, shed.get(item, 0))
                if n > 0:
                    shed[item] -= n
                    bag[item] = bag.get(item, 0) + n
            elif verb == "PLACE" and len(op) >= 2 and op[1] not in ANIMALS:
                item = op[1]
                try:
                    n = int(op[2]) if len(op) >= 3 else 1
                except (TypeError, ValueError):
                    continue
                n = min(n, int(bag.get(item, 0) or 0), max(0, cap - sum(shed.values())))
                if n > 0:
                    bag[item] -= n
                    shed[item] = shed.get(item, 0) + n
        return shed

    def _settle(self, obs, turn):
        """Observe the flow caused by `turn` and label it. Returns the label dict."""
        inventory_now = (obs.get("market") or {}).get("inventory") or {}
        drawn = town_draw(turn.step, turn.shops, self.cfg["shop_interval"], self.cfg["center_interval"])
        ours = self._our_executed(turn)
        total, opp = {}, {}
        for p in PRODUCTS:
            total[p] = int(inventory_now.get(p, 0)) - int(turn.inventory.get(p, 0)) + int(drawn.get(p, 0))
            opp[p] = total[p] - int(ours.get(p, 0))
        turn.opp_flow = opp
        if self.alignment == "next_action":
            return opp
        # legacy: the flow of `turn` is attributed by the orders of the step before it
        before = self.turns.get(turn.step - 1)
        ours_before = _requested(before.orders) if before is not None else {}
        opp_before = before.opp_flow if (before is not None and before.opp_flow) else {}
        label = {}
        for p in PRODUCTS:
            mine, theirs = ours_before.get(p, 0), opp_before.get(p, 0)
            if theirs and not mine:
                label[p] = total[p]
            elif theirs and mine:
                denom = abs(mine) + abs(theirs)
                label[p] = round(total[p] * abs(theirs) / denom) if denom else 0
            else:
                label[p] = 0
        return label

    # -- per-turn API ------------------------------------------------------------
    def observe(self, obs, configuration=None):
        """Ingest this turn's observation. Returns the forecast dict (see `forecast`)."""
        me = int(obs.get("player", 0) or 0)
        day = int(obs.get("day", 0) or 0)
        hour = int(obs.get("hour", 0) or 0)
        step = obs.get("step")
        turns_per_day = int((configuration or {}).get("turnsPerDay", 24) or 24)
        step = int(step) if step is not None else day * turns_per_day + hour
        if step == 0 or step <= self.step or self.cfg is None:
            self.reset()  # a new game (the evaluator re-imports per game; Kaggle runs one per process)
            self.cfg = config_intervals(configuration or {})
            self.cfg["hire_mult"] = int((configuration or {}).get("farmHandCostMult", 1) or 1)
            self.cfg["board_size"] = int((configuration or {}).get("boardSize", 10) or 10)

        # 1. finish row step-1 with the opponent supply executed during it. A gap
        #    (a turn we never saw) leaves zero rows behind and skips the accounting.
        last = self.turns.get(step - 1)
        if last is not None:
            supply = self._settle(obs, last)
            y = np.array([supply.get(p, 0) for p in PRODUCTS], dtype=np.int32)
            self.supply[step - 1] = y
            self.rows[step - 1, N_FEATURES:] = np.log1p(np.clip(y, 0, None)).astype(np.float32)
            self.history.update(step - 1, supply)
        self.n_complete = step

        # 2. this turn's features (history only knows sells up to step-1, as in training)
        o = dict(obs)
        o["step"] = step
        o["private"] = obs.get("private") or {}
        feats = build_features(o, me, self.cfg, self.history, self.cfg["shed_capacity"])
        x = np.array([feats[k] for k in FEATURE_NAMES], dtype=np.float32)
        if self.model is not None:
            x = (x - self.model.mean) / self.model.std
        self.rows[step, :N_FEATURES] = x
        self.step = step

        market = obs.get("market") or {}
        farms = obs.get("farms") or []
        farm = farms[me] if me < len(farms) else {}
        self.turns[step] = _Turn(
            step,
            dict(market.get("inventory") or {}),
            list((obs.get("town") or {}).get("unlocked_shops") or []),
            dict(o["private"].get("shed") or {}),
            float(farm.get("money", 0.0) or 0.0),
            int(farm.get("hires_today", 0) or 0),
            max(1, len(farm.get("unlocked_quadrants") or ())),
            [farm.get("farmer") or [0, 0]] + list(farm.get("hands") or []),
            [dict(b or {}) for b in (o["private"].get("inventories") or [])],
        )
        for old in [k for k in self.turns if k < step - 3]:
            del self.turns[old]

        # 3. forecast for steps step .. step+95 (from step = context on: day 21 at 512, day 10 at 240)
        self.pred = None
        if self.model is not None and self.n_complete >= self.min_context:
            self.pred = self.model.predict(self.context_block())
            self.pred_step = step
        return self.forecast()

    def record_action(self, action):
        """Remember what was actually submitted this turn (market orders and unit ops)."""
        turn = self.turns.get(self.step)
        if turn is None:
            return
        turn.orders = [list(o) for o in _market_orders(action)]
        units = [action.get("farmer")] if isinstance(action, dict) else [None]
        hands = action.get("hands") if isinstance(action, dict) else None
        units += list(hands) if isinstance(hands, (list, tuple)) else []
        turn.units = [list(u) if isinstance(u, (list, tuple)) else [str(u)] if u else ["PASS"] for u in units]

    def context_block(self):
        """Rows [step-context, step) -- left-padded with zeros (= feature means) early in the game."""
        end = self.n_complete
        start = end - self.context
        if start >= 0:
            return self.rows[start:end]
        block = np.zeros((self.context, N_CHANNELS), dtype=np.float32)
        if end > 0:
            block[-end:] = self.rows[:end]
        return block

    def forecast(self):
        """Policy-facing view of the latest forecast; `None` while there is no forecast."""
        if self.pred is None:
            return None
        units = np.expm1(np.clip(self.pred, 0.0, None))
        out = {"step": self.pred_step, "log": self.pred, "units": units, "products": list(PRODUCTS)}
        for k in (1, 4, 24, 96):
            out[f"units_{k}"] = {p: float(units[:k, i].sum()) for p, i in PRODUCT_INDEX.items()}
            out[f"score_{k}"] = {p: float(self.pred[:k, i].max()) for p, i in PRODUCT_INDEX.items()}
        return out
