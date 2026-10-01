"""KAD-HP-1: a structured multi-head policy for Kaggriculture that plays one turn at a time.

It trains on the replays KAD-MD-24 was encoded from (data.py: X[t] = observation t, A[t] = the
action taken from it), unchanged. KAD-MD-24 guessed 24 turns x 129 fields, each field on its own,
from a context that averages every 4 observations; in games it never planted, watered or
harvested. This model reads the current turn exactly, as its parts, and answers with a head per
kind of decision.

Inputs (inputs_at, from one file's observation rows):
  globals  the scalars of this turn and of the HISTORY turns before it: step, day, hour, shops,
           market prices and stock, both farms' money, hires, hands and quadrants, own seeds, shed
  tiles    own farm, one token per tile (kind, crop, animal, water, feed and care state, ...)
  rows     the opponent's farm, one token per row of 10 tiles
  units    own farmer (slot 0) and hands (1..32): present, position, what they carry

Heads, for every unit present:
  job      its next job: the next command that is not a move or a pass (JOBS: plant a crop, water,
           harvest, pick up an item at the shed, ...), with the amount for PICKUP and PLACE
  target   the tile where it will do that job, a pointer over the tile tokens (every job acts on
           the tile the unit stands on; shed jobs on a tile next to the shed)
  now      what it does this turn, given its job and target: a move, a pass, or the job (NOW)
and the market head: up to 10 orders in order (each sees the ones before it), the order (ORDERS:
buy seeds, animals, wheat or fertilizer, sell a product, hire, buy land) and then its amount.

policy_labels reads the labels from a replay: a unit's job and target at turn t are those of the
first job command at or after t while the same unit is on the farm (present, and moving at most
one tile a turn, which tells a hand from the one that takes its slot).

In play (Policy.act) every head takes its most likely choice, or with a temperature a draw from its
distribution (the variety expert iteration needs), and fit_plants can keep the PLANTs of a turn within
the seeds the seat holds: the engine turns every PLANT of a crop into a pass when more are asked for
than there are seeds.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from data import (ARGUMENTS, BOARD_SIZE, COMMAND_SLOTS, CROPS, FEATURE_NAMES, ITEM_TO_ID, ITEMS, MARKET_OPS,
                  MAX_HANDS, OP_TO_ID, OPERATIONS, QUANTITY_BINS, QUANTITY_OPS, STORAGE_ITEMS, TILE_FEATURE_NAMES,
                  UNIT_OPS, WindowDataset)

MOVES = ('NORTH', 'SOUTH', 'EAST', 'WEST')
NOW = MOVES + ('PASS', 'DO')                # DO: the unit does its job this turn
UNITS = MAX_HANDS + 1                        # the farmer and up to 32 hands
TILES = BOARD_SIZE * BOARD_SIZE
MARKET_SLOTS = COMMAND_SLOTS - UNITS         # 10
HISTORY = 8


def _classes(ops):
    """('NONE', None), then (op, item) for each op: one class per item it takes, else one."""
    classes = [('NONE', None)]
    for op in ops:
        classes += [(op, item) for item in ARGUMENTS[op]] if op in ARGUMENTS else [(op, None)]
    return tuple(classes)


JOB_OPS = tuple(op for op in UNIT_OPS if op not in MOVES + ('PASS',))
JOBS = _classes(JOB_OPS)          # 40: NONE = no job ahead
ORDERS = _classes(MARKET_OPS)     # 22: NONE = an empty order slot


def _table(classes, empty):
    """int64 [operation id, item id] -> class (the codec's ids); `empty` for anything else."""
    table = np.full((len(OPERATIONS), len(ITEMS)), empty, dtype=np.int64)
    for k, (op, item) in enumerate(classes):
        if op != 'NONE':
            table[OP_TO_ID[op], ITEM_TO_ID[item] if item else 0] = k
    return table


JOB_TABLE = _table(JOBS, -1)       # -1: not a job (NONE, a move, PASS)
ORDER_TABLE = _table(ORDERS, 0)    # 0: NONE
JOB_AMOUNT = np.array([op in QUANTITY_OPS for op, _ in JOBS])
ORDER_AMOUNT = np.array([op in QUANTITY_OPS for op, _ in ORDERS])
NOW_TABLE = np.full(len(OPERATIONS), NOW.index('PASS'), dtype=np.int64)   # NONE is a pass
for _op in UNIT_OPS:
    NOW_TABLE[OP_TO_ID[_op]] = NOW.index(_op) if _op in NOW else NOW.index('DO')
# the codec triple (operation, item, amount) of each class, for comparing and decoding
JOB_CODE = np.array([(OP_TO_ID[op], ITEM_TO_ID[item] if item else 0) for op, item in JOBS])
ORDER_CODE = np.array([(OP_TO_ID[op], ITEM_TO_ID[item] if item else 0) for op, item in ORDERS])
NOW_CODE = np.array([OP_TO_ID[op] if op != 'DO' else -1 for op in NOW])
PLANT_JOBS = np.array([JOBS.index(('PLANT', crop)) for crop in CROPS])   # the job class of each crop's PLANT

# Where the parts of an observation row (data.encode_observation) are.
_INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}
GLOBAL = np.array([i for i, name in enumerate(FEATURE_NAMES)
                   if not name.startswith(('own.unit.', 'opponent.unit.', 'own.tile.', 'opponent.tile.',
                                           'own.inventory.'))])
OWN_TILES = np.array([[_INDEX[f'own.tile.{y}.{x}.{k}'] for k in TILE_FEATURE_NAMES]
                      for y in range(BOARD_SIZE) for x in range(BOARD_SIZE)])
OPPONENT_ROWS = np.array([[_INDEX[f'opponent.tile.{y}.{x}.{k}'] for x in range(BOARD_SIZE) for k in TILE_FEATURE_NAMES]
                          for y in range(BOARD_SIZE)])
UNIT_FEATURES = np.array([[_INDEX[f'own.unit.{u}.{k}'] for k in ('exists', 'x', 'y')]
                          + [_INDEX[f'own.inventory.{u}.{item}'] for item in STORAGE_ITEMS] for u in range(UNITS)])
POSITION_SCALE = BOARD_SIZE - 1   # data.SCALES['position']: x and y are stored / 9


def unit_tiles(rows):
    """int64 [..., UNITS]: the tile (y * 10 + x) each unit stands on, from observation rows."""
    x = np.rint(rows[..., UNIT_FEATURES[:, 1]] * POSITION_SCALE).astype(np.int64).clip(0, BOARD_SIZE - 1)
    y = np.rint(rows[..., UNIT_FEATURES[:, 2]] * POSITION_SCALE).astype(np.int64).clip(0, BOARD_SIZE - 1)
    return y * BOARD_SIZE + x


def policy_labels(X, A):
    """int64 [T, UNITS * 5 + MARKET_SLOTS * 2 + 129] of one file (a seat of a game): per unit its
    job, the job's amount, its target tile, what it does now and whether it is present; per market
    slot the order and its amount; then the action itself (A[t], to score whole commands). A unit
    with no job ahead has job NONE and its own tile as target."""
    T = len(X)
    command = A.reshape(T, COMMAND_SLOTS, 3)
    unit = command[:, :UNITS]
    present = X[:, UNIT_FEATURES[:, 0]] > 0.5
    tile = unit_tiles(X)
    x, y = tile % BOARD_SIZE, tile // BOARD_SIZE
    job = JOB_TABLE[unit[..., 0], unit[..., 1]]
    same = np.zeros((T, UNITS), dtype=bool)   # the same unit from turn t to t + 1
    same[:-1] = present[:-1] & present[1:] & (np.abs(x[1:] - x[:-1]) + np.abs(y[1:] - y[:-1]) <= 1)
    following = np.full((T, UNITS), -1, dtype=np.int64)   # the turn of each unit's next job
    ahead = np.full(UNITS, -1, dtype=np.int64)
    for t in range(T - 1, -1, -1):
        ahead = np.where(job[t] >= 0, t, np.where(same[t], ahead, -1))
        following[t] = ahead
    has = following >= 0
    turn = np.where(has, following, np.arange(T)[:, None])
    slot = np.arange(UNITS)[None, :]
    next_job = np.where(has, job[turn, slot], 0)
    amount = np.where(has & JOB_AMOUNT[next_job], unit[turn, slot, 2], 0)
    target = tile[turn, slot]
    now = NOW_TABLE[unit[..., 0]]
    units = np.stack([next_job, amount, target, now, present.astype(np.int64)], -1)
    market = command[:, UNITS:]
    order = ORDER_TABLE[market[..., 0], market[..., 1]]
    orders = np.stack([order, np.where(ORDER_AMOUNT[order], market[..., 2], 0)], -1)
    return np.concatenate([units.reshape(T, -1), orders.reshape(T, -1), A.reshape(T, -1)], 1)


def inputs_at(X, t, history=HISTORY):
    """The model's inputs at turn t of one seat, from its observation rows X[:t + 1] (a replay
    file's, or the ones a live player has seen)."""
    back = t - np.arange(history + 1)            # this turn, then the ones before it
    seen = back >= 0
    globals_ = np.zeros((history + 1, len(GLOBAL)), dtype=np.float32)
    globals_[seen] = X[back[seen]][:, GLOBAL]
    row = X[t]
    return {'globals': globals_, 'globals_mask': seen, 'tiles': row[OWN_TILES], 'rows': row[OPPONENT_ROWS],
            'units': row[UNIT_FEATURES], 'units_mask': row[UNIT_FEATURES[:, 0]] > 0.5, 'unit_tile': unit_tiles(row)}


class PolicyWindows(WindowDataset):
    """WindowDataset's files, anchors, splits and weights, with one turn per sample: the model's
    inputs at the anchor and its labels (policy_labels, computed once per file read)."""

    def __init__(self, directory, split, stride=7, history=HISTORY, **kwargs):
        super().__init__(directory, split, stride=stride, **kwargs)
        self.history = history

    def read(self, index):
        X, A = super().read(index)
        return X, policy_labels(X, A)

    def window(self, X, labels, t):
        sample = inputs_at(X, t, self.history)
        row = labels[t]
        sample['unit_labels'] = row[:UNITS * 5].reshape(UNITS, 5)
        sample['market_labels'] = row[UNITS * 5:UNITS * 5 + MARKET_SLOTS * 2].reshape(MARKET_SLOTS, 2)
        sample['command'] = row[UNITS * 5 + MARKET_SLOTS * 2:].reshape(COMMAND_SLOTS, 3)
        sample['anchor'] = np.int64(t)
        return sample


@dataclass
class PolicyConfig:
    width: int = 384
    layers: int = 8
    heads: int = 6
    dropout: float = 0.1
    history: int = HISTORY
    market_layers: int = 2
    condition_dim: int = 0   # len(model.CONDITIONS) when the game's conditions are an input
    teams: int = 0           # size of the team vocabulary (0: no team input; id 0 = unknown team)


def _head(width, classes):
    return nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, classes))


class Policy(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = c = config
        w = c.width
        self.globals_in = nn.Linear(len(GLOBAL), w)
        self.tiles_in = nn.Linear(OWN_TILES.shape[1], w)
        self.rows_in = nn.Linear(OPPONENT_ROWS.shape[1], w)
        self.units_in = nn.Linear(UNIT_FEATURES.shape[1], w)
        self.age = nn.Embedding(c.history + 1, w)     # 0: this turn
        self.place = nn.Embedding(TILES, w)            # a tile, and a unit standing on it
        self.row = nn.Embedding(BOARD_SIZE, w)
        self.slot = nn.Embedding(UNITS, w)             # 0: the farmer, 1..32: hands
        self.kind = nn.Embedding(4, w)                 # globals, tiles, rows, units
        if c.condition_dim:
            self.condition = nn.Sequential(nn.Linear(2 * c.condition_dim, w), nn.SiLU(), nn.Linear(w, w))
            nn.init.zeros_(self.condition[2].weight)
            nn.init.zeros_(self.condition[2].bias)
        if c.teams:
            self.team = nn.Embedding(c.teams + 1, w)
            nn.init.normal_(self.team.weight, std=.02)
        layer = nn.TransformerEncoderLayer(w, c.heads, 4 * w, c.dropout, activation='gelu', batch_first=True,
                                           norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, c.layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(w)
        self.job = _head(w, len(JOBS))
        self.job_embed = nn.Embedding(len(JOBS), w)
        self.amount = _head(w, QUANTITY_BINS)
        self.target_query = nn.Linear(w, w)
        self.target_key = nn.Linear(w, w)
        self.target_embed = nn.Linear(w, w)
        self.now = _head(w, len(NOW))
        self.slots = nn.Parameter(torch.randn(MARKET_SLOTS, w) * .02)
        self.order_embed = nn.Embedding(len(ORDERS) + 1, w)   # the last one starts the list
        self.amount_embed = nn.Embedding(QUANTITY_BINS, w)
        market = nn.TransformerDecoderLayer(w, c.heads, 4 * w, c.dropout, activation='gelu', batch_first=True,
                                            norm_first=True)
        self.market = nn.TransformerDecoder(market, c.market_layers)
        self.order = _head(w, len(ORDERS))
        self.order_amount = _head(w, QUANTITY_BINS)
        self.register_buffer('job_amount', torch.as_tensor(JOB_AMOUNT))
        self.register_buffer('order_amount_used', torch.as_tensor(ORDER_AMOUNT))

    def config_dict(self):
        return asdict(self.config)

    def encode(self, batch, condition=None, team=None):
        """Token states (batch, tokens, width) and the padding mask: globals, tiles, rows, units."""
        c, kind = self.config, self.kind.weight
        tokens = torch.cat([
            self.globals_in(batch['globals'].float()) + self.age.weight + kind[0],
            self.tiles_in(batch['tiles'].float()) + self.place.weight + kind[1],
            self.rows_in(batch['rows'].float()) + self.row.weight + kind[2],
            self.units_in(batch['units'].float()) + self.slot.weight + self.place(batch['unit_tile'].long()) + kind[3],
        ], 1)
        if c.condition_dim:
            if condition is None:
                condition = torch.zeros(len(tokens), 2 * c.condition_dim, device=tokens.device)
            tokens = tokens + self.condition(condition.float())[:, None]
        if c.teams:
            if team is None:
                team = torch.zeros(len(tokens), dtype=torch.long, device=tokens.device)
            tokens = tokens + self.team(team.long())[:, None]
        pad = torch.cat([~batch['globals_mask'].bool(),
                         torch.zeros(len(tokens), TILES + BOARD_SIZE, dtype=torch.bool, device=tokens.device),
                         ~batch['units_mask'].bool()], 1)
        return self.norm(self.encoder(tokens, src_key_padding_mask=pad)), pad

    def split(self, h):
        first = self.config.history + 1
        return h[:, first:first + TILES], h[:, -UNITS:]

    def unit_heads(self, h, job, target):
        """Logits of job, amount, target and now for every unit, the last three given each unit's
        job and target (the true ones in training, the chosen ones in play)."""
        tiles, units = self.split(h)
        job_logits = self.job(units)
        z = units + self.job_embed(job)
        target_logits = torch.einsum('buw,btw->but', self.target_query(z), self.target_key(tiles))
        target_logits = target_logits / math.sqrt(self.config.width)
        at = torch.gather(tiles, 1, target[..., None].expand(-1, -1, tiles.shape[-1]))
        return job_logits, self.amount(z), target_logits, self.now(z + self.target_embed(at))

    def market_state(self, h, pad, orders, amounts):
        """The market decoder's state of each order slot, given the orders and amounts before it (so
        a slot's state does not depend on its own order or anything after it)."""
        start = torch.full((len(h), 1), len(ORDERS), dtype=torch.long, device=h.device)
        before = torch.cat([start, orders[:, :-1]], 1)
        before_amounts = torch.cat([torch.zeros_like(start), amounts[:, :-1]], 1)
        x = self.slots + self.order_embed(before) + self.amount_embed(before_amounts)
        causal = torch.triu(torch.ones(MARKET_SLOTS, MARKET_SLOTS, dtype=torch.bool, device=h.device), 1)
        return self.market(x, h, tgt_mask=causal, memory_key_padding_mask=pad)

    def market_heads(self, h, pad, orders, amounts):
        """Logits of each order slot given the ones before it, and of its amount given its order."""
        d = self.market_state(h, pad, orders, amounts)
        return self.order(d), self.order_amount(d + self.order_embed(orders))

    def forward(self, batch, condition=None, team=None):
        """Training: every head's logits, teacher-forced on the labels in `batch`."""
        h, pad = self.encode(batch, condition, team)
        units, market = batch['unit_labels'].long(), batch['market_labels'].long()
        return self.unit_heads(h, units[..., 0], units[..., 2]) + self.market_heads(h, pad, market[..., 0], market[..., 1])

    @torch.no_grad()
    def act(self, batch, condition=None, team=None, temperature=0.0, generator=None, seeds=None, empty=None):
        """Play: every head's choice, each given the ones it depends on: the most likely one, or with
        temperature > 0 a draw from the head's distribution at that temperature (noise from
        `generator`, a CPU torch.Generator, so a game replays exactly). With `seeds` (batch, crops) and
        `empty` (batch, tiles), the PLANTs done this turn are fitted to them (fit_plants). Returns per
        unit (job, amount, target, now) and per market slot (order, amount), (batch, ...) long."""
        def pick(logits):
            return _pick(logits, temperature, generator)

        h, pad = self.encode(batch, condition, team)
        tiles, units = self.split(h)
        job_logits = self.job(units)
        job = pick(job_logits)
        z = units + self.job_embed(job)
        amount = pick(self.amount(z))
        target_logits = torch.einsum('buw,btw->but', self.target_query(z), self.target_key(tiles))
        target = pick(target_logits / math.sqrt(self.config.width))   # the scale unit_heads trains
        target = torch.where(job > 0, target, batch['unit_tile'].long())
        at = torch.gather(tiles, 1, target[..., None].expand(-1, -1, tiles.shape[-1]))
        now = pick(self.now(z + self.target_embed(at)))
        if seeds is not None:
            job, now = fit_plants(job, job_logits, now, batch['unit_tile'], batch['units_mask'], seeds, empty)
        orders = torch.zeros(len(h), MARKET_SLOTS, dtype=torch.long, device=h.device)
        amounts = torch.zeros_like(orders)
        for k in range(MARKET_SLOTS):
            # one decoder pass per slot, not two: slot k's state does not depend on its own order
            # (market_state); the heads see every slot, as in training, so the choices are the same bits
            d = self.market_state(h, pad, orders, amounts)
            orders[:, k] = pick(self.order(d)[:, k])
            amounts[:, k] = pick(self.order_amount(d + self.order_embed(orders))[:, k])
        return job, amount, target, now, orders, amounts


def _pick(logits, temperature=0.0, generator=None):
    """The most likely class of each row, or with temperature > 0 a draw from softmax(logits /
    temperature) (Gumbel-max; the uniform noise comes from `generator` on the CPU)."""
    if temperature <= 0:
        return logits.argmax(-1)
    u = torch.rand(logits.shape, generator=generator).clamp_(1e-10, 1 - 1e-7).to(logits.device)
    return (logits.float() / temperature - torch.log(-torch.log(u))).argmax(-1)


def fit_plants(job, job_logits, now, unit_tile, present, seeds, empty=None):
    """The (job, now) of every unit with the PLANTs done this turn (now == DO) fitted to what the
    engine carries out: it turns every PLANT of a crop into a pass when the seat's units ask for more
    of it in a turn than the seat has seeds, and a PLANT on a tile that is not empty does nothing. In
    the engine's unit order (the farmer, then the hands) a unit keeps its crop while that crop's seeds
    last, else plants the crop with seeds left that its job head rates highest; it passes when no crop
    has seeds left, or when its tile is not empty (`empty`, tiles y * 10 + x; None: not checked) or
    another unit plants there this turn. Shapes: job, now, unit_tile (batch, UNITS); job_logits
    (batch, UNITS, len(JOBS)); present (batch, UNITS); seeds (batch, len(CROPS)); empty (batch, TILES)."""
    device = job.device
    do, passes = NOW.index('DO'), NOW.index('PASS')
    crop_of = {int(k): c for c, k in enumerate(PLANT_JOBS)}
    logits = job_logits[..., torch.as_tensor(PLANT_JOBS, device=job_logits.device)].float().cpu().tolist()
    jobs, nows = job.cpu().tolist(), now.cpu().tolist()   # one read from the device, not one per unit
    tiles, here = unit_tile.cpu().tolist(), present.bool().cpu().tolist()
    seeds = torch.as_tensor(seeds).tolist()
    empty = None if empty is None else torch.as_tensor(empty).bool().tolist()
    for b in range(len(jobs)):
        left, free = [int(n) for n in seeds[b]], None if empty is None else list(empty[b])
        for u in range(len(jobs[b])):
            if not here[b][u] or nows[b][u] != do or jobs[b][u] not in crop_of:
                continue
            crop = crop_of[jobs[b][u]]
            if left[crop] <= 0:
                options = [c for c in range(len(CROPS)) if left[c] > 0]
                crop = max(options, key=lambda c: logits[b][u][c]) if options else None
            if crop is None or (free is not None and not free[tiles[b][u]]):
                nows[b][u] = passes
                continue
            jobs[b][u] = int(PLANT_JOBS[crop])
            left[crop] -= 1
            if free is not None:
                free[tiles[b][u]] = False
    return (torch.tensor(jobs, dtype=job.dtype, device=device), torch.tensor(nows, dtype=now.dtype, device=device))


def _ce(logits, labels, keep):
    """Mean cross-entropy over the kept items; 0 (in the graph) when none is kept. A masked mean over
    every item, so the shapes stay fixed (XLA compiles one graph) and nothing waits on the device."""
    keep = keep.float()
    ce = F.cross_entropy(logits.float().flatten(0, -2), (labels * keep.long()).flatten(), reduction='none')
    return (ce * keep.flatten()).sum() / keep.sum().clamp(min=1.)


def policy_loss(model, outputs, batch):
    """The sum of the heads' mean cross-entropies, and each of them (detached tensors: reading them
    waits for the device, so the training loop does it only when it logs)."""
    job_logits, amount_logits, target_logits, now_logits, order_logits, order_amount_logits = outputs
    units, market = batch['unit_labels'].long(), batch['market_labels'].long()
    job, amount, target, now, present = units.unbind(-1)
    present = present.bool()
    order, order_amount = market.unbind(-1)
    parts = {'job': _ce(job_logits, job, present),
             'amount': _ce(amount_logits, amount, present & model.job_amount[job]),
             'target': _ce(target_logits, target, present & (job > 0)),
             'now': _ce(now_logits, now, present),
             'order': _ce(order_logits, order, torch.ones_like(order, dtype=torch.bool)),
             'order_amount': _ce(order_amount_logits, order_amount, model.order_amount_used[order])}
    return sum(parts.values()), {k: v.detach() for k, v in parts.items()}


def commands(job, amount, now, orders, amounts):
    """The codec triples (operation, item, amount) the chosen heads amount to: int64 (batch, 43, 3),
    slots as data.encode_action (farmer, 32 hands, 10 market orders)."""
    job_code = torch.as_tensor(JOB_CODE, device=job.device)
    order_code = torch.as_tensor(ORDER_CODE, device=job.device)
    now_code = torch.as_tensor(NOW_CODE, device=job.device)
    do = now == NOW.index('DO')
    unit = torch.stack([torch.where(do, job_code[job, 0], now_code[now]),
                        torch.where(do, job_code[job, 1], 0),
                        torch.where(do & torch.as_tensor(JOB_AMOUNT, device=job.device)[job], amount, 0)], -1)
    market = torch.stack([order_code[orders, 0], order_code[orders, 1],
                          torch.where(torch.as_tensor(ORDER_AMOUNT, device=job.device)[orders], amounts, 0)], -1)
    return torch.cat([unit, market], 1)
