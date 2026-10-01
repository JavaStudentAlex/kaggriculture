"""KAD-HP-1 as a copilot of the graph runtime: its forward pass in numpy, because a Kaggle agent gets no GPU and torch
cannot be relied on in the agent sandbox (the first oracle submission timed out importing it).

The model is research/action_diffusion/policy.py's Policy, trained on Kaggle's published top-player replays; the
weights come from export_kad.py (float16 in weights.npz, used as float32), with the prompt it plays with (the day's best
team, a win) folded into one token bias. kad_data.py is research/action_diffusion/data.py verbatim: the observation
encoder the model was trained on.

KadAdvisor keeps our encoded observations of the game (the model reads this turn and the 8 before it) and, when asked,
runs the model once with every head at its most likely choice (Policy.act at temperature 0) and returns what a top
player would do in our place: the market orders slot by slot with the probability of each order class at each slot
(buy land, seeds, animals, ...), and our units' next jobs. Nothing here acts: the stages that use it decide.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from . import kad_data as D

MOVES = ('NORTH', 'SOUTH', 'EAST', 'WEST')
NOW = MOVES + ('PASS', 'DO')
UNITS = D.MAX_HANDS + 1                  # the farmer and up to 32 hands
TILES = D.BOARD_SIZE * D.BOARD_SIZE
MARKET_SLOTS = D.COMMAND_SLOTS - UNITS   # 10
HISTORY = 8
PUBLIC = ('farms', 'market', 'town', 'day', 'hour')   # the view research/action_diffusion/play.py encodes
WEIGHTS = Path(__file__).resolve().parent / 'weights.npz'


def _classes(ops):
    classes = [('NONE', None)]
    for op in ops:
        classes += [(op, item) for item in D.ARGUMENTS[op]] if op in D.ARGUMENTS else [(op, None)]
    return tuple(classes)


JOB_OPS = tuple(op for op in D.UNIT_OPS if op not in MOVES + ('PASS',))
JOBS = _classes(JOB_OPS)           # 40: NONE = no job ahead
ORDERS = _classes(D.MARKET_OPS)    # 22: NONE = an empty order slot
_INDEX = {name: i for i, name in enumerate(D.FEATURE_NAMES)}
GLOBAL = np.array([i for i, name in enumerate(D.FEATURE_NAMES)
                   if not name.startswith(('own.unit.', 'opponent.unit.', 'own.tile.', 'opponent.tile.',
                                           'own.inventory.'))])
OWN_TILES = np.array([[_INDEX[f'own.tile.{y}.{x}.{k}'] for k in D.TILE_FEATURE_NAMES]
                      for y in range(D.BOARD_SIZE) for x in range(D.BOARD_SIZE)])
OPPONENT_ROWS = np.array([[_INDEX[f'opponent.tile.{y}.{x}.{k}'] for x in range(D.BOARD_SIZE)
                           for k in D.TILE_FEATURE_NAMES] for y in range(D.BOARD_SIZE)])
UNIT_FEATURES = np.array([[_INDEX[f'own.unit.{u}.{k}'] for k in ('exists', 'x', 'y')]
                          + [_INDEX[f'own.inventory.{u}.{item}'] for item in D.STORAGE_ITEMS] for u in range(UNITS)])
POSITION_SCALE = D.BOARD_SIZE - 1


def unit_tiles(rows):
    x = np.rint(rows[..., UNIT_FEATURES[:, 1]] * POSITION_SCALE).astype(np.int64).clip(0, D.BOARD_SIZE - 1)
    y = np.rint(rows[..., UNIT_FEATURES[:, 2]] * POSITION_SCALE).astype(np.int64).clip(0, D.BOARD_SIZE - 1)
    return y * D.BOARD_SIZE + x


def inputs_at(X, t, history=HISTORY):
    """policy.inputs_at: the model's inputs at turn t from the observation rows X[:t + 1]."""
    back = t - np.arange(history + 1)
    seen = back >= 0
    globals_ = np.zeros((history + 1, len(GLOBAL)), dtype=np.float32)
    globals_[seen] = X[back[seen]][:, GLOBAL]
    row = X[t]
    return {'globals': globals_, 'globals_mask': seen, 'tiles': row[OWN_TILES], 'rows': row[OPPONENT_ROWS],
            'units': row[UNIT_FEATURES], 'units_mask': row[UNIT_FEATURES[:, 0]] > 0.5, 'unit_tile': unit_tiles(row)}


def _erf(x):
    """Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7): numpy has no erf, and scipy may be missing."""
    a = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * a)
    y = 1.0 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t \
        * np.exp(-a * a)
    return np.sign(x) * y


def gelu(x):
    """torch's exact (erf) GELU, the activation of every layer and head of the model."""
    return 0.5 * x * (1.0 + _erf(x * (1.0 / math.sqrt(2.0))))


def layer_norm(x, w, b, eps=1e-5):
    mean = x.mean(-1, keepdims=True)
    var = ((x - mean) ** 2).mean(-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * w + b


def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


class KadModel:
    """Policy's inference in numpy (one position; eval mode, so no dropout)."""

    def __init__(self, params, config, token_bias):
        self.p = params
        self.width, self.layers, self.heads = int(config['width']), int(config['layers']), int(config['heads'])
        self.market_layers, self.history = int(config.get('market_layers', 2)), int(config.get('history', HISTORY))
        self.token_bias = token_bias
        self.causal = np.triu(np.ones((MARKET_SLOTS, MARKET_SLOTS), dtype=bool), 1)

    @classmethod
    def load(cls, path=WEIGHTS):
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z['__meta__']))
            params = {k: z[k].astype(np.float32) for k in z.files if not k.startswith('__')}
        return cls(params, meta['config'], params.pop('token_bias')), meta

    def _lin(self, x, name):
        return x @ self.p[name + '.weight'].T + self.p[name + '.bias']

    def _ln(self, x, name):
        return layer_norm(x, self.p[name + '.weight'], self.p[name + '.bias'])

    def _head(self, x, name):
        return self._lin(gelu(self._lin(self._ln(x, name + '.0'), name + '.1')), name + '.3')

    def _kv(self, x, name):
        w = self.width
        W, b = self.p[name + '.in_proj_weight'], self.p[name + '.in_proj_bias']
        return x @ W[w:2 * w].T + b[w:2 * w], x @ W[2 * w:].T + b[2 * w:]

    def _attend(self, x, name, kv, key_pad=None, causal=None):
        w, h = self.width, self.heads
        d = w // h
        q = x @ self.p[name + '.in_proj_weight'][:w].T + self.p[name + '.in_proj_bias'][:w]
        k, v = kv
        q = q.reshape(len(q), h, d).transpose(1, 0, 2)
        k = k.reshape(len(k), h, d).transpose(1, 0, 2)
        v = v.reshape(len(v), h, d).transpose(1, 0, 2)
        s = q @ k.transpose(0, 2, 1) / math.sqrt(d)
        if key_pad is not None:
            s = np.where(key_pad[None, None, :], -np.inf, s)
        if causal is not None:
            s = np.where(causal[None], -np.inf, s)
        o = (softmax(s) @ v).transpose(1, 0, 2).reshape(len(x), w)
        return self._lin(o, name + '.out_proj')

    def encode(self, batch):
        p = self.p
        kind = p['kind.weight']
        tokens = np.concatenate([
            self._lin(batch['globals'], 'globals_in') + p['age.weight'] + kind[0],
            self._lin(batch['tiles'], 'tiles_in') + p['place.weight'] + kind[1],
            self._lin(batch['rows'], 'rows_in') + p['row.weight'] + kind[2],
            self._lin(batch['units'], 'units_in') + p['slot.weight'] + p['place.weight'][batch['unit_tile']] + kind[3],
        ], 0) + self.token_bias
        pad = np.concatenate([~batch['globals_mask'].astype(bool), np.zeros(TILES + D.BOARD_SIZE, dtype=bool),
                              ~batch['units_mask'].astype(bool)])
        x = tokens.astype(np.float32)
        for i in range(self.layers):
            name = f'encoder.layers.{i}'
            y = self._ln(x, name + '.norm1')
            x = x + self._attend(y, name + '.self_attn', self._kv(y, name + '.self_attn'), key_pad=pad)
            y = self._ln(x, name + '.norm2')
            x = x + self._lin(gelu(self._lin(y, name + '.linear1')), name + '.linear2')
        return self._ln(x, 'norm'), pad

    def market(self, h, pad):
        """Greedy decoding of the 10 order slots (Policy.act at temperature 0): the orders, their amount bins and
        each slot's order-class probabilities (given the orders chosen before it)."""
        p = self.p
        memory = [self._kv(h, f'market.layers.{i}.multihead_attn') for i in range(self.market_layers)]
        orders = np.zeros(MARKET_SLOTS, dtype=np.int64)
        amounts = np.zeros(MARKET_SLOTS, dtype=np.int64)
        probs = np.zeros((MARKET_SLOTS, len(ORDERS)), dtype=np.float32)
        for k in range(MARKET_SLOTS):
            before = np.concatenate([[len(ORDERS)], orders[:-1]])
            before_amounts = np.concatenate([[0], amounts[:-1]])
            x = p['slots'] + p['order_embed.weight'][before] + p['amount_embed.weight'][before_amounts]
            for i in range(self.market_layers):
                name = f'market.layers.{i}'
                y = self._ln(x, name + '.norm1')
                x = x + self._attend(y, name + '.self_attn', self._kv(y, name + '.self_attn'), causal=self.causal)
                y = self._ln(x, name + '.norm2')
                x = x + self._attend(y, name + '.multihead_attn', memory[i], key_pad=pad)
                y = self._ln(x, name + '.norm3')
                x = x + self._lin(gelu(self._lin(y, name + '.linear1')), name + '.linear2')
            logits = self._head(x[k], 'order')
            probs[k] = softmax(logits)
            orders[k] = int(np.argmax(logits))
            amounts[k] = int(np.argmax(self._head(x[k] + p['order_embed.weight'][orders[k]], 'order_amount')))
        return orders, amounts, probs

    def jobs(self, h):
        """Each unit's job-class probabilities (the job head; its choice does not depend on the other heads)."""
        units = h[-UNITS:]
        return softmax(self._head(units, 'job'))

    def units_act(self, h, batch):
        """Policy.act's unit heads at temperature 0: per unit (all 33 slots) the job, its target tile and what the
        unit does now (NOW), with the probability of the chosen job and of the chosen `now`."""
        p, w = self.p, self.width
        tiles, units = h[self.history + 1:self.history + 1 + TILES], h[-UNITS:]
        job_probs = softmax(self._head(units, 'job'))
        job = job_probs.argmax(-1)
        z = units + p['job_embed.weight'][job]
        target_logits = self._lin(z, 'target_query') @ self._lin(tiles, 'target_key').T / math.sqrt(w)
        target = np.where(job > 0, target_logits.argmax(-1), batch['unit_tile'])
        now_probs = softmax(self._head(z + self._lin(tiles[target], 'target_embed'), 'now'))
        now = now_probs.argmax(-1)
        rows = np.arange(len(job))
        return {'job': job, 'job_p': job_probs[rows, job], 'target': target, 'now': now,
                'now_p': now_probs[rows, now], 'job_probs': job_probs}


class KadAdvisor:
    """One game's copilot: `see` every observation (cheap), `advise` when a decision is due (one forward pass)."""

    def __init__(self, path=WEIGHTS):
        self.path = Path(path)
        self.model = None
        self.meta = None
        self.rows, self.step, self.seat = [], None, None
        self.calls, self.ms = 0, 0.0
        self.error = ''

    def see(self, obs):
        seat, step = int(obs.get('player', 0) or 0), int(obs.get('step', 0) or 0)
        if self.step is not None and step <= self.step:
            self.rows = []
        self.step, self.seat = step, seat
        view = {k: obs.get(k) for k in PUBLIC}
        view.update(private=obs.get('private'), player=seat, step=step)
        if view.get('day') is None:     # live observations carry them; fill a view that lacks them from the step
            view['day'] = step // 24
        if view.get('hour') is None:
            view['hour'] = step % 24
        self.rows.append(D.encode_observation(view, seat))
        del self.rows[:-(HISTORY + 1)]

    def advise(self):
        """What the model would do this turn in our place, or None before any observation."""
        if not self.rows:
            return None
        started = time.perf_counter()
        if self.model is None:
            self.model, self.meta = KadModel.load(self.path)
        rows = np.stack(self.rows)
        batch = inputs_at(rows, len(rows) - 1, self.model.history)
        h, pad = self.model.encode(batch)
        orders, amounts, probs = self.model.market(h, pad)
        units = self.model.units_act(h, batch)
        present = batch['units_mask'].astype(bool)
        self.calls += 1
        ms = (time.perf_counter() - started) * 1000.0
        self.ms += ms
        return {'orders': [ORDERS[o] for o in orders], 'amounts': amounts.tolist(), 'probs': probs,
                'order_index': orders.tolist(),
                'order_prob': {cls: float(probs[:, i].max()) for i, cls in enumerate(ORDERS)},
                'first_prob': {cls: float(probs[0, i]) for i, cls in enumerate(ORDERS)},
                'jobs': units['job_probs'][present], 'units': units, 'present': present,
                'unit_tile': batch['unit_tile'], 'ms': ms}

    def probability(self, advice, op, item=None):
        """The largest probability the model gave the order (op, item) at any slot of its greedy decoding."""
        return float(advice['order_prob'].get((op, item), 0.0)) if advice else 0.0

    def plant(self, advice, crop):
        """The mean probability over our present units that planting `crop` is their next job."""
        if not advice or not len(advice['jobs']):
            return 0.0
        return float(advice['jobs'][:, JOBS.index(('PLANT', crop))].mean())
