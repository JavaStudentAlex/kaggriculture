"""KAD copilot: KAD-HP-1, the policy trained on Kaggle's published top-player replays, edits our action on a cadence
(optional turn stage, 2026-09-29; the model runs in numpy: kad/kad_numpy.py).

The model reads this turn and the 8 before it (it sees every turn from `_KC_FROM_STEP` - 8), and on turns where
step % `_KC_EVERY` == 0 between `_KC_FROM_STEP` and `_KC_TO_STEP` it says what a top player would do in our place. It
stays out of the opening (the engine's first-day tape) and the last days (the engine's liquidation). Two levers, each on
its own switch so that games can tell which one helps:

- sells (`_KC_SELL`): a SELL of a product in `_KC_SELL_ITEMS` that the model's greedy market decoding places at some
  slot with at least `_KC_SELL_P` probability is appended after the engine's orders, for the model's amount (bin 101:
  all), capped at the shed's stock of it less what the engine already sells and `_KC_KEEP`. Never into the engine's
  empty [] entries (deliberate one-slot delays: both seats' orders clear index by index), at most `_KC_MAX_ORDERS` a
  turn, never WHEAT (feed and seed stock) or FERTILIZER by default.
- hands (`_KC_HANDS`): a hand the engine left on PASS does the model's job where it stands, when the model's next job
  for that unit is in `_KC_JOBS`, its target is the unit's own tile, the model would do it now (DO) with at least
  `_KC_JOB_P`, and the tile allows it: WATER on a growing crop, HARVEST on a ripe ongoing crop (TOMATO, STRAWBERRY) or
  an animal with produce. A unit never moves: the engine's route tapes assume where it is.

Anything that fails turns the copilot off for the rest of the game (the engine's action stands), unless the graph
requires KAD (`require_kad`): then the failure ends the game, so a game without working advice never counts.

`last` holds the latest advice as plain JSON for the tactic stage (info['kad']): the step, the model's orders with
amount and probability, P(BUY_LAND), and for each of our units its next job, target tile, what it would do now and
the job's probability.
"""
from __future__ import annotations

import json
import os

try:
    from kad import kad_numpy, kad_torch
except ImportError:
    try:
        from hazel_runtime.kad import kad_numpy, kad_torch
    except ImportError:
        from hazel_runtime.kad import kad_numpy
        kad_torch = None

PARAMETERS = {
    '_KC_EVERY': 4,             # the model runs on turns with step % _KC_EVERY == 0
    '_KC_FROM_STEP': 24,        # first turn it may act (after the opening day)
    '_KC_TO_STEP': 671,         # last turn it may act (the engine's endgame liquidation owns days 28-29)
    '_KC_SELL': True,           # lever 1: the model's confident sales, appended after the engine's orders
    '_KC_SELL_P': 0.5,          # least probability of the SELL at its slot
    '_KC_SELL_ITEMS': ('MILK', 'WOOL', 'STRAWBERRY', 'EGG', 'CARROT', 'TOMATO', 'MELON'),
    '_KC_KEEP': 0,              # units of each product the copilot never sells
    '_KC_MAX_ORDERS': 2,        # most sales it adds in a turn
    '_KC_HANDS': True,          # lever 2: idle hands do the model's job where they stand
    '_KC_JOB_P': 0.5,           # least probability of the job, and of doing it now
    '_KC_JOBS': ('WATER', 'HARVEST'),
    '_KC_BACKEND': 'auto',      # backend: 'auto' (prefers PyTorch on GPU if available), 'torch', or 'numpy'
    '_KC_DEVICE': 'auto',       # device: 'auto', 'cuda', 'cuda:0', 'cpu'
    '_KC_TEMPERATURE': 0.0,     # sampling temperature (0.0 = greedy, > 0.0 = sampling)
    '_KC_CHECKPOINT': 'default',# model weights path or 'default'
}
NOTES = {
    '_KC_EVERY': 'KAD runs on turns with step % every == 0 (one forward pass, ~0.15 s on a core, ~0.4 s on Kaggle)',
    '_KC_FROM_STEP': 'first turn the copilot acts (the engine owns the opening day)',
    '_KC_TO_STEP': 'last turn the copilot acts (the engine owns its endgame liquidation)',
    '_KC_SELL': 'append the model\'s confident SELL orders after the engine\'s (never into its empty [] slots)',
    '_KC_SELL_P': 'least probability of a SELL at its slot of the model\'s greedy market decoding',
    '_KC_SELL_ITEMS': 'products the copilot may sell (never WHEAT: feed and seed stock)',
    '_KC_KEEP': 'units of each product the copilot leaves in the shed',
    '_KC_MAX_ORDERS': 'most sales the copilot adds in one turn',
    '_KC_HANDS': 'a hand the engine left on PASS does the model\'s job on its own tile (never moves)',
    '_KC_JOB_P': 'least probability of the model\'s job for the unit and of doing it now',
    '_KC_JOBS': 'jobs an idle hand may do: WATER (growing crop), HARVEST (ripe ongoing crop or an animal)',
    '_KC_BACKEND': 'backend engine: auto, torch (GPU PyTorch), or numpy',
    '_KC_DEVICE': 'PyTorch device: auto, cuda, or cpu',
    '_KC_TEMPERATURE': 'sampling temperature for KAD proposals (0.0 = greedy, > 0.0 = sampled)',
    '_KC_CHECKPOINT': 'checkpoint path (.pt for PyTorch, .npz for numpy) or "default"',
}
LIMITS = {'_KC_EVERY': (1, 48), '_KC_FROM_STEP': (8, 719), '_KC_TO_STEP': (0, 719), '_KC_SELL_P': (0.0, 1.0),
          '_KC_KEEP': (0, 1000), '_KC_MAX_ORDERS': (0, 10), '_KC_JOB_P': (0.0, 1.0),
          '_KC_TEMPERATURE': (0.0, 5.0)}
SELLABLE = ('WHEAT', 'CARROT', 'TOMATO', 'STRAWBERRY', 'MELON', 'EGG', 'MILK', 'WOOL', 'FERTILIZER')
HAND_JOBS = ('WATER', 'HARVEST')
ONGOING = ('TOMATO', 'STRAWBERRY')
SLOTS = 10
HISTORY = kad_numpy.HISTORY
DO = kad_numpy.NOW.index('DO')
LOG = os.environ.get('KAD_COPILOT_LOG')   # a file: every intervention as a JSON line (analysis replays)


def check(params):
    p = {**PARAMETERS, **(params or {})}
    for name, (low, high) in LIMITS.items():
        if not low <= p[name] <= high:
            raise ValueError(f'{name} must be within {low}..{high}')
    if p.get('_KC_BACKEND') not in ('auto', 'torch', 'numpy'):
        raise ValueError(f"_KC_BACKEND must be 'auto', 'torch', or 'numpy'")
    if any(item not in SELLABLE for item in p['_KC_SELL_ITEMS']):
        raise ValueError(f'_KC_SELL_ITEMS: products are {list(SELLABLE)}')
    if any(job not in HAND_JOBS for job in p['_KC_JOBS']):
        raise ValueError(f'_KC_JOBS: jobs are {list(HAND_JOBS)}')
    return p


class KadCopilot:
    """One game's copilot (a new game when the step goes back); `apply` edits our action for one turn (a new dict)."""

    def __init__(self, params=None, required=False):
        self.p = check(params)
        self.required = bool(required)
        backend = (os.environ.get('KAGG_KAD_BACKEND') or self.p.get('_KC_BACKEND') or 'auto').lower()
        device = os.environ.get('KAGG_KAD_DEVICE') or self.p.get('_KC_DEVICE') or 'auto'
        checkpoint = os.environ.get('KAGG_KAD_CHECKPOINT') or self.p.get('_KC_CHECKPOINT')
        if checkpoint == 'default':
            checkpoint = None
        temp = float(self.p.get('_KC_TEMPERATURE', 0.0))
        seed = int(self.p.get('_KC_SEED', 0)) if '_KC_SEED' in self.p else None

        self.advisor = None
        if backend in ('auto', 'torch') and kad_torch is not None and getattr(kad_torch, 'TORCH_AVAILABLE', False):
            try:
                self.advisor = kad_torch.TorchKadAdvisor(checkpoint, device=device, temperature=temp, seed=seed)
            except Exception as exc:
                if backend == 'torch':
                    raise
        if self.advisor is None:
            self.advisor = kad_numpy.KadAdvisor()
        self.off = False
        self.player = None
        self.step = None
        self.report = {}
        self.last = None

    def _count(self, key, n=1):
        self.report[key] = self.report.get(key, 0) + n

    def _log(self, obs, kind, **info):
        if LOG:
            with open(LOG, 'a') as fh:
                fh.write(json.dumps({'step': int(obs.get('step', 0) or 0), 'player': self.player, 'kind': kind,
                                     **info}) + '\n')

    def apply(self, obs, action):
        step = int(obs.get('step', 0) or 0)
        if self.step is not None and step <= self.step:
            self.off, self.report, self.last = False, {}, None   # a new game
        self.step = step
        self.player = int(obs.get('player', 0) or 0)
        p = self.p
        if self.off or step > p['_KC_TO_STEP'] or step < p['_KC_FROM_STEP'] - HISTORY or not isinstance(action, dict):
            return action
        try:
            self.advisor.see(obs)
            if step < p['_KC_FROM_STEP'] or step % p['_KC_EVERY']:
                return action
            advice = self.advisor.advise()
            self._count('calls')
            self.last = self._summary(step, advice)
            out = {'farmer': action.get('farmer'),
                   'hands': [list(h) if isinstance(h, (list, tuple)) else ['PASS'] for h in (action.get('hands') or [])],
                   'market': [list(o) if isinstance(o, (list, tuple)) else [] for o in (action.get('market') or [])]}
            if p['_KC_SELL']:
                self._sell(obs, out, advice)
            if p['_KC_HANDS']:
                self._hands(obs, out, advice)
            if 'farmer' not in action:
                out.pop('farmer')
            return out
        except Exception as exc:  # noqa: BLE001 -- fail closed: the engine's action stands for the rest of the game
            self.off = True
            self._count('failures')
            self._log(obs, 'failure', error=repr(exc)[:300])
            if self.required:
                raise
            return action

    @staticmethod
    def _summary(step, advice):
        orders = []
        for k, (op, item) in enumerate(advice['orders']):
            if op != 'NONE':
                orders.append([op, item, int(advice['amounts'][k]),
                               round(float(advice['probs'][k, advice['order_index'][k]]), 3)])
        units, jobs = advice['units'], []
        for slot in range(len(units['job'])):
            if not advice['present'][slot]:
                continue
            op, item = kad_numpy.JOBS[int(units['job'][slot])]
            target = int(units['target'][slot])
            jobs.append({'unit': slot, 'job': op, 'item': item, 'target': [target % 10, target // 10],
                         'now': kad_numpy.NOW[int(units['now'][slot])], 'p': round(float(units['job_p'][slot]), 3)})
        return {'step': step, 'orders': orders, 'buy_land_p': round(advice['order_prob'].get(('BUY_LAND', None), 0.0), 3),
                'units': jobs}

    def _sell(self, obs, out, advice):
        p, market = self.p, out['market']
        shed = (obs.get('private') or {}).get('shed') or {}
        added = 0
        for k, (op, item) in enumerate(advice['orders']):
            if added >= p['_KC_MAX_ORDERS'] or len(market) >= SLOTS:
                break
            if op != 'SELL' or item not in p['_KC_SELL_ITEMS']:
                continue
            prob = float(advice['probs'][k, advice['order_index'][k]])
            if prob < p['_KC_SELL_P']:
                continue
            selling = sum(int(o[2] or 0) for o in market
                          if isinstance(o, list) and len(o) >= 3 and o[0] == 'SELL' and o[1] == item)
            free = int(shed.get(item, 0) or 0) - selling - p['_KC_KEEP']
            wanted = int(advice['amounts'][k])
            qty = min(free, free if wanted >= 101 else wanted)
            if qty <= 0:
                continue
            market.append(['SELL', item, qty])
            added += 1
            self._count('sells')
            self._count(f'sell_units_{item}', qty)
            self._log(obs, 'sell', item=item, qty=qty, p=round(prob, 3), slot=k)

    def _hands(self, obs, out, advice):
        p, units = self.p, advice['units']
        farms = obs.get('farms') or []
        farm = farms[self.player] if self.player < len(farms) else {}
        tiles = farm.get('tiles') or []
        positions = farm.get('hands') or []
        day = int(obs.get('step', 0) or 0) // 24
        for i, command in enumerate(out['hands']):
            if command != ['PASS'] or i >= len(positions):
                continue
            slot = i + 1                                   # unit slot 0 is the farmer
            if slot >= len(units['job']) or not advice['present'][slot]:
                continue
            op, _ = kad_numpy.JOBS[int(units['job'][slot])]
            if (op not in p['_KC_JOBS'] or int(units['now'][slot]) != DO or units['job_p'][slot] < p['_KC_JOB_P']
                    or units['now_p'][slot] < p['_KC_JOB_P']):
                continue
            x, y = positions[i]
            if int(units['target'][slot]) != y * 10 + x:
                continue                                   # the model's target is elsewhere: never move a unit
            tile = tiles[y][x] if 0 <= y < len(tiles) and 0 <= x < len(tiles[y]) else None
            if not isinstance(tile, dict) or not self._allowed(op, tile, day):
                continue
            out['hands'][i] = [op]
            self._count(f'hand_{op}')
            self._log(obs, 'hand', hand=i, op=op, tile=[x, y], crop=tile.get('crop'), animal=tile.get('animal'),
                      p=round(float(units['job_p'][slot]), 3))

    @staticmethod
    def _allowed(op, tile, day):
        if op == 'WATER':
            return tile.get('kind') == 'PLANT' and tile.get('crop') is not None and not tile.get('watered_today')
        if op == 'HARVEST':
            if tile.get('animal'):
                return int(tile.get('yield_units', 0) or 0) > 0
            return (tile.get('kind') == 'PLANT' and tile.get('crop') in ONGOING
                    and int(tile.get('yield_units', 0) or 0) > 0)
        return False
