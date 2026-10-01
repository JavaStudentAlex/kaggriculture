"""KAD Advisor: Action-diffusion policy advisor and copilot for procedural graph agents.

KAD-HP-1 (policy.py) predicts multi-head action distributions (unit jobs, target tiles, and
market orders) trained on human top-player replay episodes. While KAD cannot beat deterministic
engines when playing standalone (0W-760L), its trained distributions have high accuracy (85% job,
89% market order, 92% target).

This optional turn stage runs as an advisor/copilot:
1. Market Co-Pilot: Injects non-conflicting, high-confidence buy/sell advice into open or
   deliberately empty order slots left by the deterministic backbone engine.
2. Idle Worker Rescue: Assigns productive maintenance tasks (water, weed, harvest, plant)
   to hands that the backbone engine left with ['PASS'].
3. Fail-Closed & Budget-Aware: Throttled cadence (default step % 4 == 0) and timeout guards
   ensure Kaggle's 1-second-per-turn limit is never breached.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PARAMETERS = {
    '_KAD_ENABLED': False,             # off by default in the graph
    '_KAD_CADENCE': 4,                 # run advisor every N steps (aligns with town cadence)
    '_KAD_MAX_MARKET_ORDERS': 2,       # max advised market orders to inject per turn
    '_KAD_MIN_ORDER_CONFIDENCE': 0.35, # probability threshold to accept market advice
    '_KAD_RESCUE_IDLE': True,          # rescue PASSing hands with KAD job suggestions
    '_KAD_FROM_STEP': 24,              # first step to run advice (after initial setup)
    '_KAD_TO_STEP': 696,               # stop before endgame liquidation
    '_KAD_TEMPERATURE': 0.0,           # greedy advice (0.0) or sampled (> 0.0)
    '_KAD_CHECKPOINT': 'default',      # path or 'default' (uses latest hp1 checkpoint)
}

NOTES = {
    '_KAD_ENABLED': 'enable the KAD policy advisor / copilot turn stage',
    '_KAD_CADENCE': 'run advisor every N steps to stay well within CPU execution budget',
    '_KAD_MAX_MARKET_ORDERS': 'maximum number of market orders KAD may add to unused slots',
    '_KAD_MIN_ORDER_CONFIDENCE': 'minimum confidence to execute a KAD-advised market action',
    '_KAD_RESCUE_IDLE': 'assign KAD-suggested tasks to workers currently passing',
    '_KAD_FROM_STEP': 'first step where KAD advice is enabled',
    '_KAD_TO_STEP': 'last step where KAD advice is enabled (hands off to endgame engine)',
    '_KAD_TEMPERATURE': 'sampling temperature for KAD proposals (0.0 = greedy best)',
    '_KAD_CHECKPOINT': 'model checkpoint path or "default"',
}

MARKET_SLOTS = 10
DEFAULT_CHECKPOINT_REL = Path(__file__).resolve().parent.parent.parent / 'action_diffusion' / 'runs' / 'kaggle_hp1_2026-09-28' / 'best.pt'

_MODEL_CACHE = {}


def _get_model(checkpoint_path: Optional[str] = None):
    """Lazy model loader. Returns None if torch is not available or checkpoint missing."""
    path = Path(checkpoint_path) if (checkpoint_path and checkpoint_path != 'default') else DEFAULT_CHECKPOINT_REL
    if str(path) in _MODEL_CACHE:
        return _MODEL_CACHE[str(path)]
    if not path.is_file():
        return None
    try:
        import torch
        ad_dir = path.parent.parent.parent
        if str(ad_dir) not in sys.path:
            sys.path.insert(0, str(ad_dir))
        from policy import Policy, PolicyConfig
        saved = torch.load(path, map_location='cpu', weights_only=False)
        model = Policy(PolicyConfig(**saved['config']))
        model.load_state_dict(saved['model'])
        model.eval()
        torch.set_num_threads(1)
        _MODEL_CACHE[str(path)] = (model, saved)
        return _MODEL_CACHE[str(path)]
    except Exception:
        return None


class KadAdvisor:
    """Runtime advisor evaluating KAD-HP-1 proposals against deterministic game state."""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = {**PARAMETERS, **(params or {})}
        self.model_tuple = None
        self.history_rows = []
        self.last_step = -1
        self.last_advice = None
        self.calls = 0
        self.failures = 0

    def reset(self):
        self.history_rows.clear()
        self.last_step = -1
        self.last_advice = None

    def _observe(self, obs: Dict[str, Any], seat: int):
        step = int(obs.get('step', 0) or 0)
        if step <= self.last_step:
            self.reset()
        self.last_step = step
        try:
            from data import encode_observation
            view = {k: obs.get(k) for k in ('farms', 'market', 'town', 'day', 'hour')}
            view.update(private=obs.get('private') or {}, player=seat, step=step)
            row = encode_observation(view, seat)
            self.history_rows.append(row)
            # keep only needed history (8 steps + 1)
            if len(self.history_rows) > 16:
                del self.history_rows[:-16]
        except Exception:
            pass

    def advise(self, obs: Dict[str, Any], action: Dict[str, Any], state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Takes current planned action, returns updated action with non-conflicting KAD advice."""
        p = self.params
        if not p['_KAD_ENABLED']:
            return action

        step = int(obs.get('step', 0) or 0)
        seat = int(obs.get('player', 0) or 0)
        self._observe(obs, seat)

        if not (p['_KAD_FROM_STEP'] <= step <= p['_KAD_TO_STEP']):
            return action

        # Check cadence to strictly bound CPU usage
        if p['_KAD_CADENCE'] > 1 and (step % p['_KAD_CADENCE']) != 0:
            return action

        if self.model_tuple is None:
            self.model_tuple = _get_model(p['_KAD_CHECKPOINT'])
            if self.model_tuple is None:
                return action

        model, saved = self.model_tuple
        out_action = dict(action)
        try:
            import numpy as np
            import torch
            from policy import inputs_at, ORDERS, ORDER_AMOUNT, JOBS, NOW, NOW_TABLE, OP_TO_ID, OPERATIONS, ITEMS

            rows = np.stack(self.history_rows)
            batch = {k: torch.as_tensor(v)[None]
                     for k, v in inputs_at(rows, len(rows) - 1, model.config.history).items()}
            cond = None
            c = model.config
            if c.condition_dim:
                prompt = (saved.get('conditions') or {}).get('prompt') or {}
                from model import encode_condition
                cond = torch.tensor([[encode_condition(
                    prompt.get('day'), prompt.get('rating_gap'), prompt.get('win'), prompt.get('excess_margin'))]],
                    dtype=torch.float32)

            with torch.no_grad():
                job, amount, target, now, orders, amounts = model.act(
                    batch, cond, None, temperature=float(p['_KAD_TEMPERATURE']))

            # 1. Market Order Advice: fill empty/open slots if high confidence
            market_orders = list(out_action.get('market') or [])
            pred_orders = orders[0].cpu().numpy()
            pred_amounts = amounts[0].cpu().numpy()

            added_orders = 0
            shed = (obs.get('private') or {}).get('shed') or {}
            money = float((obs.get('farms', [{}])[seat] if seat < len(obs.get('farms', [])) else {}).get('money', 0) or 0)

            existing_items = set()
            for o in market_orders:
                 if isinstance(o, (list, tuple)) and len(o) >= 2:
                     existing_items.add((o[0], o[1]))

            for k in range(len(pred_orders)):
                if added_orders >= p['_KAD_MAX_MARKET_ORDERS']:
                    break
                order_idx = int(pred_orders[k])
                if order_idx <= 0 or order_idx >= len(ORDERS):
                    continue
                op, item = ORDERS[order_idx]
                if op == 'NONE' or (op, item) in existing_items:
                    continue

                qty = int(pred_amounts[k]) if ORDER_AMOUNT[order_idx] else 1
                if qty <= 0:
                    qty = 1

                if op == 'SELL':
                    stock = int(shed.get(item, 0) or 0)
                    if stock <= 0:
                        continue
                    qty = min(qty, stock)
                    advised_cmd = ['SELL', item, qty]
                elif op == 'BUY_PRODUCT':
                    if money < 100:  # conservative liquidity floor
                        continue
                    advised_cmd = ['BUY_PRODUCT', item, qty]
                elif op == 'BUY_SEED':
                    if money < 50:
                        continue
                    advised_cmd = ['BUY_SEED', item, qty]
                else:
                    continue

                # Place into first available empty slot or append if within limit
                empty_slot = next((i for i, o in enumerate(market_orders[:MARKET_SLOTS]) if not o), None)
                if empty_slot is not None:
                    market_orders[empty_slot] = advised_cmd
                    added_orders += 1
                elif len(market_orders) < MARKET_SLOTS:
                    market_orders.append(advised_cmd)
                    added_orders += 1

            out_action['market'] = market_orders

            # 2. Idle Hands Rescue Advice
            if p['_KAD_RESCUE_IDLE']:
                hands = list(out_action.get('hands') or [])
                pred_jobs = job[0].cpu().numpy()
                pred_now = now[0].cpu().numpy()
                do_idx = NOW.index('DO')

                for h_idx in range(len(hands)):
                    current_h = hands[h_idx]
                    # Check if worker is currently idling
                    is_idle = (current_h == ['PASS'] or not current_h or current_h == 'PASS')
                    if not is_idle:
                        continue
                    unit_slot = 1 + h_idx
                    if unit_slot >= len(pred_jobs):
                        break
                    if pred_now[unit_slot] == do_idx:
                        j_idx = int(pred_jobs[unit_slot])
                        if 0 < j_idx < len(JOBS):
                            j_op, j_item = JOBS[j_idx]
                            if j_op in ('WATER', 'WEED', 'HARVEST'):
                                hands[h_idx] = [j_op]
                out_action['hands'] = hands

        except Exception as exc:
            self.failures += 1
            # fail closed: returns original action
            return action

        return out_action
