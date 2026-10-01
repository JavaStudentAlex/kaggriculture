"""PyTorch GPU/CPU inference advisor for KAD copilot inside the graph runtime."""
from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
AD_DIR = HERE.parents[2] / 'action_diffusion'
if str(AD_DIR) not in sys.path:
    sys.path.insert(0, str(AD_DIR))

try:
    from . import kad_data as D
except ImportError:
    try:
        import kad_data as D
    except ImportError:
        import data as D

try:
    import torch
    import torch.nn.functional as F
    from policy import (
        JOBS, MARKET_SLOTS, NOW, ORDERS, QUANTITY_BINS, UNITS,
        Policy, PolicyConfig, inputs_at, _pick, JOB_AMOUNT, ORDER_AMOUNT
    )
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

PUBLIC = ('farms', 'market', 'town', 'day', 'hour')
HISTORY = 8
TILES = D.BOARD_SIZE * D.BOARD_SIZE
DEFAULT_CHECKPOINT = AD_DIR / 'runs' / 'kaggle_hp1_2026-09-28' / 'best.pt'


def _pick_device(spec: Optional[str] = None):
    if not TORCH_AVAILABLE:
        return None
    spec = (spec or os.environ.get('KAGG_KAD_DEVICE', 'auto')).strip().lower()
    if spec != 'auto':
        return torch.device(spec)
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        return torch.device('cpu')
    n = torch.cuda.device_count()
    order = [(os.getpid() + k) % n for k in range(n)]
    for idx in order:
        try:
            free, _total = torch.cuda.mem_get_info(idx)
        except Exception:
            continue
        if free >= (512 << 20):  # at least 512 MB free
            return torch.device(f'cuda:{idx}')
    return torch.device('cpu')


def _encode_condition(day, rating_gap, win, excess_margin):
    """Encode scalar conditions into float32 [8]."""
    day_idx = 0.0
    if isinstance(day, str):
        try:
            parts = [int(p) for p in day.split('-')]
            day_idx = float(parts[1] * 31 + parts[2] - 7 * 31 - 30) / 60.0
        except Exception:
            day_idx = 0.0
    gap = float(rating_gap or 0.0) / 1000.0
    w = 1.0 if win else -1.0 if win is not None else 0.0
    em = float(excess_margin or 0.0)
    vals = [day_idx, gap, w, em]
    out = []
    for v in vals:
        out.extend([math.sin(v * math.pi), math.cos(v * math.pi)])
    return out


class TorchKadAdvisor:
    """GPU-accelerated PyTorch advisor matching KadAdvisor's public interface."""

    def __init__(self, checkpoint: Optional[Path | str] = None, device: Optional[str] = None,
                 temperature: float = 0.0, seed: Optional[int] = None):
        if not TORCH_AVAILABLE:
            raise RuntimeError('PyTorch is not available for TorchKadAdvisor')
        self.checkpoint_path = Path(checkpoint) if checkpoint else DEFAULT_CHECKPOINT
        if not self.checkpoint_path.is_file():
            # Try finding best.pt in known locations
            alt = HERE.parent.parent / 'action_diffusion' / 'runs' / 'kaggle_hp1_2026-09-28' / 'best.pt'
            if alt.is_file():
                self.checkpoint_path = alt
        self.device = _pick_device(device)
        self.temperature = float(temperature)
        self.generator = torch.Generator(device='cpu').manual_seed(seed) if seed is not None else None
        self.model: Optional[Policy] = None
        self.conditions_tensor: Optional[torch.Tensor] = None
        self.team_tensor: Optional[torch.Tensor] = None
        self.rows: list = []
        self.step: Optional[int] = None
        self.seat: Optional[int] = None
        self.calls = 0
        self.ms = 0.0

    def load_model(self):
        if self.model is not None:
            return
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f'KAD checkpoint not found: {self.checkpoint_path}')
        saved = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)
        cfg = PolicyConfig(**saved['config'])
        model = Policy(cfg)
        model.load_state_dict(saved['model'])
        model.eval()
        self.model = model.to(self.device)

        conditions = saved.get('conditions') or {}
        if cfg.condition_dim:
            prompt = conditions.get('prompt') or {}
            c_vec = _encode_condition(prompt.get('day'), prompt.get('rating_gap'),
                                     prompt.get('win'), prompt.get('excess_margin'))
            self.conditions_tensor = torch.tensor([c_vec], dtype=torch.float32, device=self.device)
        if cfg.teams:
            name = conditions.get('prompt_team')
            vocabulary = conditions.get('teams') or []
            t_idx = vocabulary.index(name) + 1 if name in vocabulary else 0
            self.team_tensor = torch.tensor([t_idx], dtype=torch.long, device=self.device)

    def see(self, obs: Dict[str, Any]):
        seat = int(obs.get('player', 0) or 0)
        step = int(obs.get('step', 0) or 0)
        if self.step is not None and step <= self.step:
            self.rows = []
        self.step = step
        self.seat = seat
        view = {k: obs.get(k) for k in PUBLIC}
        view.update(private=obs.get('private'), player=seat, step=step)
        if view.get('day') is None:
            view['day'] = step // 24
        if view.get('hour') is None:
            view['hour'] = step % 24
        self.rows.append(D.encode_observation(view, seat))
        del self.rows[:-(HISTORY + 1)]

    def advise(self) -> Optional[Dict[str, Any]]:
        if not TORCH_AVAILABLE:
            return None
        with torch.no_grad():
            return self._advise_impl()

    def _advise_impl(self) -> Optional[Dict[str, Any]]:
        if not self.rows:
            return None
        self.load_model()
        started = time.perf_counter()

        rows = np.stack(self.rows)
        raw_batch = inputs_at(rows, len(rows) - 1, self.model.config.history)
        batch = {k: torch.as_tensor(v)[None].to(self.device) for k, v in raw_batch.items()}

        def pick(logits):
            return _pick(logits, self.temperature, self.generator)

        h, pad = self.model.encode(batch, self.conditions_tensor, self.team_tensor)
        tiles, units = self.model.split(h)

        job_logits = self.model.job(units)[0]                     # [UNITS, len(JOBS)]
        job_probs = torch.softmax(job_logits, -1)
        job = pick(job_logits)
        job_p = job_probs[torch.arange(UNITS, device=self.device), job]

        z = units[0] + self.model.job_embed(job)
        tiles_flat = tiles[0]
        target_logits = torch.einsum('uw,tw->ut', self.model.target_query(z), self.model.target_key(tiles_flat))
        target_logits = target_logits / math.sqrt(self.model.config.width)
        target = pick(target_logits)
        target = torch.where(job > 0, target, batch['unit_tile'][0].long())

        at = tiles_flat[target]
        now_logits = self.model.now(z + self.model.target_embed(at))
        now_probs = torch.softmax(now_logits, -1)
        now = pick(now_logits)
        now_p = now_probs[torch.arange(UNITS, device=self.device), now]

        orders = torch.zeros(1, MARKET_SLOTS, dtype=torch.long, device=self.device)
        amounts = torch.zeros(1, MARKET_SLOTS, dtype=torch.long, device=self.device)
        probs = torch.zeros(MARKET_SLOTS, len(ORDERS), dtype=torch.float32, device=self.device)

        for k in range(MARKET_SLOTS):
            d = self.model.market_state(h, pad, orders, amounts)
            slot_logits = self.model.order(d)[0, k]
            p_k = torch.softmax(slot_logits, -1)
            probs[k] = p_k
            orders[0, k] = pick(slot_logits)
            amt_logits = self.model.order_amount(d[0, k:k+1] + self.model.order_embed(orders[0, k:k+1]))[0, 0]
            amounts[0, k] = pick(amt_logits)

        orders_np = orders[0].cpu().numpy()
        amounts_np = amounts[0].cpu().numpy()
        probs_np = probs.cpu().numpy()
        present_np = raw_batch['units_mask'].astype(bool)

        units_dict = {
            'job': job.cpu().numpy(),
            'job_p': job_p.cpu().numpy(),
            'target': target.cpu().numpy(),
            'now': now.cpu().numpy(),
            'now_p': now_p.cpu().numpy(),
            'job_probs': job_probs.cpu().numpy()
        }

        self.calls += 1
        ms = (time.perf_counter() - started) * 1000.0
        self.ms += ms

        return {
            'orders': [ORDERS[o] for o in orders_np],
            'amounts': amounts_np.tolist(),
            'probs': probs_np,
            'order_index': orders_np.tolist(),
            'order_prob': {cls: float(probs_np[:, i].max()) for i, cls in enumerate(ORDERS)},
            'first_prob': {cls: float(probs_np[0, i]) for i, cls in enumerate(ORDERS)},
            'jobs': units_dict['job_probs'][present_np],
            'units': units_dict,
            'present': present_np,
            'unit_tile': raw_batch['unit_tile'],
            'ms': ms
        }

    def probability(self, advice: Optional[Dict[str, Any]], op: str, item: Optional[str] = None) -> float:
        return float(advice['order_prob'].get((op, item), 0.0)) if advice else 0.0

    def plant(self, advice: Optional[Dict[str, Any]], crop: str) -> float:
        if not advice or not len(advice['jobs']):
            return 0.0
        return float(advice['jobs'][:, JOBS.index(('PLANT', crop))].mean())

