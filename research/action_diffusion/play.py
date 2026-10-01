"""KAD-MD-24 and KAD-HP-1 as Kaggriculture agents: each turn the model reads the observations seen so
far and one turn is played; it replans every turn. load_player picks the class from the checkpoint.

An arena bundle (make_arena.py) holds this file, model.py, policy.py, data.py and a main.py
(arena_main.py) whose `agent` is a player. Every choice is the most likely one.
- Player (KAD-MD-24, model.py): the last 64 observations, the first turn of its 24-turn chunk.
  Settings: `guidance`, classifier-free guidance toward the prompt (1 = the conditioned model as
  trained, 0 = unconditioned, > 1 pushes further toward the prompt), and `prompt`, "default" for
  the checkpoint's training prompt (the newest day, the day's top team, a typical win) or a dict of
  model.CONDITIONS values.
- PolicyPlayer (KAD-HP-1, policy.py): this turn and the scalars of the 8 before it; each unit does
  its job when the now head says DO, else moves or passes; the market head's orders in its order.
  Settings: `prompt` as above, and `team`, "default" for the checkpoint's prompt team (the newest
  day's best-rated team), a team name, or None (unknown); `temperature` (0: every head's most likely
  choice; > 0: a draw at that temperature, from a generator seeded with `sample_seed`, so the same
  game replays exactly); `fit_seeds` (policy.fit_plants: the turn's PLANTs stay within the seeds
  held and the empty tiles, as the engine would otherwise drop them); `device` ('cpu', or 'cuda'
  for a GPU; arena_main.py makes the GPU visible to the bundle's process).

Commands are decoded as data.decode_action does, with two differences: market orders keep their
slot (an empty order stays [], as the ladder agents send it: both seats' orders clear index by
index), and a SELL of the overflow bin (101, "more than 100") sells the whole stock of the item.
"""
from __future__ import annotations

import numpy as np
import torch

from data import (ARGUMENTS, BOARD_SIZE, COMMAND_SLOTS, CROPS, FEATURE_DIM, ITEMS, MARKET_OPS, OPERATIONS,
                  QUANTITY_OPS, UNIT_OPS, encode_observation)
from model import ActionDiffusion, Config, encode_condition

PUBLIC = ('farms', 'market', 'town', 'day', 'hour')


def decode(row, hand_count, shed):
    """The action dict of one predicted turn (int[129]); `shed` is the seat's private shed."""
    fields = np.asarray(row).reshape(COMMAND_SLOTS, 3)

    def command(slot, market):
        op, item, quantity = OPERATIONS[int(slot[0])], ITEMS[int(slot[1])], int(slot[2])
        if op == 'NONE' or op not in (MARKET_OPS if market else UNIT_OPS):
            return None
        result = [op]
        if op in ARGUMENTS:
            if item not in ARGUMENTS[op]:
                return None
            result.append(item)
        if op in QUANTITY_OPS:
            if op == 'SELL' and quantity == 101:
                quantity = max(quantity, int(shed.get(item, 0) or 0))
            result.append(quantity)
        return result

    market = [command(slot, True) or [] for slot in fields[33:]]
    while market and not market[-1]:
        market.pop()
    action = {'hands': [command(fields[1 + i], False) or ['PASS'] for i in range(hand_count)],
              'market': market}
    farmer = command(fields[0], False)
    if farmer is not None:
        action['farmer'] = farmer
    return action


def _condition(saved, prompt):
    """The prompt's condition tensor (1, 8), or None (unconditioned)."""
    if prompt == 'default':
        prompt = (saved.get('conditions') or {}).get('prompt') or {}
    if prompt is None:
        return None
    return torch.tensor([encode_condition(prompt.get('day'), prompt.get('rating_gap'), prompt.get('win'),
                                          prompt.get('excess_margin'))], dtype=torch.float32)


class _Seat:
    """The observation rows of the game being played (a step that does not advance starts a new one)."""

    def __init__(self, keep):
        self.keep, self.rows, self.step = keep, [], None

    def see(self, obs):
        seat, step = int(obs['player']), int(obs['step'])
        if self.step is not None and step <= self.step:
            self.rows = []
        self.step = step
        view = {k: obs.get(k) for k in PUBLIC}
        view.update(private=obs['private'], player=seat, step=step)
        self.rows.append(encode_observation(view, seat))
        del self.rows[:-self.keep]
        return seat


class Player:
    """KAD-MD-24 (model.py), one seat of one game at a time."""

    def __init__(self, saved, guidance=1.0, prompt='default'):
        torch.set_num_threads(1)
        self.model = ActionDiffusion(Config(**saved['config']))
        self.model.load_state_dict(saved['model'])
        self.model.eval()
        c = self.model.config
        self.condition = _condition(saved, prompt) if c.condition_dim else None
        self.guidance = float(guidance)
        self.noisy = self.model.sizes[None, None, :].expand(1, c.horizon, -1)
        self.seat = _Seat(c.context)

    @torch.no_grad()
    def __call__(self, obs):
        c = self.model.config
        seat = self.seat.see(obs)
        rows = self.seat.rows
        x = np.zeros((1, c.context, FEATURE_DIM), dtype=np.float32)
        mask = np.zeros((1, c.context), dtype=bool)
        x[0, -len(rows):] = np.stack(rows)
        mask[0, -len(rows):] = True
        x, mask, t = torch.from_numpy(x), torch.from_numpy(mask), torch.ones(1)
        logits = self.model(x, mask, self.noisy, t, self.condition)[0, 0]
        if self.condition is not None and self.guidance != 1.0:
            free = self.model(x, mask, self.noisy, t)[0, 0]
            logits = free + self.guidance * (logits - free)
        return decode(logits.argmax(-1).numpy(), len(obs['farms'][seat].get('hands') or []),
                      obs['private'].get('shed') or {})


class PolicyPlayer:
    """KAD-HP-1 (policy.py), one seat of one game at a time."""

    def __init__(self, saved, prompt='default', team='default', temperature=0.0, sample_seed=0, fit_seeds=False,
                 device='cpu'):
        from policy import Policy, PolicyConfig
        torch.set_num_threads(1)
        self.device = torch.device(device)
        self.model = Policy(PolicyConfig(**saved['config']))
        self.model.load_state_dict(saved['model'])
        self.model.eval().to(self.device)
        c = self.model.config
        conditions = saved.get('conditions') or {}
        self.condition = _condition(saved, prompt) if c.condition_dim else None
        self.team = None
        if c.teams:
            name = conditions.get('prompt_team') if team == 'default' else team
            vocabulary = conditions.get('teams') or []
            self.team = torch.tensor([vocabulary.index(name) + 1 if name in vocabulary else 0])
        if self.condition is not None:
            self.condition = self.condition.to(self.device)
        if self.team is not None:
            self.team = self.team.to(self.device)
        self.temperature = float(temperature)
        self.generator = torch.Generator().manual_seed(int(sample_seed)) if self.temperature > 0 else None
        self.fit_seeds = bool(fit_seeds)
        self.seat = _Seat(c.history + 1)

    @torch.no_grad()
    def __call__(self, obs):
        from policy import commands, inputs_at
        seat = self.seat.see(obs)
        rows = np.stack(self.seat.rows)
        batch = {k: torch.as_tensor(v)[None].to(self.device)
                 for k, v in inputs_at(rows, len(rows) - 1, self.model.config.history).items()}
        seeds = empty = None
        if self.fit_seeds:
            held = obs['private'].get('seeds') or {}
            seeds = torch.tensor([[int(held.get(crop, 0) or 0) for crop in CROPS]])
            tiles = obs['farms'][seat].get('tiles') or []
            empty = torch.tensor([[y < len(tiles) and x < len(tiles[y]) and tiles[y][x] is None
                                   for y in range(BOARD_SIZE) for x in range(BOARD_SIZE)]])
        job, amount, _, now, orders, amounts = self.model.act(batch, self.condition, self.team, self.temperature,
                                                              self.generator, seeds, empty)
        return decode(commands(job, amount, now, orders, amounts)[0].reshape(-1).cpu().numpy(),
                      len(obs['farms'][seat].get('hands') or []), obs['private'].get('shed') or {})


def load_player(checkpoint, **settings):
    """Player or PolicyPlayer, as the checkpoint is KAD-MD-24's or KAD-HP-1's (it carries `policy`)."""
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    return (PolicyPlayer if 'policy' in saved else Player)(saved, **settings)
