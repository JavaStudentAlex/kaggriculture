"""Rival model: which family of agent the rival plays, from its public farm, and our counter to it.

The rival_counter stage uses `MirrorTracker`: does the rival still play exactly as we do? Ladder rivals
are mostly copies of public agents, and on the same seed a copy of our own engine has our money and our
farm step for step. Alder Ford's 66 recorded games (2026-09-27) split cleanly at step 92:
- copies of our engine (the public "Demand-Preserving" agent and its near-copies) keep our money past
  step 92 (their own changes show from step 160 to 661, or never): class `mirror`;
- "buy N / sell N-5" openers (the Forecast family, most of the 2965 family, the engine's 09-27 version)
  sold their 3 surplus wheat at step 0; at step 92 our engine sells its 3 and the rival does not, so the
  rival's money first falls behind ours at step 92 by the price of 3 wheat (-$87 to -$90): `nsell_opener`
  (with the sign reversed, a rival that sells at 92 while we sold at step 0: `wheat92_seller`);
- other openings (13-wheat, "buy 8 / sell 8", 2965 variants) differ from step 1: `other_opening`;
- anything else: `other`.
The class is final once decided (step 92 or 93, or the step of an earlier divergence).

Every observation carries both farms in public: money, hands, land quadrants, and the tiles with each
crop, pasture and coop (only inventories, seeds and the shed are private). `farm_vector` reduces a farm
to numbers (FIELDS); `RivalModel` keeps the rival's vectors and, at each checkpoint of the model file,
turns their trajectory into a candidate list: the families still consistent with it and a belief over
them. The same `farm_vector` builds the offline data (research/procedural_graph/rival/rival_features.py),
so the classifier sees in play exactly what it was fitted on.

Model file (JSON, fitted offline by rival/fit_rival_model.py):
    {"clusters": [...], "checkpoints": [24, 48, ...], "drop": [field names left out],
     "scale": {"<T>": {"mean": [...], "sd": [...]}},
     "prototypes": {"<T>": [[cluster, [standardized features]], ...]}}
At checkpoint T the belief is a softmax over clusters of -(d_c^2 - d_min^2) / (2 * tau^2), d_c = the
distance from the rival's trajectory to cluster c's nearest prototype; the candidate list holds the
clusters with belief >= `min_belief`. `decision` is sticky: the last family whose belief reached the
confidence stays chosen until another one does (graph_runtime applies its counters).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

MODEL_FILE = 'rival_model.json'   # the optional nearest-prototype model (RivalModel), next to this file
# Parameters of the rival_counter turn node (graph `parameters` on it).
PARAMETERS = {}
NOTES = {}
CLASSES = {
    'mirror': 'plays exactly as we do past step 92 (a copy of our engine; its own changes come later or never)',
    'nsell_opener': 'money first falls behind ours at step 92 by the price of 3 wheat: a "buy N / sell N-5" '
                    'opener (Forecast family, most 2965, the public engine of 09-27)',
    'wheat92_seller': 'money first moves ahead of ours at step 92 by the price of 3 wheat: it sells the 3 '
                      'opening wheat we sold at step 0 (the old public engine when we run a buy-N/sell-N-5 engine)',
    'other_opening': 'differs from us before step 92 (13-wheat, "buy 8 / sell 8", 2965 variants, others)',
    'other': 'differs first at step 92 in another way',
}
DECIDE_STEP = 93   # a rival still equal to us here is a mirror


def _state(farm):
    """The public farm fields a copy of our engine shares with us (money, hands, land, crops, herd)."""
    return farm_vector(farm)[:-2]           # not the farmer's position
CROPS = ('WHEAT', 'CARROT', 'TOMATO', 'STRAWBERRY', 'MELON')
ANIMALS = ('COW', 'SHEEP', 'GOOSE')
FIELDS = ('money', 'hands', 'land', 'hires_today') + tuple(f'crop_{c}' for c in CROPS) + \
    tuple(f'animal_{a}' for a in ANIMALS) + ('empty_pasture', 'empty_coop', 'farmer_x', 'farmer_y')


def farm_vector(farm):
    """One farm's public state as numbers, in FIELDS order."""
    crops = dict.fromkeys(CROPS, 0)
    animals = dict.fromkeys(ANIMALS, 0)
    empty_pasture = empty_coop = 0
    for row in farm.get('tiles') or []:
        for tile in row or []:
            if not isinstance(tile, dict):
                continue
            kind = tile.get('kind')
            if kind == 'PLANT' and tile.get('crop') in crops:
                crops[tile['crop']] += 1
            elif kind in ('PASTURE', 'COOP'):
                animal = tile.get('animal')
                if animal in animals:
                    animals[animal] += 1
                elif kind == 'PASTURE':
                    empty_pasture += 1
                else:
                    empty_coop += 1
    farmer = farm.get('farmer') or [-1, -1]
    return [float(farm.get('money') or 0), len(farm.get('hands') or []), len(farm.get('unlocked_quadrants') or []),
            int(farm.get('hires_today') or 0)] + [crops[c] for c in CROPS] + [animals[a] for a in ANIMALS] + \
        [empty_pasture, empty_coop, int(farmer[0]), int(farmer[1])]


def trajectory_features(rows, t_max, drop, ours=None):
    """The classifier's input at checkpoint t_max: the rival's vector every 24 steps up to t_max (fields
    in `drop` left out) plus log money; rows[t] is the rival's vector at step t. With `ours` (our own
    vectors, same steps) the rival minus us instead: on the same seed a rival that plays our engine has
    our farm, so the difference is near zero, and other families differ from us in their own way."""
    keep = [i for i, f in enumerate(FIELDS) if f not in drop]
    money = FIELDS.index('money')
    out = []
    for t in range(24, t_max + 1, 24):
        r = rows[min(t, len(rows) - 1)]
        if ours is None:
            out += [r[i] for i in keep] + [math.log1p(max(0.0, r[money]))]
        else:
            o = ours[min(t, len(ours) - 1)]
            out += [r[i] - o[i] for i in keep] + [math.log1p(max(0.0, r[money])) - math.log1p(max(0.0, o[money]))]
    return out


class RivalModel:
    """The rival's trajectory and the candidate list of its family, updated once per turn."""

    def __init__(self, model, tau=1.0, min_belief=0.02):
        self.model = model if isinstance(model, dict) else json.loads(Path(model).read_text())
        self.clusters = list(self.model['clusters'])
        self.checkpoints = sorted(int(t) for t in self.model['checkpoints'])
        self.drop = set(self.model.get('drop', ()))
        self.relative = bool(self.model.get('relative', False))
        self.tau, self.min_belief = float(tau), float(min_belief)
        self.reset()

    def reset(self):
        self.rows = []
        self.ours = []
        self.candidates = list(self.clusters)
        self.belief = {c: 1.0 / len(self.clusters) for c in self.clusters}
        self.decision = None

    def update(self, obs):
        """Record the rival's farm at this step; at a checkpoint, refresh the belief. Returns the belief."""
        step = int(obs.get('step', 0) or 0)
        if step == 0 or step < len(self.rows):
            self.reset()
        farms = obs.get('farms') or []
        me = int(obs.get('player', 0) or 0)
        if len(farms) != 2:
            return self.belief
        return self.observe(step, farm_vector(farms[1 - me]), farm_vector(farms[me]))

    def observe(self, step, vec, ours=None):
        """The rival's (and our) vector at `step` (update() without the observation; offline evaluation uses it)."""
        while len(self.rows) < step:          # a missed turn keeps the last known state
            self.rows.append(self.rows[-1] if self.rows else vec)
            self.ours.append(self.ours[-1] if self.ours else ours)
        self.rows.append(vec)
        self.ours.append(ours)
        if step in self.checkpoints:
            self._classify(step)
        return self.belief

    def _classify(self, t):
        key = str(t)
        scale, protos = self.model['scale'][key], self.model['prototypes'][key]
        x = trajectory_features(self.rows, t, self.drop, self.ours if self.relative else None)
        z = [(a - m) / s for a, m, s in zip(x, scale['mean'], scale['sd'])]
        best = {}
        for cluster, p in protos:
            d = sum((a - b) ** 2 for a, b in zip(z, p))
            best[cluster] = min(d, best.get(cluster, math.inf))
        if not best:
            return
        lo = min(best.values())
        w = {c: math.exp(-(d - lo) / (2 * self.tau ** 2)) for c, d in best.items()}
        total = sum(w.values())
        self.belief = {c: v / total for c, v in w.items()}
        self.candidates = sorted((c for c in self.belief if self.belief[c] >= self.min_belief),
                                 key=lambda c: -self.belief[c])

    def decide(self, confidence):
        """The sticky decision: the top family once its belief reaches `confidence`, kept until another
        family's does (None before any)."""
        top, belief = self.top()
        if belief >= confidence:
            self.decision = top
        return self.decision

    def top(self):
        """(most likely family, its belief)."""
        c = max(self.belief, key=self.belief.get)
        return c, self.belief[c]


class MirrorTracker:
    """The rival's class relative to our own play (CLASSES), decided once in the game's first 93 steps."""

    clusters = tuple(CLASSES)

    def __init__(self):
        self.reset()

    def reset(self):
        self.diverged_at = None
        self.gap = None
        self.decision = None
        self.last_step = -1

    def update(self, obs):
        step = int(obs.get('step', 0) or 0)
        if step == 0 or step <= self.last_step:
            self.reset()
        self.last_step = step
        farms = obs.get('farms') or []
        if len(farms) != 2 or self.decision is not None:
            return self.decision
        me = int(obs.get('player', 0) or 0)
        rival, ours = _state(farms[1 - me]), _state(farms[me])
        return self.observe(step, rival, ours)

    def observe(self, step, rival, ours):
        """One step of both farms' state (update() without the observation)."""
        if self.decision is not None:
            return self.decision
        if self.diverged_at is None and (abs(rival[0] - ours[0]) > 0.5 or rival[1:] != ours[1:]):
            self.diverged_at, self.gap = step, rival[0] - ours[0]
        d = self.diverged_at
        if d is not None and d < 92:
            self.decision = 'other_opening'
        elif d == 92:
            self.decision = ('nsell_opener' if -120 <= self.gap <= -60 else
                             'wheat92_seller' if 60 <= self.gap <= 120 else 'other')
        elif step >= DECIDE_STEP:
            self.decision = 'mirror'
        return self.decision

    def decide(self, confidence=None):
        return self.decision


def check_counters(counters, clusters, engine_names, guard_names):
    """The node's `counters` ({family: {NAME: value}}) validated: known families, and names that are
    switchable engine constants or oracle_guard parameters. Returns a copy; ValueError otherwise."""
    if not isinstance(counters, dict):
        raise ValueError('rival_counter counters must be an object {family: {NAME: value}}')
    out = {}
    for family, settings in counters.items():
        if family not in clusters:
            raise ValueError(f'unknown rival family {family!r}; the model knows {sorted(clusters)}')
        if not isinstance(settings, dict) or not settings:
            raise ValueError(f'counters of {family} must be a non-empty object {{NAME: value}}')
        for name in settings:
            if name not in engine_names and name not in guard_names:
                raise ValueError(f'{name} cannot be set per rival family (switchable engine constants and '
                                 f'_OG_* guard parameters can)')
        out[family] = dict(settings)
    return out
