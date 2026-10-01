"""Rival emulation: run a public agent's code in the rival's place, from our own view of the game.

Every observation shows both farms in full (money, tiles with their yields, positions, land) and the market;
only the shed, the seeds and what each worker carries are private, and the environment is deterministic apart
from weeds and shop unlocks at the end of a day, which are public once they happen. So when the rival runs a
public agent we have, its private state can be rebuilt: it starts empty, as the environment starts it, and at
every step the agent acts on the rival's view (our public state with the rival's seat and the rebuilt private
state); at the next step that action and the one we dispatched are simulated with the environment's own
interpreter, and if the result equals what we now observe, the rival still plays that agent and the rebuilt
private state is exact. Checked on Birch Hollow's recorded games (research/procedural_graph/rival/
emulate_replay.py): five rivals that ran the public engine (old or 09-27) for all 719 moves were emulated
exactly from our side alone, and one that left the engine at move 418 was dropped at that move.

Each engine in `_EM_ENGINES` (directories under engines/) is a hypothesis, run from step 0 in its own module
namespace; a hypothesis whose simulation disagrees with an observation is dropped for the rest of the game.
Once the surviving hypotheses have matched `_EM_LOCK` steps and agree, their action for the current step is
the rival's, known before we send ours:
- `identity()` names the engine (the rival_counter stage then uses the counter family `engine:<name>`);
- with `_EM_RACE`, `race()` rearranges our market orders: both seats' order lists clear index by index, the
  two orders at one index unit by unit, so where a product is sold or bought matters. Keeping exactly the
  same orders, it moves ours to the slots that earn us the most against the rival's known orders (simulated
  with the interpreter's market), and only to arrangements that commit exactly the same units for both of us.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

try:
    import engines
except ImportError:
    from hazel_runtime import engines

PARAMETERS = {'_EM_ENGINES': ['tetsutani_demand', 'tetsutani_demand_0927'], '_EM_LOCK': 12, '_EM_RACE': True}
NOTES = {
    '_EM_ENGINES': 'public engines (hazel_runtime/engines/<name>) the rival may run; each is emulated from step 0 '
                   'and dropped at its first disagreement with what we observe',
    '_EM_LOCK': 'steps a hypothesis must have matched before its prediction of the rival is used',
    '_EM_RACE': "rearrange our market orders against the rival's known orders of this turn (same orders, the slots "
                'that earn us the most; nothing bought or sold changes)',
}
PUBLIC_FARM = ('money', 'tiles', 'farmer', 'hands', 'unlocked_quadrants', 'hires_today')
TRADES = ('SELL', 'BUY_PRODUCT')
MARKET_SLOTS = 10
_ENV = None


def environment():
    """The game's own interpreter (kaggle_environments), imported on first use."""
    global _ENV
    if _ENV is None:
        from kaggle_environments.envs.kaggriculture import kaggriculture
        _ENV = kaggriculture
    return _ENV


def _plain(value):
    """A JSON-like deep copy (observations arrive as Structs)."""
    return json.loads(json.dumps(value))


def _differs(a, b):
    """Whether two JSON-like values differ (numbers compared as numbers)."""
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) != set(b) or any(_differs(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) != len(b) or any(_differs(x, y) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return abs(a - b) > 1e-9
    return a != b


def _content(private):
    """A private state compared by content (a live observation keeps the inventories' insertion order, which
    decides what a full shed keeps; the comparison does not depend on it)."""
    return {'shed': {k: v for k, v in (private.get('shed') or {}).items() if v},
            'seeds': {k: v for k, v in (private.get('seeds') or {}).items() if v},
            'inventories': [{k: v for k, v in inv.items() if v} for inv in private.get('inventories') or []]}


def _state(public, privates, actions):
    obs = [SimpleNamespace(player=i, private=privates[i], farms=public['farms'], market=public['market'],
                           town=public['town'], day=public['day'], hour=public['hour'], step=public['step'])
           for i in range(2)]
    return obs, [SimpleNamespace(observation=obs[i], action=actions[i], status='ACTIVE', reward=0) for i in range(2)]


def _env_for(configuration):
    cfg = dict(configuration or {})
    cfg['weedSpawnChance'] = 0.0      # weeds are taken from the observation instead
    return SimpleNamespace(configuration=SimpleNamespace(**cfg), done=False, info={'seed': 0})


def simulate(public, privates, actions, configuration):
    """One step of the environment on a rebuilt state: (farms, market, privates) after it."""
    public, privates, actions = copy.deepcopy(public), copy.deepcopy(privates), copy.deepcopy(actions)
    obs, state = _state(public, privates, actions)
    environment().interpreter(state, _env_for(configuration))
    return obs[0].farms, obs[0].market, privates


def agrees(farms, market, observation):
    """Whether a simulated step equals the observed one (weeds that spawned at the end of the day copied in)."""
    seen = observation.get('farms') or []
    for i, farm in enumerate(farms):
        for y, row in enumerate(seen[i]['tiles']):
            for x, tile in enumerate(row):
                if tile == {'kind': 'WEED'} and farm['tiles'][y][x] is None:
                    farm['tiles'][y][x] = {'kind': 'WEED'}
        if _differs({k: farm[k] for k in PUBLIC_FARM}, {k: seen[i][k] for k in PUBLIC_FARM}):
            return False
    return not _differs(market['inventory'], observation['market']['inventory'])


class Hypothesis:
    """The rival runs `engine`: its rebuilt private state and its action for the current step."""

    def __init__(self, name, engine):
        self.name, self.engine = name, engine
        self.private = None
        self.matched = 0
        self.alive = True
        self.predicted = None
        self.dropped_at = None


class RivalEmulator:
    def __init__(self, engines_dir, names, lock=PARAMETERS['_EM_LOCK'], race=PARAMETERS['_EM_RACE']):
        if not names:
            raise ValueError('_EM_ENGINES must name at least one engine')
        self.hypotheses = [Hypothesis(n, engines.Engine(Path(engines_dir) / n)) for n in names]
        self.lock, self.race_enabled = int(lock), bool(race)
        self.pending = None
        self.last_step = -1
        self.races = 0          # turns whose orders race() rearranged
        self.race_gain = 0.0    # money those rearrangements added in the simulated market
        self.reset()

    def reset(self):
        for h in self.hypotheses:
            h.private = environment()._new_private()
            h.matched, h.alive, h.predicted, h.dropped_at = 0, True, None, None
        self.pending = None
        self.races, self.race_gain = 0, 0.0

    def alive(self):
        return [h for h in self.hypotheses if h.alive]

    def begin(self, obs, configuration):
        """First thing in a turn: check the previous step against this observation, then have every surviving
        hypothesis act on the rival's view of this one."""
        obs = _plain(obs)
        step = int(obs.get('step', 0) or 0)
        if step == 0 or step <= self.last_step:
            self.reset()
        self.last_step = step
        me = int(obs.get('player', 0) or 0)
        if self.pending is not None:
            if self.pending['step'] == step - 1 and 'action' in self.pending:
                self._verify(obs, me)
            else:           # a turn we did not see through: the rebuilt state can no longer be trusted
                for h in self.alive():
                    h.alive, h.dropped_at = False, step
        public = {k: obs[k] for k in ('farms', 'market', 'town', 'day', 'hour')}
        public['step'] = step
        self.pending = {'step': step, 'public': public, 'private': obs.get('private') or {},
                        'configuration': _plain(configuration or {})}
        for h in self.alive():
            view = copy.deepcopy(dict(public, player=1 - me, private=h.private,
                                      remainingOverageTime=obs.get('remainingOverageTime', 60)))
            try:
                h.predicted = _plain(h.engine.agent(view, configuration))
            except Exception:
                h.alive, h.dropped_at = False, step

    def finish(self, action):
        """Last thing in a turn: the action we dispatch (the next turn's check simulates it)."""
        if self.pending is not None:
            self.pending['action'] = _plain(action)

    def _verify(self, obs, me):
        p = self.pending
        for h in self.alive():
            actions, privates = [None, None], [None, None]
            actions[me], actions[1 - me] = p['action'], h.predicted
            privates[me], privates[1 - me] = p['private'], h.private
            try:
                farms, market, privates = simulate(p['public'], privates, actions, p['configuration'])
                ok = agrees(farms, market, obs) and not _differs(_content(privates[me]),
                                                                 _content(obs.get('private') or {}))
            except Exception:
                ok = False
            if ok:
                h.private, h.matched = privates[1 - me], h.matched + 1
            else:
                h.alive, h.dropped_at = False, obs.get('step')

    def prediction(self):
        """The rival's action for the current step, or None: every surviving hypothesis has matched `lock`
        steps and they agree on it."""
        alive = self.alive()
        if not alive or any(h.matched < self.lock or h.predicted is None for h in alive):
            return None
        first = json.dumps(alive[0].predicted, sort_keys=True)
        return alive[0].predicted if all(json.dumps(h.predicted, sort_keys=True) == first for h in alive) else None

    def identity(self):
        """The engine the rival runs, once a single hypothesis survives with `lock` matched steps."""
        alive = self.alive()
        return alive[0].name if len(alive) == 1 and alive[0].matched >= self.lock else None

    def race(self, action, obs):
        """Our final `action` with its market orders rearranged against the rival's predicted action of this
        turn, or unchanged. Every arrangement is simulated as a whole step (both seats' workers act before the
        market, so what reaches the shed first matters) and must commit exactly the same units for both seats."""
        rival = self.prediction()
        orders = (action or {}).get('market') if isinstance(action, dict) else None
        if not self.race_enabled or rival is None or not isinstance(orders, list) or self.pending is None:
            return action
        trades = lambda os_: {o[1] for o in os_ if isinstance(o, (list, tuple)) and len(o) >= 3 and o[0] in TRADES}
        shared = trades(orders) & trades(list(rival.get('market') or [])[:MARKET_SLOTS])
        if not shared:
            return action
        me = int(obs.get('player', 0) or 0)
        p = self.pending
        rival_private = self.alive()[0].private

        def outcome(market):
            actions, privates = [None, None], [None, None]
            actions[me], actions[1 - me] = dict(action, market=market), rival
            privates[me], privates[1 - me] = p['private'], rival_private
            farms, after, privates = simulate(p['public'], privates, actions, p['configuration'])
            commits = json.dumps([_content(privates[0]), _content(privates[1]), after['inventory'],
                                  [len(f['hands']) for f in farms], [f['unlocked_quadrants'] for f in farms]],
                                 sort_keys=True)
            return farms[me]['money'], commits

        base_money, base_commits = outcome(orders)
        movable = [o for o in orders if isinstance(o, (list, tuple)) and len(o) >= 3 and o[0] in TRADES
                   and o[1] in shared]
        best, best_money, budget = list(orders), base_money, 60
        for _ in range(3):
            improved = False
            for order in movable:
                i = next(k for k, o in enumerate(best) if o is order)
                for target in range(min(MARKET_SLOTS, len(best))):
                    if target == i or budget <= 0:
                        continue
                    budget -= 1
                    trial = list(best)
                    trial.insert(target, trial.pop(i))
                    money, commits = outcome(trial)
                    if commits == base_commits and money > best_money + 1e-9:
                        best, best_money, improved = trial, money, True
                        i = target
            if not improved or budget <= 0:
                break
        if best_money > base_money:
            self.races += 1
            self.race_gain += best_money - base_money
            return dict(action, market=[list(o) if isinstance(o, (list, tuple)) else o for o in best])
        return action
