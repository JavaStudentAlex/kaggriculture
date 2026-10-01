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

PARAMETERS = {'_EM_ENGINES': ['tetsutani_demand', 'tetsutani_demand_0927'], '_EM_LOCK': 12, '_EM_RACE': True,
              '_EM_REPAIR': 0}
NOTES = {
    '_EM_ENGINES': 'public engines (hazel_runtime/engines/<name>) the rival may run; each is emulated from step 0 '
                   'and dropped at its first disagreement with what we observe',
    '_EM_LOCK': 'steps a hypothesis must have matched before its prediction of the rival is used',
    '_EM_RACE': "rearrange our market orders against the rival's known orders of this turn (same orders, the slots "
                'that earn us the most; nothing bought or sold changes)',
    '_EM_REPAIR': 'at most this many steps a game in which a hypothesis whose farm actions were right but whose market '
                  'list was not is kept alive with the market list that reproduces the observation (copies of a '
                  'public engine often change only their orders); 0 drops it at the first difference',
}
REPAIR_SIMULATIONS = 12    # market lists tried per repair
# Candidate D (2026-09-30): when those fail, a wider search of single edits of the predicted list (drops, blanks,
# moves, quantities, swaps); a list that repaired one hypothesis is tried first on the others. Copies of the
# public engine that the first 12 lists missed were reproduced by such an edit in every case checked.
WIDE_SIMULATIONS = 120     # more lists per hypothesis and repair (0 = off)
WIDE_STEP_SIMULATIONS = 160    # for all hypotheses in one step
WIDE_GAME_SIMULATIONS = 5000   # in one game
WIDE_MIN_OVERAGE = 25.0    # no wide search once the overage bank is below this many seconds
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
        self.repairs = 0


class RivalEmulator:
    def __init__(self, engines_dir, names, lock=PARAMETERS['_EM_LOCK'], race=PARAMETERS['_EM_RACE'],
                 repair=PARAMETERS['_EM_REPAIR']):
        if not names:
            raise ValueError('_EM_ENGINES must name at least one engine')
        self.hypotheses = [Hypothesis(n, engines.Engine(Path(engines_dir) / n)) for n in names]
        self.lock, self.race_enabled = int(lock), bool(race)
        self.repair_limit = int(repair)
        self.pending = None
        self.last_step = -1
        self.races = 0          # turns whose orders race() rearranged
        self.race_gain = 0.0    # money those rearrangements added in the simulated market
        self.reset()

    def reset(self):
        for h in self.hypotheses:
            h.private = environment()._new_private()
            h.matched, h.alive, h.predicted, h.dropped_at, h.repairs = 0, True, None, None, 0
        self.pending = None
        self.races, self.race_gain = 0, 0.0
        self.repairs = 0
        self.wide_used = 0

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
        results = []
        for h in self.alive():
            actions, privates = [None, None], [None, None]
            actions[me], actions[1 - me] = p['action'], h.predicted
            privates[me], privates[1 - me] = p['private'], h.private
            market = None
            try:
                farms, market, privates = simulate(p['public'], privates, actions, p['configuration'])
                ok = agrees(farms, market, obs) and not _differs(_content(privates[me]),
                                                                 _content(obs.get('private') or {}))
            except Exception:
                ok = False
            results.append((h, ok, privates, market))
        # a repair only when no hypothesis reproduced the step as predicted: the rival left every engine we run
        repairable = not any(ok for _, ok, _, _ in results)
        found, budget = None, [WIDE_STEP_SIMULATIONS]
        for h, ok, privates, market in results:
            if not ok and repairable and self.repairs < self.repair_limit:
                try:
                    repaired, orders = self._repair(h, p, me, obs, market, found, budget)
                except Exception:
                    repaired, orders = None, None
                if repaired is not None:
                    ok, privates, found = True, repaired, orders
                    h.repairs += 1
                    self.repairs += 1
            if ok:
                h.private, h.matched = privates[1 - me], h.matched + 1
            else:
                h.alive, h.dropped_at = False, obs.get('step')

    def _repair(self, h, p, me, obs, failed_market, found=None, budget=None):
        """The privates after the step if the hypothesis's farm actions with another market list reproduce the
        observation, else None. Tried in order, at most REPAIR_SIMULATIONS lists: each product's order changed by
        the units the observed market inventory differs from the failed simulation (an order added or dropped
        when needed), one hire more or less when the rival's hands differ, then single edits of the predicted
        list (an order moved to another slot, an order dropped)."""
        rival = 1 - me
        predicted = h.predicted if isinstance(h.predicted, dict) else {}
        base = [list(o) if isinstance(o, (list, tuple)) else [] for o in (predicted.get('market') or [])]
        seen = (obs.get('market') or {}).get('inventory') or {}
        failed = (failed_market or {}).get('inventory') or {}
        diff = {k: int(round(seen.get(k, 0) - failed.get(k, 0))) for k in set(seen) | set(failed)}
        diff = {k: v for k, v in diff.items() if v}
        candidates = []

        def adjusted(orders):
            out = [list(o) for o in orders]
            for item, d in sorted(diff.items()):
                sell = next((o for o in out if len(o) >= 3 and o[0] == 'SELL' and o[1] == item), None)
                buy = next((o for o in out if len(o) >= 3 and o[0] == 'BUY_PRODUCT' and o[1] == item), None)
                if sell is not None:
                    sell[2] = int(sell[2]) + d
                    if sell[2] <= 0:
                        out.remove(sell)
                elif buy is not None:
                    buy[2] = int(buy[2]) - d
                    if buy[2] <= 0:
                        out.remove(buy)
                elif d > 0:
                    out.append(['SELL', item, d])
                else:
                    out.append(['BUY_PRODUCT', item, -d])
            return out

        hands_seen = len(((obs.get('farms') or [{}, {}])[rival] or {}).get('hands') or [])
        hands_pred = len((p['public']['farms'][rival] or {}).get('hands') or []) + sum(
            1 for o in base if o and o[0] == 'HIRE')
        if diff:
            first = adjusted(base)
            candidates.append(first)
            moved = [o for o in first if o and o not in base]
            for o in moved:
                for target in range(min(MARKET_SLOTS, len(first))):
                    trial = [list(x) for x in first]
                    i = next(k for k, x in enumerate(trial) if x == o)
                    if target != i:
                        trial.insert(target, trial.pop(i))
                        candidates.append(trial)
        if hands_seen != hands_pred:
            for orders in (adjusted(base) if diff else base,):
                trial = [list(x) for x in orders]
                if hands_seen < hands_pred:
                    k = max((i for i, x in enumerate(trial) if x and x[0] == 'HIRE'), default=None)
                    if k is not None:
                        trial.pop(k)
                        candidates.insert(0, trial)
                else:
                    trial.append(['HIRE'])
                    candidates.insert(0, trial)
        for i in range(len(base)):
            for target in range(min(MARKET_SLOTS, len(base))):
                if target != i and base[i]:
                    trial = [list(x) for x in base]
                    trial.insert(target, trial.pop(i))
                    candidates.append(trial)
            if base[i]:
                candidates.append([list(x) for k, x in enumerate(base) if k != i])
        def attempt(orders):
            actions, privates = [None, None], [None, None]
            actions[me], actions[rival] = p['action'], dict(predicted, market=orders)
            privates[me], privates[rival] = p['private'], h.private
            farms, market, after = simulate(p['public'], privates, actions, p['configuration'])
            if agrees(farms, market, obs) and not _differs(_content(after[me]), _content(obs.get('private') or {})):
                return after
            return None

        tried = set()
        if found is not None:           # the list that repaired another hypothesis this step
            tried.add(json.dumps(found))
            after = attempt([list(x) for x in found])
            if after is not None:
                return after, found
        n = 0
        for orders in candidates:
            key = json.dumps(orders)
            if key in tried or len(orders) > MARKET_SLOTS:
                continue
            tried.add(key)
            n += 1
            if n > REPAIR_SIMULATIONS:
                break
            after = attempt(orders)
            if after is not None:
                return after, orders
        if (not WIDE_SIMULATIONS or budget is None or self.wide_used >= WIDE_GAME_SIMULATIONS
                or float(obs.get('remainingOverageTime', 60) or 0) < WIDE_MIN_OVERAGE):
            return None, None
        n = 0
        for orders in self._wide(base, diff, h.private):
            key = json.dumps(orders)
            if key in tried or len(orders) > MARKET_SLOTS:
                continue
            if n >= WIDE_SIMULATIONS or budget[0] <= 0 or self.wide_used >= WIDE_GAME_SIMULATIONS:
                break
            tried.add(key)
            n += 1
            budget[0] -= 1
            self.wide_used += 1
            after = attempt(orders)
            if after is not None:
                return after, orders
        return None, None

    @staticmethod
    def _wide(base, diff, private):
        """More single edits of the predicted market list, most common first: the inventory adjustment on the buy
        order, an order dropped, blanked or set to 0 in place, an order moved one or two slots, a trade's quantity
        changed (doubled, +-1, +-2, halved, the rival's whole stock of it), an order moved further, two swapped."""
        size = len(base)
        if diff:
            out = [list(o) for o in base]
            for item, d in sorted(diff.items()):
                buy = next((o for o in out if len(o) >= 3 and o[0] == 'BUY_PRODUCT' and o[1] == item), None)
                if buy is not None:
                    buy[2] = int(buy[2]) - d
                    if buy[2] <= 0:
                        out.remove(buy)
            yield out
        for i in range(size):
            if base[i]:
                yield [list(x) for k, x in enumerate(base) if k != i]
                yield [list(x) if k != i else [] for k, x in enumerate(base)]
                if len(base[i]) >= 3 and base[i][0] in TRADES:
                    yield [list(x) if k != i else [x[0], x[1], 0] for k, x in enumerate(base)]

        def moved(i, j):
            trial = [list(x) for x in base]
            trial.insert(j, trial.pop(i))
            return trial
        for dist in (1, 2):
            for i in range(size):
                for j in (i - dist, i + dist):
                    if base[i] and 0 <= j < size:
                        yield moved(i, j)
        stock = {}
        for bag in [(private or {}).get('shed') or {}] + list((private or {}).get('inventories') or []):
            for k, v in (bag or {}).items():
                stock[k] = stock.get(k, 0) + int(v or 0)
        for i in range(size):
            o = base[i]
            if len(o) >= 3 and o[0] in TRADES:
                q = int(o[2])
                options = [2 * q, q + 1, q - 1, q + 2, q - 2, q // 2]
                if o[0] == 'SELL':
                    options.append(stock.get(o[1], 0))
                for q2 in options:
                    if q2 > 0 and q2 != q:
                        yield [list(x) if k != i else [x[0], x[1], q2] for k, x in enumerate(base)]
        for dist in range(3, size):
            for i in range(size):
                for j in (i - dist, i + dist):
                    if base[i] and 0 <= j < size:
                        yield moved(i, j)
        for i in range(size):
            for j in range(i + 2, size):
                if base[i] or base[j]:
                    trial = [list(x) for x in base]
                    trial[i], trial[j] = trial[j], trial[i]
                    yield trial

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
        return (alive[0].name if len(alive) == 1 and alive[0].matched >= self.lock and not alive[0].repairs
                else None)

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
