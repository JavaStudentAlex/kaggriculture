"""Land plot: an evolvable land purchase for a ladder engine (optional turn stage).

The public engine's route tapes buy the second quadrant (NE) on day 6 and the third (SW) on day 11 in every
game and never the fourth (SE), and no engine constant moves those purchases. The ladder's top teams own three
quadrants by day 9, and 43% of them own four by day 12 (tape mining of 09-22..09-28). This stage is the
evolution's lever on land. From day `_LP_DAY` (until `_LP_LAST_BUY_DAY`) it buys SE, buying SW first if the tape
has not bought it yet (the tape's own later BUY_LAND then does nothing: the environment ignores it once every
quadrant is owned). It hires `_LP_WORKERS` hands every day, after the engine's own hires of the day and no
earlier than hour `_LP_HIRE_HOUR` (the tapes hire at hours 0-1, the engine's fertilizer tours at hours 1-3), and
farms the first `_LP_TILES` SE tiles with `_LP_USE`:

- a crop (WHEAT, CARROT, TOMATO, STRAWBERRY, MELON): plant, water every day, harvest (one-time crops at peak
  yield), replant until `_LP_LAST_PLANT_DAY`;
- GOOSE: build coops, buy geese and place them, then feed, care for and harvest them every day and collect their
  fertilizer.

It buys the seeds, geese and goose feed itself and sells what the plot produces: while the price is at least
`_LP_SELL_RATIO` x the product's base price, and everything from day 28. Workers carry the day's produce into the
shed (the environment drops every inventory there at the end of the day; on the last day they walk it in).
The engine's own orders keep their slots and indices; the stage's orders are appended while the list is shorter
than 10. While the rival's farm is a copy of ours (same quadrants, `_LP_MIRROR` or more of the occupied tiles
alike: a rival running our public engine), no land is bought: the public engine races a rival it recognises as
its clone tile for tile, and a fourth quadrant ends that recognition, which in test games handed a 0927-engine
rival $11-15k more income than in the mirror race (09-29). SE is land the engine never plans for: the tapes do not use it, and the engine's two SE
layers (the tomato and sheep plots) run only while SE is locked. The shed-access tile (5, 5), where hired hands
appear, is left alone.

Parameters (graph `parameters` on the land_plot turn node):
"""
from __future__ import annotations

PARAMETERS = {
    '_LP_DAY': 12,               # first day the stage may buy the land
    '_LP_LAST_BUY_DAY': 22,      # no land is bought after this day (the plot is then off for the game)
    '_LP_USE': 'GOOSE',          # WHEAT, CARROT, TOMATO, STRAWBERRY, MELON or GOOSE
    '_LP_TILES': 8,              # SE tiles farmed: crop tiles, or coops with one goose each (1-24)
    '_LP_WORKERS': 1,            # hands hired every day for the plot (1-4)
    '_LP_HIRE_HOUR': 4,          # earliest hour of the day the plot's hands are hired
    '_LP_LAST_PLANT_DAY': 25,    # crops: nothing is planted after this day
    '_LP_MIN_MONEY': 2000,       # money that must remain after the land purchase (and the engine's buys)
    '_LP_RESERVE': 300,          # money kept back from every later plot purchase and hire
    '_LP_SELL_RATIO': 0.7,       # the plot's produce is sold while price >= ratio x base price (all from day 28)
    '_LP_MIRROR': 0.9,           # no land while this share of occupied tiles matches the rival's (1.01: always buy)
}
NOTES = {
    '_LP_DAY': 'first day the land may be bought (SE; SW first when the tape has not bought it yet: +$2,000)',
    '_LP_LAST_BUY_DAY': 'no land is bought after this day; the plot is then off for the game',
    '_LP_USE': 'what the plot does: WHEAT, CARROT, TOMATO, STRAWBERRY, MELON (a crop) or GOOSE (coops + geese)',
    '_LP_TILES': 'SE tiles used (1-24): crop tiles, or coops with one goose ($300) each',
    '_LP_WORKERS': 'hands hired every day for the plot (1-4); the day\'s k-th extra hire costs fib(hires so far)',
    '_LP_HIRE_HOUR': 'earliest hour the plot\'s hands are hired (after every engine hire of the day)',
    '_LP_LAST_PLANT_DAY': 'crops: last day a seed is planted (one-time crops also need to ripen by day 29)',
    '_LP_MIN_MONEY': 'money that must remain after the land purchase and the engine\'s own buys of that turn',
    '_LP_RESERVE': 'money kept back from every later plot purchase (seeds, geese, feed) and hire',
    '_LP_SELL_RATIO': 'the plot\'s produce is sold while price >= ratio x base price; all of it from day 28',
    '_LP_MIRROR': 'no land is bought while the rival has our quadrants and this share of occupied tiles alike (a copy of '
                  'our public engine, which races its clones; the 4th quadrant ends the race in its favour); 1.01 = '
                  'buy against every rival',
}
USES = ('WHEAT', 'CARROT', 'TOMATO', 'STRAWBERRY', 'MELON', 'GOOSE')
SEED_COST = {'WHEAT': 10, 'CARROT': 20, 'TOMATO': 50, 'STRAWBERRY': 100, 'MELON': 80}
PEAK_AGE = {'WHEAT': 4, 'CARROT': 3, 'MELON': 10}        # one-time crops: age of the largest harvest
FIRST_YIELD = {'TOMATO': 8, 'STRAWBERRY': 10}           # ongoing crops: age of the first yield
ANIMAL_COST = {'GOOSE': 300, 'COW': 400, 'SHEEP': 500}
LAND_ORDER = ('NE', 'SW', 'SE')                         # the environment's BUY_LAND order
LAND_PRICES = (1000, 2000, 4000)
SHED_ACCESS = ((4, 4), (5, 4), (4, 5), (5, 5))
SLOTS = 10
LAST_DAY = 29
HOLD_CAP = 10                                           # stock above this is sold whatever the price
CROWDED = 70                                            # shed items at which the plot's stock is sold at any price
BATCH = {'STRAWBERRY': 6, 'MELON': 6, 'TOMATO': 10}     # most units of these sold in one turn (price impact)
LIMITS = {'_LP_DAY': (1, 28), '_LP_LAST_BUY_DAY': (1, 28), '_LP_TILES': (1, 24), '_LP_WORKERS': (1, 4),
          '_LP_HIRE_HOUR': (0, 20), '_LP_LAST_PLANT_DAY': (0, 28), '_LP_MIN_MONEY': (0, 100000),
          '_LP_RESERVE': (0, 100000), '_LP_SELL_RATIO': (0.0, 5.0), '_LP_MIRROR': (0.0, 1.01)}
# SE without the shed-access tile (5, 5), in a serpentine from the shed so each worker's share is compact.
PLOT = tuple([(x, 5) for x in range(6, 10)] + [(x, 6) for x in range(9, 4, -1)] + [(x, 7) for x in range(5, 10)]
             + [(x, 8) for x in range(9, 4, -1)] + [(x, 9) for x in range(5, 10)])
try:
    from engine_contract import MARKET_PARAMS
except ImportError:
    from hazel_runtime.engine_contract import MARKET_PARAMS
BASE_PRICE = {item: spec['base'] for item, spec in MARKET_PARAMS.items()}


def check(params):
    """The stage's settings with defaults filled in; ValueError on a value it cannot run with."""
    p = {**PARAMETERS, **(params or {})}
    if p['_LP_USE'] not in USES:
        raise ValueError(f"_LP_USE must be one of {list(USES)}")
    for name, (lo, hi) in LIMITS.items():
        if not lo <= p[name] <= hi:
            raise ValueError(f'{name} must be between {lo} and {hi}')
    return p


def _fib(n):
    """The environment's hire cost after n hires today (1, 1, 2, 3, 5, 8, ...)."""
    a, b = 1, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def _walk(pos, target):
    if pos[0] != target[0]:
        return ['EAST' if pos[0] < target[0] else 'WEST']
    if pos[1] != target[1]:
        return ['SOUTH' if pos[1] < target[1] else 'NORTH']
    return None


def _dist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _qty(order):
    try:
        return max(0, int(order[2])) if len(order) > 2 else 1
    except (TypeError, ValueError):
        return 0


def _place(orders, order):
    """Append `order` while the list is shorter than 10. The engine's own entries, its deliberate empty ones
    included, keep their indices: both seats' lists clear index by index, so an order put into an empty entry
    would clear together with the rival's order at that index (a wheat purchase lifting its wheat sale)."""
    if len(orders) < SLOTS:
        orders.append(order)
        return True
    return False


def _free_slots(orders):
    return max(0, SLOTS - len(orders))


def _spend(orders, farm, prices):
    """Upper bound on what `orders` cost (sales ignored), and the hires among them."""
    cost, hires, land = 0, 0, 0
    hired = int(farm.get('hires_today', 0) or 0)
    owned = len(farm.get('unlocked_quadrants') or ['NW']) - 1
    for o in orders:
        if not isinstance(o, (list, tuple)) or not o:
            continue
        op, item = o[0], (o[1] if len(o) > 1 else None)
        if op == 'HIRE':
            cost += _fib(hired + hires)
            hires += 1
        elif op == 'BUY_LAND':
            if owned + land < len(LAND_PRICES):
                cost += LAND_PRICES[owned + land]
            land += 1
        elif op == 'BUY_SEED':
            cost += SEED_COST.get(item, 100) * _qty(o)
        elif op == 'BUY_ANIMAL':
            cost += ANIMAL_COST.get(item, 500) * _qty(o)
        elif op == 'BUY_PRODUCT':
            cost += (int(prices.get(item, 100) or 100) + 5) * _qty(o)
    return cost, hires


def similarity(farm, rival):
    """The public engine's clone test (_r37_similarity): 0 unless both own the same quadrants, else the share of
    tiles occupied on either farm that hold the same crop or animal on both (0 below 8 occupied tiles)."""
    if sorted(farm.get('unlocked_quadrants') or []) != sorted(rival.get('unlocked_quadrants') or []):
        return 0.0
    matches = total = 0
    for row_a, row_b in zip(farm.get('tiles') or [], rival.get('tiles') or []):
        for a, b in zip(row_a, row_b):
            sa = (a.get('crop'), a.get('animal')) if isinstance(a, dict) else (None, None)
            sb = (b.get('crop'), b.get('animal')) if isinstance(b, dict) else (None, None)
            if sa != (None, None) or sb != (None, None):
                total += 1
                matches += sa == sb
    return matches / total if total >= 8 else 0.0


def _new_state():
    return {'last_step': -1, 'day': -1, 'phase': 'wait', 'land_step': None, 'workers': {}, 'pending': None,
            'hire_day': -1, 'seeds': 0, 'seed_step': None, 'planted': {}, 'stock': {}, 'seen': {}, 'late': {},
            'geese_bought': 0, 'geese_step': None, 'placing': {}, 'placed_ever': 0, 'fed_day': -1}


class LandPlot:
    """Per-player plot state across a game; `apply` edits our action for one turn (a new dict)."""

    def __init__(self, params=None):
        self.p = check(params)
        self.use = self.p['_LP_USE']
        self.goose = self.use == 'GOOSE'
        self.product = 'EGG' if self.goose else self.use
        self.products = ('EGG', 'FERTILIZER') if self.goose else (self.use,)
        self.tiles = PLOT[:self.p['_LP_TILES']]
        self.states = {}
        self.report = {}

    def _count(self, key, n=1):
        self.report[key] = self.report.get(key, 0) + n

    # ---- the turn ------------------------------------------------------------------------------------------
    def apply(self, obs, action, tape_rest=None):
        """Our action with the plot's orders and worker commands. `tape_rest`: the engine's remaining tape
        actions of the day (to hire after its hires), or None when the engine has no tape."""
        step = int(obs.get('step', 0) or 0)
        player = int(obs.get('player', 0) or 0)
        st = self.states.get(player)
        if st is None or step <= st['last_step']:
            st = self.states[player] = _new_state()
            self.report = {}
        st['last_step'] = step
        day, hour = step // 24, step % 24
        if day < self.p['_LP_DAY'] or st['phase'] == 'off' or not isinstance(action, dict):
            return action
        farms = obs.get('farms') or []
        if player >= len(farms):
            return action
        farm, private = farms[player], obs.get('private') or {}
        out = {'farmer': action.get('farmer'), 'hands': [list(h) if isinstance(h, (list, tuple)) else ['PASS']
                                                         for h in (action.get('hands') or [])],
               'market': [list(o) if isinstance(o, (list, tuple)) else [] for o in (action.get('market') or [])]}
        prices = (obs.get('market') or {}).get('prices') or {}
        self._account(st, farm, private, step, day)
        if st['phase'] == 'wait':
            rival = farms[1 - player] if len(farms) > 1 else {}
            if similarity(farm, rival) >= self.p['_LP_MIRROR']:
                self._count('mirror_turns')
            else:
                self._buy_land(st, farm, prices, out, step, day, hour)
        if st['phase'] == 'farm':
            self._hire(st, farm, prices, out, day, hour, tape_rest)
            self._inputs(st, farm, private, prices, out, day, hour)
            self._work(st, farm, private, out, step, day, hour)
        self._sell(st, farm, private, prices, out, day)
        return out

    # ---- bookkeeping from the observation --------------------------------------------------------------------
    def _account(self, st, farm, private, step, day):
        unlocked = farm.get('unlocked_quadrants') or []
        if st['phase'] == 'wait' and 'SE' in unlocked:
            if st['land_step'] is not None:
                st['phase'] = 'farm'
                self._count('land_day', day)
            else:
                st['phase'] = 'off'           # the engine owns SE: its plan, not ours
        elif st['phase'] == 'wait':
            st['land_step'] = None
        invs = private.get('inventories') or []
        if st['day'] != day:                  # a new day: last night every inventory went into the shed
            carried = {}
            for seen in list(st['seen'].values()) + [st['late']]:
                for item, n in seen.items():
                    carried[item] = carried.get(item, 0) + n
            for item, n in carried.items():
                st['stock'][item] = st['stock'].get(item, 0) + n
            st.update(day=day, workers={}, pending=None, seen={}, late={})
        pending = st['pending']
        hands = farm.get('hands') or []
        if pending is not None:
            count = min(pending['count'], len(hands) - pending['first'] + 1)   # hires the money did not cover
            for k in range(max(0, count)):
                share = self.tiles[k * len(self.tiles) // count:(k + 1) * len(self.tiles) // count]
                st['workers'][pending['first'] + k] = {'tiles': list(share)}
            self._count('hands_confirmed', max(0, count))
            if count < pending['count']:
                self._count('hire_shortfalls', pending['count'] - max(0, count))
            st['pending'] = None
        # produce our workers put into the shed during the day (DROP / PLACE there) moves to the stock
        for actor in st['workers']:
            inv = invs[actor] if actor < len(invs) and isinstance(invs[actor], dict) else {}
            seen = st['seen'].get(actor, {})
            for item in self.products:
                now, before = int(inv.get(item, 0) or 0), seen.get(item, 0)
                if now < before:
                    st['stock'][item] = st['stock'].get(item, 0) + before - now
            st['seen'][actor] = {item: int(inv.get(item, 0) or 0) for item in self.products}
        for pos in list(st['planted']):
            tile = self._tile(farm, pos)
            if not (isinstance(tile, dict) and tile.get('kind') == 'PLANT'):
                st['seeds'] += 1              # the PLANT did not happen: the seed is still ours
            else:
                self._count('plants')
            del st['planted'][pos]
        if not self.goose and st['seed_step'] is not None and step > st['seed_step']:
            # a purchase the money did not cover, or our seeds planted by the engine's own units
            st['seeds'] = max(0, min(st['seeds'], int((private.get('seeds') or {}).get(self.use, 0) or 0)))
        for pos in list(st['placing']):
            tile = self._tile(farm, pos)
            if isinstance(tile, dict) and tile.get('animal') == 'GOOSE':
                st['placed_ever'] += 1
                self._count('geese_placed')
            del st['placing'][pos]
        if self.goose and st['geese_step'] is not None and step > st['geese_step']:
            carried = sum(int((invs[a] if a < len(invs) and isinstance(invs[a], dict) else {}).get('GOOSE', 0) or 0)
                          for a in st['workers'])
            unplaced = min(st['geese_bought'] - st['placed_ever'],
                           int((private.get('shed') or {}).get('GOOSE', 0) or 0) + carried)
            st['geese_bought'] = st['placed_ever'] + max(0, unplaced)

    @staticmethod
    def _tile(farm, pos):
        tiles = farm.get('tiles') or []
        x, y = pos
        return tiles[y][x] if 0 <= y < len(tiles) and 0 <= x < len(tiles[y]) else 'LOCKED'

    # ---- land ------------------------------------------------------------------------------------------------
    def _buy_land(self, st, farm, prices, out, step, day, hour):
        if day > self.p['_LP_LAST_BUY_DAY']:
            st['phase'] = 'off'
            return
        unlocked = farm.get('unlocked_quadrants') or ['NW']
        if st['land_step'] is not None or hour > 20:
            return
        if any(o and o[0] == 'BUY_LAND' for o in out['market']):
            return                            # the engine buys land this turn; ours follows next turn
        missing = [q for q in LAND_ORDER if q not in unlocked]
        owned = len(unlocked) - 1
        cost = sum(LAND_PRICES[owned + i] for i in range(len(missing)) if owned + i < len(LAND_PRICES))
        spend, _ = _spend(out['market'], farm, prices)
        money = float(farm.get('money', 0) or 0)
        if not missing or money - spend - cost < self.p['_LP_MIN_MONEY'] or _free_slots(out['market']) < len(missing):
            return
        for _ in missing:
            _place(out['market'], ['BUY_LAND'])
        st['land_step'] = step
        self._count('land_orders', len(missing))

    # ---- hands -----------------------------------------------------------------------------------------------
    def _work_left(self, farm, day):
        if self.goose:
            return day <= LAST_DAY
        if day <= self.p['_LP_LAST_PLANT_DAY']:
            return True
        return any(isinstance(self._tile(farm, t), dict) and self._tile(farm, t).get('kind') == 'PLANT'
                   for t in self.tiles)

    def _hire(self, st, farm, prices, out, day, hour, tape_rest):
        if st['hire_day'] == day or st['pending'] is not None or not self._work_left(farm, day):
            return
        if hour < self.p['_LP_HIRE_HOUR'] or hour > 20 or (day == LAST_DAY and hour > 12):
            return
        if any(o and o[0] == 'HIRE' for o in out['market']):
            return                            # engine hires this turn: ours must come after them
        if tape_rest is not None and any(o and o[0] == 'HIRE' for a in tape_rest if isinstance(a, dict)
                                         for o in (a.get('market') or [])):
            return                            # the tape still hires today
        count = self.p['_LP_WORKERS']
        spend, _ = _spend(out['market'], farm, prices)
        hired = int(farm.get('hires_today', 0) or 0)
        money = float(farm.get('money', 0) or 0) - spend - self.p['_LP_RESERVE']
        while count and sum(_fib(hired + k) for k in range(count)) > money:
            count -= 1
        count = min(count, _free_slots(out['market']))
        if count <= 0:
            self._count('hire_declines')
            return
        for _ in range(count):
            _place(out['market'], ['HIRE'])
        st['pending'] = {'first': len(farm.get('hands') or []) + 1, 'count': count}
        st['hire_day'] = day
        self._count('hires', count)

    # ---- seeds, geese, feed ----------------------------------------------------------------------------------
    def _budget(self, farm, prices, out):
        spend, _ = _spend(out['market'], farm, prices)
        return float(farm.get('money', 0) or 0) - spend - self.p['_LP_RESERVE']

    def _inputs(self, st, farm, private, prices, out, day, hour):
        if not st['workers'] and st['pending'] is None:
            return
        if self.goose:
            self._goose_inputs(st, farm, private, prices, out, day)
            return
        crop = self.use
        if not self._can_plant(day, hour):
            return
        empty = sum(1 for t in self.tiles if self._tile(farm, t) is None
                    or (isinstance(self._tile(farm, t), dict) and self._tile(farm, t).get('kind') == 'WEED'))
        need = empty - st['seeds']
        if need <= 0:
            return
        need = min(need, int(self._budget(farm, prices, out) // SEED_COST[crop]))
        if need > 0 and _place(out['market'], ['BUY_SEED', crop, need]):
            st['seeds'] += need
            st['seed_step'] = st['last_step']
            self._count('seeds_bought', need)

    def _can_plant(self, day, hour):
        if day > self.p['_LP_LAST_PLANT_DAY'] or hour > 21:
            return False
        if self.use in PEAK_AGE:
            return day + PEAK_AGE[self.use] <= LAST_DAY
        return day + FIRST_YIELD[self.use] <= LAST_DAY - 1

    def _our_geese(self, farm):
        return sum(1 for t in self.tiles if isinstance(self._tile(farm, t), dict)
                   and self._tile(farm, t).get('animal') == 'GOOSE')

    def _goose_inputs(self, st, farm, private, prices, out, day):
        placed = self._our_geese(farm)
        unplaced = max(0, st['geese_bought'] - st['placed_ever'])
        want = len(self.tiles) - placed - unplaced
        if day <= LAST_DAY - 5 and want > 0:  # a goose lays from 4 days after it is placed
            n = min(want, int(self._budget(farm, prices, out) // ANIMAL_COST['GOOSE']))
            if n > 0 and _place(out['market'], ['BUY_ANIMAL', 'GOOSE', n]):
                st['geese_bought'] += n
                st['geese_step'] = st['last_step']
                self._count('geese_bought', n)
        # feed: one wheat per goose a day, bought with the day's hires (the hands pick it up next turn)
        if st['pending'] is not None and st['fed_day'] != day:
            price = int(prices.get('WHEAT', 25) or 25) + 5
            need = min(placed + unplaced, int(self._budget(farm, prices, out) // price))
            if need > 0 and _place(out['market'], ['BUY_PRODUCT', 'WHEAT', need]):
                self._count('feed_bought', need)
            st['fed_day'] = day

    # ---- worker commands -------------------------------------------------------------------------------------
    def _work(self, st, farm, private, out, step, day, hour):
        if not st['workers']:
            return
        hands = farm.get('hands') or []
        invs = private.get('inventories') or []
        seeds_left = int((private.get('seeds') or {}).get(self.use, 0) or 0) if not self.goose else 0
        if not self.goose:                    # seeds the engine's own units plant this turn come first
            units = [out.get('farmer')] + [h for i, h in enumerate(out['hands'], 1) if i not in st['workers']]
            seeds_left -= sum(1 for u in units if isinstance(u, (list, tuple)) and len(u) > 1
                              and u[0] == 'PLANT' and u[1] == self.use)
        shed = dict(private.get('shed') or {})
        for actor in sorted(st['workers']):
            if actor > len(hands):
                continue
            pos = tuple(hands[actor - 1])
            inv = invs[actor] if actor < len(invs) and isinstance(invs[actor], dict) else {}
            if self.goose:
                cmd = self._goose_command(st, farm, shed, st['workers'][actor]['tiles'], pos, inv, step, day, hour)
            else:
                cmd, seeds_left = self._crop_command(st, farm, st['workers'][actor]['tiles'], pos, inv,
                                                     step, day, hour, seeds_left)
            tile = self._tile(farm, pos)
            if hour == 23 and isinstance(tile, dict):   # tonight's drop, after the last snapshot of the day
                if cmd[0] == 'HARVEST' and int(tile.get('yield_units', 0) or 0) > 0:
                    item = self.product if self.goose else tile.get('crop')
                    st['late'][item] = st['late'].get(item, 0) + int(tile.get('yield_units', 0) or 0)
                elif cmd[0] == 'COLLECT_FERTILIZER' and tile.get('fertilizer_available'):
                    st['late']['FERTILIZER'] = st['late'].get('FERTILIZER', 0) + 1
            if cmd[0] == 'PLACE':
                st['placing'][pos] = step
            while len(out['hands']) < actor:
                out['hands'].append(['PASS'])
            out['hands'][actor - 1] = cmd

    def _deliver(self, pos, inv, step):
        """On the last day, walk the carried produce into the shed before the final sale."""
        carrying = sum(int(inv.get(item, 0) or 0) for item in self.products)
        if not carrying:
            return None
        home = min(SHED_ACCESS, key=lambda s: _dist(pos, s))
        if step < LAST_DAY * 24 + 16 - _dist(pos, home):
            return None
        return _walk(pos, home) or ['DROP']

    def _crop_command(self, st, farm, tiles, pos, inv, step, day, hour, seeds_left):
        if day == LAST_DAY:
            cmd = self._deliver(pos, inv, step)
            if cmd:
                return cmd, seeds_left
        can_plant = self._can_plant(day, hour) and st['seeds'] > 0 and seeds_left > 0
        tasks = []
        for t in tiles:
            tile = self._tile(farm, t)
            if tile is None:
                if can_plant:
                    tasks.append((3, t, ['PLANT', self.use]))
            elif isinstance(tile, dict) and tile.get('kind') == 'WEED':
                tasks.append((2, t, ['DIG']))
            elif isinstance(tile, dict) and tile.get('kind') == 'PLANT':
                if self._ripe(tile, step, day):
                    tasks.append((1, t, ['HARVEST']))
                elif not tile.get('watered_today') and day < LAST_DAY:
                    tasks.append((0, t, ['WATER']))
        if not tasks:
            return ['PASS'], seeds_left
        prio, target, cmd = min(tasks, key=lambda v: (_dist(pos, v[1]), v[0]))
        walk = _walk(pos, target)
        if walk:
            return walk, seeds_left
        if cmd[0] == 'PLANT':
            st['seeds'] -= 1
            st['planted'][target] = step
            seeds_left -= 1
        elif cmd[0] == 'HARVEST':
            self._count('harvests')
        return cmd, seeds_left

    def _ripe(self, tile, step, day):
        units = int(tile.get('yield_units', 0) or 0)
        if units <= 0:
            return False
        crop = tile.get('crop')
        if crop in FIRST_YIELD or day >= LAST_DAY:
            return True                       # ongoing crops: every yield as it comes
        age = day - int(tile.get('planted_day', day) or day)
        decay = int(tile.get('max_lifespan_step', -1) or -1)
        return age >= PEAK_AGE.get(crop, 4) or (decay >= 0 and step >= decay)

    def _goose_command(self, st, farm, shed, tiles, pos, inv, step, day, hour):
        """One goose hand's command. Hands appear at the shed each morning and load the day's feed and every goose
        their share still lacks in one visit; then, tile by tile: feed and place, build the coops (a coop is built
        and its goose placed on the next turn), care for, harvest and collect fertilizer. A trip back to the shed
        is made for hungry geese first, and for geese only once every coop is built."""
        if day == LAST_DAY:
            cmd = self._deliver(pos, inv, step)
            if cmd:
                return cmd
        wheat, geese_in_hand = int(inv.get('WHEAT', 0) or 0), int(inv.get('GOOSE', 0) or 0)
        hungry = homes = 0                    # geese not fed today; coops and free tiles still without a goose
        urgent, build, care = [], [], []
        for t in tiles:
            tile = self._tile(farm, t)
            if tile is None:
                homes += 1
                build.append((t, ['BUILD_COOP']))
            elif not isinstance(tile, dict):
                continue
            elif tile.get('kind') == 'WEED':
                homes += 1
                build.append((t, ['DIG']))
            elif tile.get('kind') == 'COOP' and tile.get('animal') is None:
                homes += 1
                if geese_in_hand > 0:
                    urgent.append((t, ['PLACE', 'GOOSE', 1]))
            elif tile.get('kind') == 'COOP' and tile.get('animal') == 'GOOSE':
                if not tile.get('fed_today'):
                    # a goose survives its first day unfed: no shed trip for one placed today
                    hungry += int(tile.get('placed_day', day) if tile.get('placed_day') is not None else day) < day
                    if wheat > 0:
                        urgent.append((t, ['FEED']))
                        continue
                if not tile.get('cared_today'):
                    care.append((t, ['CARE']))
                elif int(tile.get('yield_units', 0) or 0) > 0:
                    care.append((t, ['HARVEST']))
                elif tile.get('fertilizer_available'):
                    care.append((t, ['COLLECT_FERTILIZER']))
        shed_wheat = int(shed.get('WHEAT', 0) or 0)
        ours = min(int(shed.get('GOOSE', 0) or 0), max(0, st['geese_bought'] - st['placed_ever'] - geese_in_hand))
        need_wheat = min(hungry - wheat, shed_wheat) if hungry > wheat else 0
        need_geese = min(homes - geese_in_hand, ours) if homes > geese_in_hand else 0
        if pos in SHED_ACCESS:
            if need_wheat > 0:
                shed['WHEAT'] = shed_wheat - need_wheat
                self._count('feed_pickups', need_wheat)
                return ['PICKUP', 'WHEAT', need_wheat]
            if need_geese > 0:
                shed['GOOSE'] = int(shed.get('GOOSE', 0) or 0) - need_geese
                return ['PICKUP', 'GOOSE', need_geese]
        home = min(SHED_ACCESS, key=lambda s: _dist(pos, s))
        for group, trip in ((urgent, False), (None, need_wheat > 0), (build, False), (None, need_geese > 0),
                            (care, False)):
            if group is None:
                if trip:
                    return _walk(pos, home) or ['PASS']
            elif group:
                target, cmd = min(group, key=lambda v: _dist(pos, v[0]))
                walk = _walk(pos, target)
                if walk:
                    return walk
                self._count({'PLACE': 'place_orders', 'FEED': 'feeds', 'HARVEST': 'harvests',
                             'COLLECT_FERTILIZER': 'fertilizer_collected'}.get(cmd[0], 'other_work'))
                return cmd
        return ['PASS']

    # ---- sales -----------------------------------------------------------------------------------------------
    def _sell(self, st, farm, private, prices, out, day):
        shed = private.get('shed') or {}
        crowded = sum(int(v or 0) for v in shed.values()) >= CROWDED   # the engine needs the room for its produce
        for item in self.products:
            n = st['stock'].get(item, 0)
            if n <= 0:
                continue
            price = int(prices.get(item, 0) or 0)
            if price <= 1 or (day < 28 and n < HOLD_CAP and not crowded
                              and price < self.p['_LP_SELL_RATIO'] * BASE_PRICE.get(item, 100)):
                continue
            selling = sum(_qty(o) for o in out['market'] if o and o[0] == 'SELL' and len(o) > 1 and o[1] == item)
            keep = 0
            if item == 'WHEAT':               # the engine's animals eat shed wheat: never sell their two days' feed
                keep = 2 * sum(1 for row in farm.get('tiles') or [] for t in row
                               if isinstance(t, dict) and t.get('animal'))
            qty = min(n, int(shed.get(item, 0) or 0) - selling - keep, BATCH.get(item, n))
            if qty <= 0:
                if int(shed.get(item, 0) or 0) <= 0:
                    st['stock'][item] = 0     # the shed has none left (the engine used or sold it)
                continue
            if _place(out['market'], ['SELL', item, qty]):
                st['stock'][item] = n - qty
                self._count(f'sold_{item.lower()}', qty)
