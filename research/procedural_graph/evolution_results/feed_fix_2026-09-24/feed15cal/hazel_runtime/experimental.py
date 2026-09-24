"""Opt-in research stages, not a promoted policy. No seat-priority assumption.

Only contiguous SELL blocks may be reordered. All other orders are dependency
barriers. Capacity sales are appended, never substituted for production orders.
Unknown unit inventory effects disable quantity-changing market extensions.
"""
from __future__ import annotations

from copy import deepcopy
import math

try:
    from engine_contract import CROPS, PRODUCTS, SHOPS, market_price
except ImportError:
    from hazel_runtime.engine_contract import CROPS, PRODUCTS, SHOPS, market_price

STAGES = (
    ('experimental_state', 'Project known unit inventory effects and actual input needs'),
    ('shop_phase', 'Compute post-market town consumption and conditional next quotes'),
    ('capacity_liquidation', 'Append net headroom relief without spending protected inputs'),
    ('opponent_pressure', 'Rank contiguous sells by opponent pressure symmetrically'),
    ('idle_opportunities', 'Guarded idle weed reclamation; unsupported planting is a no-op'),
    ('order_arbitration', 'Stock/reserve/cap checks with dependency barriers'),
)
SWITCHES = ('shop_phase', 'capacity_liquidation', 'opponent_pressure',
            'idle_opportunities', 'order_arbitration')


def settings(raw):
    raw = raw or {}
    if not isinstance(raw, dict) or set(raw) - set(SWITCHES):
        raise ValueError('Unknown experimental switch')
    if any(type(value) is not bool for value in raw.values()):
        raise ValueError('Experimental switches must be booleans')
    out = {key: raw.get(key, False) for key in SWITCHES}
    if any(out[key] for key in ('capacity_liquidation', 'opponent_pressure')) and not out['order_arbitration']:
        raise ValueError('Market extensions require order_arbitration')
    return out


def cfg(config, key, default):
    return config.get(key, default) if isinstance(config, dict) else getattr(config, key, default)


def quantity(value):
    return max(0, int(value))


def sales(orders):
    result = {}
    for order in orders:
        if isinstance(order, (list, tuple)) and len(order) >= 3 and order[0] == 'SELL':
            result[order[1]] = result.get(order[1], 0) + quantity(order[2])
    return result


def unit_actions(action, farm):
    hands = list(action.get('hands') or [])
    return [action.get('farmer') or ['PASS']] + [
        hands[i] if i < len(hands) else ['PASS'] for i in range(len(farm.get('hands', [])))]


class Extensions:
    def __init__(self, switches):
        self.switches = settings(switches)
        self.telemetry = []
        self.ctx = None

    def event(self, stage, reason, **values):
        self.telemetry.append(dict(stage=stage, reason=reason, **values))

    def start(self):
        self.telemetry = []
        self.ctx = None

    def run(self, stage, obs, player, action, state, configuration):
        if stage == 'experimental_state':
            if any(self.switches.values()):
                self.ctx = self.context(obs, player, action, configuration)
            else:
                self.event(stage, 'all_disabled')
            return action
        if not self.switches[stage]:
            self.event(stage, 'disabled')
            return action
        if self.ctx is None:
            self.event(stage, 'missing_extension_context')
            return action
        # Every stage is transactional: caller retains prior action on exception.
        result = deepcopy(action)
        try:
            getattr(self, stage)(obs, player, result, state or {}, configuration)
            return result
        except Exception as exc:
            self.event(stage, 'extension_failed_closed', error=repr(exc))
            return action

    def context(self, obs, player, action, config):
        farm = deepcopy(obs['farms'][player])
        private = deepcopy(obs['private'])
        shed = private['shed']
        positions = [farm['farmer']] + farm.get('hands', [])
        inventories = private.get('inventories', [])
        inventories += [{} for _ in range(max(0, len(positions) - len(inventories)))]
        cap = int(cfg(config, 'shedCapacity', 100))
        tpd = max(1, int(cfg(config, 'turnsPerDay', 24)))
        step = int(obs.get('step', obs.get('day', 0) * tpd + obs.get('hour', 0)))
        last = int(cfg(config, 'episodeSteps', 720)) - 1
        day = step // tpd
        board = len(farm['tiles'])
        adjacent = {(board // 2 - 1, board // 2 - 1), (board // 2, board // 2 - 1),
                    (board // 2 - 1, board // 2), (board // 2, board // 2)}
        safe = True
        for pos, inv, unit in zip(positions, inventories, unit_actions(action, farm)):
            op = unit[0]
            x, y = pos
            tile = farm['tiles'][y][x]
            if op == 'PICKUP' and tuple(pos) in adjacent and len(unit) >= 2:
                item = unit[1]
                n = min(quantity(unit[2] if len(unit) > 2 else 1), quantity(shed.get(item, 0)))
                shed[item] = shed.get(item, 0) - n
                inv[item] = inv.get(item, 0) + n
            elif op == 'DROP' and tuple(pos) in adjacent:
                lost = 0
                for item, n in list(inv.items()):
                    take = min(quantity(n), max(0, cap - sum(shed.values())))
                    shed[item] = shed.get(item, 0) + take
                    lost += quantity(n) - take
                inv.clear()
                if lost:
                    self.event('experimental_state', 'drop_overflow_precedes_market_cannot_repair', units=lost)
            elif op in ('FEED', 'FERTILIZE'):
                item = 'WHEAT' if op == 'FEED' else 'FERTILIZER'
                valid = isinstance(tile, dict) and (
                    ('animal' in tile and not tile.get('fed_today')) if op == 'FEED'
                    else tile.get('kind') == 'PLANT')
                if valid and inv.get(item, 0) > 0:
                    inv[item] -= 1
                    if op == 'FEED':
                        tile['fed_today'] = True
                    else:
                        tile['fertilized_until_day'] = day + 2
            elif op in ('HARVEST', 'COLLECT_FERTILIZER', 'PLACE', 'PLANT', 'DIG'):
                # These can change yields, placements, seed competition or tile needs.
                # Do not pretend our partial projection is a full engine simulator.
                safe = False
        animals = [tile for row in farm['tiles'] for tile in row
                   if isinstance(tile, dict) and tile.get('animal')]
        plants = [tile for row in farm['tiles'] for tile in row
                  if isinstance(tile, dict) and tile.get('kind') == 'PLANT']
        # Feed one outstanding ration today and one next-day ration if it exists.
        next_day = (day + 1) * tpd <= last
        feed_need = sum(not tile.get('fed_today', False) for tile in animals) + len(animals) * next_day
        fertilizer_need = 0
        for tile in plants:
            crop = CROPS.get(tile.get('crop'))
            if not crop or tile.get('fertilized_until_day', -1) >= day:
                continue
            age = day - tile.get('planted_day', day)
            useful = crop['ongoing'] or (crop['max_yield_day'] + 1) // 2 <= age <= crop['max_yield_day']
            if useful and not tile.get('watered_today') and (day + max(0, crop['first_yield_day'] - age)) * tpd <= last:
                fertilizer_need += 1
        carried = {item: sum(quantity(inv.get(item, 0)) for inv in inventories) for item in PRODUCTS}
        reserve = {'WHEAT': max(0, feed_need - carried['WHEAT']),
                   'FERTILIZER': max(0, fertilizer_need - carried['FERTILIZER'])}
        # No arbitrary floor when no animals or eligible plants exist.
        projected = sum(shed.values()) + sum(sum(inv.values()) for inv in inventories)
        orders = action.get('market') or []
        remaining = sales(orders)
        net_sales = sum(min(n, max(0, quantity(shed.get(item, 0)) - reserve.get(item, 0)))
                        for item, n in remaining.items())
        incoming_buys = sum(quantity(o[2]) for o in orders if len(o) >= 3 and o[0] in ('BUY_PRODUCT', 'BUY_ANIMAL'))
        ctx = dict(farm=farm, private=private, shed=shed, reserve=reserve, safe=safe,
                   cap=cap, order_cap=min(10, max(1, int(cfg(config, 'maxMarketOrdersPerTurn', 10)))),
                   step=step, last=last, tpd=tpd, projected=projected,
                   shortage=max(0, projected + incoming_buys - net_sales - cap), future_prices={})
        self.event('experimental_state', 'projected' if safe else 'unknown_unit_effects_quantity_changes_guarded',
                   reserve=reserve, projected=projected, scheduled_sales=net_sales,
                   incoming_buys=incoming_buys, shortage=ctx['shortage'])
        return ctx

    def shop_phase(self, obs, player, action, state, config):
        ctx = self.ctx
        consumption = {item: 0 for item in PRODUCTS}
        if ctx['step'] % max(1, int(cfg(config, 'townShopSellInterval', 4))) == 0:
            for shop in obs.get('town', {}).get('unlocked_shops', []):
                if shop not in SHOPS:
                    self.event('shop_phase', 'unknown_shop_guarded', shop=shop)
                    continue
                products = SHOPS[shop]
                for item in products:
                    consumption[item] += 2 if len(products) == 1 else 1
        if ctx['step'] % max(1, int(cfg(config, 'townCenterSellInterval', 24))) == 0:
            for item in PRODUCTS:
                if item != 'FERTILIZER':
                    consumption[item] += 1
        market = obs.get('market', {})
        inventory = market.get('inventory', {})
        future = {}
        for item in PRODUCTS:
            if item in inventory:
                future[item] = market_price(item, inventory[item] - consumption[item], market.get('params'))
        ctx['future_prices'] = future
        self.event('shop_phase', 'post_market_consumption_not_same_turn_premium', consumption=consumption,
                   current_quotes=market.get('prices', {}), conditional_next_quotes=future,
                   assumption='zero_intervening_trades; quantities are town consumption, not player revenue',
                   hold_reason='no_sale_deferral_without_verified_funding_and_deposit_slack')

    def capacity_liquidation(self, obs, player, action, state, config):
        ctx = self.ctx
        if not ctx['safe']:
            self.event('capacity_liquidation', 'unknown_unit_effects')
            return
        if ctx['step'] % ctx['tpd'] < max(0, ctx['tpd'] - 4):
            self.event('capacity_liquidation', 'outside_predrop_window')
            return
        need = ctx['shortage']
        orders = action.setdefault('market', [])
        scheduled = sales(orders)
        # Never merge into an earlier sell across a purchase/hire barrier.
        blocked = set(scheduled) | {o[1] for o in orders if len(o) > 1 and o[0] == 'BUY_PRODUCT'}
        prices = obs.get('market', {}).get('prices', {})
        candidates = [item for item in PRODUCTS if item not in blocked and prices.get(item, 0) >= 1]
        candidates.sort(key=lambda item: (ctx['future_prices'].get(item, prices[item]), item))
        added = []
        for item in candidates:
            if need <= 0 or len(orders) >= ctx['order_cap']:
                break
            amount = min(need, max(0, quantity(ctx['shed'].get(item, 0)) - ctx['reserve'].get(item, 0)))
            if amount:
                order = ['SELL', item, amount]
                orders.append(order)
                added.append(order)
                need -= amount
        self.event('capacity_liquidation', 'net_relief' if added else 'no_safe_additional_sale',
                   added=added, unresolved=need, note='scheduled_sales_net_already; current DROP cannot be rescued')

    def opponent_pressure(self, obs, player, action, state, config):
        forecast = state.get('oracle') or {}
        scores = forecast.get('score_4') or {}
        def pressure(order):
            value = scores.get(order[1], 0)
            return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else 0.0
        orders = action.setdefault('market', [])
        before = deepcopy(orders)
        start = 0
        while start < len(orders):
            if orders[start][0] != 'SELL':
                start += 1
                continue
            end = start
            while end < len(orders) and orders[end][0] == 'SELL':
                end += 1
            orders[start:end] = sorted(orders[start:end], key=lambda order: -pressure(order))
            start = end
        self.event('opponent_pressure', 'ranked_sell_blocks' if before != orders else 'no_reorder',
                   seat=player, score_semantics='ranking_signal_not_calibrated_units',
                   quote_semantics='both_seats_quoted_before_commit; no_seat_priority')

    def idle_opportunities(self, obs, player, action, state, config):
        ctx = self.ctx
        farm, private = obs['farms'][player], obs['private']
        positions = [farm['farmer']] + farm.get('hands', [])
        units = unit_actions(action, farm)
        tiles = farm['tiles']
        seeds = private.get('seeds', {})
        seed_opportunity = any(seeds.get(crop, 0) > 0 and (obs.get('day', 0) + data['first_yield_day'] + 1) * ctx['tpd'] <= ctx['last']
                               for crop, data in CROPS.items())
        # Existing live production beats speculative land reclamation.
        urgent = any(isinstance(tile, dict) and (
            tile.get('yield_units', 0) > 0 or
            (tile.get('kind') == 'PLANT' and not tile.get('watered_today')) or
            (tile.get('animal') and not tile.get('fed_today')))
            for row in tiles for tile in row)
        claimed = {tuple(pos) for pos, unit in zip(positions, units) if unit != ['PASS']}
        # Claim destinations too; never redirect an already active unit.
        moves = {'NORTH': (0, -1), 'SOUTH': (0, 1), 'EAST': (1, 0), 'WEST': (-1, 0)}
        for (x, y), unit in zip(positions, units):
            if unit[0] in moves:
                dx, dy = moves[unit[0]]
                claimed.add((x + dx, y + dy))
        added = []
        inventories = private.get('inventories', [])
        for index, ((x, y), unit) in enumerate(zip(positions, units)):
            if unit != ['PASS'] or urgent or not seed_opportunity or (x, y) in claimed:
                continue
            if index < len(inventories) and any(inventories[index].values()):
                continue  # preserve deposit/input routes
            tile = tiles[y][x]
            target = None
            if isinstance(tile, dict) and tile.get('kind') == 'WEED':
                units[index] = ['DIG']
                target = (x, y)
            elif ctx['step'] % ctx['tpd'] < ctx['tpd'] - 2:
                for direction, (dx, dy) in moves.items():
                    nx, ny = x + dx, y + dy
                    if 0 <= ny < len(tiles) and 0 <= nx < len(tiles[ny]) and (nx, ny) not in claimed:
                        cell = tiles[ny][nx]
                        if isinstance(cell, dict) and cell.get('kind') == 'WEED':
                            units[index] = [direction]
                            target = (nx, ny)
                            break
            if target is not None:
                claimed.add(target)
                added.append(dict(unit=index, action=units[index], target_xy=list(target)))
        action['farmer'], action['hands'] = units[0], units[1:]
        self.event('idle_opportunities', 'weed_reclamation' if added else 'no_safe_idle_weed_opportunity', added=added,
                   plant_reason='guarded_noop_future_maintenance_and_payoff_not_verified; DIG_before_PLANT; seeds_pre_market')

    def order_arbitration(self, obs, player, action, state, config):
        ctx = self.ctx
        out, seen = [], set()
        shed_dict = ctx['shed'] if isinstance(ctx.get('shed'), dict) else {}
        reserve_dict = ctx['reserve'] if isinstance(ctx.get('reserve'), dict) else {}
        budget = {item: max(0, quantity(shed_dict.get(item, 0)) - reserve_dict.get(item, 0)) for item in PRODUCTS}
        rejected = []
        for order in action.get('market', []):
            if not isinstance(order, (list, tuple)) or not order:
                rejected.append('malformed_order')
                continue
            order = list(order)
            if len(out) >= ctx['order_cap']:
                rejected.append('order_cap_suffix')
                continue
            if order[0] == 'SELL':
                if len(order) != 3 or order[1] not in PRODUCTS:
                    rejected.append('invalid_sell')
                    continue
                item = order[1]
                if item in seen:
                    rejected.append('duplicate_sell_dependency_barrier_not_merged')
                    continue
                seen.add(item)
                # Unknown effects: retain original quantity rather than pretend
                # to know stock. Still reject duplicate sell orders and cap.
                amount = quantity(order[2])
                if ctx['safe']:
                    amount = min(amount, budget[item])
                if not amount:
                    rejected.append('no_unreserved_stock')
                    continue
                order[2] = amount
            out.append(order)
        action['market'] = out
        self.event('order_arbitration', 'stable_dependency_barriers', rejected=rejected,
                   quantity_guard=ctx['safe'], policy='no_HIRE_priority; no_sell_relocation_across_nonSELL; no_assumed_buys')
