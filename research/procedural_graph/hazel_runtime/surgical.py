"""Opt-in replacements at frozen champion statement boundaries.

No champion globals/source bytes are changed. Score thresholds are heuristic
ranking cutoffs, not calibrated probabilities or predictions of exact peaks.
"""
from __future__ import annotations

import ast


def enabled(config):
    if config is None:
        return False
    if not isinstance(config, dict) or set(config) - {'enabled'}:
        raise ValueError('surgical config must contain only enabled')
    value = config.get('enabled', False)
    if type(value) is not bool:
        raise ValueError('surgical.enabled must be boolean')
    return value


def _at(source, line):
    """Anchor injected statements to their original graph stage, not snippet lines."""
    nodes = ast.parse(source).body
    for node in nodes:
        for child in ast.walk(node):
            if hasattr(child, 'lineno'):
                for attr, value in (('lineno', line), ('end_lineno', line),
                                    ('col_offset', 0), ('end_col_offset', 0)):
                    setattr(child, attr, value)
    return nodes


_DEFERRED = '''
_surgical_expanding = 0 <= hour <= 3 and any(
    isinstance(o, (list, tuple)) and o and o[0] in ('HIRE', 'BUY_SEED')
    for o in (base_orders or []))
for d_it in list(deferred):
    d_q, d_step = deferred[d_it]
    held_d = int(shed.get(d_it, 0) or 0)
    q_d = min(int(d_q), held_d - _already_selling(mkt, d_it))
    if q_d <= 0 or step - d_step > _DEFERRED_TTL:
        deferred.pop(d_it, None)
        continue
    # At most seven occupied slots after injection; no eviction, no reordering.
    if not _surgical_expanding and len(mkt) < 7 and _add_sell(mkt, d_it, q_d):
        deferred.pop(d_it, None)
'''

# This guard also prevents the preceding pressure valve and later routine
# fertilizer sales from consuming the floor before/after headroom executes.
_FERTILIZER_GUARD = '''
if it == 'FERTILIZER' and step < 696:
    _fert_held = max(0, int(shed.get(it, 0) or 0))
    _fert_floor = min(_fert_held, max(0, int(st.get('n_plants', 0))) * 2)
    q = min(q, max(0, _fert_held - _already_selling(orders, it) - _fert_floor))
'''

_HEADROOM = '''
if hour >= 22 and step < 712:
    def _surgical_need():
        # Orders are stock-bounded individually by product; duplicate or oversized
        # base sells must not manufacture headroom that cannot actually execute.
        scheduled = sum(min(max(0, int(q or 0)), max(0, _already_selling(mkt, it)))
                        for it, q in shed.items())
        return max(0, shed_used + int(st.get('carried', 0) or 0)
                   - scheduled - (_SHED_CAPACITY - _HEADROOM_MARGIN))
    cheapest = sorted(
        ((float(prices.get(it, 0) or 0), int(q or 0), it)
         for it, q in shed.items() if int(q or 0) > 0),
        key=lambda x: (x[2] != 'WHEAT', x[0], -x[1]))
    for price_h, qty_held, item in cheapest:
        need = _surgical_need()
        if need <= 0:
            break
        if price_h < 1.0:
            continue
        reserve = feed_reserve if item == 'WHEAT' else (
            min(qty_held, max(0, int(st.get('n_plants', 0))) * 2)
            if item == 'FERTILIZER' and step < 696 else 0)
        avail = qty_held - _already_selling(mkt, item) - reserve
        qty = min(avail, need)
        if qty > 0:
            # Deliberately unprotected: never lock cheap clearance ahead of value.
            _add_sell(mkt, item, qty)
'''

_LUXURY = '''
if _ENABLE_ORACLE_FRONTRUN and _ORACLE_FROM_STEP <= step < 712:
    _lux_scores = _oracle_scores(st, 'score_24')
    _lux_ranked = sorted(
        ((it, float(_lux_scores.get(it, 0) or 0)) for it in ('WOOL', 'MILK')),
        key=lambda pair: -pair[1])
    for item, score in _lux_ranked:
        # A ranking cutoff only: neither a sale probability nor an exact peak.
        if not 0.30 <= score < float('inf'):
            continue
        price = float(prices.get(item, 0) or 0)
        min_ratio = 0.70 if item == 'WOOL' else 0.60
        if not min_ratio * _BASE_PRICE[item] <= price < float('inf'):
            continue
        already = _already_selling(mkt, item)
        avail = int(shed.get(item, 0) or 0) - already
        qty = min(avail, max(0, min(6, _ORACLE_FRONTRUN_BATCH) - already))
        if qty > 0:
            _add_sell(mkt, item, qty)
'''


def rewrite_market(function):
    """Rewrite only known source anchors, fail closed on an unexpected source AST."""
    anchors = {statement.lineno: statement for statement in function.body}
    for line, kind in ((314, ast.FunctionDef), (365, ast.For),
                       (458, ast.If), (480, ast.If)):
        if not isinstance(anchors.get(line), kind):
            raise ValueError(f'Surgical source boundary mismatch at {line}')
    add_sell = anchors[314]
    if add_sell.name != '_add_sell':
        raise ValueError('Surgical sell helper mismatch')
    add_sell.body[:0] = _at(_FERTILIZER_GUARD, 314)
    body = []
    for statement in function.body:
        if statement.lineno == 365:
            body.extend(_at(_DEFERRED, 365))
        elif statement.lineno == 458:
            body.extend(_at(_HEADROOM, 458))
        else:
            body.append(statement)
        if statement.lineno == 480:
            body.extend(_at(_LUXURY, 514))
    function.body = body
    return function


def idle_dispatch(obs, player, base, state, champion):
    """Replace the faulty champion rescues, starting ONLY from backbone actions.

    Actor positions are (x,y), tiles are [y][x], inventory 0 is the farmer.
    Never move speculatively; reserve non-PASS tile work across all actors before
    assigning rescue so an idle farmer cannot duplicate a later hand's work.
    """
    positions = [state.get('farmer')] + list(state.get('hands') or [])
    hand_actions = list(base.get('hands') or [])
    raw = [base.get('farmer')] + [
        hand_actions[i] if i < len(hand_actions) else None
        for i in range(len(positions) - 1)]
    actions = [list(a) if isinstance(a, (list, tuple)) and a else
               ([a] if isinstance(a, str) else ['PASS']) for a in raw]
    tiles = state.get('tiles') or []
    inventories = (obs.get('private') or {}).get('inventories') or []
    claimed = set()
    moves = {'NORTH', 'SOUTH', 'EAST', 'WEST', 'PASS'}
    for pos, action in zip(positions, actions):
        if isinstance(pos, (list, tuple)) and len(pos) == 2 and action[0] not in moves:
            claimed.add(tuple(pos))
    for i, (pos, action) in enumerate(zip(positions, actions)):
        if action[0] != 'PASS' or (i and not champion._ENABLE_HAND_RESCUE):
            continue
        if not isinstance(pos, (list, tuple)) or len(pos) != 2:
            continue
        x, y = pos
        if not isinstance(x, int) or not isinstance(y, int):
            continue
        if (x, y) in claimed or not (0 <= y < len(tiles) and 0 <= x < len(tiles[y])):
            continue
        tile = tiles[y][x]
        if not isinstance(tile, dict):
            continue
        op = None
        if tile.get('kind') == 'WEED':
            op = 'DIG'
        elif (tile.get('kind') == 'PLANT' and not tile.get('watered_today')
              and int(tile.get('consecutive_unwatered', 0) or 0) >= champion._WATER_RESCUE_THRESHOLD):
            op = 'WATER'
        elif tile.get('animal') and not tile.get('fed_today'):
            unfed = int(tile.get('consecutive_unfed', 0) or 0)
            bag = inventories[i] if i < len(inventories) else {}
            if ((unfed >= champion._FEED_RESCUE_THRESHOLD or state['hour'] >= 18)
                    and isinstance(bag, dict) and int(bag.get('WHEAT', 0) or 0) > 0):
                op = 'FEED'
        if op:
            actions[i] = [op]
            claimed.add((x, y))
    return actions[0], actions[1:]
