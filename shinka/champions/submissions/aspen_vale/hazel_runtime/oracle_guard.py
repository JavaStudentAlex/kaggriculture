"""Oracle guard: the predictor's lever on a ladder engine's market orders (optional turn stage).

The oracle (kagg_oracle.py) forecasts the opponent's sales per product; `score_4` is the largest
predicted opponent sale over the next 4 turns (log1p units, about P(sale) at the reference
thresholds). When the opponent is about to dump a product we hold, selling first gets the
price before their units push it down. The stage adds such SELL orders to the engine's own
list and changes nothing else: every engine order keeps its slot (both seats' orders clear
index by index, and ladder engines use empty [] entries as deliberate one-slot delays), a new
order fills the first empty entry or is appended while the list is shorter than 10, and stock
the engine already sells, or keeps back (_OG_KEEP), is never sold twice.

Parameters (graph `parameters` on the oracle_guard turn node; defaults = Hazel's front-run):
"""
from __future__ import annotations

PARAMETERS = {
    '_OG_SCORE': 0.30,          # score_4 at/above which the opponent is about to sell (front-run)
    '_OG_STRONG_SCORE': 0.60,   # at/above this, sell all free stock of the product
    '_OG_BATCH': 6,             # units per front-run order below the strong score
    '_OG_PRICE_RATIO': 0.60,    # front-run only while price >= ratio x the product's base price
    '_OG_ITEMS': ('MILK', 'WOOL', 'STRAWBERRY', 'EGG', 'CARROT', 'TOMATO'),  # never WHEAT (feed)
    '_OG_KEEP': 0,              # units of each product left in the shed
    '_OG_FROM_STEP': 256,       # the forecast starts at the checkpoint's context length
    '_OG_TO_STEP': 696,         # the engine's own endgame liquidation owns the last day
    '_OG_MAX_ORDERS': 2,        # front-run orders added per turn
}
NOTES = {
    '_OG_SCORE': 'score_4 threshold: the opponent is predicted to sell this product within 4 turns',
    '_OG_STRONG_SCORE': 'at/above this score_4 all free stock of the product is sold',
    '_OG_BATCH': 'units per front-run order below the strong score',
    '_OG_PRICE_RATIO': 'minimum price as a share of the base price (never sell into a crashed market)',
    '_OG_ITEMS': 'products the guard may sell (wheat is feed and seed stock; melon forecasts are weak)',
    '_OG_KEEP': 'units of each product always left in the shed',
    '_OG_FROM_STEP': 'first step the guard acts (no forecast before 256)',
    '_OG_TO_STEP': 'last step the guard acts',
    '_OG_MAX_ORDERS': 'front-run orders the guard may add in one turn',
}
try:
    from engine_contract import MARKET_PARAMS
except ImportError:
    from hazel_runtime.engine_contract import MARKET_PARAMS
BASE_PRICE = {item: spec['base'] for item, spec in MARKET_PARAMS.items()}   # the engine's base prices
SLOTS = 10


def _selling(orders, item):
    return sum(int(o[2] or 0) for o in orders
               if isinstance(o, (list, tuple)) and len(o) >= 3 and o[0] == 'SELL' and o[1] == item)


def apply(obs, orders, state, params):
    """The engine's market orders plus the guard's front-run sells (a new list; input untouched)."""
    p = {**PARAMETERS, **(params or {})}
    step = int(obs.get('step', 0) or 0)
    forecast = (state or {}).get('oracle') or {}
    scores = forecast.get('score_4') if isinstance(forecast, dict) else None
    out = [list(o) if isinstance(o, (list, tuple)) else o for o in (orders or [])]
    if not scores or not (p['_OG_FROM_STEP'] <= step <= p['_OG_TO_STEP']):
        return out
    shed = (state or {}).get('shed') or {}
    prices = (state or {}).get('prices') or {}
    added = 0
    for item, score in sorted(((i, float(scores.get(i) or 0.0)) for i in p['_OG_ITEMS']), key=lambda kv: -kv[1]):
        if added >= p['_OG_MAX_ORDERS'] or score < p['_OG_SCORE']:
            break
        price = float(prices.get(item, 0) or 0)
        if price < p['_OG_PRICE_RATIO'] * BASE_PRICE.get(item, 100):
            continue
        free = int(shed.get(item, 0) or 0) - _selling(out, item) - int(p['_OG_KEEP'])
        qty = free if score >= p['_OG_STRONG_SCORE'] else min(free, int(p['_OG_BATCH']))
        if qty <= 0:
            continue
        order = ['SELL', item, qty]
        slot = next((i for i, o in enumerate(out[:SLOTS]) if not o), None)
        if slot is not None:
            out[slot] = order
        elif len(out) < SLOTS:
            out.append(order)
        else:
            break
        added += 1
    return out
