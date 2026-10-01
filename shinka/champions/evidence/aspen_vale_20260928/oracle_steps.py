"""Step by step: would the predictor's forecasts have let us beat the rival in Aspen Vale's lost games? (cliproxyapi)

Our seat of every game is replayed through the predictor's live tracker (hazel_runtime/kagg_oracle.OpponentTracker:
observe() our observation, record_action() the action we sent), exactly as our agent runs it, so at every step from
turn 256 we have the forecast it would have had of the rival's sales: score_4 per product, the largest predicted sale
(log1p units) at this step and the next three. Every step is also re-run through the environment's interpreter (as in
flows_av.py), which gives both seats' executed sales unit by unit, with the price and the market inventory before each.

The oracle guard's rule (hazel_runtime/oracle_guard.apply, with the settings the old engine's islands evolved) is then
played over the recorded game step by step: where score_4 >= threshold, the price is at least ratio x base and we hold
free stock (shed - what the engine sells this turn - keep - what the guard already sold earlier), it sells that stock
now (all of it at/above the strong score, else a batch). First-order value of each such sale against what happened:
  - ours: the units at this step's prices (the engine's price curve from this step's inventory, our own units moving
    it) minus what the same units fetched when we really sold them (our next sales of the product; nothing if never);
  - rival: while our units sit in the market earlier than they really did, each unit the rival sells of the product
    meets that much more inventory; the price difference is the rival's loss.
margin change = ours + the rival's loss. Reactions (the engine selling differently later, the rival answering) are
not simulated: the backtest of the validation rounds plays the whole game for that. Forecast quality: whether the rival
really sold the product in those 4 steps (precision) and how many of the rival's selling steps were forecast (recall).

    python oracle_steps.py <index.json> <replay dir> <model dir> <out.json> [--workers 4]
"""
import argparse
import collections
import copy
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

PG = Path('/home/alex/kagg-evo/repo/research/procedural_graph')
os.environ.setdefault('KAGG_OPP_MODEL_SRC', str(PG / 'hazel_runtime' / 'opponent_model'))
os.environ.setdefault('KAGG_ORACLE_BACKEND', 'numpy')
for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):   # one core per game, as on Kaggle
    os.environ.setdefault(_var, '1')
sys.path.insert(0, str(PG / 'hazel_runtime'))
sys.path.insert(0, str(PG / 'rival'))
import kagg_oracle  # noqa: E402
import oracle_guard  # noqa: E402
from emulate_replay import K, simulate  # noqa: E402

EVOLVED = {'_OG_SCORE': 0.5, '_OG_STRONG_SCORE': 0.6, '_OG_BATCH': 4, '_OG_PRICE_RATIO': 0.75, '_OG_KEEP': 2,
           '_OG_FROM_STEP': 256, '_OG_TO_STEP': 696, '_OG_MAX_ORDERS': 2}
GUARD_ITEMS = ('MILK', 'WOOL', 'STRAWBERRY')
OTHER_ITEMS = ('EGG', 'CARROT', 'TOMATO', 'MELON')
VARIANTS = {   # name -> (items, parameter overrides)
    'evolved_0.5': (GUARD_ITEMS, {}),
    'evolved_0.3': (GUARD_ITEMS, {'_OG_SCORE': 0.3}),
    'evolved_0.7': (GUARD_ITEMS, {'_OG_SCORE': 0.7, '_OG_STRONG_SCORE': 0.9}),
    'all_items_0.5': (GUARD_ITEMS + OTHER_ITEMS, {}),
    'other_items_0.5': (OTHER_ITEMS, {}),
}
LOG = []
_commit = K._commit_unit


def _logged_commit(op, item, price, farm, private, market, shed_capacity=100):
    inventory = market['inventory'].get(item)
    ok = _commit(op, item, price, farm, private, market, shed_capacity)
    if ok:
        LOG.append((id(farm), op, item, price, inventory))
    return ok


K._commit_unit = _logged_commit


def market_log(steps, configuration):
    """{seat: {item: [(step, price, inventory before), ...]}}: every SELL unit the market committed, step = the
    observation the order was decided on."""
    sales = {0: collections.defaultdict(list), 1: collections.defaultdict(list)}
    for t in range(1, len(steps)):
        seen = steps[t - 1][0]['observation']
        public = {k: seen[k] for k in ('farms', 'market', 'town', 'day', 'hour')}
        public['step'] = seen.get('step', t - 1)
        privates = [copy.deepcopy(steps[t - 1][s]['observation']['private']) for s in range(2)]
        actions = [steps[t][s].get('action') or {} for s in range(2)]
        LOG.clear()
        farms, _, _, _ = simulate(public, privates, actions, configuration)
        seat_of = {id(farms[s]): s for s in range(2)}
        for fid, op, item, price, inventory in LOG:
            if op == 'SELL':
                sales[seat_of[fid]][item].append((t - 1, price, inventory))
    return sales


def forecasts(steps, configuration, ours, model):
    """{step: score_4 per product}: the live tracker fed our observations and our actions."""
    tracker = kagg_oracle.OpponentTracker(model)
    out = {}
    for t in range(len(steps) - 1):
        obs = dict(steps[t][0]['observation'], player=ours,
                   private=steps[t][ours]['observation'].get('private') or {})
        obs['step'] = t
        forecast = tracker.observe(obs, configuration)
        tracker.record_action(steps[t + 1][ours].get('action') or {})
        if forecast is not None:
            out[t] = {p: float(v) for p, v in forecast['score_4'].items()}
    return out


def selling(orders, item):
    return sum(int(o[2] or 0) for o in orders if isinstance(o, list) and len(o) >= 3 and o[0] == 'SELL' and o[1] == item)


def play_guard(steps, ours, scores, sales, items, params):
    """The guard's rule over the recorded game; per firing its first-order value (see the module docstring)."""
    p = {**EVOLVED, **params}
    rival = 1 - ours
    own = {i: sales[ours].get(i, []) for i in items}
    nxt = {i: 0 for i in items}                  # our first real sale not yet reached / matched
    moved = {i: [] for i in items}               # [front-run step, the real sale step it replaces or None]
    firings = []
    last = min(p['_OG_TO_STEP'], len(steps) - 2)
    for s in range(p['_OG_FROM_STEP'], last + 1):
        score = scores.get(s)
        if not score:
            continue
        obs = steps[s][0]['observation']
        shed = (steps[s][ours]['observation'].get('private') or {}).get('shed') or {}
        prices, inventory = obs['market']['prices'], obs['market']['inventory']
        orders = (steps[s + 1][ours].get('action') or {}).get('market') or []
        added = 0
        for item, sc in sorted(((i, score.get(i, 0.0)) for i in items), key=lambda kv: -kv[1]):
            if added >= p['_OG_MAX_ORDERS'] or sc < p['_OG_SCORE']:
                break
            if prices[item] < p['_OG_PRICE_RATIO'] * oracle_guard.BASE_PRICE.get(item, 100):
                continue
            early = sum(1 for fs, rs in moved[item] if rs is None or rs > s)   # sold early, still in the real shed
            free = int(shed.get(item, 0)) - selling(orders, item) - p['_OG_KEEP'] - early
            qty = free if sc >= p['_OG_STRONG_SCORE'] else min(free, p['_OG_BATCH'])
            if qty <= 0:
                continue
            added += 1
            while nxt[item] < len(own[item]) and own[item][nxt[item]][0] <= s:
                nxt[item] += 1
            now = sum(K.market_price(item, inventory[item] + early + j) for j in range(qty))
            real = own[item][nxt[item]:nxt[item] + qty]
            nxt[item] += len(real)
            then = sum(price for _, price, _ in real)
            for k in range(qty):
                moved[item].append((s, real[k][0] if k < len(real) else None))
            truth = sum(1 for e, _, _ in sales[rival].get(item, []) if s <= e <= s + 3)
            firings.append({'step': s, 'item': item, 'score': round(sc, 3), 'qty': qty, 'now': now, 'then': then,
                            'never_sold': qty - len(real), 'rival_units_4': truth})
    rival_loss = 0.0
    for item in items:
        if not moved[item]:
            continue
        for e, price, inventory in sales[rival].get(item, []):
            shift = sum(1 for fs, rs in moved[item] if fs <= e and (rs is None or rs > e))
            if shift:
                rival_loss += price - K.market_price(item, inventory + shift)
    ours_gain = sum(f['now'] - f['then'] for f in firings)
    return {'firings': firings, 'ours': ours_gain, 'rival_loss': rival_loss, 'margin_change': ours_gain + rival_loss}


def recall(scores, sales, rival, items, threshold):
    """Of the rival's selling steps of these products, the share forecast (score_4 >= threshold) 0-3 steps before."""
    hit = total = 0
    for item in items:
        for e in sorted({e for e, _, _ in sales[rival].get(item, []) if e >= 259}):
            total += 1
            hit += any(scores.get(s, {}).get(item, 0.0) >= threshold for s in range(e - 3, e + 1))
    return hit, total


def one_game(args):
    game, replay_dir, model_dir = args
    replay = json.load(open(f"{replay_dir}/episode-{game['id']}-replay.json"))
    steps, configuration = replay['steps'], replay['configuration']
    ours = game['seat']
    model = kagg_oracle.get_model(model_dir)
    scores = forecasts(steps, configuration, ours, model)
    sales = market_log(steps, configuration)
    out = {'id': game['id'], 'team': game['team'], 'margin': game['diff'], 'seat': ours, 'variants': {}}
    for name, (items, params) in VARIANTS.items():
        r = play_guard(steps, ours, scores, sales, items, params)
        threshold = {**EVOLVED, **params}['_OG_SCORE']
        r['recall'] = recall(scores, sales, 1 - ours, items, threshold)
        out['variants'][name] = r
    out['score_4'] = {s: v for s, v in scores.items() if s % 1 == 0}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('index')
    ap.add_argument('replays')
    ap.add_argument('model')
    ap.add_argument('out')
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    games = sorted(json.load(open(args.index)), key=lambda g: g['diff'])
    with Pool(args.workers) as pool:
        results = pool.map(one_game, [(g, args.replays, args.model) for g in games])
    json.dump(results, open(args.out, 'w'))
    for name in VARIANTS:
        rows = [(r, r['variants'][name]) for r in results]
        fired = [f for _, v in rows for f in v['firings']]
        right = sum(1 for f in fired if f['rival_units_4'] > 0)
        hit = sum(v['recall'][0] for _, v in rows)
        total = sum(v['recall'][1] for _, v in rows)
        change = [v['margin_change'] for _, v in rows]
        flips = sum(1 for r, v in rows if r['margin'] + v['margin_change'] > 0)
        print(f"\n== {name}: {len(fired)} guard sales in {len(rows)} games; forecast right (rival sold it within 4 "
              f"steps) {right}/{len(fired)}; rival selling steps forecast {hit}/{total}; margin change total "
              f"${sum(change):+,.0f} (ours ${sum(v['ours'] for _, v in rows):+,.0f}, rival's loss "
              f"${sum(v['rival_loss'] for _, v in rows):+,.0f}); better in {sum(c > 0 for c in change)} games, worse in "
              f"{sum(c < 0 for c in change)}; losses it would flip (first order): {flips}")
        for r, v in rows:
            if v['firings']:
                items = collections.Counter(f['item'] for f in v['firings'])
                print(f"  {r['id']} {r['margin']:+8,.0f} {r['team'][:16]:16s} sales {dict(items)} right "
                      f"{sum(1 for f in v['firings'] if f['rival_units_4'] > 0)}/{len(v['firings'])} | ours "
                      f"${v['ours']:+,.0f} rival's loss ${v['rival_loss']:+,.0f} = ${v['margin_change']:+,.0f}")


if __name__ == '__main__':
    main()
