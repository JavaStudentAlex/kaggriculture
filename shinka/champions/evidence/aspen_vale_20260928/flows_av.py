"""Exact money flows of both seats in recorded games (cliproxyapi): every recorded step is re-run through the
environment's own interpreter (research/procedural_graph/rival/emulate_replay.simulate) from the observations before
it and the two recorded actions, and every unit the market commits is logged with its price: sales and purchases by
product, seeds, animals, hires and land. The simulated money of both seats must equal the next recorded observation
at every step (the count of steps where it does not is reported), so the split by product is exact, not estimated.

    python flows_av.py <index.json> <replay dir> <out.json>

Per game and seat: 'sell' {product: [units, dollars]}, 'buy' {BUY_PRODUCT/BUY_SEED/BUY_ANIMAL:item: [units, dollars]},
'hire' / 'land' dollars, 'by_day' {product: [dollars sold on each game day]}, 'money_mismatch' steps.
"""
import collections
import copy
import json
import sys

sys.path.insert(0, '/home/alex/kagg-evo/repo/research/procedural_graph/rival')
from emulate_replay import K, simulate  # noqa: E402

LOG = []
_commit, _hire, _land = K._commit_unit, K._do_hire, K._do_buy_land


def commit(op, item, price, farm, private, market, shed_capacity=100):
    ok = _commit(op, item, price, farm, private, market, shed_capacity)
    if ok:
        LOG.append((id(farm), op, item, price))
    return ok


def hire(farm, *args, **kw):
    before = farm['money']
    _hire(farm, *args, **kw)
    LOG.append((id(farm), 'HIRE', '', before - farm['money']))


def land(farm, *args, **kw):
    before = farm['money']
    _land(farm, *args, **kw)
    LOG.append((id(farm), 'BUY_LAND', '', before - farm['money']))


K._commit_unit, K._do_hire, K._do_buy_land = commit, hire, land


def flows(replay):
    steps, configuration = replay['steps'], replay['configuration']
    out = [{'sell': collections.defaultdict(lambda: [0, 0.0]), 'buy': collections.defaultdict(lambda: [0, 0.0]),
            'hire': 0.0, 'land': 0.0, 'by_day': collections.defaultdict(lambda: [0.0] * 31), 'money_mismatch': 0}
           for _ in range(2)]
    for t in range(1, len(steps)):
        seen = steps[t - 1][0]['observation']
        public = {k: seen[k] for k in ('farms', 'market', 'town', 'day', 'hour')}
        public['step'] = seen.get('step', t - 1)
        public = copy.deepcopy(public)
        privates = [copy.deepcopy(steps[t - 1][s]['observation']['private']) for s in range(2)]
        actions = [steps[t][s].get('action') or {} for s in range(2)]
        LOG.clear()
        # simulate() deep-copies its inputs; the farms it returns are the objects the market phase committed to
        farms, market, town, privates = simulate(public, privates, actions, configuration)
        seat_of = {id(farms[s]): s for s in range(2)}
        day = int(seen.get('day', 0))
        for fid, op, item, price in LOG:
            s = seat_of.get(fid)
            if s is None:
                raise SystemExit(f'step {t}: a commit on a farm that is not one of the two seats')
            o = out[s]
            if op == 'SELL':
                o['sell'][item][0] += 1
                o['sell'][item][1] += price
                o['by_day'][item][min(day, 30)] += price
            elif op in ('HIRE', 'BUY_LAND'):
                o['hire' if op == 'HIRE' else 'land'] += price
            else:
                o['buy'][f'{op}:{item}'][0] += 1
                o['buy'][f'{op}:{item}'][1] += price
        for s in range(2):
            if abs(farms[s]['money'] - steps[t][0]['observation']['farms'][s]['money']) > 1e-6:
                out[s]['money_mismatch'] += 1
    for o in out:
        o['sell'], o['buy'], o['by_day'] = dict(o['sell']), dict(o['buy']), dict(o['by_day'])
    return out


def main():
    index, replays, dst = json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3]
    result = {}
    for g in sorted(index, key=lambda g: g['diff']):
        replay = json.load(open(f"{replays}/episode-{g['id']}-replay.json"))
        f = flows(replay)
        ours, rival = f[g['seat']], f[1 - g['seat']]
        result[g['id']] = {'team': g['team'], 'orate': g['orate'], 'diff': g['diff'], 'seat': g['seat'],
                           'ours': ours, 'rival': rival}
        net = lambda o: {p: v[1] - o['buy'].get(f'BUY_PRODUCT:{p}', [0, 0.0])[1] for p, v in o['sell'].items()}
        n_o, n_r = net(ours), net(rival)
        gaps = sorted(((p, n_o.get(p, 0) - n_r.get(p, 0)) for p in set(n_o) | set(n_r)), key=lambda kv: kv[1])
        print(f"{g['id']} {g['diff']:+8,.0f} {g['team'][:16]:16s} mismatch {ours['money_mismatch']}/{rival['money_mismatch']} | "
              + ', '.join(f'{p} {v:+,.0f}' for p, v in gaps[:4]), flush=True)
    json.dump(result, open(dst, 'w'), indent=1, ensure_ascii=False)


if __name__ == '__main__':
    main()
