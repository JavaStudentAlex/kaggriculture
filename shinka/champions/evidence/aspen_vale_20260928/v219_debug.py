"""Why the tomato investment (the 09-27 engine's layer V219) did not fire for Aspen Vale in one recorded game
(cliproxyapi; one game, in-process). Our bundle is loaded as kaggle_environments loads it (main.py exec'd with its
directory appended to sys.path, the last callable is the agent); the rival replays its recorded moves. Every engine
module that has the layer gets its eligibility check (_v219_qualifies), its request (_v219_request) and its per-player
state logged from day 17 hour 23 to day 18 hour 6, so a state reset (a second call in the same step) or a declined
request shows. The final money must equal the recorded game's.

    python v219_debug.py <bundle dir> <replay json> <our seat>
"""
import gc
import json
import sys
import types
from pathlib import Path

from kaggle_environments import make

bundle, replay_path, ours = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
replay = json.load(open(replay_path))
steps = replay['steps']
seed = (replay.get('info') or {}).get('seed') or replay['configuration'].get('seed')

sys.path.append(str(bundle))
scope = {'__name__': 'bundle_main', '__file__': str(bundle / 'main.py')}   # arena bundles read __file__
exec(compile((bundle / 'main.py').read_text(), str(bundle / 'main.py'), 'exec'), scope)
agent = [v for v in scope.values() if callable(v)][-1]
patched = {}
LOG = []


def patch():
    # hazel_runtime.engines.Engine execs an engine's source into a plain dict (kaggle_environments'
    # get_last_callable), so the engine's globals are found among the dicts, not the modules
    found = {f'ns@{id(d):x}': types.SimpleNamespace(ns=d) for d in gc.get_objects()
             if isinstance(d, dict) and '_v219_request' in d and '_V219_STATES' in d}
    for name, module in found.items():
        if name in patched:
            continue
        patched[name] = module
        qualifies, request = module.ns['_v219_qualifies'], module.ns['_v219_request']

        def q(obs, native, _f=qualifies, _n=name, _ns=module.ns):
            result = _f(obs, native)
            farm = obs['farms'][obs['player']]
            late = sorted({r for r, tape in _ns['_IMPL'].chassis.routes.items() for a in tape[432:719]
                           if any(o and o[0] == 'BUY_LAND' for o in a.get('market', []))
                           or any(c == ['PLANT', 'TOMATO'] for c in [a.get('farmer')] + a.get('hands', []))})
            parts = {'tiles': len(farm['tiles']), 'quads': sorted(farm['unlocked_quadrants']), 'money': farm['money'],
                     'tomato_price': obs['market']['prices']['TOMATO'], 'CROP_MIN_PRICE': _ns.get('CROP_MIN_PRICE'),
                     'shops': sum(s in ('PIZZA_SHOP', 'FARMERS_MARKET') for s in obs['town']['unlocked_shops']),
                     'se_locked': all(farm['tiles'][y][x] == 'LOCKED' for y in (5, 6) for x in range(5, 10)),
                     'tomato_seeds': obs['private']['seeds'].get('TOMATO', 0),
                     'routes_with_late_land_or_tomato': late[:12], 'n_routes': len(_ns['_IMPL'].chassis.routes)}
            LOG.append((int(obs['step']), _n, 'qualifies', int(obs['player']), result, parts))
            return result

        def r(obs, action, state, native, _f=request, _n=name):
            out = _f(obs, action, state, native)
            step = int(obs['step'])
            if 431 <= step <= 438:
                LOG.append((step, _n, 'request', int(obs['player']), out is not action,
                            {k: state.get(k) for k in ('eligible', 'committed', 'requested_day', 'last_step')},
                            len(out.get('market') or [])))
            return out
        module.ns['_v219_qualifies'], module.ns['_v219_request'] = q, r


def ours_fn(obs, config):
    step = int(obs['step'])
    if 431 <= step <= 438:
        for name, module in patched.items():
            st = module.ns['_V219_STATES'].get(ours)
            LOG.append((step, name, 'before', ours, None if st is None else
                        {k: st.get(k) for k in ('eligible', 'committed', 'requested_day', 'last_step')}))
    out = agent(obs, config)
    patch()
    return out


def rival_fn(obs, config):
    return steps[int(obs['step']) + 1][1 - ours].get('action') or {}


env = make('kaggriculture', configuration={'seed': seed}, debug=False)
env.run([ours_fn, rival_fn] if ours == 0 else [rival_fn, ours_fn])
money = [s['reward'] for s in env.steps[-1]]
recorded = [s['reward'] for s in steps[-1]]
print('final rewards', money, 'recorded', recorded, 'equal' if money == recorded else 'DIFFERENT')
for row in LOG:
    print(*row)
for name, module in patched.items():
    print(name, 'report', module.ns.get('_V219_REPORT'))
    # the sale reordering raises on an empty order entry (leo_pi guards it); both callers count the error and skip
    print(name, 'R37 stats', {k: v for k, v in (module.ns.get('_R37_STATS') or {}).items() if 'error' in k or 'reorder' in k},
          '| S839', module.ns.get('_S839_REPORT'))
