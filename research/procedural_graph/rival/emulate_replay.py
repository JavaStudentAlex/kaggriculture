#!/usr/bin/env python3
"""Can we run the rival's engine from our own view of the game? Feasibility test on recorded games (read-only).

    python emulate_replay.py BUNDLE_DIR INDEX_JSON REPLAY_DIR [EPISODE ...]

Every observation shows both farms in full (money, tiles with their yields, positions, land) and the market;
only the shed, the seeds and what each worker carries are private. The environment is deterministic apart
from weeds and shop unlocks at the end of a day, both public once they happen. So the rival's private state
can be rebuilt from ours: it starts empty, as the environment starts it, and at every step

1. the bundle (a copy of the engine the rival may run) acts on the rival's view: our public state with the
   rival's seat and the rebuilt private state;
2. the step is simulated with the environment's own interpreter, on our recorded action and the bundle's;
   the weeds and shops the record shows appear are copied into the simulation;
3. the simulated public state must equal the next recorded one; the rebuilt private state is then the
   simulation's.

Only our seat's observations and actions are used; the rival's recorded actions (the truth) only score the
test. Our own private state is carried from the simulation too, and compared by content: when a full shed
(capacity 100) refuses part of a drop, what is kept depends on the order the items entered each worker's
inventory, which the replay file loses (it stores keys sorted) but a live agent's observation keeps. One JSON line per game: the steps where the bundle's action equals the rival's, and the first step
where the simulation and the record disagree (with the first difference).
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parents[3]  # the repo
sys.path.insert(0, str(HERE / 'shinka' / 'champions' / 'ladder'))

from kaggle_environments.envs.kaggriculture import kaggriculture as K  # noqa: E402
from kaggle_environments.utils import structify  # noqa: E402
from match_ladder_games import canon, load_agent  # noqa: E402

PUBLIC_FARM = ('money', 'tiles', 'farmer', 'hands', 'unlocked_quadrants', 'hires_today')


def first_difference(a, b, path=''):
    """The first path where two JSON-like values differ, or None."""
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            d = first_difference(a.get(k), b.get(k), f'{path}.{k}')
            if d:
                return d
        return None
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return f'{path}: length {len(a)} != {len(b)}'
        for i, (x, y) in enumerate(zip(a, b)):
            d = first_difference(x, y, f'{path}[{i}]')
            if d:
                return d
        return None
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return None if abs(a - b) < 1e-9 else f'{path}: {a} != {b}'
    return None if a == b else f'{path}: {json.dumps(a)[:80]} != {json.dumps(b)[:80]}'


def simulate(public, privates, actions, configuration):
    """One step of the environment on a rebuilt state; returns (farms, market, town, privates) after it."""
    public = copy.deepcopy(public)
    privates = copy.deepcopy(privates)
    obs = [SimpleNamespace(player=i, private=privates[i], farms=public['farms'], market=public['market'],
                           town=public['town'], day=public['day'], hour=public['hour'], step=public['step'])
           for i in range(2)]
    state = [SimpleNamespace(observation=obs[i], action=copy.deepcopy(actions[i]), status='ACTIVE', reward=0)
             for i in range(2)]
    cfg = dict(configuration, weedSpawnChance=0.0)  # weeds are copied from the record instead
    env = SimpleNamespace(configuration=SimpleNamespace(**cfg), done=False, info={'seed': 0})
    K.interpreter(state, env)
    return obs[0].farms, obs[0].market, obs[0].town, privates


def emulate(bundle, game, replay):
    agent = load_agent(bundle)
    ours, theirs = game['seat'], 1 - game['seat']
    steps = replay['steps']
    configuration = replay['configuration']
    rival_private = K._new_private()
    our_private = copy.deepcopy(steps[0][ours]['observation']['private'])
    equal, action_diff, sim_diff = 0, None, None
    for t in range(len(steps) - 1):
        seen = steps[t][ours]['observation']
        public = {k: seen[k] for k in ('farms', 'market', 'town', 'day', 'hour')}
        public['step'] = seen.get('step', t)
        view = copy.deepcopy(dict(public, player=theirs, private=rival_private,
                                  remainingOverageTime=seen.get('remainingOverageTime', 60)))
        got = agent(structify(view), structify(configuration))
        want = steps[t + 1][theirs].get('action')
        if action_diff is None:
            if canon(got) == canon(want):
                equal += 1
            else:
                action_diff = {'step': t, 'bundle': json.loads(canon(got)), 'recorded': json.loads(canon(want))}
        actions = [None, None]
        actions[ours], actions[theirs] = steps[t + 1][ours].get('action') or {}, got
        privates = [None, None]
        privates[ours], privates[theirs] = our_private, rival_private
        farms, market, town, privates = simulate(public, privates, actions, configuration)
        nxt = steps[t + 1][ours]['observation']
        for i, farm in enumerate(farms):  # the weeds that spawned at the end of the day
            for y, row in enumerate(nxt['farms'][i]['tiles']):
                for x, tile in enumerate(row):
                    if tile == {'kind': 'WEED'} and farm['tiles'][y][x] is None:
                        farm['tiles'][y][x] = {'kind': 'WEED'}
        diff = (first_difference({i: {k: farms[i][k] for k in PUBLIC_FARM} for i in range(2)},
                                 {i: {k: nxt['farms'][i][k] for k in PUBLIC_FARM} for i in range(2)}, 'farms')
                or first_difference(market['inventory'], nxt['market']['inventory'], 'market.inventory')
                or first_difference(privates[ours], nxt['private'], 'our private'))
        if diff:
            sim_diff = {'step': t, 'difference': diff}
            break
        our_private, rival_private = privates[ours], privates[theirs]
    return {'bundle': bundle.name, 'episode': game['id'], 'rival': game.get('team'),
            'actions_equal_until': action_diff['step'] if action_diff else equal, 'of': len(steps) - 1,
            'first_action_difference': action_diff, 'simulation_exact_until': sim_diff['step'] if sim_diff else
            len(steps) - 1, 'first_simulation_difference': sim_diff}


def main():
    bundle, index, replays = Path(sys.argv[1]).resolve(), Path(sys.argv[2]), Path(sys.argv[3])
    wanted = {int(x) for x in sys.argv[4:]}
    for game in json.loads(index.read_text()):
        if wanted and game['id'] not in wanted:
            continue
        replay = json.loads((replays / f"episode-{game['id']}-replay.json").read_text())
        print(json.dumps(emulate(bundle, game, replay)), flush=True)


if __name__ == '__main__':
    main()
