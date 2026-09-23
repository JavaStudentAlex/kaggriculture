"""Source-integrated surgical regression tests; no model/network/engine import."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import types
import unittest
from unittest.mock import patch

from hazel_runtime import graph_runtime, surgical

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'hazel_runtime' / 'champion.py'
FROZEN_HASH = '9c8d1431dc2401eb7aa63091cb72d84f749273ed8462dfb75fbfe4f1618d568f'


def champion():
    tree = ast.parse(SOURCE.read_text())
    module = types.ModuleType('surgical_test_champion')
    module.__file__ = str(SOURCE)
    tree.body = [node for node in tree.body if getattr(node, 'lineno', 0) >= 125]
    exec(compile(tree, str(SOURCE), 'exec'), module.__dict__)
    return module


def observation(step=310, shed=None):
    tiles = [[None for _ in range(10)] for _ in range(10)]
    return {'step': step, 'day': step // 24, 'hour': step % 24, 'player': 0,
            'farms': [{'money': 5000, 'farmer': [2, 5], 'hands': [], 'tiles': tiles}],
            'private': {'shed': shed or {}, 'inventories': [{}], 'seeds': {}},
            'market': {'prices': {'WHEAT': 25, 'FERTILIZER': 100,
                                  'WOOL': 200, 'MILK': 160, 'CARROT': 35}},
            'town': {'unlocked_shops': []}}


def graph(c, enabled=True):
    data = json.loads((ROOT / 'policy_graph.json').read_text())
    data['surgical'] = {'enabled': enabled}
    data['experimental'] = {'switches': {name: False for name in
                           ('shop_phase', 'capacity_liquidation', 'opponent_pressure',
                            'idle_opportunities', 'order_arbitration')}}
    # Avoid modifying the parent-owned manifest or writing temporary files.
    with patch.object(graph_runtime.json, 'loads', return_value=data), patch.dict(os.environ):
        os.environ.pop('KAGG_GRAPH_EXPERIMENTS', None)
        return graph_runtime.HazelGraph(c, ROOT / 'policy_graph.json')


def checkpoint(g, obs, state, before, base=()):
    gen = g.market_generator(obs, 0, list(base), state)
    for stage in gen:
        if stage == before:
            snapshot = copy.deepcopy({key: gen.gi_frame.f_locals[key]
                                     for key in ('mkt', 'protected_items', 'deferred')})
            gen.close()
            return snapshot
    raise AssertionError(f'Missing checkpoint {before}')


def sold(orders, item):
    return sum(o[2] for o in orders if len(o) >= 3 and o[:2] == ['SELL', item])


class SurgicalGraphTests(unittest.TestCase):
    def setUp(self):
        self.c = champion()
        self.g = graph(self.c)

    def test_source_is_frozen_and_stages_are_preserved(self):
        self.assertEqual(hashlib.sha256(SOURCE.read_bytes()).hexdigest(), FROZEN_HASH)
        obs = observation()
        self.g.market(obs, 0, [], self.c.farm_state(obs, 0))
        self.assertEqual(self.g.last_trace, self.g.market_ids)
        tree = ast.parse(SOURCE.read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'evolve_market_orders')
        surgical.rewrite_market(fn)
        self.assertTrue(any(n.lineno == 458 for n in fn.body))
        self.assertTrue(any(n.lineno == 514 for n in fn.body))

    def test_headroom_waits_for_22_not_20_or_21(self):
        for hour in (20, 21, 22, 23):
            obs = observation(288 + hour, {'FERTILIZER': 20, 'WHEAT': 20, 'CARROT': 20})
            obs['private']['inventories'] = [{'WOOL': 45}]
            obs['market']['prices']['FERTILIZER'] = 1
            obs['farms'][0]['tiles'][0][0] = {'kind': 'PLANT'}
            state = self.c.farm_state(obs, 0)
            after = checkpoint(self.g, obs, state, 'oracle_frontrun')
            self.assertEqual(sold(after['mkt'], 'WHEAT'), 7 if hour >= 22 else 0)
            self.assertEqual(sold(after['mkt'], 'FERTILIZER'), 0)
            self.assertEqual(after['protected_items'], set())
        old = graph(self.c, False)
        obs['hour'] = 20
        obs['step'] = 308
        before = checkpoint(old, obs, self.c.farm_state(obs, 0), 'oracle_frontrun')
        self.assertEqual(sold(before['mkt'], 'FERTILIZER'), 7)

    def test_headroom_nets_sales_and_keeps_feed_and_fertilizer(self):
        obs = observation(310, {'WHEAT': 10, 'FERTILIZER': 8, 'CARROT': 42})
        obs['private']['inventories'] = [{'WOOL': 50}]
        state = self.c.farm_state(obs, 0)
        state.update(n_animals=3, n_plants=3)
        base = [['SELL', 'CARROT', 5], ['HIRE', 1]]
        result = checkpoint(self.g, obs, state, 'oracle_frontrun', base)
        self.assertEqual([o[:2] for o in result['mkt'][:2]], [o[:2] for o in base])
        # Consolidation may enlarge the funding sale but must not move HIRE ahead.
        self.assertEqual(sold(result['mkt'], 'WHEAT'), 4)
        self.assertEqual(sold(result['mkt'], 'FERTILIZER'), 0)
        self.assertEqual(sold(result['mkt'], 'CARROT'), 8)
        self.assertEqual(result['protected_items'], set())

    def test_headroom_fertilizer_floor_and_endgame_release(self):
        for step, plants, expected in ((310, 3, 4), (310, 8, 0),
                                      (310, 0, 10), (694, 3, 4), (718, 3, 10)):
            obs = observation(step, {'FERTILIZER': 10})
            obs['private']['inventories'] = [{'WOOL': 100}]
            state = self.c.farm_state(obs, 0)
            state['n_plants'] = plants
            # 718 skips headroom and exercises final full-stock liquidation instead.
            result = (self.g.market(obs, 0, [], state) if step == 718 else
                      checkpoint(self.g, obs, state, 'oracle_frontrun')['mkt'])
            self.assertEqual(sold(result, 'FERTILIZER'), expected)
        obs = observation(696, {'FERTILIZER': 10})
        state = self.c.farm_state(obs, 0)
        state['n_plants'] = 8
        self.assertGreater(sold(self.g.market(obs, 0, [], state), 'FERTILIZER'), 0)

    def test_pressure_and_routine_cannot_consume_fertilizer_floor(self):
        obs = observation(308, {'FERTILIZER': 90})
        state = self.c.farm_state(obs, 0)
        state['n_plants'] = 45
        self.assertEqual(sold(self.g.market(obs, 0, [], state), 'FERTILIZER'), 0)
        baseline = graph(self.c, False)
        self.assertGreater(sold(baseline.market(obs, 0, [], state), 'FERTILIZER'), 0)

    def test_net_sales_are_stock_bounded(self):
        obs = observation(310, {'WHEAT': 20, 'CARROT': 40})
        obs['private']['inventories'] = [{'WOOL': 80}]
        result = checkpoint(self.g, obs, self.c.farm_state(obs, 0), 'oracle_frontrun',
                            [['SELL', 'CARROT', 1000]])
        self.assertEqual(sold(result['mkt'], 'WHEAT'), 2)

    def test_deferred_uses_less_than_seven_slots_no_eviction_and_cleans_expired(self):
        obs = observation(300, {'WOOL': 20, 'MILK': 20})
        for count in (6, 7, 9, 10):
            self.c._deferred_sells = {0: {'WOOL': [4, 299], 'MILK': [2, 270]}}
            base = [['BUY_SEED', 'WHEAT', 1] for _ in range(count)]
            result = checkpoint(self.g, obs, self.c.farm_state(obs, 0), 'wage_liquidity', base)
            self.assertEqual(result['mkt'][:count], base)
            self.assertEqual(len(result['mkt']), 7 if count == 6 else count)
            self.assertNotIn('MILK', result['deferred'])
            self.assertEqual('WOOL' in result['deferred'], count >= 7)
        self.c._deferred_sells = {0: {'WOOL': [4, 299], 'MILK': [2, 299]}}
        result = checkpoint(self.g, obs, self.c.farm_state(obs, 0), 'wage_liquidity',
                            [['BUY_SEED', 'WHEAT', 1]] * 6)
        self.assertEqual(len(result['mkt']), 7)
        self.assertEqual(len(result['deferred']), 1)

    def test_deferred_morning_expansion_gate_and_funding_dependency(self):
        for hour in range(5):
            for op in ('HIRE', 'BUY_SEED'):
                obs = observation(288 + hour, {'WOOL': 20, 'CARROT': 3})
                self.c._deferred_sells = {0: {'WOOL': [4, 287]}}
                base = [['SELL', 'CARROT', 3], [op, 1]]
                result = checkpoint(self.g, obs, self.c.farm_state(obs, 0), 'wage_liquidity', base)
                self.assertEqual(result['mkt'][:2], base)
                self.assertEqual(sold(result['mkt'], 'WOOL'), 0 if hour <= 3 else 4)
        obs = observation(288, {'WOOL': 20})
        self.c._deferred_sells = {0: {'WOOL': [4, 287]}}
        result = checkpoint(self.g, obs, self.c.farm_state(obs, 0), 'wage_liquidity')
        self.assertEqual(sold(result['mkt'], 'WOOL'), 4)

    def test_luxury_24_step_supplement_runs_at_original_oracle_stage(self):
        obs = observation(301, {'WOOL': 20, 'MILK': 20})  # no routine cadence
        state = self.c.farm_state(obs, 0, {'score_4': {'WOOL': 0.01},
                                         'score_24': {'WOOL': 0.4, 'MILK': 0.8}})
        before = checkpoint(self.g, obs, state, 'oracle_frontrun')
        after = checkpoint(self.g, obs, state, 'town_and_fertilizer')
        self.assertEqual(before['mkt'], [])
        self.assertEqual(after['mkt'], [['SELL', 'MILK', 6], ['SELL', 'WOOL', 6]])
        self.assertEqual(self.g.market(obs, 0, [], state), after['mkt'])
        self.assertEqual(graph(self.c, False).market(obs, 0, [], state), [])
        obs['market']['prices'].update(WOOL=139, MILK=95)
        self.assertEqual(self.g.market(obs, 0, [], self.c.farm_state(obs, 0, state['oracle'])), [])

    def test_luxury_caps_batch_net_of_existing_and_has_warmup(self):
        obs = observation(301, {'WOOL': 20, 'MILK': 3})
        forecast = {'score_24': {'WOOL': 10, 'MILK': 20}}
        result = checkpoint(self.g, obs, self.c.farm_state(obs, 0, forecast),
                            'town_and_fertilizer', [['SELL', 'WOOL', 4]])
        self.assertEqual(sold(result['mkt'], 'WOOL'), 6)
        self.assertEqual(sold(result['mkt'], 'MILK'), 3)
        obs['step'], obs['day'], obs['hour'] = 253, 10, 13
        self.assertEqual(self.g.market(obs, 0, [], self.c.farm_state(obs, 0, forecast)), [])

    def run_agent(self, obs, base, enabled=True):
        self.c._mohui = types.SimpleNamespace(kaggle_agent_v66_meta_closed_loop=lambda *_: copy.deepcopy(base))
        self.c._oracle_observe = lambda *_: None
        self.c._oracle_record = lambda *_: None
        g = graph(self.c, enabled)
        result = g.agent(obs)
        self.assertIsNone(g.last_error)
        self.assertEqual(g.fallback_count, 0)
        return result

    def test_agent_replaces_farmer_dispatch_with_xy_and_preserves_nonpass(self):
        obs = observation()
        # Transposed tile is a weed. DIG would destroy the actual valuable crop!
        obs['farms'][0]['tiles'][2][5] = {'kind': 'WEED'}
        obs['farms'][0]['tiles'][5][2] = {'kind': 'PLANT', 'consecutive_unwatered': 1}
        base = {'farmer': ['PASS'], 'hands': [], 'market': []}
        self.assertEqual(self.run_agent(obs, base)['farmer'], ['WATER'])
        self.assertEqual(self.run_agent(obs, base, False)['farmer'], ['DIG'])
        base['farmer'] = ['HARVEST']
        self.assertEqual(self.run_agent(obs, base)['farmer'], ['HARVEST'])

    def test_agent_feed_requires_actor_bag_not_shed_for_each_unit(self):
        obs = observation(310, {'WHEAT': 30})
        obs['farms'][0]['hands'] = [[3, 6], [4, 7]]
        for x, y in ([2, 5], [3, 6], [4, 7]):
            obs['farms'][0]['tiles'][y][x] = {'kind': 'PASTURE', 'animal': 'COW', 'consecutive_unfed': 1}
        base = {'farmer': ['PASS'], 'hands': [['PASS'], ['PASS']], 'market': []}
        obs['private']['inventories'] = [{}, {'WHEAT': 1}, {}]
        result = self.run_agent(obs, base)
        self.assertEqual(result['farmer'], ['PASS'])
        self.assertEqual(result['hands'], [['FEED'], ['PASS']])
        obs['private']['shed'] = {}
        obs['private']['inventories'] = [{'WHEAT': 1}, {}, {'WHEAT': 1}]
        result = self.run_agent(obs, base)
        self.assertEqual(result['farmer'], ['FEED'])
        self.assertEqual(result['hands'], [['PASS'], ['FEED']])

    def test_agent_hands_no_duplicate_maintenance_or_speculative_move(self):
        obs = observation()
        obs['farms'][0]['hands'] = [[2, 5], [2, 5], [3, 6]]
        obs['farms'][0]['tiles'][5][2] = {'kind': 'WEED'}
        base = {'farmer': ['PASS'], 'hands': [['PASS']] * 3, 'market': []}
        result = self.run_agent(obs, base)
        self.assertEqual(result['farmer'], ['DIG'])
        self.assertEqual(result['hands'], [['PASS']] * 3)
        base['hands'][1] = ['DIG']
        result = self.run_agent(obs, base)
        self.assertEqual(result['farmer'], ['PASS'])
        self.assertEqual(result['hands'], [['PASS'], ['DIG'], ['PASS']])
        base['farmer'] = ['EAST']
        base['hands'] = [['PASS']] * 3
        result = self.run_agent(obs, base)
        self.assertEqual(result['farmer'], ['EAST'])
        self.assertEqual(result['hands'], [['DIG'], ['PASS'], ['PASS']])

    def test_disabled_equivalence_720_turns_including_deferred_state(self):
        reference, compiled = champion(), champion()
        g = graph(compiled, False)
        rng = random.Random(55)
        products = list(reference._BASE_PRICE)
        for step in range(720):
            obs = observation(step, {p: rng.randrange(24) for p in products})
            obs['private']['inventories'] = [{'WOOL': rng.randrange(35)}]
            obs['market']['prices'] = {p: rng.randrange(1, 300) for p in products}
            forecast = {key: {p: rng.random() for p in products}
                        for key in ('score_4', 'score_24', 'units_24')}
            base = [['SELL', rng.choice(products), rng.randrange(1, 12)]
                    for _ in range(rng.randrange(11))]
            if step % 13 == 0:
                base.insert(1, ['HIRE', 1])
            want = reference.evolve_market_orders(obs, 0, base, reference.farm_state(obs, 0, forecast))
            got = g.market(obs, 0, base, compiled.farm_state(obs, 0, forecast))
            self.assertEqual(got, want, f'turn {step}')
            self.assertEqual(compiled._deferred_sells, reference._deferred_sells)

    def test_disabled_agent_channels_match_original_with_nontrivial_tiles(self):
        obs = observation(310, {'WHEAT': 5, 'FERTILIZER': 6})
        obs['farms'][0]['hands'] = [[2, 5], [3, 6]]
        obs['farms'][0]['tiles'][2][5] = {'kind': 'WEED'}
        obs['farms'][0]['tiles'][6][3] = {'kind': 'PLANT', 'consecutive_unwatered': 1}
        base = {'farmer': ['PASS'], 'hands': [['PASS'], ['DROP']],
                'market': [['SELL', 'FERTILIZER', 2], ['HIRE', 1]]}
        state = self.c.farm_state(obs, 0)
        want = self.c._sanitize({
            'farmer': self.c.evolve_farmer_action(obs, 0, base['farmer'], state),
            'hands': self.c.evolve_hand_actions(obs, 0, base['hands'], state),
            'market': self.c.evolve_market_orders(obs, 0, base['market'], state),
        }, base, 2)
        self.assertEqual(self.run_agent(obs, base, False), want)

    def test_headroom_accounts_for_preceding_pressure_sales(self):
        obs = observation(310, {'WHEAT': 30, 'FERTILIZER': 10, 'CARROT': 50})
        obs['private']['inventories'] = [{'WOOL': 25}]
        state = self.c.farm_state(obs, 0)
        state['n_plants'] = 5
        pressure = checkpoint(self.g, obs, state, 'predrop_headroom')
        headroom = checkpoint(self.g, obs, state, 'oracle_frontrun')
        self.assertEqual(pressure['mkt'], headroom['mkt'])
        self.assertEqual(sold(headroom['mkt'], 'FERTILIZER'), 0)

    def test_config_validation_and_opening_early_returns(self):
        self.assertFalse(surgical.enabled(None))
        for bad in (True, {'enabled': 1}, {'unknown': True}):
            with self.assertRaises(ValueError):
                surgical.enabled(bad)
        for step in (0, 1):
            obs = observation(step, {'WHEAT': 35})
            state = self.c.farm_state(obs, 0)
            self.assertEqual(self.g.market(obs, 0, [], state),
                             self.c.evolve_market_orders(obs, 0, [], state))


if __name__ == '__main__':
    unittest.main()
