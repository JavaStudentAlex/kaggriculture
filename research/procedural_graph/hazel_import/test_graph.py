"""Source-equivalence and fail-closed graph regression tests (stdlib unittest)."""
import ast
import copy
import json
import random
import tempfile
import types
import unittest
from pathlib import Path
from graph_runtime import HazelGraph, validate_chain, MARKET_STAGES, TURN_STAGES

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'payload/agents/submission_hazel_weir/champion.py'
GRAPH = HERE / 'policy_graph.json'


def pure_champion():
    # Execute the actual source's pure functions/constants, without model imports.
    tree = ast.parse(SOURCE.read_text())
    body = [n for n in tree.body if n.lineno >= 125]
    c = types.ModuleType('pure_champion')
    c.__file__ = str(SOURCE)
    exec(compile(ast.Module(body=body, type_ignores=[]), str(SOURCE), 'exec'), c.__dict__)
    return c


def observation(step=200):
    farm = {'money': 5000, 'farmer': [0, 0], 'hands': [], 'tiles': [[None]], 'unlocked_quadrants': ['NW']}
    return {'step': step, 'day': step // 24, 'hour': step % 24, 'player': 0,
            'farms': [farm, copy.deepcopy(farm)], 'private': {'shed': {}, 'inventories': [], 'seeds': {}},
            'town': {'unlocked_shops': ['BAKERY', 'YARN_STORE']},
            'market': {'prices': {'WHEAT': 25, 'WOOL': 200, 'FERTILIZER': 100}}}


class GraphTests(unittest.TestCase):
    def test_market_equivalence_720_turns_randomized(self):
        c = pure_champion()
        g = HazelGraph(c, GRAPH)
        rng = random.Random(42)
        original_queue, graph_queue = {}, {}
        stages = set()
        for step in range(720):
            obs = observation(step)
            products = list(c._BASE_PRICE)
            obs['private']['shed'] = {p: rng.randrange(30) for p in products}
            obs['private']['inventories'] = [{'WHEAT': rng.randrange(30)}]
            obs['market']['prices'] = {p: rng.randrange(0, 300) for p in products}
            obs['farms'][0]['money'] = rng.randrange(10000)
            obs['farms'][0]['hands'] = [[0, 0]] * rng.randrange(12)
            forecast = {key: {p: rng.random() * (3 if key == 'units_24' else 1) for p in products}
                        for key in ['score_4', 'score_24', 'units_24']}
            state = c.farm_state(obs, 0, forecast)
            state['n_animals'] = rng.randrange(20)
            base = [['SELL', rng.choice(products), rng.randrange(1, 30)] for _ in range(rng.randrange(11))]
            c._deferred_sells = original_queue
            expected = c.evolve_market_orders(copy.deepcopy(obs), 0, copy.deepcopy(base), copy.deepcopy(state))
            original_queue = copy.deepcopy(c._deferred_sells)
            c._deferred_sells = graph_queue
            g.last_trace = []
            actual = g.market(copy.deepcopy(obs), 0, copy.deepcopy(base), copy.deepcopy(state))
            graph_queue = copy.deepcopy(c._deferred_sells)
            self.assertEqual(expected, actual, f'step {step}')
            self.assertEqual(original_queue, graph_queue, f'deferred state step {step}')
            stages.update(g.last_trace)
        self.assertEqual(stages, {s[0] for s in MARKET_STAGES})

    def test_opening_uses_actual_35_and_30_not_prose_one(self):
        c = pure_champion(); g = HazelGraph(c, GRAPH)
        for step, expected in [(0, ['BUY_PRODUCT', 'WHEAT', 35]), (1, ['SELL', 'WHEAT', 30])]:
            obs = observation(step)
            self.assertEqual(g.market(obs, 0, [], c.farm_state(obs, 0))[0], expected)
            self.assertEqual(g.last_trace[-1], 'opening_scalp')

    def test_missing_duplicate_unknown_or_disconnected_node_rejected(self):
        raw = json.loads(GRAPH.read_text())
        ids = [s[0] for s in MARKET_STAGES]
        for mutation in ['missing', 'duplicate', 'unknown', 'edge', 'binding', 'order']:
            graph = copy.deepcopy(raw['market'])
            if mutation == 'missing': graph['nodes'].pop()
            if mutation == 'duplicate': graph['nodes'].append(graph['nodes'][0])
            if mutation == 'unknown': graph['nodes'][0]['id'] = 'decorative'
            if mutation == 'edge': graph['edges'].pop()
            if mutation == 'binding': graph['nodes'][0]['binding'] = 'no_handler'
            if mutation == 'order': graph['edges'].reverse()
            with self.assertRaises(ValueError, msg=mutation): validate_chain(graph, ids)

    def test_source_hash_and_boundaries_rejected(self):
        raw = json.loads(GRAPH.read_text())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'graph.json'
            for mutation in ['hash', 'line']:
                graph = copy.deepcopy(raw)
                if mutation == 'hash': graph['source']['champion_sha256'] = '0' * 64
                else: graph['market']['nodes'][0]['source_lines'][0] += 1
                path.write_text(json.dumps(graph))
                with self.assertRaises(ValueError): HazelGraph(pure_champion(), path)

    def test_preserves_hire_routine_order_and_drop(self):
        c = pure_champion(); g = HazelGraph(c, GRAPH); obs = observation(50)
        base = [['HIRE', 1], ['BUY_SEED', 'WHEAT', 5], ['SELL', 'TOMATO', 3]]
        self.assertEqual(g.market(obs, 0, base, c.farm_state(obs, 0)), base)
        self.assertEqual(c._sanitize({'farmer': ['DROP'], 'hands': [], 'market': base}, {}, 0)['farmer'], ['DROP'])

    def test_turn_order_and_fallback_match_original(self):
        for failure in [None, 'state', 'farmer', 'hands', 'market', 'backbone']:
            c = pure_champion(); g = HazelGraph(c, GRAPH)
            def setup():
                events = []
                def action(name, value):
                    def f(*args):
                        events.append(name)
                        if name == failure: raise ValueError('injected')
                        return copy.deepcopy(value)
                    return f
                c._mohui = types.SimpleNamespace(kaggle_agent_v66_meta_closed_loop=action('backbone', {'farmer': ['DROP'], 'hands': [], 'market': [['HIRE', 1]]}))
                c._oracle_observe = action('oracle', None)
                c._oracle_record = action('record', None)
                c.farm_state = action('state', {})
                c.evolve_farmer_action = action('farmer', ['WATER'])
                c.evolve_hand_actions = action('hands', [])
                c.evolve_market_orders = action('market', [['SELL', 'WOOL', 3]])
                g.market = c.evolve_market_orders
                return events
            events = setup(); expected = c.agent(observation()); expected_events = events[:]
            events = setup(); actual = g.agent(observation())
            self.assertEqual(expected, actual, failure)
            self.assertEqual(expected_events, events, failure)


if __name__ == '__main__':
    unittest.main(verbosity=2)
