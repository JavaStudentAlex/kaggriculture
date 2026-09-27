"""Ladder-engine graphs: the engine catalog and parameters, the graph builder, the new edit keys
and the oracle guard."""
from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / 'hazel_runtime'))

import graph_edits  # noqa: E402
import make_ladder_graph  # noqa: E402
from arena import payload  # noqa: E402
from hazel_runtime import engines, oracle_guard  # noqa: E402

ENGINE = 'tetsutani_demand'
SOURCE = '''
import json
_SETTINGS = {'hand_align': True, 'sell_lead': False}
SEED_PRICE = {'WHEAT': 10}
V9_RATIO = 1.8          # carrot ratio
_R110 = {('BAKERY', 'PIZZA_SHOP'): 0, ('YARN_STORE', 'BAKERY'): 9}
_RUN_REPORT = {'calls': 0}
_EMPTY = {}
BLOB = "%s"
TWICE = 1
TWICE = 2
_CHAIN = {'x': _SETTINGS['hand_align']}
def agent(obs, config=None):
    return {"farmer": ["PASS"], "hands": [], "market": [["SELL", "WHEAT", int(V9_RATIO * 10)], [],
            ["HIRE"]] if _SETTINGS["sell_lead"] or _R110[("BAKERY", "PIZZA_SHOP")] else []}
''' % ('x' * 300)


class EngineCatalogTests(unittest.TestCase):
    def test_catalog_keeps_policy_constants_only(self):
        cat = engines.catalog(SOURCE)
        self.assertEqual(set(cat), {'_SETTINGS', 'V9_RATIO', '_R110'})     # facts, state, blobs, rebound out
        self.assertEqual(cat['V9_RATIO']['note'], 'carrot ratio')

    def test_tuple_keys_round_trip(self):
        default = engines.catalog(SOURCE)['_R110']['default']
        as_json = engines.to_json(default)
        self.assertEqual(as_json, {'BAKERY|PIZZA_SHOP': 0, 'YARN_STORE|BAKERY': 9})
        self.assertEqual(engines.from_json(as_json, default, '_R110'), default)
        with self.assertRaises(ValueError):
            engines.from_json({'BAKERY': 0}, default, '_R110')
        with self.assertRaises(ValueError):
            engines.from_json('2.0', 1.8, 'V9_RATIO')

    def test_rewrite_runs_the_new_value_at_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / 'toy'
            (d / 'agent').mkdir(parents=True)
            (d / 'agent' / 'agent.py').write_text(SOURCE)
            import hashlib
            (d / 'SOURCE.json').write_text(json.dumps({'name': 'toy', 'entry': 'agent.py', 'files': {
                'agent.py': hashlib.sha256(SOURCE.encode()).hexdigest()}}))
            plain = engines.Engine(d)
            self.assertEqual(plain.agent({'step': 0})['market'], [])
            tuned = engines.Engine(d, {'_SETTINGS': {'hand_align': False, 'sell_lead': True}, 'V9_RATIO': 2.5})
            self.assertEqual(tuned.agent({'step': 0})['market'], [['SELL', 'WHEAT', 25], [], ['HIRE']])
            self.assertEqual(tuned.policy.__globals__['_CHAIN'], {'x': False})   # load-time use sees it
            with self.assertRaises(ValueError):
                engines.Engine(d, {'SEED_PRICE': {'WHEAT': 1}})


class LadderGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        files, meta = make_ladder_graph.install_engine(ENGINE)
        cls.graph = make_ladder_graph.ladder_graph(json.loads(make_ladder_graph.BASE.read_text()), ENGINE, meta, files)
        cls.constants = graph_edits.catalog()

    def test_builder_starts_as_the_engine(self):
        nodes = {n['id']: n for n in self.graph['turn']['nodes']}
        self.assertEqual(nodes['backbone']['engine'], ENGINE)
        self.assertEqual([nodes[k].get('enabled') for k in ('farmer', 'hands', 'market', 'oracle_guard')],
                         [False, False, False, False])
        ids = [n['id'] for n in self.graph['turn']['nodes']]
        self.assertEqual(ids[ids.index('market') + 1], 'oracle_guard')
        pairs = [(e['source'], e['target']) for e in self.graph['turn']['edges']]
        self.assertEqual(pairs, list(zip(ids, ids[1:])))
        pins = self.graph['provenance']['runtime_bundle_hashes']
        entry = json.loads((make_ladder_graph.LADDER / ENGINE / 'SOURCE.json').read_text())['entry']
        self.assertIn(f'engines/{ENGINE}/agent/{entry}', pins)
        self.assertIn('oracle_guard.py', pins)

    def test_engine_parameter_edit(self):
        g = graph_edits.apply_edit(self.graph, {'engine_parameters': {'V9_CARROT_RATIO': 2.2}}, self.constants)
        self.assertEqual(graph_edits.diff(self.graph, g, self.constants), ['engine V9_CARROT_RATIO: 2.0 -> 2.2'])
        self.assertNotEqual(graph_edits.settings_key(g, self.constants), graph_edits.settings_key(self.graph, self.constants))
        back = graph_edits.apply_edit(g, {'engine_parameters': {'V9_CARROT_RATIO': None}}, self.constants)
        self.assertEqual(graph_edits.settings_key(back, self.constants), graph_edits.settings_key(self.graph, self.constants))
        for bad in ({'NOPE': 1}, {'V9_CARROT_RATIO': 'x'}, {'SEED_PRICE': {'WHEAT': 1}}):
            with self.assertRaises(ValueError):
                graph_edits.apply_edit(self.graph, {'engine_parameters': bad}, self.constants)

    def test_tuple_keyed_engine_table(self):
        spec = graph_edits.engine_catalog(self.graph)['_R110_OLD_SHOPS']
        table = engines.to_json(spec['default'])
        key = next(iter(table))
        table[key] = 3
        g = graph_edits.apply_edit(self.graph, {'engine_parameters': {'_R110_OLD_SHOPS': table}}, self.constants)
        self.assertEqual(len(graph_edits.diff(self.graph, g, self.constants)), 1)

    def test_inert_settings_are_refused(self):
        with self.assertRaisesRegex(ValueError, 'no effect'):
            graph_edits.apply_edit(self.graph, {'parameters': {'_ORACLE_FRONTRUN_SCORE': 0.4}}, self.constants)
        with self.assertRaisesRegex(ValueError, 'no effect'):
            graph_edits.apply_edit(self.graph, {'parameters': {'_OG_SCORE': 0.4}}, self.constants)
        with self.assertRaisesRegex(ValueError, 'no effect'):
            graph_edits.apply_edit(self.graph, {'stages': {'oracle_frontrun': False}}, self.constants)
        # the same settings with their channel switched on in the same edit
        g = graph_edits.apply_edit(self.graph, {'channels': {'oracle_guard': True},
                                                'parameters': {'_OG_SCORE': 0.4}}, self.constants)
        self.assertEqual(sorted(graph_edits.diff(self.graph, g, self.constants)),
                         ['_OG_SCORE (oracle_guard): 0.3 -> 0.4', 'channel oracle_guard: off -> on'])
        g = graph_edits.apply_edit(self.graph, {'channels': {'market': True},
                                                'parameters': {'_ORACLE_FRONTRUN_SCORE': 0.4}}, self.constants)
        self.assertIn('channel market: off -> on', graph_edits.diff(self.graph, g, self.constants))

    def test_experimental_switches(self):
        g = graph_edits.apply_edit(self.graph, {'experimental': {'order_arbitration': True}}, self.constants)
        self.assertEqual(graph_edits.settings(g, self.constants)['experimental'], {'order_arbitration': True})
        with self.assertRaises(ValueError):
            graph_edits.apply_edit(self.graph, {'experimental': {'opponent_pressure': True}}, self.constants)
        with self.assertRaises(ValueError):
            graph_edits.apply_edit(self.graph, {'experimental': {'nope': True}}, self.constants)

    def test_migration_edit_with_channels_and_engine_parameters(self):
        c = self.constants
        donor = graph_edits.apply_edit(self.graph, {'channels': {'oracle_guard': True}, 'parameters': {'_OG_SCORE': 0.5},
                                                    'engine_parameters': {'V9_CARROT_RATIO': 2.2}}, c)
        own = graph_edits.apply_edit(self.graph, {'engine_parameters': {'V9_CARROT_RATIO': 2.4}}, c)
        edit = graph_edits.migration_edit(own, donor, self.graph, c)
        self.assertEqual(edit, {'parameters': {'_OG_SCORE': 0.5}, 'channels': {'oracle_guard': True}})
        mixed = graph_edits.settings(graph_edits.apply_edit(own, edit, c), c)
        self.assertEqual(mixed['engine_parameters'], {'V9_CARROT_RATIO': 2.4})   # its own value stays
        self.assertTrue(mixed['channels']['oracle_guard'])
        # a channel the donor switched back to its default (on) is switched on in the recipient
        market = graph_edits.apply_edit(self.graph, {'channels': {'market': True}}, c)
        self.assertEqual(graph_edits.migration_edit(self.graph, market, self.graph, c), {'channels': {'market': True}})

    def test_mohui_graph_refuses_engine_edits(self):
        base = json.loads(make_ladder_graph.BASE.read_text())
        with self.assertRaisesRegex(ValueError, 'ladder engine'):
            graph_edits.apply_edit(base, {'engine_parameters': {'V9_CARROT_RATIO': 2.0}}, self.constants)

    def test_bundle_pins_only_its_own_engine(self):
        # made from the old engine's seed graph, it still pins that engine's files, which the bundle leaves out
        path = HERE / 'evolution_results' / 'ladder_2026-09-27' / 'seed_graph_engine0927.json'
        graph = json.loads(path.read_text())
        self.assertTrue(any(k.startswith('engines/tetsutani_demand/') for k in
                            graph['provenance']['runtime_bundle_hashes']))
        with tempfile.TemporaryDirectory() as tmp:
            dst = payload.write_graph_bundle(Path(tmp) / 'b', graph, 'test')
            pins = json.loads((dst / 'policy_graph.json').read_text())['provenance']['runtime_bundle_hashes']
            self.assertEqual(sorted(p.name for p in (dst / 'hazel_runtime' / 'engines').iterdir()),
                             ['tetsutani_demand_0927'])
            self.assertFalse([k for k in pins if k.startswith('engines/tetsutani_demand/')])
            self.assertIn('engines/tetsutani_demand_0927/agent/main.py', pins)
            for relative, digest in pins.items():
                self.assertEqual(payload.sha(dst / 'hazel_runtime' / relative), digest, relative)

    def test_prompt_lists_engine_controls(self):
        text = graph_edits.describe_controls(self.graph, self.constants)
        self.assertIn('PRODUCTION ENGINE: tetsutani_demand', text)
        self.assertIn('V9_CARROT_RATIO | float | 2.0', text)
        self.assertIn('ORACLE GUARD PARAMETERS', text)
        self.assertNotIn('SEED_PRICE |', text)


class OracleGuardTests(unittest.TestCase):
    def state(self, scores, shed, prices=None):
        return {'oracle': {'score_4': scores}, 'shed': shed,
                'prices': prices or {'MILK': 150, 'WOOL': 200, 'STRAWBERRY': 120, 'EGG': 50}}

    def test_fills_the_first_empty_slot_and_keeps_the_others(self):
        orders = [['SELL', 'MILK', 2], [], ['HIRE'], []]
        out = oracle_guard.apply({'step': 300}, orders, self.state({'WOOL': 0.4}, {'WOOL': 20, 'MILK': 5}), {})
        self.assertEqual(out, [['SELL', 'MILK', 2], ['SELL', 'WOOL', 6], ['HIRE'], []])
        self.assertEqual(orders, [['SELL', 'MILK', 2], [], ['HIRE'], []])     # input untouched

    def test_stock_already_sold_price_floor_and_limits(self):
        st = self.state({'MILK': 0.9, 'WOOL': 0.5, 'EGG': 0.7}, {'MILK': 5, 'WOOL': 3, 'EGG': 40},
                        {'MILK': 150, 'WOOL': 200, 'EGG': 20})
        out = oracle_guard.apply({'step': 300}, [['SELL', 'MILK', 2]], st, {})
        # MILK strong: all free stock (5 - 2 already sold); EGG under 0.6 x 50: skipped; WOOL: batch capped by stock
        self.assertEqual(out, [['SELL', 'MILK', 2], ['SELL', 'MILK', 3], ['SELL', 'WOOL', 3]])
        self.assertEqual(oracle_guard.apply({'step': 100}, [], st, {}), [])        # before the forecast
        self.assertEqual(oracle_guard.apply({'step': 300}, [], {'shed': {}}, {}), [])   # no forecast
        full = [['HIRE']] * 10
        self.assertEqual(oracle_guard.apply({'step': 300}, full, st, {}), full)     # no free slot
        kept = oracle_guard.apply({'step': 300}, [], st, {'_OG_KEEP': 3, '_OG_MAX_ORDERS': 1})
        self.assertEqual(kept, [['SELL', 'MILK', 2]])


if __name__ == '__main__':
    unittest.main()
