"""Oracle-required runs reject opt-outs before validation or game scheduling."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
from types import SimpleNamespace

import graph_edits
import highcpu_island_evolution as evo
from hazel_runtime.graph_runtime import HazelGraph
from shinka_graph_mab import UCB1Bandit

ROOT = Path(__file__).resolve().parent


def oracle_seed():
    graph = json.loads((ROOT / 'evolution_results/ladder_2026-09-26/next_counter_emulator_graph.json').read_text())
    return graph_edits.apply_edit(graph, {
        'channels': {'oracle_guard': True},
        'parameters': {'_OG_ITEMS': ['MILK', 'WOOL', 'STRAWBERRY'], '_OG_SCORE': 0.5},
    })


def node(graph, name):
    return next(n for n in graph['turn']['nodes'] if n['id'] == name)


class OracleRequirementTests(unittest.TestCase):
    def test_guard_and_nonempty_products_required(self):
        seed = oracle_seed()
        evo.check_required_oracle(seed)
        for change in ({'enabled': False}, {'parameters': {'_OG_ITEMS': []}},
                       {'parameters': {'_OG_MAX_ORDERS': 0}},
                       {'parameters': {'_OG_BATCH': 0}},
                       {'parameters': {'_OG_KEEP': 100}},
                       {'parameters': {'_OG_ITEMS': ['NONEXISTENT']}},
                       {'parameters': {'_OG_SCORE': 1e9}},
                       {'parameters': {'_OG_PRICE_RATIO': 1e9}},
                       {'parameters': {'_OG_FROM_STEP': 720}}):
            graph = copy.deepcopy(seed)
            node(graph, 'oracle_guard').update(change)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'required oracle'):
                evo.check_required_oracle(graph)

    def test_family_counters_cannot_disable_guard(self):
        graph = graph_edits.apply_edit(oracle_seed(), {
            'channels': {'rival_counter': True},
            'counters': {'engine:tetsutani_demand_0927': {'_OG_ITEMS': []}},
        })
        with self.assertRaisesRegex(ValueError, 'engine:tetsutani_demand_0927'):
            evo.check_required_oracle(graph)
        counter = node(graph, 'rival_counter')['counters']['engine:tetsutani_demand_0927']
        counter['_OG_ITEMS'] = ['MILK']
        evo.check_required_oracle(graph)

    def test_tuning_thresholds_and_production_remains_possible(self):
        graph = graph_edits.apply_edit(oracle_seed(), {
            'parameters': {'_OG_SCORE': 0.4, '_OG_KEEP': 1},
            'engine_parameters': {'_CA_MARGIN': -22.0},
        })
        evo.check_required_oracle(graph)

    def loop(self, run, validate=None, **kwargs):
        return evo.EditEvolution(
            run, Mock(), Mock(), Mock(), UCB1Bandit(run / 'bandit.json'),
            validate or Mock(), Mock(), 'facts', seed_graph=oracle_seed(),
            islands=[('Island-Oracle', 'oracle decisions', ('oracle_guard',))], **kwargs)

    def test_requirement_persists_on_resume_and_is_in_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            loop = self.loop(run, require_oracle=True)
            loop.persist()
            resumed = self.loop(run)  # forgetting the CLI flag must not remove the contract
            self.assertTrue(resumed.require_oracle)
            self.assertIn('REQUIRED ORACLE', resumed.current_knowledge())
            graph = copy.deepcopy(resumed.state['islands'][0]['graph'])
            node(graph, 'oracle_guard')['enabled'] = False
            path = run / 'bad.json'
            path.write_text(json.dumps(graph))
            with self.assertRaisesRegex(ValueError, 'required oracle'):
                resumed.validate(path)
            resumed._validate.assert_not_called()

    def test_queued_opt_out_is_rejected_and_valid_edit_is_returned(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / 'candidates').mkdir()
            queue = run / 'queue.json'
            queue.write_text(json.dumps([
                {'island': 'Island-Oracle', 'edit': {'channels': {'oracle_guard': False}}},
                {'island': 'Island-Oracle', 'edit': {'parameters': {'_OG_SCORE': 0.4}}},
            ]))
            validate = Mock()
            loop = self.loop(run, validate=validate, require_oracle=True, queue_path=queue)
            candidate = loop.queued(loop.state['islands'][0], 1)
            self.assertEqual(candidate['edit'], {'parameters': {'_OG_SCORE': 0.4}})
            validate.assert_called_once()

    def test_new_requirement_rejects_existing_oracle_off_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            loop = self.loop(run)
            node(loop.state['islands'][0]['graph'], 'oracle_guard')['enabled'] = False
            loop.persist()
            with self.assertRaisesRegex(ValueError, 'required oracle'):
                self.loop(run, require_oracle=True)


    def test_model_opt_out_gets_retry_feedback(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / 'candidates').mkdir()
            loop = self.loop(run, require_oracle=True, candidates=1, retries=2)
            loop.mutator.mutate_edit.side_effect = [
                ({'channels': {'oracle_guard': False}}, 'disable'),
                ({'parameters': {'_OG_SCORE': 0.4}}, 'tune'),
            ]
            proposals = loop.propose(loop.state['islands'][0], 1, 'results', [])
            self.assertEqual(len(proposals), 1)
            self.assertIn('required oracle', loop.mutator.mutate_edit.call_args_list[1].args[7])
            loop._validate.assert_called_once()

    def test_mixing_opt_out_never_reaches_gauntlet(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / 'candidates').mkdir()
            loop = self.loop(run, require_oracle=True)
            donor = copy.deepcopy(loop.state['seed_graph'])
            node(donor, 'oracle_guard')['enabled'] = False
            self.assertIsNone(loop.migrate(loop.state['islands'][0],
                                          {'after': 6, 'donor': 'old-island', 'graph': donor}))
            loop.gauntlet.evaluate.assert_not_called()

    def test_runtime_requires_loaded_model_and_forecasts_after_warmup(self):
        engine = HazelGraph.__new__(HazelGraph)
        engine.oracle_needed = engine.oracle_required = True
        engine.champion = SimpleNamespace(
            _TRACKER=SimpleNamespace(model=object(), min_context=256),
            ORACLE_STATS={'errors': 0}, _oracle_observe=Mock(return_value=None))
        self.assertIsNone(engine.observe_oracle({'step': 255}, {}))
        with self.assertRaisesRegex(RuntimeError, 'after warmup'):
            engine.observe_oracle({'step': 256}, {})
        forecast = {'score_4': {'MILK': 0.5}}
        engine.champion._oracle_observe.return_value = forecast
        self.assertIs(engine.observe_oracle({'step': 256}, {}), forecast)
        engine.champion.ORACLE_STATS['errors'] = 1
        with self.assertRaisesRegex(RuntimeError, 'unavailable'):
            engine.observe_oracle({'step': 257}, {})
        engine.champion.ORACLE_STATS['errors'] = 0
        engine.champion._TRACKER.model = None
        with self.assertRaisesRegex(RuntimeError, 'unavailable'):
            engine.observe_oracle({'step': 0}, {})
        engine.oracle_required = False
        engine.champion._oracle_observe.return_value = None
        self.assertIsNone(engine.observe_oracle({'step': 256}, {}))  # legacy fallback preserved

    def test_initial_checkpoint_records_contract_before_any_games(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            loop = self.loop(run, require_oracle=True)
            loop.run(iterations=0)
            saved = json.loads((run / 'checkpoint.json').read_text())
            self.assertTrue(saved['require_oracle'])
            self.assertTrue(saved['seed_graph']['require_oracle'])
            loop.gauntlet.evaluate.assert_not_called()


if __name__ == '__main__':
    unittest.main()
