"""Edit-based graph evolution: edits reach execution, the judge sees diffs, the paired
gauntlet decides, and the loop writes only to its run directory. No LLM, network or
game is used: the model replies, judge votes and game results are fakes."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import graph_edits
import graph_gauntlet
import highcpu_island_evolution as evo
import sift_graph_judge
from bam_graph_mutator import BAMGraphMutator
from hazel_runtime import graph_runtime
from shinka_graph_mab import UCB1Bandit
from test_surgical_graph import champion, observation

ROOT = Path(__file__).resolve().parent


def seed_graph():
    graph = json.loads((ROOT / 'policy_graph.json').read_text())
    for chain in ('turn', 'market'):
        for node in graph[chain]['nodes']:
            for key in graph_edits.NODE_FEATURES:
                node.pop(key, None)
    graph['surgical'] = {'enabled': False}
    return graph


class GraphEditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.constants = graph_edits.catalog()

    def test_catalog_lists_every_evolvable_constant_with_its_stage(self):
        self.assertEqual(len(self.constants), 41)  # 40 champion constants + _TOWN_CADENCE_PHASE
        spec = self.constants['_OPENING_BUY_WHEAT_QTY']
        self.assertEqual((spec['default'], spec['type'], spec['home']), (35, 'int', 'opening_scalp'))
        self.assertEqual(self.constants['_TOWN_CADENCE_PHASE']['home'], 'town_and_fertilizer')
        self.assertIn('town_and_fertilizer', self.constants['_SHOP_SELL_BATCH_MAX']['stages'])
        inert = sorted(k for k, v in self.constants.items() if not v['used'])
        self.assertEqual(inert, ['_ANIMAL_YIELD', '_ENABLE_ORACLE_HOLD', '_ORACLE_HOLD_SCORE',
                                 '_WAGE_RESERVE_FLOOR'])
        with self.assertRaisesRegex(ValueError, 'never reads'):
            graph_edits.apply_edit(seed_graph(), {'parameters': {'_WAGE_RESERVE_FLOOR': 900.0}})

    def test_parameter_edit_is_placed_on_its_stage_and_executes(self):
        graph = graph_edits.apply_edit(seed_graph(), {'parameters': {'_OPENING_BUY_WHEAT_QTY': 13,
                                                                      '_OPENING_SELL_WHEAT_QTY': 9}})
        node = next(n for n in graph['market']['nodes'] if n['id'] == 'opening_scalp')
        self.assertEqual(node['parameters'], {'_OPENING_BUY_WHEAT_QTY': 13, '_OPENING_SELL_WHEAT_QTY': 9})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'graph.json'
            path.write_text(json.dumps(graph))
            c = champion()
            engine = graph_runtime.HazelGraph(c, path)
            orders = engine.market(observation(0), 0, [], c.farm_state(observation(0), 0, None))
        self.assertIn(['BUY_PRODUCT', 'WHEAT', 13], [list(o) for o in orders])

    def test_update_reset_and_type_checks(self):
        base = graph_edits.apply_edit(seed_graph(), {'parameters': {'_SHOP_SELL_BATCH_MAX': 6}})
        again = graph_edits.apply_edit(base, {'parameters': {'_SHOP_SELL_BATCH_MAX': 3}})
        holders = [n['id'] for c in ('turn', 'market') for n in again[c]['nodes']
                   if '_SHOP_SELL_BATCH_MAX' in (n.get('parameters') or {})]
        self.assertEqual(holders, ['town_and_fertilizer'])
        reset = graph_edits.apply_edit(again, {'parameters': {'_SHOP_SELL_BATCH_MAX': None}})
        self.assertEqual(graph_edits.settings(reset)['parameters'], {})
        self.assertEqual(graph_edits.apply_edit(seed_graph(), {'parameters': {'_PRICE_THRESHOLD_RATIO': 1}})
                         ['market']['nodes'][11]['parameters'], {'_PRICE_THRESHOLD_RATIO': 1.0})
        for bad in ({'parameters': {'_OPENING_BUY_WHEAT_QTY': '13'}},
                    {'parameters': {'_OPENING_BUY_WHEAT_QTY': 13.5}},
                    {'parameters': {'_ENABLE_SHED_PRESSURE': 1}},
                    {'parameters': {'_NOT_A_CONSTANT': 1}},
                    {'parameters': {'_OPENING_BUY_WHEAT_QTY': 35}},  # changes nothing
                    {'guidance': 'sell more'}, {}, []):
            with self.assertRaises(ValueError, msg=bad):
                graph_edits.apply_edit(seed_graph(), bad)

    def test_stage_order_and_surgical_edits(self):
        graph = graph_edits.apply_edit(seed_graph(), {'stages': {'investment_freeze': False},
                                                      'dispatch_order': 'sells_first',
                                                      'surgical': {'enabled': True,
                                                                   'overrides': {'fertilizer_guard': False}}})
        s = graph_edits.settings(graph)
        self.assertEqual(s['stages'], {'investment_freeze': False})
        self.assertEqual(s['dispatch_order'], 'sells_first')
        self.assertFalse(s['surgical']['overrides']['fertilizer_guard'])
        back = graph_edits.apply_edit(graph, {'stages': {'investment_freeze': True}, 'dispatch_order': 'submitted',
                                              'surgical': {'enabled': False}})
        self.assertEqual(graph_edits.settings_key(back), graph_edits.settings_key(seed_graph()))
        for bad in ({'stages': {'market_setup': False}}, {'stages': {'routine_dispatch': False}},
                    {'stages': {'nope': False}}, {'stages': {'investment_freeze': 0}},
                    {'dispatch_order': 'buys_first'}, {'surgical': {'enabled': True, 'overrides': {'x': 1}}}):
            with self.assertRaises(ValueError, msg=bad):
                graph_edits.apply_edit(seed_graph(), bad)

    def test_settings_key_ignores_explicit_defaults_and_prose(self):
        graph = seed_graph()
        next(n for n in graph['market']['nodes'] if n['id'] == 'opening_scalp')['parameters'] = \
            {'_OPENING_BUY_WHEAT_QTY': 35}
        graph['nodes'][0]['description'] = 'rewritten prose'
        self.assertEqual(graph_edits.settings_key(graph), graph_edits.settings_key(seed_graph()))

    def test_diff_and_controls_text(self):
        after = graph_edits.apply_edit(seed_graph(), {'parameters': {'_TOWN_CADENCE_PHASE': 3},
                                                      'dispatch_order': 'sells_first'})
        self.assertEqual(graph_edits.diff(seed_graph(), after),
                         ['_TOWN_CADENCE_PHASE (town_and_fertilizer): 0 -> 3',
                          'routine_dispatch order: submitted -> sells_first'])
        text = graph_edits.describe_controls(after)
        for name in self.constants:
            self.assertIn(name, text)
        self.assertIn('DISPATCH ORDER (routine_dispatch): sells_first', text)


class JudgeTests(unittest.TestCase):
    def proposals(self):
        return [{'model': 'm1', 'rationale': 'r1', 'changes': ['_SHOP_SELL_BATCH_MAX (town_and_fertilizer): 4 -> 6']},
                {'model': 'm2', 'rationale': 'r2', 'changes': ['routine_dispatch order: submitted -> sells_first']}]

    def test_judge_sees_the_changes_and_ranks_by_votes(self):
        seen = []

        def reply(model, system, user, **kw):
            seen.append(user)
            return '{"reasoning": "sells first frees cash", "winner": "B"}'
        with patch.object(sift_graph_judge, '_llm_call', reply):
            ranked, votes = sift_graph_judge.SIFTGraphJudge().rank_edits(self.proposals(), 'facts', 'context')
        self.assertEqual(votes, 1)
        self.assertEqual([p['model'] for p, _ in ranked], ['m2', 'm1'])
        self.assertIn('_SHOP_SELL_BATCH_MAX (town_and_fertilizer): 4 -> 6', seen[0])
        self.assertIn('routine_dispatch order: submitted -> sells_first', seen[0])

    def test_invalid_replies_are_no_votes_not_a_default_winner(self):
        for text in ('I prefer A.', '{"winner": "C"}', 'no json'):
            self.assertIsNone(sift_graph_judge._verdict(text))
        with patch.object(sift_graph_judge, '_llm_call', side_effect=RuntimeError('proxy down')):
            ranked, votes = sift_graph_judge.SIFTGraphJudge().rank_edits(self.proposals(), 'facts', 'context')
        self.assertEqual((votes, [p['model'] for p, _ in ranked]), (0, ['m1', 'm2']))

    def test_legacy_ranking_shows_json_differences_and_losses(self):
        a, b = seed_graph(), seed_graph()
        b['nodes'][3]['description'] = 'changed text deep in the graph'
        seen = []

        def reply(model, system, user, **kw):
            seen.append(user)
            return '{"winner": "A"}'
        with patch.object(sift_graph_judge, '_llm_call', reply):
            sift_graph_judge.SIFTGraphJudge().rank_candidates(
                [(a, 'ra', 'x'), (b, 'rb', 'y')], 'facts',
                {'worst_differential_losses': [{'champ': 'hazel', 'deficit': 1234}]})
        self.assertIn('changed text deep in the graph', seen[0])
        self.assertIn('1234', seen[0])


class MutatorTests(unittest.TestCase):
    def test_prose_mutation_refuses_the_merged_graph(self):
        with self.assertRaises(ValueError):
            BAMGraphMutator().mutate_graph('m', seed_graph(), {}, [], '', [])

    def test_edit_mutation_parses_the_edit(self):
        reply = '```json\n{"rationale": "why", "edit": {"dispatch_order": "sells_first"}}\n```'
        with patch.object(BAMGraphMutator, '_chat', return_value=reply) as chat:
            edit, rationale = BAMGraphMutator().mutate_edit('m', 'controls', 'focus', 'results', [], [], 'facts',
                                                            error='previous error text', ideas=['try 9/5'])
        self.assertEqual((edit, rationale), ({'dispatch_order': 'sells_first'}, 'why'))
        prompt = chat.call_args.args[2]
        self.assertIn('previous error text', prompt)
        self.assertIn('- try 9/5', prompt)
        self.assertIn('controls', prompt)
        with patch.object(BAMGraphMutator, '_chat', return_value='{"rationale": "no edit"}'):
            with self.assertRaises(ValueError):
                BAMGraphMutator().mutate_edit('m', 'c', 'f', 'r', [], [], 'k')

    def test_bandit_picks_distinct_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            bandit = UCB1Bandit(Path(tmp) / 'b.json')
            used = []
            for _ in range(3):
                used.append(bandit.select_arm(exclude=used))
        self.assertEqual(len(set(used)), 3)


class GauntletTests(unittest.TestCase):
    def test_seeds_are_disjoint_from_validation_rounds(self):
        seeds = graph_gauntlet.seed_list(40)
        import random
        held = set(random.Random(graph_gauntlet.VALIDATION_SALT).sample(range(10_000_000, 2_000_000_000), 200))
        self.assertEqual(len(set(seeds)), 40)
        self.assertFalse(held & set(seeds))
        self.assertEqual(seeds, graph_gauntlet.seed_list(40))

    def test_paired_comparison(self):
        base = {'margins': {'hazel|1|0': 100.0, 'hazel|2|1': -50.0, 'mohui|1|0': 10.0}}
        cand = {'jobs': ['hazel|1|0', 'hazel|2|1', 'mohui|1|0', 'incumbent|1|0'],
                'margins': {'hazel|1|0': 150.0, 'hazel|2|1': -50.0, 'mohui|1|0': 5.0, 'incumbent|1|0': 30.0},
                'errors': [], 'fallbacks': 0}
        v = graph_gauntlet.compare(cand, base)
        self.assertEqual((v['wins'], v['losses'], v['ties'], v['mean_change']), (2, 1, 1, 18.8))
        self.assertEqual(v['per_opponent']['incumbent'], {'wins': 1, 'losses': 0, 'ties': 0, 'n': 1, 'mean': 30.0})
        self.assertFalse(v['promote'])  # 2-1 is not significant
        many = {'jobs': [f'hazel|{i}|0' for i in range(12)], 'errors': [], 'fallbacks': 0,
                'margins': {f'hazel|{i}|0': 10.0 for i in range(12)}}
        self.assertTrue(graph_gauntlet.compare(many, {'margins': {k: 0.0 for k in many['margins']}})['promote'])
        self.assertFalse(graph_gauntlet.compare(dict(many, fallbacks=1),
                                                {'margins': {k: 0.0 for k in many['margins']}})['promote'])
        missing = dict(many, jobs=many['jobs'] + ['hazel|99|0'])
        self.assertFalse(graph_gauntlet.compare(missing, {'margins': {k: 0.0 for k in many['margins']}})['valid'])

    def test_evaluate_plays_pool_and_head_to_head_and_caches_the_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            played = []

            class FakeExecutor:
                def describe(self):
                    return {'engine': 'fake'}

                def run(self, run_dir, name, bundles):
                    played.append((name, sorted(bundles)))
                    jobs = json.loads((run_dir / 'jobs' / f'{name}.json').read_text())['jobs']
                    with (run_dir / 'games' / f'{name}.jsonl').open('a') as fh:
                        for j in jobs:
                            rewards = [1000.0, 900.0] if j['a_seat'] == 0 else [900.0, 1000.0]
                            fh.write(json.dumps(dict(j, rewards=rewards, statuses=['DONE', 'DONE'], errors=[])) + '\n')
                    (run_dir / 'logs' / f'{name}.log').write_text(
                        'graph fallback #1 [someone_else] at step 3 stage market: x\n')

            g = graph_gauntlet.Gauntlet(run, FakeExecutor(), seeds_per_opponent=4, opponents=('mohui',))
            g.prepare()
            seed = seed_graph()
            cand = graph_edits.apply_edit(seed, {'dispatch_order': 'sells_first'})
            result = g.evaluate(cand, seed)
            self.assertEqual(len(result['margins']), 8)
            self.assertEqual(result['fallbacks'], 0)
            self.assertEqual({k.split('|')[0] for k in result['margins']}, {'mohui', 'incumbent'})
            name = g.bundle(cand)
            bundle_graph = json.loads((run / 'bundles' / name / 'policy_graph.json').read_text())
            self.assertEqual(bundle_graph['name'], name)
            self.assertEqual(graph_edits.settings_key(bundle_graph), graph_edits.settings_key(cand))
            cached = g.baseline(cand)
            self.assertEqual(len(played), 1)  # the pool games were cached by evaluate()
            self.assertEqual(set(cached['margins']), {k for k in result['margins'] if k.startswith('mohui')})


    def test_fallbacks_are_attributed_to_the_candidate_graph(self):
        with tempfile.TemporaryDirectory() as tmp:
            g = graph_gauntlet.Gauntlet(Path(tmp), None, seeds_per_opponent=2, opponents=('mohui',))
            (Path(tmp) / 'logs').mkdir()
            (Path(tmp) / 'logs' / 'g_a_vs_g_b.log').write_text(
                'graph fallback #1 [g_b] at step 9 stage market: x\n'
                'graph fallback #1 [g_a] at step 400 stage market: KeyError\n')
            self.assertEqual(g.fallbacks('g_a_vs_g_b', 'g_a'), 1)
            self.assertEqual(g.fallbacks('g_a_vs_g_b', 'g_c'), 0)

    def test_ssh_executor_launches_once_and_attaches_to_a_running_job_set(self):
        calls = []

        def fake_ssh(self, command, check=True, timeout=600):
            calls.append(command)
            out = ''
            if command.startswith('pgrep'):
                out = 'yes\n' if self.script and self.script.pop(0) else ''
            elif command.startswith('grep -o'):
                marker = command.split('"')[1].split('[')[0]
                out = f'{marker}123=0\n' if marker != 'ARENA_EXIT_' else 'ARENA_EXIT_9=0\n'
            return subprocess.CompletedProcess(command, 0, out, '')

        import subprocess
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(graph_gauntlet.SSHExecutor, '_ssh', fake_ssh), \
                patch.object(graph_gauntlet.SSHExecutor, '_rsync', lambda *a, **k: None), \
                patch.object(graph_gauntlet.subprocess, 'run'), patch.object(graph_gauntlet.time, 'sleep'):
            ex = graph_gauntlet.SSHExecutor('box', '~/evo', 60, poll=0)
            ex.ready = True
            ex.script = [False]            # nothing running: launch with a fresh marker
            ex.run(Path(tmp), 'job', ['g_a'])
            launches = [c for c in calls if 'nohup' in c]
            self.assertEqual(len(launches), 1)
            self.assertIn('--workers 60', launches[0])
            calls.clear()
            ex.script = [True, True, False]  # still running from a previous loop: wait, never relaunch
            ex.run(Path(tmp), 'job', ['g_a'])
            self.assertFalse([c for c in calls if 'nohup' in c])
            self.assertTrue(all('[j]obs/job.json' in c for c in calls if c.startswith('pgrep')))


class EvolutionLoopTests(unittest.TestCase):
    def test_one_iteration_promotes_and_writes_only_the_run_directory(self):
        committed = (ROOT / 'policy_graph.json').read_bytes()
        seed = seed_graph()

        class FakeGauntlet:
            alpha = 0.05

            def __init__(self):
                self.evaluated = []

            def baseline(self, graph):
                gain = 5.0 if graph_edits.settings(graph)['dispatch_order'] == 'sells_first' else 0.0
                return {'margins': {f'hazel|{i}|{i % 2}': gain for i in range(20)}}

            def evaluate(self, graph, incumbent):
                self.evaluated.append(graph)
                margins = {f'hazel|{i}|{i % 2}': 5.0 for i in range(20)}
                margins.update({f'incumbent|{i}|{i % 2}': 1.0 for i in range(20)})
                return {'bundle': 'g_x', 'jobs': sorted(margins), 'margins': margins, 'errors': [], 'fallbacks': 0}

        class FakeMutator:
            def __init__(self):
                self.ideas = []

            def mutate_edit(self, model, controls, focus, results, history, guidance, knowledge, error, ideas=None):
                self.ideas.append(ideas)
                return {'dispatch_order': 'sells_first'}, f'{model} proposes sells first'

        class FakeJudge:
            def rank_edits(self, proposals, knowledge, context):
                return [(p, 1.0) for p in proposals], 0

        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            validated = []
            (run / 'ideas.md').write_text('# ideas\n- try opening 9/5\n* cadence phase 2\nnot a bullet\n')
            mutator = FakeMutator()
            loop = evo.EditEvolution(run, FakeGauntlet(), mutator, FakeJudge(),
                                     UCB1Bandit(run / 'bandit.json'), validated.append,
                                     lambda *a, **k: ['guidance'], 'facts', seed_graph=seed,
                                     candidates=2, supervisor_interval=1, ideas_path=run / 'ideas.md')
            loop.run(iterations=2)
            self.assertEqual(mutator.ideas[0], ['try opening 9/5', 'cadence phase 2'])
            opening, town = loop.state['islands'][:2]
            self.assertEqual(graph_edits.settings(opening['graph'])['dispatch_order'], 'sells_first')
            self.assertEqual(len(opening['history']), 1)
            # The second proposal of iteration 1 and every proposal of iteration 2 repeat
            # already-seen settings: refused before any game, fed back to the model.
            self.assertEqual(len(loop.gauntlet.evaluated), 1)
            self.assertEqual(town['history'], [])
            log = [json.loads(line) for line in (run / 'candidates.jsonl').read_text().splitlines()]
            self.assertTrue(any('already evaluated' in r.get('error', '') for r in log))
            self.assertEqual(loop.state['guidance'], ['guidance'])
            best = json.loads((run / 'best.json').read_text())
            self.assertEqual(best['changes'], ['routine_dispatch order: submitted -> sells_first'])
            self.assertTrue((run / 'checkpoint.json').exists())
            resumed = evo.EditEvolution(run, FakeGauntlet(), FakeMutator(), FakeJudge(),
                                        UCB1Bandit(run / 'bandit.json'), validated.append,
                                        lambda *a, **k: [], 'facts')
            self.assertEqual(resumed.state['next_iteration'], 3)
            self.assertIn(str(run / 'seed_graph.json'), [str(p) for p in validated])
        self.assertEqual((ROOT / 'policy_graph.json').read_bytes(), committed)

    def test_seed_must_be_the_merged_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            legacy = {'version': '1.4.0', 'nodes': [], 'edges': []}
            with self.assertRaises(ValueError):
                evo.EditEvolution(Path(tmp), None, None, None, None, None, None, '', seed_graph=legacy)


if __name__ == '__main__':
    unittest.main()
