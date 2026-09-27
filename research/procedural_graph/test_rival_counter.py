"""The rival_counter stage: the mirror tracker's classes, the counter edits and switchable engine constants."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import graph_edits
from hazel_runtime import engines, rival_model

HERE = Path(__file__).resolve().parent
LADDER_GRAPH = HERE / 'evolution_results' / 'ladder_2026-09-26' / 'opening_champion_graph.json'


def farm(money, crops=0):
    return {'money': money, 'hands': [[0, 0]], 'unlocked_quadrants': ['NW'], 'hires_today': 0, 'farmer': [4, 4],
            'tiles': [[{'kind': 'PLANT', 'crop': 'WHEAT'}] * crops]}


def classify(rival_money):
    """Our money is 1000 every step; the rival's is rival_money(step)."""
    mt = rival_model.MirrorTracker()
    for step in range(120):
        decision = mt.update({'step': step, 'player': 0, 'farms': [farm(1000), farm(rival_money(step))]})
        if decision is not None:
            return decision, step
    return None, None


class MirrorTrackerTest(unittest.TestCase):
    def test_mirror_is_decided_at_step_93(self):
        self.assertEqual(classify(lambda s: 1000), ('mirror', 93))

    def test_step_92_signatures(self):
        self.assertEqual(classify(lambda s: 1000 - 90 * (s >= 92)), ('nsell_opener', 92))
        self.assertEqual(classify(lambda s: 1000 + 87 * (s >= 92)), ('wheat92_seller', 92))
        self.assertEqual(classify(lambda s: 1000 - 500 * (s >= 92)), ('other', 92))

    def test_earlier_divergence_is_another_opening(self):
        self.assertEqual(classify(lambda s: 1000 + (s >= 1)), ('other_opening', 1))

    def test_a_farm_difference_counts(self):
        mt = rival_model.MirrorTracker()
        mt.update({'step': 0, 'player': 1, 'farms': [farm(1000), farm(1000)]})
        self.assertEqual(mt.update({'step': 5, 'player': 1, 'farms': [farm(1000, crops=2), farm(1000)]}),
                         'other_opening')

    def test_a_new_game_resets(self):
        mt = rival_model.MirrorTracker()
        for step in range(100):
            mt.update({'step': step, 'player': 0, 'farms': [farm(1000), farm(1000)]})
        self.assertEqual(mt.decision, 'mirror')
        self.assertIsNone(mt.update({'step': 0, 'player': 0, 'farms': [farm(1000), farm(1000)]}))


class CounterEditTest(unittest.TestCase):
    def setUp(self):
        self.graph = json.loads(LADDER_GRAPH.read_text())
        self.edit = {'channels': {'rival_counter': True}, 'counters': {'mirror': {'_EV_H': 12, '_ADV_LOOK': 4}}}

    def test_switching_on_inserts_the_stage_first(self):
        out = graph_edits.apply_edit(self.graph, self.edit)
        turn = out['turn']
        self.assertEqual(turn['entry'], 'rival_counter')
        self.assertEqual(turn['nodes'][0]['id'], 'rival_counter')
        self.assertEqual(turn['edges'][0], {'source': 'rival_counter', 'target': 'backbone', 'relation': 'NEXT'})
        self.assertTrue(any(n['id'] == 'rival_counter' for n in out['nodes']))
        self.assertIn('rival_model.py', out['provenance']['runtime_bundle_hashes'])
        self.assertEqual(graph_edits.settings(out)['counters'], {'mirror': {'_ADV_LOOK': 4, '_EV_H': 12}})
        self.assertIn('counter vs mirror: _EV_H "default" -> 12', graph_edits.diff(self.graph, out))

    def test_graphs_without_the_stage_keep_their_settings(self):
        self.assertNotIn('counters', graph_edits.settings(self.graph))
        self.assertNotIn('rival_counter', graph_edits.settings(self.graph)['channels'])

    def test_counters_need_the_stage(self):
        with self.assertRaisesRegex(ValueError, 'rival_counter'):
            graph_edits.apply_edit(self.graph, {'counters': {'mirror': {'_EV_H': 12}}})

    def test_unknown_class_and_name_are_refused(self):
        with self.assertRaisesRegex(ValueError, 'unknown rival family'):
            graph_edits.apply_edit(self.graph, {'channels': {'rival_counter': True}, 'counters': {'x': {'_EV_H': 12}}})
        with self.assertRaisesRegex(ValueError, 'cannot be set per rival family'):
            graph_edits.apply_edit(self.graph, {'channels': {'rival_counter': True},
                                                'counters': {'mirror': {'_NO_SUCH_THING': 1}}})

    def test_null_removes_a_class(self):
        out = graph_edits.apply_edit(self.graph, self.edit)
        out = graph_edits.apply_edit(out, {'counters': {'mirror': None, 'nsell_opener': {'_OG_SCORE': 0.6}}})
        self.assertEqual(graph_edits.settings(out)['counters'], {'nsell_opener': {'_OG_SCORE': 0.6}})

    def test_migration_carries_counters(self):
        donor = graph_edits.apply_edit(self.graph, self.edit)
        edit = graph_edits.migration_edit(self.graph, donor, self.graph)
        self.assertEqual(edit['channels'], {'rival_counter': True})
        out = graph_edits.apply_edit(self.graph, edit)
        self.assertEqual(graph_edits.settings(out)['counters'], graph_edits.settings(donor)['counters'])


class SwitchableTest(unittest.TestCase):
    def test_only_call_time_constants(self):
        source = ('A = 1\nB = 2\nC = 3\nD = 4\nTABLE = {"x": B}\n'
                  'def f(x, d=C):\n    return x + A\n'
                  'def g():\n    return D\nD_COPY = D\n')
        self.assertEqual(engines.switchable(source), {'A'})


if __name__ == '__main__':
    unittest.main()
