"""Structural/source coverage tests for the actual CURRENT merged graph."""
import hashlib
import importlib.util
import json
import unittest
from pathlib import Path
from graph_runtime import validate_chain, MARKET_STAGES, TURN_STAGES

HERE = Path(__file__).resolve().parent
PG = HERE.parent


class MergedTests(unittest.TestCase):
    def setUp(self):
        self.graph = json.loads((PG / 'policy_graph.json').read_text())

    def test_preserves_every_old_concept_with_binding(self):
        old = json.loads((HERE / 'premerge/policy_graph.json').read_text())
        nodes = {n['id']: n for n in self.graph['nodes']}
        self.assertEqual(len(nodes), len(self.graph['nodes']))
        for n in old['nodes']:
            self.assertIn(n['id'], nodes)
            self.assertTrue(nodes[n['id']]['bindings'])
            self.assertEqual(nodes[n['id']]['previous_description'], n['description'])

    def test_all_execution_stages_bound_and_all_edges_resolve(self):
        allowed = {s[0] for s in TURN_STAGES} | {s[0] for s in MARKET_STAGES}
        nodes = self.graph['nodes']
        ids = {n['id'] for n in nodes}
        self.assertEqual({b for n in nodes for b in n['bindings']}, allowed)
        for edge in self.graph['edges']:
            self.assertIn(edge['source'], ids)
            self.assertIn(edge['target'], ids)
        for key, stages in [('turn', TURN_STAGES), ('market', MARKET_STAGES)]:
            validate_chain(self.graph[key], [s[0] for s in stages])

    def test_runtime_and_frozen_copy_match_current(self):
        for relative, expected in self.graph['provenance']['runtime_bundle_hashes'].items():
            self.assertEqual(hashlib.sha256((PG / 'hazel_runtime' / relative).read_bytes()).hexdigest(), expected)
        candidate = HERE / 'merged_payload/agents/candidate_merged_graph'
        self.assertEqual((candidate / 'main.py').read_bytes(), (PG / 'agent_graph.py').read_bytes())
        self.assertEqual((candidate / 'policy_graph.json').read_bytes(), (PG / 'policy_graph.json').read_bytes())

    def test_every_archived_source_file_is_retained(self):
        source = PG.parents[1] / 'shinka/champions/submissions/hazel_weir'
        manifest = json.loads((source / 'MANIFEST.json').read_text())
        for relative in manifest['archive_files']:
            self.assertEqual((source / relative).read_bytes(), (PG / 'hazel_runtime' / relative).read_bytes(), relative)

    def test_legacy_engine_rejects_new_schema(self):
        spec = importlib.util.spec_from_file_location('legacy_engine_test', PG / 'graph_engine.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with self.assertRaises(ValueError):
            module.ProceduralGraphEngine(PG / 'policy_graph.json')
        # Historic graph remains loadable with its historic semantics.
        module.ProceduralGraphEngine(HERE / 'premerge/policy_graph.json')


if __name__ == '__main__': unittest.main(verbosity=2)
