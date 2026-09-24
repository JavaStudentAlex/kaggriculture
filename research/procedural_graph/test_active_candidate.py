"""Deployment contract tests for the latest local procedural graph."""
import hashlib
import json
from pathlib import Path
import unittest

import graph_edits

ROOT = Path(__file__).resolve().parent


class ActiveCandidateTests(unittest.TestCase):
    def test_predictor_is_hazel_weirs_checkpoint(self):
        hazel = ROOT.parents[1] / 'shinka/champions/submissions/hazel_weir/checkpoint'
        for name in ('model.safetensors', 'scaler.npz', 'config.json', 'labels.json'):
            self.assertEqual((hazel/name).read_bytes(), (ROOT/'hazel_runtime/checkpoint'/name).read_bytes(), name)
        self.assertFalse((ROOT/'hazel_runtime/checkpoint/calibration.json').exists())

    def test_policy_is_hazel_plus_graph_edits_only(self):
        graph = json.loads((ROOT/'policy_graph.json').read_text())
        self.assertFalse(graph['surgical']['enabled'])
        self.assertFalse(any(graph['experimental']['switches'].values()))
        self.assertEqual(graph_edits.settings(graph), {
            'parameters': {'_OPENING_BUY_WHEAT_QTY': 13, '_OPENING_SELL_WHEAT_QTY': 9},
            'stages': {}, 'dispatch_order': 'sells_first', 'surgical': {'enabled': False}})

    def test_all_runtime_hashes_match(self):
        graph = json.loads((ROOT/'policy_graph.json').read_text())
        for rel, expected in graph['provenance']['runtime_bundle_hashes'].items():
            self.assertEqual(hashlib.sha256((ROOT/'hazel_runtime'/rel).read_bytes()).hexdigest(), expected, rel)
        self.assertEqual(hashlib.sha256((ROOT/'agent_graph.py').read_bytes()).hexdigest(), graph['provenance']['entrypoint_sha256'])
        self.assertEqual(graph['model_checkpoint']['model_sha256'],
                         graph['provenance']['runtime_bundle_hashes']['checkpoint/model.safetensors'])


if __name__ == '__main__':
    unittest.main()
