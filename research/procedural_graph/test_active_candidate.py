"""Deployment contract tests for the latest local procedural graph."""
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent


class ActiveCandidateTests(unittest.TestCase):
    def test_promoted_predictor_byte_identity(self):
        promoted = ROOT.parents[1] / 'models/ttm_c256_h96_ft_2026-09-22'
        for name in ('model.safetensors', 'scaler.npz', 'config.json', 'labels.json'):
            self.assertEqual((promoted/name).read_bytes(), (ROOT/'hazel_runtime/checkpoint'/name).read_bytes(), name)

    def test_surgical_mode_is_active_and_broad_failed_experiment_disabled(self):
        graph = json.loads((ROOT/'policy_graph.json').read_text())
        self.assertTrue(graph['surgical']['enabled'])
        self.assertFalse(any(graph['experimental']['switches'].values()))
        self.assertIn('surgical.py', graph['provenance']['runtime_bundle_hashes'])

    def test_all_runtime_hashes_match(self):
        graph = json.loads((ROOT/'policy_graph.json').read_text())
        for rel, expected in graph['provenance']['runtime_bundle_hashes'].items():
            self.assertEqual(hashlib.sha256((ROOT/'hazel_runtime'/rel).read_bytes()).hexdigest(), expected, rel)
        self.assertEqual(hashlib.sha256((ROOT/'agent_graph.py').read_bytes()).hexdigest(), graph['provenance']['entrypoint_sha256'])


if __name__ == '__main__':
    unittest.main()
