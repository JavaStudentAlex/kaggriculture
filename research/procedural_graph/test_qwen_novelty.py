"""Offline safety tests. Live Ollama smoke test is run separately on evo-80."""
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import qwen_novelty_archive as archive
import shinka_graph_novelty as novelty
from pool_upgrade_state import save_checkpoint, load_checkpoint


def vector():
    return [1.0] + [0.0] * 4095


class QwenNoveltyTests(unittest.TestCase):
    def test_request_no_silent_truncation(self):
        graph = {"code": "x" * 20000}
        with patch.object(novelty.urllib.request, "urlopen", return_value=io.BytesIO(
                json.dumps({"embeddings": [vector()]}).encode())) as call:
            self.assertEqual(len(novelty.get_graph_embedding(graph)), 4096)
        body = json.loads(call.call_args.args[0].data)
        self.assertEqual(json.loads(body["input"]), graph)
        self.assertFalse(body["truncate"])
        self.assertEqual(body["model"], "qwen3-embedding:8b")

    def test_bad_vectors_fail_closed(self):
        for v in ([], [1.0] * 384, [0.0] * 4096, [float("nan")] * 4096):
            with self.assertRaises(ValueError):
                novelty.validate_embedding(v)
        with self.assertRaises(ValueError):
            novelty.cosine_similarity([1.0] * 384, vector())
        self.assertAlmostEqual(novelty.cosine_similarity(vector(), vector()), 1.0)

    def test_invalid_judge_does_not_accept(self):
        for content in ('nonsense', '{"is_novel":"false"}', '{}'):
            response = {"choices": [{"message": {"content": content}}]}
            with patch.object(novelty.urllib.request, "urlopen", return_value=io.BytesIO(
                    json.dumps(response).encode())), patch.object(novelty, "record_call"):
                self.assertFalse(novelty.judge_novelty_with_llm({}, {}, 1.0)[0])

    def test_archive_migration_roundtrip_and_cache(self):
        with tempfile.TemporaryDirectory() as root:
            p = Path(root)
            g = {"nodes": ["one"]}
            other = {"nodes": ["two"]}
            (p / "cand_iter_001_Island-Test.json").write_text(json.dumps(other))
            island = SimpleNamespace(island_id=0, name="Island-Test", focus_theme="test",
                                     champion_graph=g, champion_score={}, history=[],
                                     rejections=[], embeddings_archive=[[1.0] * 384])
            with patch.object(archive, "get_graph_embedding", return_value=vector()) as embed:
                report = archive.restore_archives([island], p, 2)
                self.assertEqual(embed.call_count, 2)
                self.assertEqual(report[0]["recovered_graphs"], 2)
                archive.restore_archives([island], p, 2)
                self.assertEqual(embed.call_count, 2)
            archive.register_graph(island, {"nodes": ["three"]}, vector())
            save_checkpoint(p / "checkpoint.json", [island], 2, g, {}, "old", {})
            state, restored = load_checkpoint(p / "checkpoint.json", SimpleNamespace)
            self.assertEqual(state["next_iteration"], 2)
            self.assertEqual(len(restored[0].novelty_graphs), 3)
            archive.restore_archives(restored, p, 2)
            restored[0].embeddings_archive.pop()
            with self.assertRaises(ValueError):
                archive.restore_archives(restored, p, 2)


if __name__ == "__main__":
    unittest.main()
