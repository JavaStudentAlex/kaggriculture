"""Tests of the conditioned model (model.CONDITIONS): the condition encoding, the per-file table and
prompt, condition dropout, growing a conditioned model from an unconditioned checkpoint (it must
start out predicting exactly as the checkpoint), guidance, and the loader's file index."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import inference
import train
from model import CONDITIONS, ActionDiffusion, Config, encode_condition, predict

N = len(CONDITIONS)


def tiny(condition_dim=0, seed=0):
    torch.manual_seed(seed)
    config = Config(feature_dim=6, field_sizes=[3, 4, 5], context=8, horizon=4, width=16, layers=1, heads=2,
                    context_patch=4, dropout=0.0, condition_dim=condition_dim)
    return ActionDiffusion(config).eval()


def inputs(batch=3, seed=1):
    g = torch.Generator().manual_seed(seed)
    context = torch.randn(batch, 8, 6, generator=g)
    mask = torch.ones(batch, 8, dtype=torch.bool)
    mask[0, :3] = False
    sizes = torch.tensor([3, 4, 5])
    noisy = sizes[None, None, :].expand(batch, 4, 3).clone()
    noisy[1, 0] = torch.tensor([1, 2, 3])
    return context, mask, noisy, torch.full((batch,), 0.9)


class Files:
    def __init__(self, files, others=()):
        self.files = files
        self.manifest = {"files": files + list(others)}


class TestEncoding(unittest.TestCase):
    def test_values_flags_and_clipping(self):
        row = encode_condition("2026-09-28", -150.0, 1.0, 0.05)
        self.assertEqual(len(row), 2 * N)
        np.testing.assert_allclose(row, [1.0, -1.5, 1.0, 0.5, 1, 1, 1, 1])
        self.assertEqual(encode_condition(), [0.0] * (2 * N))
        np.testing.assert_allclose(encode_condition(None, -900.0, 0.0, -0.9), [0, -3, -1, -3, 0, 1, 1, 1])
        np.testing.assert_allclose(encode_condition(None, 20.0, 0.5, 0.9), [0, 0, 0, 3, 0, 1, 1, 1])  # a tie is known

    def test_table_prompt_and_day_best(self):
        a = {"source": "x/kaggriculture-episodes-2026-09-25.zip", "team_rating": 3000.0, "win": 1.0,
             "margin": 0.10, "expected_margin": 0.02}
        b = {"source": "x/kaggriculture-episodes-2026-09-25.zip", "team_rating": 2900.0, "win": 0.0,
             "margin": -0.10, "expected_margin": -0.02, "day": "2026-09-25"}
        c = {"source": "y.zip", "day": "2026-09-01"}   # nothing known but the day
        top = {"source": "x/kaggriculture-episodes-2026-09-25.zip", "team_rating": 3050.0}   # validation split
        dataset = Files([a, b, c], others=[top])
        table = train.condition_table(dataset)
        self.assertEqual(table.shape, (3, 2 * N))
        date = (np.datetime64("2026-09-25") - np.datetime64("2026-07-30")).astype(int) / 60
        np.testing.assert_allclose(table[0], [date, -0.5, 1, 0.8, 1, 1, 1, 1], rtol=1e-6)
        np.testing.assert_allclose(table[1], [date, -1.5, -1, -0.8, 1, 1, 1, 1], rtol=1e-6)
        np.testing.assert_allclose(table[2], [(np.datetime64("2026-09-01") - np.datetime64("2026-07-30")).astype(int) / 60,
                                              0, 0, 0, 1, 0, 0, 0], rtol=1e-6)
        info = train.condition_prompt(dataset)
        self.assertEqual(info["fields"], list(CONDITIONS))
        self.assertEqual(info["prompt"], {"day": "2026-09-25", "rating_gap": 0.0, "win": 1.0, "excess_margin": 0.08})

    def test_hide_conditions(self):
        condition = torch.tensor([encode_condition("2026-09-25", -50.0, 1.0, 0.1)] * 200)
        self.assertTrue(torch.equal(train.hide_conditions(condition, 0.0, 0.0), condition))
        self.assertEqual(train.hide_conditions(condition, 1.0, 0.0).abs().sum().item(), 0)
        self.assertEqual(train.hide_conditions(condition, 0.0, 1.0).abs().sum().item(), 0)
        hidden = train.hide_conditions(condition, 0.5, 0.0)
        known = hidden[:, N:]
        self.assertTrue(torch.equal(hidden[:, :N], condition[:, :N] * known))   # a hidden value is 0 with its flag
        self.assertTrue(0.3 < known.mean().item() < 0.7)


class TestGrowing(unittest.TestCase):
    def test_conditioned_model_starts_as_the_checkpoint(self):
        plain, grown = tiny(), tiny(N, seed=5)
        train.load_weights(grown, {"config": plain.config_dict(), "model": plain.state_dict()})
        x = inputs()
        condition = torch.tensor([encode_condition("2026-09-25", -30.0, 1.0, 0.1), encode_condition(),
                                  encode_condition(None, None, 0.0, None)])
        with torch.no_grad():
            expected = plain(*x)
            torch.testing.assert_close(grown(*x), expected)
            torch.testing.assert_close(grown(*x, condition), expected)
        # the condition layers learn: their last layer gets a gradient at once
        grown.train()
        grown(*x, condition).float().logsumexp(-1).sum().backward()
        self.assertGreater(grown.condition[2].weight.grad.abs().sum().item(), 0)

    def test_old_checkpoint_config_and_refusals(self):
        plain = tiny()
        old = {k: v for k, v in plain.config_dict().items() if k != "condition_dim"}   # saved before conditions
        grown = tiny(N)
        train.load_weights(grown, {"config": old, "model": plain.state_dict()})
        with self.assertRaises(AssertionError):   # a conditioned checkpoint cannot start an unconditioned model
            train.load_weights(tiny(), {"config": grown.config_dict(), "model": grown.state_dict()})
        wider = ActionDiffusion(Config(**dict(plain.config_dict(), width=24, heads=2)))
        with self.assertRaises(AssertionError):
            train.load_weights(tiny(N), {"config": wider.config_dict(), "model": wider.state_dict()})

    def test_reloading_an_unconditioned_checkpoint_resets_the_condition_layers(self):
        plain, trained = tiny(), tiny(N, seed=3)
        with torch.no_grad():
            trained.condition[2].weight.normal_()
        model = tiny(N)
        train.load_weights(model, {"config": trained.config_dict(), "model": trained.state_dict()})
        train.load_weights(model, {"config": plain.config_dict(), "model": plain.state_dict()})
        x = inputs()
        condition = torch.tensor([encode_condition("2026-09-25", 0.0, 1.0, 0.1)] * 3)
        with torch.no_grad():
            torch.testing.assert_close(model(*x, condition), plain(*x))


class TestPrediction(unittest.TestCase):
    def test_guidance(self):
        model = tiny(N)
        with torch.no_grad():
            model.condition[2].weight.normal_(std=0.5)
        context, mask, _, _ = inputs()
        condition = torch.tensor([encode_condition("2026-09-25", 0.0, 1.0, 0.1)] * 3)
        free, _ = predict(model, context, mask, steps=2)
        same, p1 = predict(model, context, mask, steps=2, condition=condition)
        pushed, p2 = predict(model, context, mask, steps=2, condition=condition, guidance=3.0)
        self.assertEqual(pushed.shape, (3, 4, 3))
        self.assertTrue(torch.allclose(p1.sum(-1), torch.ones(3, 4, 3)))
        self.assertFalse(torch.allclose(p1, p2))   # guidance moves the distribution further
        self.assertTrue(torch.equal(predict(model, context, mask, 2, condition, 1.0)[0], same))

    def test_prompt_condition(self):
        model = tiny(N)
        model.conditions = {"prompt": {"day": "2026-09-25", "rating_gap": 0.0, "win": 1.0, "excess_margin": 0.08}}
        row = inference.prompt_condition(model, batch=2)
        self.assertEqual(row.shape, (2, 2 * N))
        np.testing.assert_allclose(row[0].numpy(), encode_condition("2026-09-25", 0.0, 1.0, 0.08), rtol=1e-6)
        self.assertIsNone(inference.prompt_condition(model, None))
        self.assertIsNone(inference.prompt_condition(tiny(), "default"))
        partial = inference.prompt_condition(model, {"win": 1.0})
        np.testing.assert_allclose(partial[0].numpy(), encode_condition(None, None, 1.0, None))


class TestLoader(unittest.TestCase):
    def test_shuffled_windows_carry_the_file_index(self):
        class Windows:
            files = [{"id": k} for k in range(5)]

            def read(self, index):
                return np.full((10, 1), index), np.zeros((10, 1))

            def anchors(self, length):
                return range(0, length, 3)

            def window(self, X, A, t):
                return {"x": X[t, 0]}

            def __len__(self):
                return 20

        items = list(train.ShuffledWindows(Windows(), 0, 1, buffer=2))
        self.assertEqual(len(items), 20)
        self.assertTrue(all(item["file"] == item["x"] for item in items))


if __name__ == "__main__":
    unittest.main()
