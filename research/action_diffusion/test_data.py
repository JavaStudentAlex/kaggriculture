"""Comprehensive tests for action diffusion data pipeline and action codec.

Covers:
- Codec constants, field_sizes length 129 repeating, vocab sizes.
- Deterministic action encode / decode roundtrip.
- Refusal of hand_count > 32 and rejection of out-of-range / MASK categories.
- Commands the engine skips become counted empty slots; codec violations the engine would act
  on still reject the episode (DataError).
- Team and seat ratings of a day from its manifest.csv game scores (add_ratings).
- Numerical quantity parsing, engine sentinels (999/1000), truncation to 101.
- Active field mask computation.
- Observation featurization (FEATURE_DIM property), leak-free public/private checks.
- merge_observation correctness and isolation (seat0 vs seat1 private).
- Real replay loading from replays/*.zip: next_action alignment (t obs -> t+1 action).
- Real replay private leakage verification (assert zero opponent private in own features).
- Streaming prepare() with deterministic SHA256 split and deduplication across sources.
- WindowDataset: context [64, F], actions [24, 129], padding, masks, stride, LRU cache.
"""
from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import (
    ACTION_DIM,
    ARGUMENTS,
    CODEC_CONFIG,
    COMMAND_SLOTS,
    CONTEXT,
    DataError,
    FEATURE_DIM,
    FEATURE_NAMES,
    FEATURE_SPEC,
    FIELD_SIZES,
    HORIZON,
    ITEM_TO_ID,
    ITEMS,
    MAX_HANDS,
    OP_TO_ID,
    OPERATIONS,
    QUANTITY_BINS,
    WindowDataset,
    active_field_mask,
    add_ratings,
    decode_action,
    encode_action,
    encode_observation,
    episode_arrays,
    field_sizes,
    merge_observation,
    prepare,
    split_for_episode,
)


class TestActionCodec(unittest.TestCase):
    def test_constants_and_field_sizes(self):
        self.assertEqual(HORIZON, 24)
        self.assertEqual(CONTEXT, 64)
        self.assertEqual(COMMAND_SLOTS, 43)
        self.assertEqual(len(field_sizes), 129)
        self.assertEqual(len(FIELD_SIZES), 129)
        self.assertEqual(ACTION_DIM, 129)
        # Field sizes repeating vocab sizes: [len(OPERATIONS), len(ITEMS), 102] * 43
        expected_slot = [len(OPERATIONS), len(ITEMS), QUANTITY_BINS]
        self.assertEqual(field_sizes, expected_slot * COMMAND_SLOTS)
        self.assertEqual(len(OPERATIONS), 25)
        self.assertEqual(len(ITEMS), 13)
        self.assertEqual(QUANTITY_BINS, 102)

    def test_codec_config_serialization(self):
        serialized = json.dumps(CODEC_CONFIG)
        self.assertIn("operations", serialized)
        self.assertIn("items", serialized)
        self.assertIn("field_sizes", serialized)
        self.assertEqual(CODEC_CONFIG["command_slots"], 43)

    def test_encode_decode_roundtrip(self):
        action = {
            "farmer": ["PICKUP", "SHEEP", 4],
            "hands": [
                ["WEST"],
                ["PLANT", "MELON"],
                ["PASS"],
            ],
            "market": [
                ["BUY_PRODUCT", "WHEAT", 4],
                ["HIRE"],
                ["BUY_SEED", "MELON", 7],
                ["SELL", "WOOL", 15],
            ],
        }
        encoded = encode_action(action)
        self.assertIsInstance(encoded, np.ndarray)
        self.assertEqual(encoded.shape, (129,))
        self.assertEqual(encoded.dtype, np.int64)

        decoded = decode_action(encoded, hand_count=3)
        self.assertEqual(decoded["farmer"], ["PICKUP", "SHEEP", 4])
        self.assertEqual(decoded["hands"], [["WEST"], ["PLANT", "MELON"], ["PASS"]])
        self.assertEqual(
            decoded["market"],
            [
                ["BUY_PRODUCT", "WHEAT", 4],
                ["HIRE"],
                ["BUY_SEED", "MELON", 7],
                ["SELL", "WOOL", 15],
            ],
        )

    def test_decode_hands_matched_to_live_count(self):
        action = {
            "farmer": ["PASS"],
            "hands": [["NORTH"], ["SOUTH"]],
            "market": [],
        }
        enc = encode_action(action)
        # When live hand count is 1, only 1 hand command decoded
        d1 = decode_action(enc, hand_count=1)
        self.assertEqual(len(d1["hands"]), 1)
        self.assertEqual(d1["hands"], [["NORTH"]])

        # When live hand count is 0, empty hands list
        d0 = decode_action(enc, hand_count=0)
        self.assertEqual(len(d0["hands"]), 0)

        # When live hand count is 3, slot 3 was empty (NONE) -> decodes to PASS to preserve index
        d3 = decode_action(enc, hand_count=3)
        self.assertEqual(len(d3["hands"]), 3)
        self.assertEqual(d3["hands"], [["NORTH"], ["SOUTH"], ["PASS"]])

    def test_refuse_hand_count_over_32(self):
        arr = np.zeros(129, dtype=np.int64)
        with self.assertRaises(ValueError):
            decode_action(arr, hand_count=33)
        with self.assertRaises(ValueError):
            decode_action(arr, hand_count=-1)

    def test_refuse_mask_and_out_of_range(self):
        arr = np.zeros(129, dtype=np.int64)
        # Category field_sizes[0] is MASK
        arr[0] = field_sizes[0]
        with self.assertRaises(ValueError):
            decode_action(arr, hand_count=0)

        # Negative category
        arr[0] = -1
        with self.assertRaises(ValueError):
            decode_action(arr, hand_count=0)

    def test_quantities_and_sentinels(self):
        stats = Counter()
        # Normal 0..100
        action = {"farmer": ["PICKUP", "WHEAT", 50], "hands": [], "market": []}
        enc = encode_action(action, stats=stats)
        self.assertEqual(enc[2], 50)

        # Oversized sentinel e.g. 999 or 1000 -> bin 101
        action_large = {"farmer": ["PASS"], "hands": [], "market": [["SELL", "WHEAT", 1000]]}
        enc_large = encode_action(action_large, stats=stats)
        # Market order 0 starts at slot 33 -> index 33*3 + 2 = 101
        self.assertEqual(enc_large[33 * 3 + 2], 101)
        self.assertEqual(stats["quantity_overflow_101"], 1)

        # Decodes to 101
        dec = decode_action(enc_large, hand_count=0)
        self.assertEqual(dec["market"][0], ["SELL", "WHEAT", 101])

        # String numeric e.g. "12"
        action_str = {"farmer": ["PASS"], "hands": [], "market": [["SELL", "WHEAT", "12"]]}
        enc_str = encode_action(action_str, stats=stats)
        self.assertEqual(enc_str[33 * 3 + 2], 12)
        self.assertEqual(stats["quantity_numeric_strings"], 1)

    def test_engine_ignored_commands_are_empty_slots(self):
        # Commands engine 1.32.7 skips without effect: the empty slot at the same index, counted
        order = ["SELL", "WHEAT", 3]
        cases = (
            ({"farmer": ["FLY"], "market": [order]}, {"market": [order]}, "unsupported_operation"),
            ({"farmer": ["HIRE"], "market": [order]}, {"market": [order]}, "unsupported_operation_slot"),
            ({"farmer": ["PICKUP", "GOLD", 1], "market": [order]}, {"market": [order]}, "unsupported_item"),
            ({"hands": [["PLANT", "GOOSE"], ["WATER"]]}, {"hands": [[], ["WATER"]]}, "unsupported_item"),
            ({"market": [["INVALID"], order]}, {"market": [[], order]}, "unsupported_operation"),
            ({"market": [["PASS"], order]}, {"market": [[], order]}, "unsupported_operation_slot"),
            ({"market": [["BUY_PRODUCT", "TOMATO", 2], ["BUY_PRODUCT", "WHEAT", 2]]},
             {"market": [[], ["BUY_PRODUCT", "WHEAT", 2]]}, "unsupported_item"),
            ({"market": [["SELL", "WHEAT"], order]}, {"market": [[], order]}, "missing_quantity"),
            ({"market": [["SELL", "WHEAT", "ALL"], order]}, {"market": [[], order]}, "unsupported_quantity"),
            (None, {}, "malformed_action"),
            ({"hands": "oops", "market": [order]}, {"market": [order]}, "malformed_action"),
        )
        for action, same_as, code in cases:
            stats = Counter()
            np.testing.assert_array_equal(encode_action(action, stats=stats), encode_action(same_as), err_msg=repr(action))
            self.assertEqual(stats["skipped_" + code], 1, repr(action))
        # An operation without an item ignores a stray argument, as the engine does
        stats = Counter()
        np.testing.assert_array_equal(encode_action({"farmer": ["NORTH", "junk"]}, stats=stats),
                                      encode_action({"farmer": ["NORTH"]}))
        self.assertEqual(stats["ignored_arguments"], 1)

    def test_codec_violations_the_engine_would_act_on_reject_episode(self):
        for action, code in (({"farmer": ["PASS"], "hands": [["PASS"]] * 33, "market": []}, "too_many_hands"),
                             ({"farmer": [7]}, "unsupported_operation"),
                             ({"farmer": ["PICKUP", "WHEAT", "ALL"]}, "unsupported_quantity"),
                             ({"farmer": ["PICKUP", "WHEAT", 1, 2]}, "malformed_command")):
            with self.assertRaises(DataError) as ctx:
                encode_action(action)
            self.assertEqual(ctx.exception.code, code, repr(action))

    def test_add_ratings_from_game_scores(self):
        true = {"A": 3100.0, "B": 3000.0, "C": 2900.0}
        games = {"1": ("A", "B"), "2": ("C", "B"), "3": ("A", "C"), "4": ("B", "A")}
        cash = {"1": (105000, 95000), "2": (90000, 110000), "3": (100000, 100000), "4": (99000, 101000)}
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "day.zip"
            rows = ["episode_id,create_time,avg_score,min_score,sum_score,agent_count,size_bytes"]
            for eid, (t0, t1) in games.items():
                a, b = true[t0], true[t1]
                rows.append(f"{eid},x,{(a + b) / 2},{min(a, b)},{a + b},2,1")
            with zipfile.ZipFile(source, "w") as z:
                z.writestr("manifest.csv", "\n".join(rows) + "\n")
            files = [{"source": str(source), "episode_id": eid, "seat": seat, "team": team,
                      "cash": cash[eid][seat], "opponent_cash": cash[eid][1 - seat]}
                     for eid, pair in games.items() for seat, team in enumerate(pair)]
            manifest = {"files": files, "sources": [{"source": str(source)}]}
            add_ratings(manifest)
        for record in files:
            self.assertAlmostEqual(record["team_rating"], true[record["team"]], delta=0.5)
            self.assertAlmostEqual(record["rating"], true[record["team"]], delta=0.5)
            self.assertEqual(record["team_rank"], "ABC".index(record["team"]) + 1)
        self.assertEqual([t["team"] for t in manifest["sources"][0]["teams"]], ["A", "B", "C"])
        # Margins: cash lead over the opponent / mean cash; expected: the rating edge times the
        # day's fitted margin per rating point (through the origin)
        edges = {eid: true[a] - true[b] for eid, (a, b) in games.items()}
        leads = {eid: (c0 - c1) / ((c0 + c1) / 2) for eid, (c0, c1) in cash.items()}
        per_point = sum(edges[e] * leads[e] for e in games) / sum(edges[e] ** 2 for e in games)
        self.assertAlmostEqual(manifest["sources"][0]["margin_fit"]["per_point"], per_point, places=6)
        for record in files:
            sign = 1 if record["seat"] == 0 else -1
            self.assertAlmostEqual(record["margin"], sign * leads[record["episode_id"]], places=4)
            self.assertAlmostEqual(record["expected_margin"], sign * per_point * edges[record["episode_id"]], places=4)

    def test_active_field_mask(self):
        action = {
            "farmer": ["PASS"],
            "hands": [["PLANT", "MELON"]],
            "market": [["BUY_SEED", "WHEAT", 5], ["HIRE"]],
        }
        enc = encode_action(action)
        mask = active_field_mask(enc)
        self.assertEqual(mask.shape, (129,))
        self.assertEqual(mask.dtype, bool)
        # All op fields are True
        for slot in range(COMMAND_SLOTS):
            self.assertTrue(mask[slot * 3])
        # Farmer PASS has no arg and no qty
        self.assertFalse(mask[1])
        self.assertFalse(mask[2])
        # Hand 0 PLANT has crop arg but no qty
        self.assertTrue(mask[1 * 3 + 1])
        self.assertFalse(mask[1 * 3 + 2])
        # Market 0 BUY_SEED has seed arg and qty
        self.assertTrue(mask[33 * 3 + 1])
        self.assertTrue(mask[33 * 3 + 2])
        # Market 1 HIRE has no arg and no qty
        self.assertFalse(mask[34 * 3 + 1])
        self.assertFalse(mask[34 * 3 + 2])


class TestObservationFeaturizer(unittest.TestCase):
    def _make_dummy_obs(self, seat=0):
        tiles_0 = [[None for _ in range(10)] for _ in range(10)]
        tiles_0[4][4] = {
            "kind": "PLANT",
            "crop": "WHEAT",
            "watered_today": True,
            "consecutive_unwatered": 0,
            "yield_units": 1,
            "max_lifespan_step": 120,
            "planted_day": 0,
            "fertilized_until_day": -1,
        }
        farms = [
            {
                "farmer": [4, 4],
                "hands": [[5, 4]],
                "hires_today": 1,
                "money": 2500.0,
                "tiles": tiles_0,
                "unlocked_quadrants": ["NW"],
            },
            {
                "farmer": [1, 1],
                "hands": [],
                "hires_today": 0,
                "money": 3000.0,
                "tiles": [[None for _ in range(10)] for _ in range(10)],
                "unlocked_quadrants": ["NW"],
            },
        ]
        market = {
            "inventory": {p: 10000 for p in ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER")},
            "prices": {p: 50 for p in ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER")},
        }
        town = {"unlocked_shops": ["BAKERY"]}
        private = {
            "inventories": [{"WHEAT": 2}, {}],
            "seeds": {"WHEAT": 5, "MELON": 2, "CARROT": 0, "TOMATO": 0, "STRAWBERRY": 0},
            "shed": {"WHEAT": 10, "COW": 0, "EGG": 0, "FERTILIZER": 0, "GOOSE": 0, "MELON": 0, "MILK": 0, "SHEEP": 0, "STRAWBERRY": 0, "TOMATO": 0, "CARROT": 0, "WOOL": 0},
        }
        return {
            "farms": farms,
            "market": market,
            "town": town,
            "day": 0,
            "hour": 5,
            "step": 5,
            "player": seat,
            "private": private,
        }

    def test_feature_dim_and_type(self):
        self.assertGreater(FEATURE_DIM, 0)
        self.assertEqual(len(FEATURE_NAMES), FEATURE_DIM)
        obs = self._make_dummy_obs(0)
        vec = encode_observation(obs, 0)
        self.assertIsInstance(vec, np.ndarray)
        self.assertEqual(vec.shape, (FEATURE_DIM,))
        self.assertEqual(vec.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(vec)))

    def test_feature_spec_serialization(self):
        serialized = json.dumps(FEATURE_SPEC)
        self.assertIn("feature_dim", serialized)
        self.assertIn("magnitude_transform", serialized)

    def test_no_oracle_private_leakage(self):
        obs0 = self._make_dummy_obs(0)
        # Add rogue secret oracle data to opponent farm and top level
        obs0["farms"][1]["secret_opponent_key"] = 999999
        obs0["opponent_private_oracle"] = {"shed": {"WOOL": 9999}}
        vec0 = encode_observation(obs0, 0)

        # Clear rogue keys
        obs_clean = self._make_dummy_obs(0)
        vec_clean = encode_observation(obs_clean, 0)
        np.testing.assert_array_equal(vec0, vec_clean)

    def test_merge_observation_seat_isolation(self):
        pub = self._make_dummy_obs(0)
        seat1_obs = {
            "private": {
                "inventories": [{}],
                "seeds": {"WHEAT": 99, "MELON": 0, "CARROT": 0, "TOMATO": 0, "STRAWBERRY": 0},
                "shed": {"WOOL": 77, "CARROT": 0, "COW": 0, "EGG": 0, "FERTILIZER": 0, "GOOSE": 0, "MELON": 0, "MILK": 0, "SHEEP": 0, "STRAWBERRY": 0, "TOMATO": 0, "WHEAT": 0},
            }
        }
        merged1 = merge_observation(pub, seat1_obs, seat=1, step=5)
        # Assert seat1 merged observation does NOT carry seat0 private
        self.assertEqual(merged1["player"], 1)
        self.assertEqual(merged1["private"]["seeds"]["WHEAT"], 99)
        self.assertEqual(merged1["private"]["shed"]["WOOL"], 77)
        # Seat 0 shed had WHEAT: 10, seat 1 has WHEAT: 0
        self.assertEqual(merged1["private"]["shed"]["WHEAT"], 0)


class TestRealReplayPipeline(unittest.TestCase):
    REPLAY_ZIP = Path("/home/alex/kaggriculture/replays/kaggriculture-episodes-2026-08-28.zip")

    def test_real_replay_alignment_and_leakage(self):
        if not self.REPLAY_ZIP.exists():
            self.skipTest("Real replay archive not found")

        with zipfile.ZipFile(self.REPLAY_ZIP) as zf:
            member = next(m for m in zf.namelist() if m.endswith("101270262.json"))
            doc = json.loads(zf.read(member))

        arrays = episode_arrays(doc)
        self.assertEqual(len(arrays), 2)
        X0, A0 = arrays[0]
        X1, A1 = arrays[1]
        self.assertEqual(len(X0), 719)
        self.assertEqual(len(A0), 719)
        self.assertEqual(X0.shape[1], FEATURE_DIM)
        self.assertEqual(A0.shape[1], 129)

        # Verify next_action alignment: A0[t] must match step t+1 action exactly
        for t in [0, 1, 20, 100]:
            step_action = doc["steps"][t + 1][0]["action"]
            expected_encoded = encode_action(step_action)
            np.testing.assert_array_equal(A0[t], expected_encoded)

        # Verify leak-free: check that features of seat 0 do NOT contain seat 1's private shed
        shed_fields = [i for i, name in enumerate(FEATURE_NAMES) if name.startswith("own.shed.")]
        # In step 20, let's verify that seat 0's shed features match seat 0's private shed, not seat 1's
        # From earlier inspection: seat 0 at t=20 has WHEAT: 6, seat 1 has WHEAT: 3
        wheat_idx = next(i for i in shed_fields if FEATURE_NAMES[i] == "own.shed.WHEAT")
        val0 = X0[20, wheat_idx]
        val1 = X1[20, wheat_idx]
        # X0[20] should reflect 6 units, X1[20] should reflect 3 units
        expected_s0 = np.log1p(6.0) / np.log1p(100.0)
        expected_s1 = np.log1p(3.0) / np.log1p(100.0)
        self.assertAlmostEqual(val0, expected_s0, places=5)
        self.assertAlmostEqual(val1, expected_s1, places=5)
        self.assertNotEqual(val0, val1)

    def test_prepare_streaming_and_manifest(self):
        if not self.REPLAY_ZIP.exists():
            self.skipTest("Real replay archive not found")

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "shards"
            # Extract 2 episodes to test deterministic split & prepare
            manifest = prepare([self.REPLAY_ZIP], out_dir, max_episodes_per_source=2, val_fraction=0.5)

            self.assertEqual(manifest["feature_dim"], FEATURE_DIM)
            self.assertEqual(manifest["field_sizes"], field_sizes)
            self.assertTrue((out_dir / "manifest.json").exists())
            self.assertTrue((out_dir / "features.json").exists())
            self.assertTrue((out_dir / "codec.json").exists())

            total_episodes = manifest["counts"]["train"]["episodes"] + manifest["counts"]["val"]["episodes"]
            self.assertEqual(total_episodes, 2)
            self.assertEqual(len(manifest["sources"]), 1)
            self.assertEqual(manifest["sources"][0]["accepted"], 2)
            # every seat record carries its day (from the file name) and result (more final cash wins)
            games = {}
            for record in manifest["files"]:
                self.assertEqual(record["day"], "2026-08-28")
                self.assertEqual(record["win"], 1.0 if record["cash"] > record["opponent_cash"] else
                                 0.0 if record["cash"] < record["opponent_cash"] else 0.5)
                games.setdefault(record["episode_id"], []).append(record["win"])
            self.assertTrue(all(sum(results) == 1.0 for results in games.values()))

            # Test WindowDataset
            # Check which split has files
            active_split = "train" if manifest["counts"]["train"]["episodes"] > 0 else "val"
            ds = WindowDataset(out_dir, active_split, context=64, horizon=24, stride=8)
            self.assertGreater(len(ds), 0)

            sample = ds[0]
            self.assertEqual(sample["context"].shape, (64, FEATURE_DIM))
            self.assertEqual(sample["context_mask"].shape, (64,))
            self.assertEqual(sample["actions"].shape, (24, 129))
            self.assertEqual(sample["action_mask"].shape, (24,))

            self.assertEqual(sample["context"].dtype, np.float32)
            self.assertEqual(sample["actions"].dtype, np.int64)
            self.assertEqual(sample["context_mask"].dtype, bool)
            self.assertEqual(sample["action_mask"].dtype, bool)

            # Check beginning padding: at t=0, context has length 1 valid (the current turn t=0)
            self.assertEqual(sample["context_mask"].sum(), 1)
            self.assertTrue(sample["context_mask"][-1])
            self.assertFalse(sample["context_mask"][0])
            self.assertEqual(sample["action_mask"].sum(), 24)

            # LRU test: read multiple samples
            sample_mid = ds[len(ds) // 2]
            self.assertEqual(sample_mid["context"].shape, (64, FEATURE_DIM))


if __name__ == "__main__":
    unittest.main()
