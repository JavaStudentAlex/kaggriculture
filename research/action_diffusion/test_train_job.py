"""Tests of the Kaggle job pieces around the data: combine() (which episodes train and which
validate), the day split of the data notebooks, and the generated notebook scripts."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_kaggle_kernel
import train_job

HEADER = {"version": 1, "alignment": "next_action", "feature_dim": 3, "field_sizes": [2], "codec": {},
          "split": {"val_fraction": 0.1}}


def write_day(root, day, records):
    """A day directory as data.prepare leaves it: manifest.json with (episode, split) records."""
    folder = root / "days" / day
    folder.mkdir(parents=True)
    files = [{"path": f"{split}/episode_{eid}_seat{seat}.npz", "episode_id": eid, "seat": seat, "split": split,
              "length": 10} for eid, split in records for seat in (0, 1)]
    (folder / "manifest.json").write_text(json.dumps(dict(HEADER, files=files)))


class TestCombine(unittest.TestCase):
    def test_held_out_episodes_of_other_days_train(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_day(root, "2026-09-25", [("a", "train"), ("b", "val")])
            # "b" again in an older day's archive: a validation game never trains
            write_day(root, "2026-09-24", [("c", "train"), ("d", "val"), ("b", "val")])
            used, n_train, n_val = train_job.combine(root, ["2026-09-25", "2026-09-24", "2026-09-23"],
                                                     ["2026-09-25"])
            self.assertEqual(used, ["2026-09-25", "2026-09-24"])
            manifest = json.loads((root / "current" / "manifest.json").read_text())
            split = {(f["episode_id"], f["path"].split("/")[2]): f["split"] for f in manifest["files"]}
            self.assertEqual(split, {("a", "2026-09-25"): "train", ("b", "2026-09-25"): "val",
                                     ("c", "2026-09-24"): "train", ("d", "2026-09-24"): "train"})
            self.assertEqual((n_train, n_val), (6, 2))
            self.assertEqual(manifest["days"], used)
            self.assertTrue(all(f["path"].startswith("../days/") for f in manifest["files"]))

    def test_no_day_encoded(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                train_job.combine(Path(tmp), ["2026-09-25"], ["2026-09-25"])

    def test_packed_day_unpacks_where_combine_finds_it(self):
        """The data notebook's days/<day>.tar, unpacked by the training notebook, is an encoded day."""
        with tempfile.TemporaryDirectory() as tmp:
            prep, train = Path(tmp) / "prep", Path(tmp) / "train"
            write_day(prep, "2026-09-25", [("a", "train"), ("b", "val")])
            with tarfile.open(Path(tmp) / "2026-09-25.tar", "w") as tar:
                tar.add(prep / "days" / "2026-09-25", arcname="2026-09-25")
            (train / "days").mkdir(parents=True)
            with tarfile.open(Path(tmp) / "2026-09-25.tar") as tar:
                tar.extractall(train / "days", filter="data")
            self.assertTrue(train_job.extract_day(train, "2026-09-25", workers=1))   # present: not encoded again
            self.assertEqual(train_job.combine(train, ["2026-09-25"], ["2026-09-25"]), (["2026-09-25"], 2, 2))


class TestSeatWeights(unittest.TestCase):
    """train.seat_weights: the fresher days weigh more (--recency-halving)."""

    class Files:
        def __init__(self, files):
            self.files = files
            self.manifest = {"files": files}

    def test_recency_halving(self):
        import train
        files = self.Files([{"day": "2026-09-25", "source": "/tmp/ad/replays/2026-09-25.zip"},
                            {"source": "/tmp/ad/replays/2026-09-11.zip"},   # older records: day from the source
                            {"day": "2026-08-28", "source": "x.zip"},
                            {"source": "no-date.zip"}])
        weights = train.seat_weights(files, recency_halving=14)
        self.assertEqual(weights[0], 1.0)
        self.assertAlmostEqual(weights[1], 0.5)
        self.assertAlmostEqual(weights[2], 0.25)
        self.assertEqual(weights[3], 1.0)   # unknown day: not down-weighted
        self.assertEqual(train.seat_weights(files), [1.0] * 4)

    def test_recency_times_team_and_game(self):
        import train
        files = self.Files([
            {"day": "2026-09-25", "source": "a", "team_rating": 3000.0, "margin": 0.0, "expected_margin": 0.0},
            {"day": "2026-09-18", "source": "b", "team_rating": 2950.0, "margin": 0.1, "expected_margin": 0.0},
            {"day": "2026-09-18", "source": "b", "team_rating": 3000.0, "margin": -0.1, "expected_margin": 0.0}])
        weights = train.seat_weights(files, halving=50, margin_doubling=0.1, recency_halving=7)
        self.assertAlmostEqual(weights[0], 1.0)
        self.assertAlmostEqual(weights[1], 0.5 * 2 * 0.5)   # team 50 below its day's best, 1 doubling, 7 days old
        self.assertAlmostEqual(weights[2], 1.0 * 0.5 * 0.5)


class TestKernels(unittest.TestCase):
    ROWS = [{"date": f"2026-09-{d:02d}", "episode_count": str(c)}
            for d, c in zip(range(25, 5, -1), [600, 620, 900, 650, 700, 930, 610, 640, 880, 660,
                                               700, 720, 800, 690, 610, 650, 700, 760, 820, 600])]

    def test_split_days_balances_episodes(self):
        groups = make_kaggle_kernel.split_days(self.ROWS, 5)
        self.assertEqual(sorted(d for g in groups for d in g), sorted(r["date"] for r in self.ROWS))
        count = {r["date"]: int(r["episode_count"]) for r in self.ROWS}
        loads = [sum(count[d] for d in g) for g in groups]
        self.assertLessEqual(max(loads) - min(loads), max(count.values()))
        self.assertTrue(all(g == sorted(g, reverse=True) for g in groups))

    def test_select_days(self):
        rows = sorted(self.ROWS, key=lambda r: r["date"], reverse=True)
        self.assertEqual(len(make_kaggle_kernel.select_days("all", rows)), 20)
        self.assertEqual([r["date"] for r in make_kaggle_kernel.select_days("last:2", rows)],
                         ["2026-09-25", "2026-09-24"])
        with self.assertRaises(SystemExit):
            make_kaggle_kernel.select_days("2026-10-01", rows)

    def test_generated_scripts_compile(self):
        code = make_kaggle_kernel.code_archive()
        prep = make_kaggle_kernel.PREP_LAUNCHER.format(code=code, days=["2026-09-25", "2026-09-24"])
        compile(prep, "kaggle_prep.py", "exec")
        for prepared in (True, False):
            train = make_kaggle_kernel.LAUNCHER.format(code=code, job_args=["--days", "all"],
                                                       train_args=["--batch-size", "64"], prepared=prepared)
            compile(train, "kaggle_train.py", "exec")
            self.assertIn(f"PREPARED = {prepared!r}", train)
        meta = json.loads(make_kaggle_kernel.metadata("u/k-data-1", "kaggle_prep.py", gpu=False))
        self.assertEqual((meta["enable_gpu"], meta["enable_internet"], meta["kernel_sources"]), ("false", "true", []))
        self.assertNotIn("machine_shape", meta)
        meta = json.loads(make_kaggle_kernel.metadata("u/k", "kaggle_train.py", gpu=True, dataset_sources=["u/w"],
                                                      kernel_sources=["u/k-data-1"]))
        self.assertEqual((meta["machine_shape"], meta["kernel_sources"]), ("NvidiaTeslaT4", ["u/k-data-1"]))


if __name__ == "__main__":
    unittest.main()
