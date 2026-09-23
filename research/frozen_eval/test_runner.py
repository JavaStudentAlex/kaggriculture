"""Standard-library unit tests; opt in to real local engine/RPC tests.

python -B -m unittest discover -s research/frozen_eval -p test_runner.py -v
FROZEN_EVAL_INTEGRATION=1 .venv/bin/python -B -m unittest discover \
    -s research/frozen_eval -p test_runner.py -v

Integration tests use temporary trivial PASS agents, never the real candidate,
cloud, evolution, or networking. Fixture observations/results are not benchmark
results. All temporary payload/output artifacts are automatically removed.
"""
from __future__ import annotations

import copy
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from typing import Any

import runner


def manifest_fixture() -> dict[str, Any]:
    return dict(evaluation_id="frozen-test", candidate_id="candidate_iter27",
                opponents=[dict(id=f"opponent_{i}", label=f"Opponent {i}", kind="fixture")
                           for i in range(15)],
                seeds={"0": list(range(1_000_000, 1_000_100)),
                       "1": list(range(2_000_000, 2_000_100))},
                files={"placeholder": "0" * 64})


def good_record(job, fingerprint="fingerprint", rewards=None):
    record = runner.base_record(job, fingerprint)
    record.update(statuses=["DONE", "DONE"], states=720,
                  rewards=[10.0, 5.0] if rewards is None else rewards,
                  resolved_seed=job["seed"])
    return runner.finish_record(record)


class CoverageTests(unittest.TestCase):
    def test_exact_3000_and_600_per_shard(self):
        manifest = manifest_fixture()
        all_jobs = []
        shard_ids = []
        for shard in range(5):
            jobs = runner.build_jobs(manifest, shard_index=shard)
            self.assertEqual(len(jobs), 600)
            self.assertEqual({j["opponent_id"] for j in jobs},
                             {o["id"] for o in manifest["opponents"]})
            self.assertEqual({j["seed_index"] for j in jobs}, set(range(shard * 20, (shard + 1) * 20)))
            for opponent in manifest["opponents"]:
                for seat in (0, 1):
                    selected = [j for j in jobs if j["opponent_id"] == opponent["id"] and j["seat"] == seat]
                    self.assertEqual(len(selected), 20)
                    self.assertEqual(len({j["seed"] for j in selected}), 20)
            shard_ids.append({j["job_id"] for j in jobs})
            all_jobs.extend(jobs)
        self.assertEqual(len(all_jobs), 3000)
        self.assertEqual(len({j["job_id"] for j in all_jobs}), 3000)
        for a in range(5):
            for b in range(a + 1, 5):
                self.assertFalse(shard_ids[a] & shard_ids[b])
        for opponent in manifest["opponents"]:
            seat_sets = [{j["seed"] for j in all_jobs if j["opponent_id"] == opponent["id"] and j["seat"] == s}
                         for s in (0, 1)]
            self.assertEqual(list(map(len, seat_sets)), [100, 100])
            self.assertFalse(seat_sets[0] & seat_sets[1])
            self.assertFalse((seat_sets[0] | seat_sets[1]) & runner.LEGACY_SEEDS)

    def test_smoke_and_later_wave(self):
        manifest = manifest_fixture()
        for shard in range(5):
            smoke = runner.build_jobs(manifest, shard_index=shard, smoke=True)
            self.assertEqual(len(smoke), 30)
            self.assertEqual({j["seed_index"] for j in smoke}, {shard})
            self.assertEqual(smoke, runner.build_jobs(manifest, shard_index=shard, seed_count=5))
        first = runner.build_jobs(manifest, seed_start=0, seed_count=50)
        later = runner.build_jobs(manifest, seed_start=50, seed_count=50)
        self.assertFalse({j["job_id"] for j in first} & {j["job_id"] for j in later})
        self.assertEqual({j["seed_index"] for j in later}, set(range(50, 60)))
        subset = runner.build_jobs(manifest, smoke=True, opponents=["opponent_2"])
        self.assertEqual(len(subset), 2)

    def test_bad_shards_and_selection(self):
        for kwargs in (dict(shards=0), dict(shard_index=5), dict(seed_count=7),
                       dict(seed_start=-1), dict(seed_start=1), dict(seed_count=0),
                       dict(opponents=["missing"]), dict(opponents=[])):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                runner.build_jobs(manifest_fixture(), **kwargs)

    def test_reject_seed_overlap_duplicates_and_legacy(self):
        for change in ("duplicate", "overlap", "legacy", "excluded", "bool", "too_few"):
            manifest = manifest_fixture()
            if change == "duplicate":
                manifest["seeds"]["0"][1] = manifest["seeds"]["0"][0]
            elif change == "overlap":
                manifest["seeds"]["1"][0] = manifest["seeds"]["0"][0]
            elif change == "legacy":
                manifest["seeds"]["0"][0] = 101
            elif change == "excluded":
                manifest["excluded_seeds"] = [manifest["seeds"]["0"][0]]
            elif change == "bool":
                manifest["seeds"]["0"][0] = True
            else:
                manifest["seeds"]["0"].pop()
            with self.subTest(change=change), self.assertRaises(ValueError):
                runner.build_jobs(manifest)

    def test_roster_and_manifest_fail_closed(self):
        for change in ("duplicate", "short", "unsafe", "seed_override"):
            manifest = manifest_fixture()
            if change == "duplicate":
                manifest["opponents"][1] = manifest["opponents"][0]
            elif change == "short":
                manifest["opponents"].pop()
            elif change == "unsafe":
                manifest["candidate_id"] = "../candidate"
            else:
                manifest["configuration"] = {"seed": 17}
            with self.subTest(change=change), self.assertRaises(ValueError):
                runner.build_jobs(manifest)


class AccountingTests(unittest.TestCase):
    def test_wld_candidate_relative(self):
        self.assertEqual(runner.classify(["DONE"] * 2, 720, [2, 1], 0, []), ("W", []))
        self.assertEqual(runner.classify(["DONE"] * 2, 720, [2, 1], 1, []), ("L", []))
        self.assertEqual(runner.classify(["DONE"] * 2, 720, [0, 0], 1, []), ("D", []))
        self.assertEqual(runner.classify(["DONE"] * 2, 720, [-1, -2], 0, []), ("W", []))

    def test_errors_never_count_as_wins_or_draws(self):
        cases = [(["DONE", "ERROR"], 720, [1000, 0], []),
                 (["ACTIVE", "DONE"], 720, [1000, 0], []),
                 (["DONE"] * 2, 719, [1000, 0], []),
                 (["DONE"] * 2, 721, [1000, 0], []),
                 (["DONE"] * 2, 720, [float("nan"), 0], []),
                 (["DONE"] * 2, 720, [0, float("inf")], []),
                 (["DONE"] * 2, 720, [None, 0], []),
                 (["DONE"] * 2, 720, [True, 0], []),
                 (["DONE"] * 2, 720, [1000, 0], [{"message": "RPC error"}]),
                 (["DONE"] * 2, 720, [0, 0], [{"message": "RPC error"}])]
        for statuses, states, rewards, errors in cases:
            with self.subTest(case=(statuses, states, rewards, errors)):
                self.assertEqual(runner.classify(statuses, states, rewards, 0, errors)[0], "INVALID")

    def test_summary_preserves_invalids_and_pending_denominators(self):
        manifest = manifest_fixture()
        jobs = runner.build_jobs(manifest, smoke=True)
        records = [good_record(jobs[0]), good_record(jobs[1]),
                   good_record(jobs[2], rewards=[0, 0]),
                   runner.failure_record(jobs[3], "fingerprint", "Error", "broken opponent")]
        summary = runner.make_summary(manifest, jobs, {r["job_id"]: r for r in records},
                                      "fingerprint", {}, status="deadline", elapsed=10,
                                      new_results=4, workers=4)
        totals = summary["overall"]
        self.assertEqual([totals[k] for k in ("W", "L", "D", "invalid", "valid", "attempted", "remaining")],
                         [1, 1, 1, 1, 3, 4, 26])
        self.assertEqual(totals["win_rate_valid_only"], 1 / 3)
        self.assertEqual(totals["wins_over_planned"], 1 / 30)
        self.assertEqual(summary["estimated_remaining_seconds"], 65)
        self.assertFalse(summary["complete"])
        self.assertFalse(summary["benchmark_pass"])
        self.assertEqual(len(summary["per_opponent"]), 15)
        self.assertEqual(summary["per_opponent"]["opponent_1"]["seats"]["1"]["invalid"], 1)


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "results.jsonl"
        self.jobs = runner.build_jobs(manifest_fixture(), smoke=True)
        self.records = [good_record(self.jobs[0]),
                        runner.failure_record(self.jobs[1], "fingerprint", "Timeout", "too slow")]

    def write(self, records):
        self.path.write_text("".join(json.dumps(r) + "\n" for r in records))

    def test_resume_validated_ids_including_invalid_attempts(self):
        self.write(self.records)
        loaded = runner.load_results(self.path, self.jobs, "fingerprint")
        self.assertEqual(set(loaded), {j["job_id"] for j in self.jobs[:2]})
        self.assertEqual(loaded[self.jobs[1]["job_id"]]["outcome"], "INVALID")
        self.assertEqual(len([j for j in self.jobs if j["job_id"] not in loaded]), 28)

    def test_duplicate_rejected(self):
        self.write(self.records + self.records[:1])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            runner.load_results(self.path, self.jobs, "fingerprint")

    def test_mismatched_or_forged_rows_rejected(self):
        changes = {"source_fingerprint": "changed", "seed": -1, "seat": 1,
                   "job_id": "invented", "evaluation_id": "wrong", "candidate_id": "other",
                   "opponent_kind": "wrong", "engine_version": "9", "outcome": "D",
                   "states": 719, "rewards": [None, 0], "resolved_seed": 42,
                   "valid": False, "errors": [{"message": "hidden error"}],
                   "timing": {"total_seconds": -1}, "length": 0, "schema_version": 99}
        for key, value in changes.items():
            records = copy.deepcopy(self.records)
            records[0][key] = value
            self.write(records)
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.load_results(self.path, self.jobs, "fingerprint")

    def test_truncated_json_nan_and_wrong_wave_rejected(self):
        for text in ('{"job_id":', json.dumps(self.records[0]), "\n",
                     json.dumps(dict(self.records[0], rewards=[float("nan"), 0])) + "\n"):
            self.path.write_text(text)
            with self.subTest(text=text[:30]), self.assertRaises(ValueError):
                runner.load_results(self.path, self.jobs, "fingerprint")
        self.write(self.records)
        other_jobs = runner.build_jobs(manifest_fixture(), seed_start=5, seed_count=5)
        with self.assertRaises(ValueError):
            runner.load_results(self.path, other_jobs, "fingerprint")

    def test_empty_and_missing(self):
        self.assertEqual(runner.load_results(self.path, self.jobs, "fingerprint"), {})
        self.path.touch()
        self.assertEqual(runner.load_results(self.path, self.jobs, "fingerprint"), {})


class PayloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = manifest_fixture()
        self.manifest["files"] = {}
        for name in ["pool_upgrade_bundle_agent.py", "agents/candidate_iter27/main.py"] + [
                f"agents/{o['id']}/main.py" for o in self.manifest["opponents"]]:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# fixture\n")
            self.manifest["files"][name] = runner.sha256_file(path)

    def test_valid_payload_and_modification_rejected(self):
        self.assertEqual(runner.verify_payload(self.manifest, self.root),
                         str(self.root / "pool_upgrade_bundle_agent.py"))
        (self.root / "agents/candidate_iter27/main.py").write_text("# changed\n")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            runner.verify_payload(self.manifest, self.root)

    def test_unhashed_bundle_file_rejected(self):
        (self.root / "agents/candidate_iter27/injected.py").touch()
        with self.assertRaisesRegex(ValueError, "unhashed"):
            runner.verify_payload(self.manifest, self.root)

    def test_path_traversal_and_symlinks_rejected(self):
        for path in ("../outside", "/absolute", "agents/../outside", "agents//other"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                runner.payload_path(self.root, path)
        (self.root / "agents/candidate_iter27/link.py").symlink_to(self.root / "pool_upgrade_bundle_agent.py")
        with self.assertRaisesRegex(ValueError, "symlink"):
            runner.verify_payload(self.manifest, self.root)

    def test_atomic_summary(self):
        path = self.root / "summary.json"
        runner.atomic_json(path, {"complete": False})
        runner.atomic_json(path, {"complete": True})
        self.assertEqual(json.loads(path.read_text()), {"complete": True})
        self.assertFalse(list(self.root.glob("*.tmp")))


class MatchLifecycleTests(unittest.TestCase):
    def test_actual_python_socket_connection_denial(self):
        code = (
            "import socket,runner; runner.deny_network(); s=socket.socket(); "
            "s.connect(('127.0.0.1',9))"
        )
        result = subprocess.run([sys.executable, "-B", "-c", code],
                                cwd=Path(runner.__file__).parent, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("PermissionError: Frozen evaluation denies", result.stderr)

    def test_two_new_clients_prestart_wrapping_cleanup_and_metadata(self):
        events = []
        job = runner.build_jobs(manifest_fixture(), smoke=True)[1]

        class Client:
            def __init__(self, bundle, **kwargs):
                self.id = len([e for e in events if e[0] == "new"])
                events.append(("new", str(bundle), kwargs))
            def start(self):
                events.append(("start", self.id))
                return {"pid": self.id + 100, "network_guard": {"connection_denial": True}}
            def __call__(self, obs, config):
                events.append(("action", self.id, config))
                return {"farmer": ["PASS"]}
            def close(self):
                events.append(("close", self.id))

        class Env:
            info = {"seed": job["seed"]}
            steps = [[{"status": "DONE", "reward": 1}, {"status": "DONE", "reward": 2}]] * 720
            logs = []
            def run(self, agents):
                self_outer.assertEqual(len([e for e in events if e[0] == "start"]), 2)
                for agent in agents:
                    self_outer.assertEqual(agent.__code__.co_argcount, 2)
                    agent({}, {"test_configuration": True})

        self_outer = self
        options = dict(rpc_timeout=1, startup_timeout=1, configuration={})
        record = runner.run_match(job, "/fixture", "fingerprint", options, lambda *a, **k: Env(), Client)
        self.assertEqual(record["outcome"], "W")
        self.assertEqual(len([e for e in events if e[0] == "close"]), 2)
        self.assertEqual(record["agent_metadata"]["0"]["pid"], 100)
        self.assertEqual(record["agent_metadata"]["1"]["pid"], 101)
        self.assertFalse(record["network"]["child_socket_denial_verified_by_runner"])
        runner.run_match(job, "/fixture", "fingerprint", options, lambda *a, **k: Env(), Client)
        self.assertEqual(len([e for e in events if e[0] == "new"]), 4)

    def test_startup_exception_is_invalid_and_closes_all_created_clients(self):
        closed = []
        class Broken:
            def __init__(self, *a, **k):
                pass
            def start(self):
                raise RuntimeError("import failed")
            def close(self):
                closed.append(True)
        job = runner.build_jobs(manifest_fixture(), smoke=True)[0]
        record = runner.run_match(job, "/fixture", "fingerprint",
                                  dict(rpc_timeout=1, startup_timeout=1, configuration={}),
                                  None, Broken)
        self.assertEqual(record["outcome"], "INVALID")
        self.assertEqual(closed, [True])
        runner.validate_record(record, {job["job_id"]: job}, "fingerprint")


@unittest.skipUnless(os.environ.get("FROZEN_EVAL_INTEGRATION") == "1", "opt-in real engine/RPC tests")
class RealIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.assertEqual(importlib.metadata.version("kaggle-environments"), "1.32.7")
        self.temp = tempfile.TemporaryDirectory(prefix="frozen-eval-test-")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "payload"
        self.root.mkdir()
        self.output = self.base / "output"
        self.manifest_path = self.base / "manifest.json"
        self.manifest = manifest_fixture()
        helper = Path(__file__).resolve().parents[2] / "shinka/evolution/pool_upgrade_bundle_agent.py"
        shutil.copyfile(helper, self.root / "pool_upgrade_bundle_agent.py")
        body = "def agent(obs, config):\n    assert config['episodeSteps'] == 720\n    return {'farmer': ['PASS'], 'hands': [], 'market': []}\n"
        for agent_id in [self.manifest["candidate_id"], *(o["id"] for o in self.manifest["opponents"])]:
            path = self.root / "agents" / agent_id / "main.py"
            path.parent.mkdir(parents=True)
            path.write_text(body)
        self.freeze()

    def freeze(self):
        self.manifest["files"] = {p.relative_to(self.root).as_posix(): runner.sha256_file(p)
                                  for p in self.root.rglob("*") if p.is_file()}
        self.manifest_path.write_text(json.dumps(self.manifest))

    def invoke(self, *extra):
        command = [sys.executable, "-B", str(Path(runner.__file__).resolve()),
                   "--manifest", str(self.manifest_path), "--root", str(self.root),
                   "--output", str(self.output), "--shards", "100", "--shard-index", "0",
                   "--seed-count", "100", "--opponents", "opponent_0", "--workers", "1",
                   "--summary-interval", "0.1", *extra]
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=180)
        self.assertIn(result.returncode, (0, 2), result.stdout[-15000:])
        return result, json.loads((self.output / "summary.json").read_text())

    def test_real_full_episodes_fresh_processes_and_resume(self):
        result, summary = self.invoke()
        self.assertEqual(result.returncode, 0, result.stdout[-15000:])
        self.assertTrue(summary["benchmark_pass"], summary)
        self.assertEqual(summary["overall"]["D"], 2)
        records = [json.loads(line) for line in (self.output / "results.jsonl").read_text().splitlines()]
        self.assertEqual([r["states"] for r in records], [720, 720])
        pids = [r["agent_metadata"][str(seat)]["pid"] for r in records for seat in (0, 1)]
        self.assertEqual(len(set(pids)), 4)
        self.assertEqual(len({r["worker_pid"] for r in records}), 1)
        for record in records:
            self.assertEqual(record["statuses"], ["DONE", "DONE"])
            self.assertEqual(record["resolved_seed"], record["seed"])
            self.assertFalse(record["network"]["child_socket_denial_verified_by_runner"])
        before = (self.output / "results.jsonl").read_bytes()
        _, resumed = self.invoke()
        self.assertEqual(resumed["new_results_this_invocation"], 0)
        self.assertEqual((self.output / "results.jsonl").read_bytes(), before)

    def test_deadline_leaves_unfinished_jobs_pending(self):
        _, summary = self.invoke("--deadline-seconds", "0.01")
        self.assertEqual(summary["status"], "deadline")
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["overall"]["attempted"], 0)
        self.assertEqual(summary["overall"]["remaining"], 2)

    def test_real_rpc_failure_is_invalid_never_win(self):
        (self.root / "agents/opponent_0/main.py").write_text(
            "def agent(obs, config):\n    raise RuntimeError('intentional fixture failure')\n")
        self.freeze()
        result, summary = self.invoke()
        self.assertEqual(result.returncode, 2)
        self.assertTrue(summary["complete"])
        self.assertFalse(summary["benchmark_pass"])
        self.assertEqual(summary["overall"]["invalid"], 2)
        self.assertEqual(summary["overall"]["W"], 0)
        self.assertEqual(summary["overall"]["D"], 0)

    def test_watchdog_kills_hung_agents_and_replaces_worker(self):
        (self.root / "agents/opponent_0/main.py").write_text(
            "import time\ndef agent(obs, config):\n    time.sleep(60)\n    return {}\n")
        self.freeze()
        result, summary = self.invoke("--match-timeout", "1.0", "--rpc-timeout", "60")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(summary["overall"]["invalid"], 2)
        records = [json.loads(line) for line in (self.output / "results.jsonl").read_text().splitlines()]
        self.assertTrue(all(r["errors"][0]["type"] == "MatchWatchdogTimeout" for r in records))
        worker_pids = {r["worker_pid"] for r in records}
        self.assertEqual(len(worker_pids), 2)
        # Verify no live worker OR agent descendant remains in either killed group.
        for path in Path("/proc").glob("[0-9]*/stat"):
            try:
                fields = path.read_text().rsplit(")", 1)[1].split()
                state, process_group = fields[0], int(fields[2])
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
            self.assertFalse(process_group in worker_pids and state != "Z",
                             f"live descendant left in killed worker group: {path}")


if __name__ == "__main__":
    unittest.main()
