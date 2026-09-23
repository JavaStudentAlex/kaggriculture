#!/usr/bin/env python3
"""Frozen, CPU-only Kaggriculture evaluation; never evolves or uploads anything.

Manifest paths are relative to --root. Required: evaluation_id, candidate_id,
opponents=[{id,label,kind}] (15 entries), seeds={"0":[...],"1":[...]}, and
files={relative_path:sha256}. Every file in each referenced agents/<id>/ bundle
(including main.py) and bundle_agent_path must be listed. bundle_agent_path
 defaults to pool_upgrade_bundle_agent.py at the payload root. The helper may
be an instrumented frozen copy. It is never edited by this runner.

Seeds are reused across opponents for paired comparisons, NOT across seats or
indices. Known legacy evaluation seeds and optional excluded_seeds are rejected;
other historical freshness is the manifest producer's responsibility. Each wave
selects [seed_start, seed_start+seed_count), then splits that interval into equal
contiguous shard slices. --smoke is shorthand for --seed-count SHARDS (one seed
per seat/opponent/shard). A new wave can use a separate output directory. An
existing results file must belong to the exact selected job set and fingerprint.
Invalid attempts are retained on resume, not silently retried or counted as wins.

Linux workers own separate process groups. Each loads the engine once, creates
TWO new BundleAgents per match, prestarts both, and closes both. The controller
kills the entire worker group on a wall-clock match timeout (including startup),
then replaces the worker. Deadline/signal interruption leaves unfinished jobs
pending rather than fabricating outcomes. Child RPCs have independent bounds.

The runner and workers deny Python socket connection/send audit events. That
hook does NOT survive exec into BundleAgent children. Child network evidence is
recorded verbatim from helper metadata; absent instrumented helper evidence,
this runner explicitly does NOT claim child network isolation. This is a trusted
bundle evaluation tool, not a security sandbox.
"""
from __future__ import annotations

# Must precede engine/numpy/helper imports, including in spawned workers.
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
             "OMP_THREAD_LIMIT", "NUMBA_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ.update(CUDA_VISIBLE_DEVICES="", KAGG_ORACLE_BACKEND="numpy",
                  KAGG_ORACLE_DEVICE="cpu", PYTHONDONTWRITEBYTECODE="1")

import argparse
import collections
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import multiprocessing
import re
import signal
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True
ENGINE_VERSION = "1.32.7"
STATES = 720
SCHEMA_VERSION = 1
LEGACY_SEEDS = frozenset([101 * i for i in range(1, 21)] +
                         [70001 + 101 * i for i in range(1, 21)])
IDENTITY_FIELDS = ("job_id", "evaluation_id", "candidate_id", "opponent_id",
                   "opponent_label", "opponent_kind", "seat", "seed_index", "seed")
_NETWORK_HOOK = False


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def safe_id(value):
    return (isinstance(value, str) and value not in (".", "..") and
            re.fullmatch(r"[A-Za-z0-9_.-]+", value) is not None)


def validate_manifest(manifest):
    require(isinstance(manifest, dict), "manifest must be an object")
    require(isinstance(manifest.get("evaluation_id"), str) and
            bool(manifest["evaluation_id"]), "missing evaluation_id")
    require(safe_id(manifest.get("candidate_id")), "unsafe candidate_id")
    opponents = manifest.get("opponents")
    require(isinstance(opponents, list) and len(opponents) >= 1,
            "manifest must contain at least 1 opponent entry")
    ids = []
    for opponent in opponents:
        require(isinstance(opponent, dict) and safe_id(opponent.get("id")),
                "unsafe opponent id")
        for field in ("label", "kind"):
            require(isinstance(opponent.get(field), str) and bool(opponent[field]),
                    f"missing opponent {field}")
        ids.append(opponent["id"])
    require(len(set(ids)) == len(ids), "duplicate opponent IDs")
    blocks = manifest.get("seeds", {})
    require(isinstance(blocks, dict) and set(blocks) == {"0", "1"},
            "seeds must have exactly the keys '0' and '1'")
    excluded = manifest.get("excluded_seeds", [])
    require(isinstance(excluded, list) and all(type(s) is int for s in excluded),
            "excluded_seeds must be an integer list")
    forbidden = LEGACY_SEEDS | set(excluded)
    for seat in ("0", "1"):
        seeds = blocks[seat]
        require(isinstance(seeds, list) and len(seeds) >= 1,
                "each seat requires at least 1 frozen seed")
        require(all(type(s) is int and 0 <= s < 2**63 for s in seeds),
                "seeds must be nonnegative 63-bit integers, not bools")
        require(len(set(seeds)) == len(seeds), "duplicate seeds within seat")
        require(not forbidden.intersection(seeds), "seeds overlap excluded/legacy seeds")
    require(not set(blocks["0"]).intersection(blocks["1"]), "seat seeds overlap")
    require(isinstance(manifest.get("files"), dict) and manifest["files"],
            "files must be a nonempty hash map")
    configuration = manifest.get("configuration", {})
    require(isinstance(configuration, dict), "configuration must be an object")
    require(not {"seed", "randomSeed", "episodeSteps"}.intersection(configuration),
            "configuration cannot override seed/episodeSteps")
    canonical(manifest)  # Reject NaN/Infinity, including in optional metadata.


def payload_path(root, relative):
    require(isinstance(relative, str) and bool(relative), "empty payload path")
    p = Path(relative)
    require(not p.is_absolute() and ".." not in p.parts and str(p) == relative,
            f"noncanonical/unsafe payload path: {relative}")
    target = root / p
    require(target.resolve().is_relative_to(root), f"path escapes payload: {relative}")
    require(not any((root / Path(*p.parts[:i])).is_symlink()
                    for i in range(1, len(p.parts) + 1)), f"symlink in payload: {relative}")
    return target


def verify_payload(manifest, root):
    """Hash all declared files and reject unhashed files in agent bundles."""
    root = Path(root).resolve()
    for relative, expected in manifest["files"].items():
        require(isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected),
                f"invalid sha256: {relative}")
        path = payload_path(root, relative)
        require(path.is_file(), f"missing payload file: {relative}")
        require(sha256_file(path) == expected, f"hash mismatch: {relative}")
    helper_relative = manifest.get("bundle_agent_path", "pool_upgrade_bundle_agent.py")
    require(helper_relative in manifest["files"], "BundleAgent helper must be hashed in files")
    for agent_id in {manifest["candidate_id"], *(o["id"] for o in manifest["opponents"])}:
        bundle = payload_path(root, f"agents/{agent_id}")
        require(f"agents/{agent_id}/main.py" in manifest["files"],
                f"missing hashed main.py for {agent_id}")
        for path in bundle.rglob("*"):
            relative = path.relative_to(root).as_posix()
            require(not path.is_symlink(), f"symlink in bundle: {relative}")
            if path.is_file():
                require(relative in manifest["files"], f"unhashed bundle file: {relative}")
    return str(payload_path(root, helper_relative))


def build_jobs(manifest, shard_index=0, shards=5, seed_start=0, seed_count=100,
               opponents=None, smoke=False):
    validate_manifest(manifest)
    require(type(shards) is int and shards > 0 and 0 <= shard_index < shards,
            "invalid shard coordinates")
    if smoke:
        seed_count = shards
    require(seed_start >= 0 and seed_count > 0 and seed_count % shards == 0,
            "seed-count must be positive and divisible by shards; seed-start must be >=0")
    require(all(seed_start + seed_count <= len(manifest["seeds"][str(s)]) for s in (0, 1)),
            "selected seed interval exceeds manifest")
    selected = None if opponents is None else set(opponents)
    all_ids = {o["id"] for o in manifest["opponents"]}
    require(selected is None or (selected and selected <= all_ids), "unknown/empty opponents")
    width = seed_count // shards
    first = seed_start + shard_index * width
    jobs = []
    # Interleave opponents and seats: a deadline should not hide a late opponent.
    for index in range(first, first + width):
        for opponent in manifest["opponents"]:
            if selected is not None and opponent["id"] not in selected:
                continue
            for seat in (0, 1):
                job: dict[str, Any] = dict(evaluation_id=manifest["evaluation_id"],
                           candidate_id=manifest["candidate_id"], opponent_id=opponent["id"],
                           opponent_label=opponent["label"], opponent_kind=opponent["kind"],
                           seat=seat, seed_index=index, seed=manifest["seeds"][str(seat)][index])
                job["job_id"] = digest(job)
                jobs.append(job)
    require(len({j["job_id"] for j in jobs}) == len(jobs), "duplicate job IDs")
    return jobs


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def classify(statuses, states, rewards, seat, errors):
    """Strict candidate-relative W/L/D; failures never become wins or draws."""
    reasons = []
    if errors:
        reasons.append("recorded_errors")
    if statuses != ["DONE", "DONE"]:
        reasons.append("non_DONE_status")
    if type(states) is not int or states != STATES:
        reasons.append("incomplete_episode")
    if not isinstance(rewards, list) or len(rewards) != 2 or not all(map(finite_number, rewards)):
        reasons.append("nonfinite_or_missing_reward")
    if reasons:
        return "INVALID", reasons
    a, b = rewards[seat], rewards[1 - seat]
    return ("W" if a > b else "L" if a < b else "D"), []


def base_record(job, fingerprint) -> dict[str, Any]:
    return dict(job, schema_version=SCHEMA_VERSION, source_fingerprint=fingerprint,
                engine_version=ENGINE_VERSION, statuses=[], states=0, length=0,
                rewards=[None, None], resolved_seed=None, errors=[],
                timing={"startup_seconds": 0.0, "play_seconds": 0.0, "total_seconds": 0.0},
                agent_metadata={}, worker_pid=os.getpid(),
                started_at=datetime.now(timezone.utc).isoformat(),
                network={"runner_python_socket_denial": True,
                         "child_socket_denial_verified_by_runner": False,
                         "child_evidence": {}})


def finish_record(record):
    outcome, reasons = classify(record["statuses"], record["states"], record["rewards"],
                                record["seat"], record["errors"])
    if type(record["resolved_seed"]) is not int or record["resolved_seed"] != record["seed"]:
        outcome = "INVALID"
        reasons.append("resolved_seed_mismatch")
    record.update(outcome=outcome, valid=outcome != "INVALID", invalid_reasons=reasons,
                  length=record["states"])
    return record


def failure_record(job, fingerprint, error_type, message, seconds=0.0):
    record = base_record(job, fingerprint)
    record["errors"].append(dict(phase="controller", type=error_type, message=message))
    record["timing"]["total_seconds"] = seconds
    return finish_record(record)


def validate_record(record, expected_jobs, fingerprint):
    require(isinstance(record, dict), "result row must be an object")
    job = expected_jobs.get(record.get("job_id"))
    require(job is not None, "unknown job ID (different wave, shard, or manifest)")
    for field in IDENTITY_FIELDS:
        require(type(record.get(field)) is type(job[field]) and record[field] == job[field],
                f"result identity mismatch: {field}")
    require(record.get("schema_version") == SCHEMA_VERSION, "result schema mismatch")
    require(record.get("source_fingerprint") == fingerprint, "source fingerprint mismatch")
    require(record.get("engine_version") == ENGINE_VERSION, "engine version mismatch")
    require(type(record.get("valid")) is bool, "missing valid boolean")
    require(isinstance(record.get("errors"), list), "missing errors list")
    require(isinstance(record.get("statuses"), list) and len(record["statuses"]) in (0, 2),
            "malformed statuses")
    require(type(record.get("states")) is int and record["states"] >= 0 and
            record.get("length") == record["states"], "malformed episode length")
    require(isinstance(record.get("rewards"), list) and len(record["rewards"]) == 2,
            "malformed rewards")
    require(all(r is None or finite_number(r) for r in record["rewards"]), "nonfinite JSON reward")
    require("resolved_seed" in record and isinstance(record.get("agent_metadata"), dict) and
            isinstance(record.get("network"), dict), "missing seed/agent/network evidence")
    timing = record.get("timing", {})
    require(all(finite_number(timing.get(k)) and timing[k] >= 0 for k in
                ("startup_seconds", "play_seconds", "total_seconds")), "malformed timing")
    check = finish_record(dict(record))
    for field in ("outcome", "valid", "invalid_reasons"):
        require(record.get(field) == check[field], f"inconsistent result accounting: {field}")
    canonical(record)
    return record


def load_results(path, jobs, fingerprint):
    expected = {job["job_id"]: job for job in jobs}
    records = {}
    if not Path(path).exists():
        return records
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            # A truncated final write is not silently repaired or discarded.
            require(line.endswith("\n") and bool(line.strip()), f"incomplete/blank row {line_number}")
            try:
                record = validate_record(json.loads(line), expected, fingerprint)
                require(record["job_id"] not in records, "duplicate result job ID")
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"invalid resume row {line_number}: {exc}") from exc
            records[record["job_id"]] = record
    return records


def hardware_info():
    memory = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            if key in ("MemTotal", "MemAvailable"):
                memory[key + "_bytes"] = int(value.split()[0]) * 1024
    except (OSError, ValueError):
        pass
    return dict(cpu_count=os.cpu_count(), affinity=sorted(os.sched_getaffinity(0)),
                memory=memory, platform=sys.platform, python=sys.version)


def make_summary(manifest, jobs, records, fingerprint, provenance, *, status,
                 elapsed, new_results, workers, interrupted_jobs=(), error=None):
    def bucket():
        return dict(planned=0, W=0, L=0, D=0, invalid=0, attempted=0, valid=0, remaining=0)
    overall = bucket()
    by_opponent = {}
    by_seat = {str(s): bucket() for s in (0, 1)}
    for job in jobs:
        opponent = by_opponent.setdefault(job["opponent_id"],
                    dict(label=job["opponent_label"], kind=job["opponent_kind"],
                         total=bucket(), seats={str(s): bucket() for s in (0, 1)}))
        groups = [overall, by_seat[str(job["seat"])], opponent["total"],
                  opponent["seats"][str(job["seat"])]]
        row = records.get(job["job_id"])
        for group in groups:
            group["planned"] += 1
            if row:
                group["attempted"] += 1
                group["valid"] += int(row["valid"])
                group["invalid" if row["outcome"] == "INVALID" else row["outcome"]] += 1
            else:
                group["remaining"] += 1
    for group in [overall, *by_seat.values(), *(g for o in by_opponent.values()
                                                for g in [o["total"], *o["seats"].values()])]:
        group["win_rate_valid_only"] = group["W"] / group["valid"] if group["valid"] else None
        group["wins_over_planned"] = group["W"] / group["planned"] if group["planned"] else None
    return dict(schema_version=SCHEMA_VERSION, evaluation_id=manifest["evaluation_id"],
                candidate_id=manifest["candidate_id"], source_fingerprint=fingerprint,
                provenance=provenance, engine_version=ENGINE_VERSION, hardware=hardware_info(),
                status=status, complete=overall["remaining"] == 0,
                all_attempts_valid=overall["invalid"] == 0,
                benchmark_pass=overall["remaining"] == 0 and overall["invalid"] == 0,
                selected_jobs=len(jobs), full_manifest_jobs=15 * sum(map(len, manifest["seeds"].values())),
                overall=overall, per_opponent=by_opponent, per_seat=by_seat,
                workers=workers, elapsed_seconds=elapsed, new_results_this_invocation=new_results,
                estimated_remaining_seconds=(elapsed / new_results * overall["remaining"]
                                             if new_results else None),
                interrupted_job_ids=list(interrupted_jobs), error=error,
                updated_at=datetime.now(timezone.utc).isoformat(),
                network_policy="Runner/workers deny Python socket connect/send audit events; "
                               "child isolation unverified by runner; inspect per-record helper evidence.",
                seed_freshness="Disjoint manifest blocks, legacy seeds excluded; other historical "
                               "freshness is supplied by manifest producer.")


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            handle.write(canonical(value) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def deny_network():
    global _NETWORK_HOOK
    if not _NETWORK_HOOK:
        def audit(event, args):
            if event in ("socket.connect", "socket.connect_ex", "socket.sendto", "socket.sendmsg"):
                raise PermissionError("Frozen evaluation denies runner/worker socket connections")
        sys.addaudithook(audit)
        _NETWORK_HOOK = True


def source_provenance(manifest, helper, options):
    version = importlib.metadata.version("kaggle-environments")
    require(version == ENGINE_VERSION, f"need kaggle-environments=={ENGINE_VERSION}; found {version}")
    distribution = importlib.metadata.distribution("kaggle-environments")
    engine_hashes = {}
    for relative in ("kaggle_environments/core.py", "kaggle_environments/agent.py",
                     "kaggle_environments/envs/kaggriculture/kaggriculture.py",
                     "kaggle_environments/envs/kaggriculture/kaggriculture.json"):
        engine_hashes[relative] = sha256_file(distribution.locate_file(relative))
    result = dict(manifest_sha256=digest(manifest), runner_sha256=sha256_file(__file__),
                  helper_sha256=sha256_file(helper), engine_version=version,
                  engine_hashes=engine_hashes,
                  python_version=".".join(map(str, sys.version_info[:3])),
                  timeouts={k: options[k] for k in ("rpc_timeout", "startup_timeout", "match_timeout")})
    return result, digest(result)


import gzip

def save_trace(env, job, trace_dir):
    """Dump full env.steps to gzipped JSON for post-hoc analysis."""
    if env is None or not env.steps:
        return None
    trace_dir = Path(trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{job['opponent_id']}_seat{job['seat']}_seed{job['seed']}.json.gz"
    path = trace_dir / fname
    # Build a compact trace: per-step observations + actions + rewards
    steps_data = []
    for step_pair in env.steps:
        step_entry = []
        for seat_state in step_pair:
            entry = {}
            if 'observation' in seat_state:
                entry['observation'] = seat_state['observation']
            if 'action' in seat_state:
                entry['action'] = seat_state['action']
            for k in ('reward', 'status', 'info'):
                if k in seat_state:
                    entry[k] = seat_state[k]
            step_entry.append(entry)
        steps_data.append(step_entry)
    trace = {
        'job_id': job['job_id'],
        'evaluation_id': job['evaluation_id'],
        'candidate_id': job['candidate_id'],
        'opponent_id': job['opponent_id'],
        'seat': job['seat'],
        'seed': job['seed'],
        'steps': steps_data,
    }
    with gzip.open(path, 'wt', encoding='utf-8', compresslevel=6) as fh:
        json.dump(trace, fh, separators=(',', ':'), allow_nan=False)
    return str(path)


def run_match(job, root, fingerprint, options, make, bundle_class):
    start = time.monotonic()
    record = base_record(job, fingerprint)
    clients = []
    env = None
    play_start = None
    try:
        ids = [job["opponent_id"], job["opponent_id"]]
        ids[job["seat"]] = job["candidate_id"]
        for seat, agent_id in enumerate(ids):
            client = bundle_class(Path(root) / "agents" / agent_id, entrypoint="main.py",
                                  timeout=options["rpc_timeout"],
                                  startup_timeout=options["startup_timeout"])
            clients.append(client)
            try:
                metadata = client.start()  # Imports/model warmup outside timed engine turns.
                record["agent_metadata"][str(seat)] = metadata
                record["network"]["child_evidence"][str(seat)] = metadata.get("network_guard")
            except Exception as exc:
                record["errors"].append(dict(seat=seat, phase="startup", type=type(exc).__name__,
                                              message=str(exc), traceback=traceback.format_exc()))
                raise
        record["timing"]["startup_seconds"] = time.monotonic() - start

        def wrap(client, seat):
            # Kaggle inspects __code__.co_argcount: passing the callable object loses config.
            def f(obs, config):
                try:
                    return client(obs, config)
                except Exception as exc:
                    record["errors"].append(dict(seat=seat, phase="action", type=type(exc).__name__,
                                                  message=str(exc), traceback=traceback.format_exc()))
                    raise
            return f

        configuration = dict(options["configuration"], episodeSteps=STATES, seed=job["seed"])
        env = make("kaggriculture", configuration=configuration, debug=False)
        record["resolved_seed"] = env.info.get("seed")
        play_start = time.monotonic()
        env.run([wrap(client, seat) for seat, client in enumerate(clients)])
        record["resolved_seed"] = env.info.get("seed")
    except Exception as exc:
        record["errors"].append(dict(phase="match", type=type(exc).__name__, message=str(exc),
                                      traceback=traceback.format_exc()))
    finally:
        before_close = time.monotonic()
        if play_start is not None:
            record["timing"]["play_seconds"] = before_close - play_start
        else:
            record["timing"]["startup_seconds"] = before_close - start
        if env is not None and env.steps:
            record["states"] = len(env.steps)
            final = env.steps[-1]
            record["statuses"] = [state.get("status") for state in final]
            rewards = [state.get("reward") for state in final]
            record["rewards"] = [r if finite_number(r) else None for r in rewards]
            # Keep intermediate errors even if an interpreter overwrites final status.
            for step_index, state_pair in enumerate(env.steps):
                for seat, state in enumerate(state_pair):
                    if state.get("status") not in ("ACTIVE", "INACTIVE", "DONE"):
                        record["errors"].append(dict(phase="engine_status", seat=seat,
                                                      step=step_index, status=state.get("status")))
            # Engine exceptions appear in stderr, not an `error` key in 1.32.7.
            for step_index, logs in enumerate(getattr(env, "logs", [])):
                if isinstance(logs, list):
                    for seat, log in enumerate(logs):
                        if isinstance(log, dict):
                            message = log.get("error") or log.get("stderr")
                            if message:
                                record["errors"].append(dict(phase="engine", seat=seat,
                                                              step=step_index, message=str(message)))
        for client in clients:
            try:
                client.close()
            except Exception as exc:
                record["errors"].append(dict(phase="close", type=type(exc).__name__, message=str(exc)))
        record["timing"]["total_seconds"] = time.monotonic() - start
        if options.get("save_traces") and env is not None and env.steps:
            trace_dir = options.get("trace_dir", ".")
            try:
                trace_path = save_trace(env, job, trace_dir)
                record["trace_path"] = trace_path
            except Exception as exc:
                record["errors"].append(dict(phase="trace_save", type=type(exc).__name__,
                                              message=str(exc)))
    return finish_record(record)


def worker_main(connection, root, helper, fingerprint, options):
    os.setsid()  # Descendant BundleAgent subprocesses inherit this killable process group.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    deny_network()
    try:
        require(importlib.metadata.version("kaggle-environments") == ENGINE_VERSION,
                "worker engine version mismatch")
        from kaggle_environments import make
        spec = importlib.util.spec_from_file_location("frozen_bundle_agent", helper)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        connection.send(("ready", os.getpid()))
        while True:
            job = connection.recv()
            if job is None:
                return
            record = run_match(job, root, fingerprint, options, make, module.BundleAgent)
            connection.send(("result", record))
    except (EOFError, BrokenPipeError):
        pass
    except BaseException:
        try:
            connection.send(("fatal", traceback.format_exc()))
        except (EOFError, BrokenPipeError):
            pass
    finally:
        connection.close()


def stop_worker(worker):
    process = worker["process"]
    if process.pid:
        # Only our setsid group, never a caller's process group.
        try:
            if os.getpgid(process.pid) == process.pid:
                os.killpg(process.pid, signal.SIGKILL)
            elif process.is_alive():
                process.kill()
        except ProcessLookupError:
            # Worker may have exited leaving descendants in its old process group.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    process.join(timeout=5)
    require(not process.is_alive(), f"unable to stop worker {process.pid}")
    worker["connection"].close()


def execute(manifest, jobs, root, helper, output, options, provenance, fingerprint):
    """Persistent non-daemon workers, bounded dispatch, no executor shutdown hangs."""
    import fcntl
    require(sys.platform == "linux", "watchdog requires Linux process groups")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    lock = open(output / ".runner.lock", "a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise ValueError("output directory is already owned by another runner")
    workers = []
    handlers = {}
    records = {}
    stop_requested = []
    interrupted = []
    status = "running"
    error = None
    result = None
    start = time.monotonic()
    new_results = 0
    results_path = output / "results.jsonl"
    expected = {j["job_id"]: j for j in jobs}

    def summary():
        value = make_summary(manifest, jobs, records, fingerprint, provenance, status=status,
                             elapsed=time.monotonic() - start, new_results=new_results,
                             workers=options["workers"], interrupted_jobs=interrupted, error=error)
        value["selection"] = options["selection"]
        atomic_json(output / "summary.json", value)
        return value

    def spawn():
        parent, child = multiprocessing.get_context("spawn").Pipe()
        process = multiprocessing.get_context("spawn").Process(
            target=worker_main, args=(child, str(root), helper, fingerprint, options), daemon=False)
        process.start()
        child.close()
        worker = dict(process=process, connection=parent, ready=False, job=None,
                      since=time.monotonic())
        workers.append(worker)
        return worker

    try:
        records = load_results(results_path, jobs, fingerprint)
        queue = collections.deque(j for j in jobs if j["job_id"] not in records)
        for sig in (signal.SIGINT, signal.SIGTERM):
            handlers[sig] = signal.signal(sig, lambda signum, frame: stop_requested.append(signum))
        summary()
        for _ in range(min(options["workers"], len(queue))):
            spawn()
        last_summary = time.monotonic()
        with open(results_path, "a", encoding="utf-8") as handle:
            def save(record):
                nonlocal new_results
                validate_record(record, expected, fingerprint)
                require(record["job_id"] not in records, "duplicate live result")
                handle.write(canonical(record) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                records[record["job_id"]] = record
                new_results += 1

            while queue or any(w["job"] for w in workers):
                now = time.monotonic()
                # Consume already delivered results even when the deadline has arrived.
                for worker in list(workers):
                    conn = worker["connection"]
                    try:
                        if conn.poll():
                            kind, payload = conn.recv()
                            if kind == "ready":
                                worker["ready"] = True
                            elif kind == "result":
                                require(worker["job"] is not None and
                                        payload.get("job_id") == worker["job"]["job_id"],
                                        "worker returned mismatched job")
                                save(payload)
                                worker["job"] = None
                            else:
                                raise RuntimeError(f"worker failure: {payload}")
                    except (EOFError, ConnectionResetError, BrokenPipeError):
                        if worker["job"] is not None:
                            record = failure_record(worker["job"], fingerprint, "WorkerExit",
                                                    "Worker exited without a result", now - worker["since"])
                            record["worker_pid"] = worker["process"].pid
                            save(record)
                        elif not worker["ready"]:
                            raise RuntimeError("worker exited during engine/helper initialization")
                        stop_worker(worker)
                        workers.remove(worker)
                        continue
                    if worker["job"] and now - worker["since"] >= options["match_timeout"]:
                        job = worker["job"]
                        stop_worker(worker)
                        workers.remove(worker)
                        record = failure_record(job, fingerprint, "MatchWatchdogTimeout",
                                                "Worker group and both agents killed at wall-clock limit",
                                                now - worker["since"])
                        record["worker_pid"] = worker["process"].pid
                        save(record)
                    elif not worker["ready"] and now - worker["since"] >= options["worker_startup_timeout"]:
                        raise TimeoutError("worker engine/helper initialization timed out")
                if stop_requested or now - start >= options["deadline_seconds"]:
                    status = "interrupted" if stop_requested else "deadline"
                    interrupted = [w["job"]["job_id"] for w in workers if w["job"]]
                    break  # In-flight jobs are pending, not fake invalids/completions.
                while queue and len(workers) < min(options["workers"], len(queue) +
                                                  sum(w["job"] is not None for w in workers)):
                    spawn()
                for worker in workers:
                    if worker["ready"] and worker["job"] is None and queue:
                        worker["job"] = queue.popleft()
                        worker["since"] = time.monotonic()
                        worker["connection"].send(worker["job"])
                if now - last_summary >= options["summary_interval"]:
                    summary()
                    last_summary = now
                time.sleep(0.05)
        if len(records) == len(jobs):
            status = "complete"
    except BaseException as exc:
        status = "error"
        error = f"{type(exc).__name__}: {exc}"
        interrupted = [w["job"]["job_id"] for w in workers if w["job"]]
        raise
    finally:
        for worker in workers:
            stop_worker(worker)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        # Do not overwrite a prior summary when its JSONL failed validation.
        if records or not results_path.exists() or results_path.stat().st_size == 0:
            result = summary()
        lock.close()
    require(result is not None, "missing final summary")
    print(canonical({"status": status, "overall": result["overall"],
                     "summary": str(output / "summary.json")}), flush=True)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shards", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--opponents", nargs="+", help="opponent IDs (space separated)")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--save-traces", action="store_true",
                        help="Dump full env.steps per match as gzipped JSON in output/traces/")
    parser.add_argument("--rpc-timeout", type=float, default=60.0)
    parser.add_argument("--startup-timeout", type=float, default=120.0)
    parser.add_argument("--match-timeout", type=float, default=900.0)
    parser.add_argument("--worker-startup-timeout", type=float, default=240.0)
    parser.add_argument("--deadline-seconds", type=float, default=36000.0)
    parser.add_argument("--summary-interval", type=float, default=30.0)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    require(args.workers > 0, "workers must be positive")
    for key in ("rpc_timeout", "startup_timeout", "match_timeout", "worker_startup_timeout",
                "deadline_seconds", "summary_interval"):
        require(finite_number(getattr(args, key)) and getattr(args, key) > 0,
                f"{key} must be finite and positive")
    manifest = json.loads(args.manifest.read_text())
    jobs = build_jobs(manifest, args.shard_index, args.shards, args.seed_start,
                      args.seed_count, args.opponents, args.smoke)
    root = args.root.resolve()
    require(not args.output.resolve().is_relative_to(root), "output must be outside frozen payload")
    helper = verify_payload(manifest, root)
    options = {k: v for k, v in vars(args).items() if k not in ("manifest", "root", "output")}
    options["configuration"] = manifest.get("configuration", {})
    options["selection"] = {k: options[k] for k in
                             ("shard_index", "shards", "seed_start", "seed_count", "opponents", "smoke")}
    if args.smoke:
        options["selection"]["seed_count"] = args.shards
    if args.save_traces:
        trace_dir = args.output.resolve() / "traces"
        trace_dir.mkdir(parents=True, exist_ok=True)
        options["save_traces"] = True
        options["trace_dir"] = str(trace_dir)
    provenance, fingerprint = source_provenance(manifest, helper, options)
    deny_network()
    result = execute(manifest, jobs, root, helper, args.output, options, provenance, fingerprint)
    # Partial deadline exits are intentional and resumable; invalid/fatal runs are not success.
    return 2 if result["overall"]["invalid"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
