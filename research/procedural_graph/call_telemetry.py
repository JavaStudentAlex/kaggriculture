"""Persisted, secret-free reachability telemetry for the evolution pipeline.

The host-side LLM callers write here via ``record_call``. Kaggle workers cannot
write to the host filesystem, so their aggregate Jev / tunneled-LLM statistics
are merged by the orchestrator after each shard result is downloaded.
"""
from __future__ import annotations

from datetime import UTC, datetime
import json
import os
from pathlib import Path
import tempfile
from typing import Any

ENV_PATH = "KAGG_CALL_TELEMETRY_PATH"


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _path() -> Path | None:
    raw = os.environ.get(ENV_PATH)
    return Path(raw) if raw else None


def _read(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _write(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".telemetry-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        Path(name).replace(path)
    finally:
        Path(name).unlink(missing_ok=True)


def reset(path: Path | str, run_id: str) -> None:
    """Start an isolated telemetry ledger for a new evolution run."""
    dest = Path(path)
    _write(dest, {
        "run_id": run_id,
        "started_utc": _now(),
        "updated_utc": _now(),
        "sources": {},
    })


def record_call(source: str, outcome: str, model: str | None = None,
                error: Exception | str | None = None) -> None:
    """Count one attempted remote call without recording secrets or prompts.

    ``outcome`` must be ``success`` or ``failure``.  Instrumentation is
    deliberately best-effort: telemetry failure must never take down evolution.
    """
    path = _path()
    if path is None or outcome not in {"success", "failure"}:
        return
    try:
        data = _read(path)
        sources = data.setdefault("sources", {})
        entry = sources.setdefault(source, {"attempted": 0, "succeeded": 0, "failed": 0, "models": {}})
        entry["attempted"] += 1
        entry["succeeded" if outcome == "success" else "failed"] += 1
        entry[f"last_{outcome}_utc"] = _now()
        if outcome == "failure" and error is not None:
            entry["last_error"] = str(error)[:300]

        if model:
            models = entry.setdefault("models", {})
            model_entry = models.setdefault(model, {"attempted": 0, "succeeded": 0, "failed": 0})
            model_entry["attempted"] += 1
            model_entry["succeeded" if outcome == "success" else "failed"] += 1
        data["updated_utc"] = _now()
        _write(path, data)
    except Exception:
        pass


def merge_worker_metrics(worker: dict[str, Any] | None) -> None:
    """Merge one completed Kaggle shard's aggregate remote-call counters.

    Expected keys are ``jev`` and ``inner_llm``; each has attempted/succeeded/
    failed. ``last_error`` is only a short exception class/message, never a key,
    URL, prompt, or response body.
    """
    path = _path()
    if path is None or not worker:
        return
    try:
        data = _read(path)
        sources = data.setdefault("sources", {})
        for name, stats in worker.items():
            if not isinstance(stats, dict):
                continue
            entry = sources.setdefault(name, {"attempted": 0, "succeeded": 0, "failed": 0, "models": {}})
            for key in ("attempted", "succeeded", "failed"):
                entry[key] += int(stats.get(key, 0) or 0)
            if stats.get("last_success_utc"):
                entry["last_success_utc"] = stats["last_success_utc"]
            if stats.get("last_failure_utc"):
                entry["last_failure_utc"] = stats["last_failure_utc"]
            if stats.get("last_error"):
                entry["last_error"] = str(stats["last_error"])[:300]
        data["updated_utc"] = _now()
        _write(path, data)
    except Exception:
        pass


def snapshot(path: Path | str) -> dict[str, Any]:
    """Return status-safe metrics plus a current reachability classification."""
    data = _read(Path(path))
    for entry in data.get("sources", {}).values():
        attempted = int(entry.get("attempted", 0) or 0)
        succeeded = int(entry.get("succeeded", 0) or 0)
        failed = int(entry.get("failed", 0) or 0)
        if attempted == 0:
            entry["reachability"] = "not-yet-called"
        elif failed == 0:
            entry["reachability"] = "reachable"
        elif succeeded == 0:
            entry["reachability"] = "unreachable"
        else:
            entry["reachability"] = "degraded"
    return data
