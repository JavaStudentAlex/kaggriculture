"""Checkpoint and score-comparability primitives for the high-CPU runner."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

CHECKPOINT_VERSION = 1


def _json_default(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def graph_fingerprint(graph):
    return hashlib.sha256(json.dumps(graph, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def evaluation_fingerprint(champions, seeds_seat0, seeds_seat1, code_files, settings):
    """Hash actual opponent/code bytes, not mtimes or pool size."""
    if not champions or not seeds_seat0 or not seeds_seat1:
        raise ValueError("An evaluation requires opponents and both seed blocks")
    def files(paths):
        return [(str(Path(p).resolve()), hashlib.sha256(Path(p).read_bytes()).hexdigest())
                for p in sorted(paths, key=lambda p: str(Path(p).resolve()))]
    return graph_fingerprint({"champions": files(champions), "code": files(code_files),
                              "seat0": list(seeds_seat0), "seat1": list(seeds_seat1),
                              "settings": settings})


def atomic_json(path, payload):
    """Durable same-directory replace: readers see all of a checkpoint or none."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False, default=_json_default)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        Path(temporary).unlink(missing_ok=True)


def save_checkpoint(path, islands, next_iteration, master_graph, baseline_score,
                    fingerprint, provenance):
    atomic_json(path, {"version": CHECKPOINT_VERSION, "next_iteration": next_iteration,
                       "master_graph": master_graph, "baseline_score": baseline_score,
                       "evaluation_fingerprint": fingerprint, "provenance": provenance,
                       "islands": [vars(island) for island in islands]})


def load_checkpoint(path, island_class):
    """Malformed checkpoints fail closed; never silently fall back to legacy state."""
    state = json.loads(Path(path).read_text(encoding="utf-8"))
    if state.get("version") != CHECKPOINT_VERSION:
        raise ValueError("Unsupported evolution checkpoint version")
    if not isinstance(state.get("next_iteration"), int) or state["next_iteration"] < 1:
        raise ValueError("Invalid checkpoint next_iteration")
    if not isinstance(state.get("master_graph"), dict) or not state.get("islands"):
        raise ValueError("Checkpoint missing master graph or islands")
    islands = []
    for record in state["islands"]:
        for key, kind in (("island_id", int), ("name", str), ("focus_theme", str),
                          ("champion_graph", dict), ("champion_score", dict),
                          ("history", list), ("rejections", list), ("embeddings_archive", list)):
            if not isinstance(record.get(key), kind):
                raise ValueError(f"Invalid checkpoint island field: {key}")
        island = island_class.__new__(island_class)
        island.__dict__.update(copy.deepcopy(record))
        islands.append(island)
    if len({i.island_id for i in islands}) != len(islands):
        raise ValueError("Duplicate checkpoint island IDs")
    return state, islands


def legacy_resume(status):
    """Only iteration position is recoverable; status contains no island graphs.

    A gauntlet/in-progress event repeats its iteration. Explicit terminal events
    have completed promotion/rejection and can advance. Counts remain provenance,
    not fabricated history entries or claims of recovered originals.
    """
    current = max(1, int(status.get("current_iteration", 1)))
    event = status.get("latest_event", {}).get("event", "")
    completed = event in {"CROWNED", "REJECTED", "Novelty Rejection"}
    return current + int(completed), {
        "mode": "legacy_master_initialization",
        "warning": "Original island graphs/history unavailable; initialized ALL islands from current master; legacy scores discarded",
        "legacy_current_iteration": current,
        "legacy_status": status.get("status"),
        "legacy_event": event,
        "legacy_island_summaries": status.get("islands", []),
        "iteration_policy": "advance_terminal_event" if completed else "repeat_unconfirmed_iteration",
    }


def valid_score(score):
    return (score.get("valid") is True and score.get("crash_count") == 0
            and score.get("total_matches", 0) > 0
            and all(isinstance(score.get(k), (int, float)) and math.isfinite(score[k])
                    for k in ("overall_win_rate", "overall_cand_cash", "overall_opp_cash")))


def comparable_score(score, graph, fingerprint):
    return (valid_score(score) and score.get("evaluation_fingerprint") == fingerprint
            and score.get("graph_fingerprint") == graph_fingerprint(graph))


def rebase_incumbents(islands, master_graph, baseline_score, fingerprint, evaluate_graph):
    """Evaluate every stale unique graph, then commit all scores together."""
    entries = [(master_graph, baseline_score)] + [(i.champion_graph, i.champion_score) for i in islands]
    cache = {graph_fingerprint(graph): copy.deepcopy(score) for graph, score in entries
             if comparable_score(score, graph, fingerprint)}
    for graph, _ in entries:
        key = graph_fingerprint(graph)
        if key not in cache:
            score = evaluate_graph(graph, f"rebase_{key[:16]}")
            if not comparable_score(score, graph, fingerprint):
                raise RuntimeError("Incumbent rebase failed/crashed; comparisons are blocked")
            cache[key] = copy.deepcopy(score)
    for island in islands:
        island.champion_score = copy.deepcopy(cache[graph_fingerprint(island.champion_graph)])
    return copy.deepcopy(cache[graph_fingerprint(master_graph)])
