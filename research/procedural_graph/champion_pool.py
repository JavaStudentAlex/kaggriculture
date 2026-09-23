"""Growing champion pool + dynamic Kaggle shard planning (Red Queen ratchet).

This module is the single source of truth for three things the distributed
island evolution needs once the opponent pool stops being a frozen list:

1.  **Induction** - turning a crowned procedural graph into a real, standalone
    champion agent file in ``shinka/champions/pool`` so every *later* candidate
    must beat it (the Red Queen ratchet).
2.  **Dynamic sharding** - splitting an arbitrary number of champions across an
    arbitrary number of Kaggle notebooks with a balanced, deterministic,
    gap-free partition.
3.  **Bundle publishing** - re-versioning the Kaggle dataset that the notebooks
    read their champions from, because a champion that only exists on the local
    disk is invisible to the cloud swarm.

Design constraints that drove the implementation
------------------------------------------------
* A champion must be *deterministic*. It is an opponent, not a candidate, so it
  must never call the LLM proxy / Jev tunnel: an opponent whose strength depends
  on network latency makes every score incomparable across generations.
* A champion must be *self-contained with respect to module globals*. Two agents
  run inside a single worker process. ``agent_jev_graph`` keeps its graph in a
  module-level ``_ACTOR``/``_FALLBACK_ENGINE`` and ignores ``KAGG_GRAPH_PATH``,
  so a graph champion that reused that module would silently fight using the
  *candidate's* graph. Generated champions therefore own a private engine
  instance bound to an inlined copy of their graph.
* Induction must be *idempotent and content-addressed*. The same graph can only
  ever become one champion, no matter how many islands rediscover it.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tarfile
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

__all__ = [
    "aggregate_shard_results",
    "assign_champions",
    "build_bundle_archive",
    "build_champion_agent_source",
    "champion_names",
    "graph_fingerprint",
    "induct_graph_champion",
    "list_champions",
    "plan_shard_split",
    "pool_manifest",
    "publish_bundle",
]

# A champion file must define `agent`; these are never champions.
_EXCLUDED_NAMES = {"__init__.py", "POOL.json"}
_CHAMPION_GLOB = "*.py"


# --------------------------------------------------------------------- hashing
def _canonical_graph_bytes(graph: dict[str, Any]) -> bytes:
    """Stable serialization so logically identical graphs hash identically.

    ``sort_keys`` makes key order irrelevant and the compact separators make
    whitespace irrelevant, so a graph that only differs by formatting or by the
    order an LLM happened to emit its keys is correctly detected as a duplicate.
    """
    return json.dumps(graph, sort_keys=True, separators=(",", ":")).encode("utf-8")


def graph_fingerprint(graph: dict[str, Any]) -> str:
    """Content address of a policy graph (hex sha256)."""
    return hashlib.sha256(_canonical_graph_bytes(graph)).hexdigest()


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------- pool reading
def list_champions(pool_dir: Path | str) -> list[Path]:
    """Every champion program in the pool, deterministically ordered.

    Sorted by filename so the partition is reproducible across machines and
    across runs. Hidden files, ``__pycache__`` and non-champion helpers are
    excluded. Returns ``[]`` for a missing directory rather than raising, so
    callers can produce their own domain-specific error.
    """
    pool = Path(pool_dir)
    if not pool.is_dir():
        return []
    out = [
        p for p in pool.glob(_CHAMPION_GLOB)
        if p.is_file()
        and p.name not in _EXCLUDED_NAMES
        and not p.name.startswith(".")
    ]
    return sorted(out, key=lambda p: p.name)


def champion_names(pool_dir: Path | str) -> list[str]:
    """Champion file names (``champ_x.py``) in partition order."""
    return [p.name for p in list_champions(pool_dir)]


# ------------------------------------------------------------- shard planning
def plan_shard_split(
    num_champions: int,
    max_shards: int,
    champions_per_shard: int | None = None,
) -> int:
    """Decide how many Kaggle notebooks to run for ``num_champions`` opponents.

    The pool grows every time a candidate is crowned, so the shard count can no
    longer be a constant. Rules:

    * never launch an empty notebook -> ``num_shards <= num_champions``
    * never exceed the concurrency ceiling -> ``num_shards <= max_shards``
    * if ``champions_per_shard`` is given, use the smallest shard count that
      keeps each notebook at or under that many champions (bounded runtime per
      notebook), still clamped by ``max_shards``.

    Returns 0 for an empty pool so the caller can fail loudly with context.
    """
    if num_champions <= 0:
        return 0
    if max_shards <= 0:
        raise ValueError(f"max_shards must be >= 1, got {max_shards}")

    if champions_per_shard is not None:
        if champions_per_shard <= 0:
            raise ValueError(
                f"champions_per_shard must be >= 1, got {champions_per_shard}")
        # ceil division: enough shards to respect the per-shard cap
        wanted = -(-num_champions // champions_per_shard)
    else:
        wanted = max_shards

    return max(1, min(num_champions, max_shards, wanted))


def assign_champions(
    champions: Sequence[Any],
    shard_id: int,
    total_shards: int,
) -> list[Any]:
    """Balanced, contiguous, gap-free partition of champions for one shard.

    Every champion is assigned to exactly one shard and shard sizes differ by at
    most one. Contiguous (rather than round-robin ``i % n``) so that adding a
    champion perturbs the assignment minimally and the split is trivially
    auditable from a log line.

    Raises on an out-of-range ``shard_id`` - a silent empty list there would
    quietly drop champions from the gauntlet and inflate the win rate.
    """
    n = len(champions)
    if total_shards <= 0:
        raise ValueError(f"total_shards must be >= 1, got {total_shards}")
    if not (0 <= shard_id < total_shards):
        raise ValueError(
            f"shard_id {shard_id} out of range for total_shards {total_shards}")
    if n == 0:
        return []

    base, extra = divmod(n, total_shards)
    # The first `extra` shards take one additional champion.
    start = shard_id * base + min(shard_id, extra)
    size = base + (1 if shard_id < extra else 0)
    return list(champions[start:start + size])


# ------------------------------------------------------- champion materialization
_CHAMPION_TEMPLATE = '''"""Auto-generated graph champion - DO NOT EDIT BY HAND.

Crowned by the distributed island evolution.
    source iteration : {iteration}
    source island    : {island}
    win rate         : {win_rate}
    mean cash        : {cash}
    graph sha256     : {fingerprint}

This file is a frozen *opponent*. It is intentionally deterministic: it never
contacts the LLM proxy, so its strength cannot drift between generations.

It also keeps its engine in this module's own namespace instead of reusing
`agent_graph` / `agent_jev_graph`, whose module-level engine globals are shared
with the candidate under test inside the same worker process.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

_HERE = Path(__file__).resolve()
# pool/ -> champions/ -> shinka/ -> <repo root>
_ROOT = _HERE.parent.parent.parent.parent
for _p in (
    _ROOT / "shinka" / "evolution",
    _ROOT / "research" / "procedural_graph",
    _ROOT / "shinka" / "champions" / "dependencies" / "mohui_v66",
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import initial as hazel
from graph_engine import ProceduralGraphEngine

# The graph is inlined so this champion is independent of policy_graph.json,
# which the evolution keeps rewriting.
GRAPH_JSON = r"""{graph_json}"""
GRAPH_FINGERPRINT = "{fingerprint}"

_ENGINE = None


def _engine() -> ProceduralGraphEngine:
    """Private engine bound to this champion's own inlined graph."""
    global _ENGINE
    if _ENGINE is None:
        tmp = Path(__file__).with_suffix(".graph.json")
        try:
            if not tmp.exists():
                tmp.write_text(GRAPH_JSON, encoding="utf-8")
            _ENGINE = ProceduralGraphEngine(tmp)
        except OSError:
            # Read-only filesystem (Kaggle input dirs are read-only): fall back
            # to a temp copy so the champion still plays.
            import tempfile
            fd, name = tempfile.mkstemp(suffix=".graph.json")
            with open(fd, "w", encoding="utf-8") as fh:
                fh.write(GRAPH_JSON)
            _ENGINE = ProceduralGraphEngine(Path(name))
    return _ENGINE


def agent(obs: dict, configuration: dict | None = None) -> dict:
    """Public entry point invoked by kaggle_environments each step."""
    player_idx = int(obs.get("player", 0) or 0)
    base = hazel._backbone_action(obs, configuration)

    farms = obs.get("farms", []) or []
    farm = farms[player_idx] if player_idx < len(farms) else {{}}
    n_hands = len(farm.get("hands") or [])

    forecast = hazel._oracle_observe(obs, configuration)

    try:
        engine = _engine()
        st = hazel.farm_state(obs, player_idx, forecast)
        guards = engine.evaluate_guards(st, obs)

        farmer_act = hazel.evolve_farmer_action(obs, player_idx, base.get("farmer"), st)
        hands_act = hazel.evolve_hand_actions(obs, player_idx, base.get("hands"), st)
        base_market = hazel.evolve_market_orders(obs, player_idx, base.get("market"), st)

        orders_by_node: dict[str, list[list[Any]]] = {{
            "wage_defense": [],
            "terminal_liquidation": [],
            "shed_headroom": [],
            "town_shop_preempt": [],
            "oracle_frontrun": [],
            "farm_execution": [],
        }}

        step = int(obs.get("step", 0) or 0)

        if guards.get("wage_defense", False):
            for o in base_market:
                if isinstance(o, (list, tuple)) and len(o) >= 3 and o[0] == "SELL":
                    orders_by_node["wage_defense"].append(list(o))

        if step >= 700:
            for o in base_market:
                if isinstance(o, (list, tuple)) and len(o) >= 3 and o[0] == "SELL":
                    if list(o) not in orders_by_node["wage_defense"]:
                        orders_by_node["terminal_liquidation"].append(list(o))

        if guards.get("shed_headroom", False):
            for o in base_market:
                if (isinstance(o, (list, tuple)) and len(o) >= 3 and o[0] == "SELL"
                        and o[1] in ("WHEAT", "FERTILIZER", "STRAWBERRY")):
                    if list(o) not in orders_by_node["wage_defense"]:
                        orders_by_node["shed_headroom"].append(list(o))

        if guards.get("town_shop_preempt", False):
            for o in base_market:
                if isinstance(o, (list, tuple)) and len(o) >= 3 and o[1] in ("CARROT", "TOMATO", "EGG"):
                    orders_by_node["town_shop_preempt"].append(list(o))

        if guards.get("oracle_frontrun", False):
            for o in base_market:
                if (isinstance(o, (list, tuple)) and len(o) >= 3 and o[0] == "SELL"
                        and o[1] in ("WOOL", "MILK")):
                    orders_by_node["oracle_frontrun"].append(list(o))

        for o in base_market:
            orders_by_node["farm_execution"].append(list(o))

        arbitrated = engine.arbitrate_market_orders(orders_by_node, max_orders=10)

        evolved = {{"farmer": farmer_act, "hands": hands_act, "market": arbitrated}}
        evolved = hazel.policy_sanitize(evolved, base, st)
    except Exception:
        evolved = base

    final = hazel._hard_sanitize(evolved, base, n_hands)
    hazel._oracle_record(final)
    return final
'''


def build_champion_agent_source(
    graph: dict[str, Any],
    iteration: Any = "?",
    island: str = "?",
    win_rate: Any = "?",
    cash: Any = "?",
) -> str:
    """Render a standalone, deterministic champion agent for ``graph``."""
    graph_json = json.dumps(graph, indent=1, sort_keys=True)
    # The graph is embedded in an r"""...""" literal; a stray triple quote or a
    # trailing backslash would break the file. JSON escaping already removes
    # backslashes-in-strings ambiguity, but guard the delimiter explicitly.
    if '"""' in graph_json:
        graph_json = graph_json.replace('"""', '\\"\\"\\"')
    return _CHAMPION_TEMPLATE.format(
        graph_json=graph_json,
        fingerprint=graph_fingerprint(graph),
        iteration=iteration,
        island=island,
        win_rate=win_rate,
        cash=cash,
    )


def induct_graph_champion(
    graph: dict[str, Any],
    pool_dir: Path | str,
    iteration: Any = "?",
    island: str = "?",
    win_rate: Any = "?",
    cash: Any = "?",
) -> tuple[Path | None, str]:
    """Add a crowned graph to the champion pool.

    Returns ``(path, status)`` where status is one of:
      * ``"created"``   - a new champion file was written
      * ``"duplicate"`` - this exact graph is already a champion (no-op)

    Content-addressed and atomic: concurrent inductions of the same graph
    converge on one file, and no reader ever observes a half-written champion.
    """
    if not isinstance(graph, dict) or not graph.get("nodes"):
        raise ValueError("refusing to induct a graph with no 'nodes'")

    pool = Path(pool_dir)
    pool.mkdir(parents=True, exist_ok=True)

    fp = graph_fingerprint(graph)

    # Duplicate check by fingerprint recorded inside existing champions.
    for existing in list_champions(pool):
        try:
            head = existing.read_text(encoding="utf-8", errors="ignore")[:4000]
        except OSError:
            continue
        if fp in head:
            return existing, "duplicate"

    safe_island = re.sub(r"[^A-Za-z0-9]+", "-", str(island)).strip("-") or "island"
    name = f"champ_evo_{int(iteration) if str(iteration).isdigit() else 0:04d}_{safe_island}_{fp[:12]}.py"
    dest = pool / name
    if dest.exists():
        return dest, "duplicate"

    source = build_champion_agent_source(
        graph, iteration=iteration, island=island, win_rate=win_rate, cash=cash)

    fd, tmp = tempfile.mkstemp(prefix=".induct-", dir=str(pool))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(source)
            fh.flush()
            os.fsync(fh.fileno())
        try:
            os.link(tmp, dest)
        except FileExistsError:
            return dest, "duplicate"
    finally:
        Path(tmp).unlink(missing_ok=True)

    # Append-only provenance log.
    try:
        with open(pool / "INDUCTED.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "file": name,
                "iteration": iteration,
                "island": island,
                "win_rate": win_rate,
                "cash": cash,
                "graph_sha256": fp,
            }) + "\n")
    except OSError:
        pass

    return dest, "created"


def pool_manifest(pool_dir: Path | str) -> dict[str, Any]:
    """Human/machine readable snapshot of the current pool."""
    champs = list_champions(pool_dir)
    return {
        "count": len(champs),
        "champions": [
            {"file": c.name, "sha256": _file_sha256(c)} for c in champs
        ],
    }


# ------------------------------------------------------------ bundle publishing
def build_bundle_archive(
    repo_dir: Path | str,
    bundle_dir: Path | str,
    archive_name: str = "champions_bundle.tar.gz",
) -> Path:
    """Rebuild the champions tarball the Kaggle notebooks mount as a dataset.

    Mirrors the layout the shard bootstrap expects (``shinka/champions/...``
    plus the code the champions import) and skips ``__pycache__`` so stale
    bytecode can never shadow a freshly added champion.
    """
    repo = Path(repo_dir)
    out_dir = Path(bundle_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    archive = out_dir / archive_name

    members = [
        repo / "shinka" / "champions" / "pool",
        repo / "shinka" / "champions" / "dependencies",
        repo / "shinka" / "evolution",
        repo / "research" / "procedural_graph",
    ]

    def _filter(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
        parts = Path(info.name).parts
        if "__pycache__" in parts:
            return None
        if info.name.endswith((".pyc", ".pyo")):
            return None
        if any(p.startswith(".") and p not in (".", "..") for p in parts):
            return None
        return info

    tmp = archive.with_suffix(".tmp")
    with tarfile.open(tmp, "w:gz") as tar:
        for m in members:
            if m.exists():
                tar.add(m, arcname=str(m.relative_to(repo)), filter=_filter)
    tmp.replace(archive)
    return archive


def publish_bundle(
    bundle_dir: Path | str,
    version_notes: str,
    kaggle_bin: str = "kaggle",
    timeout: int = 900,
) -> tuple[bool, str]:
    """Push a new version of the champions dataset to Kaggle.

    Without this the cloud notebooks keep mounting the previous snapshot and a
    newly crowned champion would never actually be played against.
    Returns ``(ok, output)`` and never raises - the caller decides whether a
    publish failure should stop the run.
    """
    bundle = Path(bundle_dir)
    meta = bundle / "dataset-metadata.json"
    if not meta.exists():
        return False, f"missing dataset-metadata.json in {bundle}"

    cmd = [kaggle_bin, "datasets", "version", "-p", str(bundle),
           "-m", version_notes, "--dir-mode", "zip"]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        return False, f"{kaggle_bin} CLI not found on PATH"
    except subprocess.TimeoutExpired:
        return False, f"kaggle datasets version timed out after {timeout}s"

    out = (res.stdout or "") + (res.stderr or "")
    return res.returncode == 0, out.strip()


# --------------------------------------------------------------- aggregation
def aggregate_shard_results(shard_results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Combine per-notebook results into one gauntlet score.

    Cash is a **match-weighted** mean. The old unweighted ``sum/len(shards)``
    was only correct while every shard held exactly the same number of
    champions; once the pool grows to a size that is not a multiple of the
    shard count, shards carry different match counts and a plain average
    silently over-weights the smallest notebook.
    """
    results = [r for r in shard_results if r]
    total_wins = 0
    total_matches = 0
    cash_num = 0.0
    opp_num = 0.0
    losses: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    champions_seen: list[str] = []
    remote_call_metrics: dict[str, dict[str, Any]] = {
        "jev": {"attempted": 0, "succeeded": 0, "failed": 0},
        "inner_llm": {"attempted": 0, "succeeded": 0, "failed": 0},
    }

    for d in results:
        m = int(d.get("total_matches", 0) or 0)
        total_wins += int(d.get("total_wins", 0) or 0)
        total_matches += m
        cash_num += float(d.get("overall_cand_cash", 0.0) or 0.0) * m
        opp_num += float(d.get("overall_opp_cash", 0.0) or 0.0) * m
        losses.extend(d.get("worst_losses", []) or [])
        traces.extend(d.get("mined_loss_traces", []) or [])
        champions_seen.extend(d.get("assigned_champions", []) or [])
        for source, stats in (d.get("remote_call_metrics", {}) or {}).items():
            bucket = remote_call_metrics.setdefault(
                source, {"attempted": 0, "succeeded": 0, "failed": 0})
            for key in ("attempted", "succeeded", "failed"):
                bucket[key] += int(stats.get(key, 0) or 0)
            if stats.get("last_error"):
                bucket["last_error"] = str(stats["last_error"])[:300]

    denom = max(1, total_matches)
    mean_cand = cash_num / denom
    mean_opp = opp_num / denom

    losses.sort(key=lambda x: x.get("deficit", 0.0), reverse=True)
    traces.sort(key=lambda x: x.get("deficit", 0.0), reverse=True)

    return {
        "num_shards_evaluated": len(results),
        "total_matches": total_matches,
        "total_wins": total_wins,
        "overall_win_rate": round(total_wins / denom, 3),
        "overall_cand_cash": round(mean_cand, 2),
        "overall_opp_cash": round(mean_opp, 2),
        "cash_delta": round(mean_cand - mean_opp, 2),
        "champions_played": sorted(set(champions_seen)),
        "worst_losses": losses[:10],
        "mined_loss_traces": traces[:8],
        "remote_call_metrics": remote_call_metrics,
        "shard_summaries": [
            {
                "shard_id": d.get("shard_id"),
                "champ_name": d.get("champ_name"),
                "assigned_champions": d.get("assigned_champions"),
                "matches": d.get("total_matches"),
                "win_rate": d.get("overall_win_rate"),
                "cand_cash": d.get("overall_cand_cash"),
                "opp_cash": d.get("overall_opp_cash"),
            } for d in results
        ],
    }
