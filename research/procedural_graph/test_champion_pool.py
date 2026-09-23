"""Behavioural tests for champion_pool: growth, partitioning, aggregation."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))

import champion_pool as cp

FAILS = []
PASSES = []


def check(name, cond, detail=""):
    if cond:
        PASSES.append(name)
    else:
        FAILS.append(f"{name} :: {detail}")


# ---------------------------------------------------------------- partitioning
def test_partition_is_total_and_balanced():
    """Every champion assigned exactly once; sizes differ by <= 1."""
    for n in range(1, 40):
        for shards in range(1, 13):
            champs = [f"c{i}" for i in range(n)]
            buckets = [cp.assign_champions(champs, s, shards) for s in range(shards)]
            flat = [c for b in buckets for c in b]

            check(f"partition total n={n} s={shards}",
                  sorted(flat) == sorted(champs),
                  f"got {len(flat)} want {n}")
            check(f"partition no dupes n={n} s={shards}",
                  len(flat) == len(set(flat)), "duplicate champion")

            sizes = [len(b) for b in buckets]
            if n >= shards:
                check(f"partition balanced n={n} s={shards}",
                      max(sizes) - min(sizes) <= 1, f"sizes={sizes}")


def test_partition_rejects_bad_shard_id():
    try:
        cp.assign_champions(["a", "b"], 5, 2)
        check("bad shard_id raises", False, "no exception")
    except ValueError:
        check("bad shard_id raises", True)
    try:
        cp.assign_champions(["a"], 0, 0)
        check("zero total_shards raises", False, "no exception")
    except ValueError:
        check("zero total_shards raises", True)


def test_partition_empty_pool():
    check("empty pool -> empty bucket", cp.assign_champions([], 0, 3) == [])


# ------------------------------------------------------------- shard planning
def test_plan_never_exceeds_limits():
    check("plan empty pool -> 0", cp.plan_shard_split(0, 5) == 0)
    check("plan 1 champ -> 1 shard", cp.plan_shard_split(1, 5) == 1)
    check("plan 3 champs max5 -> 3", cp.plan_shard_split(3, 5) == 3,
          f"got {cp.plan_shard_split(3, 5)}")
    check("plan 10 champs max5 -> 5", cp.plan_shard_split(10, 5) == 5)
    check("plan 40 champs max5 -> 5", cp.plan_shard_split(40, 5) == 5)
    # never more shards than champions (no empty notebooks)
    for n in range(1, 30):
        for m in range(1, 12):
            s = cp.plan_shard_split(n, m)
            check(f"plan no empty notebook n={n} m={m}", 1 <= s <= min(n, m),
                  f"got {s}")


def test_plan_respects_per_shard_cap():
    # 20 champions, cap 2 per shard -> wants 10, but ceiling is 5
    check("cap clamped by max", cp.plan_shard_split(20, 5, champions_per_shard=2) == 5)
    # 6 champions, cap 2 -> 3 shards
    check("cap honoured", cp.plan_shard_split(6, 10, champions_per_shard=2) == 3,
          f"got {cp.plan_shard_split(6, 10, champions_per_shard=2)}")
    try:
        cp.plan_shard_split(5, 5, champions_per_shard=0)
        check("bad cap raises", False, "no exception")
    except ValueError:
        check("bad cap raises", True)


# ------------------------------------------------------------------ induction
GRAPH_A = {
    "version": "1.0",
    "name": "graph_a",
    "nodes": [{"id": "state_audit"}, {"id": "farm_execution"}],
    "edges": [{"from": "state_audit", "to": "farm_execution", "priority": 1}],
}
GRAPH_B = {
    "version": "1.0",
    "name": "graph_b",
    "nodes": [{"id": "state_audit"}, {"id": "wage_defense"}],
    "edges": [{"from": "state_audit", "to": "wage_defense", "priority": 1}],
}


def test_fingerprint_is_order_insensitive():
    g1 = {"nodes": [1], "edges": [2], "name": "x"}
    g2 = {"name": "x", "edges": [2], "nodes": [1]}
    check("fingerprint key-order stable", cp.graph_fingerprint(g1) == cp.graph_fingerprint(g2))
    check("fingerprint distinguishes graphs",
          cp.graph_fingerprint(GRAPH_A) != cp.graph_fingerprint(GRAPH_B))


def test_induction_grows_pool_and_dedupes():
    with tempfile.TemporaryDirectory() as td:
        pool = Path(td) / "pool"
        check("empty pool lists nothing", cp.list_champions(pool) == [])

        p1, s1 = cp.induct_graph_champion(GRAPH_A, pool, iteration=3, island="Island-Capital")
        check("first induction creates", s1 == "created", s1)
        check("pool grew to 1", len(cp.list_champions(pool)) == 1)

        # same graph again -> duplicate, pool must not grow
        p2, s2 = cp.induct_graph_champion(GRAPH_A, pool, iteration=9, island="Island-Other")
        check("re-induct is duplicate", s2 == "duplicate", s2)
        check("pool still 1", len(cp.list_champions(pool)) == 1)
        check("duplicate returns same file", p1 == p2)

        # different graph -> grows
        cp.induct_graph_champion(GRAPH_B, pool, iteration=4, island="Island-Arbitrage")
        check("pool grew to 2", len(cp.list_champions(pool)) == 2)

        # provenance log written
        log = pool / "INDUCTED.jsonl"
        check("provenance log exists", log.exists())
        lines = [json.loads(x) for x in log.read_text().splitlines() if x.strip()]
        check("provenance has 2 entries", len(lines) == 2, f"got {len(lines)}")

        # POOL.json / non-.py files never counted as champions
        (pool / "POOL.json").write_text("{}", encoding="utf-8")
        check("POOL.json excluded", len(cp.list_champions(pool)) == 2)

        # ordering deterministic
        check("listing sorted",
              [p.name for p in cp.list_champions(pool)] == sorted(p.name for p in cp.list_champions(pool)))


def test_induction_rejects_empty_graph():
    with tempfile.TemporaryDirectory() as td:
        try:
            cp.induct_graph_champion({"nodes": []}, Path(td))
            check("empty graph rejected", False, "no exception")
        except ValueError:
            check("empty graph rejected", True)


def test_generated_champion_is_valid_python_and_deterministic():
    src = cp.build_champion_agent_source(GRAPH_A, iteration=1, island="Island-Capital")
    compile(src, "champ.py", "exec")
    check("generated champion compiles", True)
    check("champion defines agent", "def agent(" in src)
    check("champion has no LLM/tunnel use",
          "typesafe" not in src.lower() and "8317" not in src and "urllib" not in src)
    check("champion does not reuse shared agent modules",
          "import agent_jev_graph" not in src and "import agent_graph" not in src)
    check("champion embeds its own graph", "GRAPH_JSON" in src)
    # the embedded graph round-trips
    start = src.index('GRAPH_JSON = r"""') + len('GRAPH_JSON = r"""')
    end = src.index('"""', start)
    check("embedded graph round-trips", json.loads(src[start:end]) == GRAPH_A)


# ----------------------------------------------------------------- aggregation
def test_aggregate_weights_by_matches():
    """Unequal shard sizes must not distort the mean cash."""
    shards = [
        {"shard_id": 0, "total_matches": 140, "total_wins": 140,
         "overall_cand_cash": 1000.0, "overall_opp_cash": 10.0,
         "assigned_champions": ["a", "b"]},
        {"shard_id": 1, "total_matches": 70, "total_wins": 35,
         "overall_cand_cash": 4000.0, "overall_opp_cash": 40.0,
         "assigned_champions": ["c"]},
    ]
    agg = cp.aggregate_shard_results(shards)
    check("total matches summed", agg["total_matches"] == 210, agg["total_matches"])
    check("wins summed", agg["total_wins"] == 175)
    check("win rate over matches", agg["overall_win_rate"] == round(175 / 210, 3),
          agg["overall_win_rate"])
    # match-weighted: (1000*140 + 4000*70)/210 = 2000.0  (unweighted would be 2500)
    check("cash match-weighted", abs(agg["overall_cand_cash"] - 2000.0) < 0.01,
          f"got {agg['overall_cand_cash']} (unweighted would be 2500)")
    check("champions tracked", agg["champions_played"] == ["a", "b", "c"])


def test_aggregate_handles_empty_and_missing():
    agg = cp.aggregate_shard_results([])
    check("empty aggregate no crash", agg["total_matches"] == 0)
    check("empty aggregate zero winrate", agg["overall_win_rate"] == 0.0)
    agg2 = cp.aggregate_shard_results([None, {}, {"total_matches": 10, "total_wins": 5}])
    check("sparse shards tolerated", agg2["total_matches"] == 10, agg2["total_matches"])


def test_manifest():
    with tempfile.TemporaryDirectory() as td:
        pool = Path(td)
        cp.induct_graph_champion(GRAPH_A, pool, iteration=1, island="I")
        man = cp.pool_manifest(pool)
        check("manifest counts", man["count"] == 1)
        check("manifest has sha", len(man["champions"][0]["sha256"]) == 64)


for fn in sorted([v for k, v in list(globals().items()) if k.startswith("test_")],
                 key=lambda f: f.__name__):
    try:
        fn()
    except Exception as exc:
        import traceback
        FAILS.append(f"{fn.__name__} raised: {exc}\n{traceback.format_exc()}")

print(f"passed checks: {len(PASSES)}")
if FAILS:
    print(f"FAILURES ({len(FAILS)}):")
    for f in FAILS[:25]:
        print("  -", f)
    sys.exit(1)
print("ALL TESTS PASSED")
