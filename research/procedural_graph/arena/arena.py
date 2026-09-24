#!/usr/bin/env python3
"""Head-to-head Kaggriculture arena on CPU with process-isolated agents.

Each job plays one full game: bundle `a` in seat `a_seat` against bundle `b`. Every
agent runs in its own interpreter through bundle_agent.BundleAgent (stdlib JSONL
RPC), one fresh pair of children per game. Results go to --out as one JSON line
per game (resumable: finished (tag, seed, a_seat) jobs are skipped) with rewards,
statuses, timing and, per seat, the SELL units it submitted and their value at the
decision-time quote (submitted, not executed: an order can exceed the shed stock).
--trace-dir keeps a compact per-step trace (actions, sheds, carried items, cash,
prices, market inventory, shops) for divergence analysis.
"""
from __future__ import annotations

import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
os.environ.update(CUDA_VISIBLE_DEVICES="", KAGG_ORACLE_BACKEND="numpy", KAGG_ORACLE_DEVICE="cpu",
                  PYTHONDONTWRITEBYTECODE="1")

import argparse
import gzip
import importlib.util
import json
import multiprocessing as mp
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]


def _load_bundle_agent():
    spec = importlib.util.spec_from_file_location("arena_bundle_agent", HERE / "bundle_agent.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.BundleAgent


def ledger(env, seat):
    """Per-product SELL units a seat submitted and their value at the decision-time quote."""
    sold = {p: 0 for p in PRODUCTS}
    value = {p: 0.0 for p in PRODUCTS}
    steps = env.steps
    for t in range(1, len(steps)):
        prices = ((steps[t - 1][0].get("observation") or {}).get("market") or {}).get("prices") or {}
        action = steps[t][seat].get("action")
        orders = (action.get("market") or []) if isinstance(action, dict) else []
        for o in orders:
            if isinstance(o, (list, tuple)) and len(o) >= 3 and o[0] == "SELL" and o[1] in sold:
                sold[o[1]] += int(o[2] or 0)
                value[o[1]] += int(o[2] or 0) * float(prices.get(o[1], 0) or 0)
    return {"submitted_sell_units": sold, "submitted_sell_value_at_quote": {k: round(v) for k, v in value.items()}}


def compact_trace(env):
    out = []
    for pair in env.steps:
        row = []
        for s in pair:
            obs = s.get("observation") or {}
            row.append({"action": s.get("action"), "reward": s.get("reward"), "status": s.get("status"),
                        "shed": (obs.get("private") or {}).get("shed"),
                        "inv": (obs.get("private") or {}).get("inventories")})
        obs0 = pair[0].get("observation") or {}
        out.append({"seats": row,
                    "money": [f.get("money") for f in (obs0.get("farms") or [])],
                    "prices": (obs0.get("market") or {}).get("prices"),
                    "minv": (obs0.get("market") or {}).get("inventory"),
                    "shops": (obs0.get("town") or {}).get("unlocked_shops")})
    return out


def play(job, bundle_class, opts):
    from kaggle_environments import make
    t0 = time.monotonic()
    rec = dict(job)
    rec.update(errors=[], rewards=None, statuses=None)
    clients = []
    env = None
    try:
        paths = [job["b"], job["b"]]
        paths[job["a_seat"]] = job["a"]
        for seat, bundle in enumerate(paths):
            c = bundle_class(Path(bundle), entrypoint="main.py", timeout=opts["rpc_timeout"],
                             startup_timeout=opts["startup_timeout"])
            clients.append(c)
            c.start()
        rec["startup_s"] = round(time.monotonic() - t0, 2)

        def wrap(client, seat):
            def f(obs, config):
                try:
                    return client(obs, config)
                except Exception as exc:
                    rec["errors"].append(dict(seat=seat, phase="action", type=type(exc).__name__,
                                              message=str(exc)[:2000]))
                    raise
            return f

        env = make("kaggriculture", configuration={"seed": job["seed"]}, debug=False)
        t1 = time.monotonic()
        env.run([wrap(c, s) for s, c in enumerate(clients)])
        rec["play_s"] = round(time.monotonic() - t1, 2)
    except Exception as exc:
        rec["errors"].append(dict(phase="match", type=type(exc).__name__, message=str(exc)[:2000],
                                  tb=traceback.format_exc()[-3000:]))
    finally:
        for c in clients:
            try:
                c.close()
            except Exception:
                pass
    if env is not None and env.steps:
        final = env.steps[-1]
        rec["states"] = len(env.steps)
        rec["statuses"] = [s.get("status") for s in final]
        rec["rewards"] = [s.get("reward") for s in final]
        try:
            rec["ledger"] = [ledger(env, 0), ledger(env, 1)]
        except Exception as exc:
            rec["errors"].append(dict(phase="ledger", message=repr(exc)))
        if opts.get("trace_dir"):
            try:
                td = Path(opts["trace_dir"])
                td.mkdir(parents=True, exist_ok=True)
                p = td / f"{job['tag']}_seed{job['seed']}_aseat{job['a_seat']}.json.gz"
                with gzip.open(p, "wt", compresslevel=6) as fh:
                    json.dump(compact_trace(env), fh, separators=(",", ":"))
                rec["trace"] = p.name
            except Exception as exc:
                rec["errors"].append(dict(phase="trace", message=repr(exc)))
    rec["total_s"] = round(time.monotonic() - t0, 2)
    return rec


def worker(q_in, q_out, opts):
    os.setsid()
    bundle_class = _load_bundle_agent()
    while True:
        job = q_in.get()
        if job is None:
            return
        try:
            q_out.put(play(job, bundle_class, opts))
        except Exception as exc:
            q_out.put(dict(job, errors=[dict(phase="worker", message=repr(exc))]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", help="candidate bundle dir (main.py inside)")
    ap.add_argument("--b", help="opponent bundle dir")
    ap.add_argument("--tag", default="match")
    ap.add_argument("--seeds", help="comma list, or start:count (both seats each)")
    ap.add_argument("--jobs", help="jobs.json manifest; bundle paths relative to --root")
    ap.add_argument("--root", default=".")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    ap.add_argument("--trace-dir", default=None)
    ap.add_argument("--rpc-timeout", type=float, default=60.0)
    ap.add_argument("--startup-timeout", type=float, default=180.0)
    ap.add_argument("--deadline", type=float, default=0.0, help="stop dispatching after N seconds")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    if args.jobs:
        manifest = json.loads(Path(args.jobs).read_text())
        jobs = [dict(j, a=str(root / j["a"]), b=str(root / j["b"]))
                for i, j in enumerate(manifest["jobs"]) if i % args.shards == args.shard]
    else:
        if ":" in args.seeds:
            s, n = args.seeds.split(":")
            seeds = list(range(int(s), int(s) + int(n)))
        else:
            seeds = [int(x) for x in args.seeds.split(",") if x]
        jobs = [dict(tag=args.tag, a=str(Path(args.a).resolve()), b=str(Path(args.b).resolve()),
                     seed=sd, a_seat=st) for sd in seeds for st in (0, 1)]
    out = Path(args.out)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            try:
                r = json.loads(line)
                if r.get("statuses") and not r.get("errors"):
                    done.add((r["tag"], r["seed"], r["a_seat"]))
            except Exception:
                pass
    todo = [j for j in jobs if (j["tag"], j["seed"], j["a_seat"]) not in done]
    opts = dict(rpc_timeout=args.rpc_timeout, startup_timeout=args.startup_timeout, trace_dir=args.trace_dir)
    print(f"arena: {len(todo)} games to play ({len(jobs) - len(todo)} already done), workers={args.workers}",
          flush=True)
    ctx = mp.get_context("spawn")
    q_in, q_out = ctx.Queue(), ctx.Queue()
    procs = [ctx.Process(target=worker, args=(q_in, q_out, opts), daemon=True) for _ in range(args.workers)]
    for p in procs:
        p.start()
    t0 = time.monotonic()
    sent = 0
    for j in todo[: args.workers]:
        q_in.put(j)
        sent += 1
    received = 0
    stats = {}
    with out.open("a") as fh:
        while received < sent:
            r = q_out.get()
            received += 1
            fh.write(json.dumps(r, separators=(",", ":")) + "\n")
            fh.flush()
            rw = r.get("rewards")
            st = stats.setdefault(r["tag"], [0, 0, 0, 0.0])
            if rw and None not in rw and not r.get("errors"):
                a, b = rw[r["a_seat"]], rw[1 - r["a_seat"]]
                st[0] += a > b
                st[1] += a < b
                st[2] += a == b
                st[3] += a - b
            n = max(1, st[0] + st[1] + st[2])
            print(f"[{time.monotonic() - t0:7.0f}s] {r['tag']} seed {r['seed']} a_seat {r['a_seat']} "
                  f"rewards {rw} err {len(r.get('errors') or [])} | {r['tag']} {st[0]}W-{st[1]}L-{st[2]}T "
                  f"mean margin {st[3] / n:+.1f} | {r.get('total_s')}s", flush=True)
            if sent < len(todo) and not (args.deadline and time.monotonic() - t0 > args.deadline):
                q_in.put(todo[sent])
                sent += 1
    for _ in procs:
        q_in.put(None)
    for p in procs:
        p.join(timeout=10)
    for tag, st in sorted(stats.items()):
        n = max(1, st[0] + st[1] + st[2])
        print(f"ARENA_SUMMARY {tag} {st[0]}W-{st[1]}L-{st[2]}T mean margin {st[3] / n:+.1f}", flush=True)
    print("ARENA_DONE", flush=True)


if __name__ == "__main__":
    main()
