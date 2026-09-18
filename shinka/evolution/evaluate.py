#!/usr/bin/env python3
"""Digital Red Queen evaluator for the Kaggriculture Shinka evolution.

PROTOCOL (per candidate, against EVERY champion in the pool)
------------------------------------------------------------
    20 games with the candidate on SEAT 0, each on a different seed
  + 20 games with the candidate on SEAT 1, each on a different, DISJOINT seed
  = 40 games per champion.

The two seed blocks are disjoint so a seat-specific overfit cannot be laundered
into the other seat's score, and both blocks are fixed across generations so all
candidates in a run are graded on identical worlds.

CROWNING GATE
-------------
A candidate is crowned into the pool (and so becomes an opponent for every later
candidate - the Red Queen ratchet) only if its overall win rate across ALL
champions reaches CROWN_THRESHOLD (0.75 = 75%). Ties never count as wins. A
candidate that crashes even once is never crowned.

Shinka contract: writes metrics.json ({combined_score, public, private,
text_feedback}) and correct.json ({correct, error}) into --results_dir.

    python evaluate.py --program_path <prog.py> --results_dir <dir> [--workers N]
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import multiprocessing
import os
import shutil
import sys
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

# --------------------------------------------------------------------- config
REPO = Path(os.environ.get("KAGG_TASK_DIR", Path.home() / "kaggriculture")).resolve()
HISTORY_DIR = Path(os.environ.get("KAGG_HISTORY_DIR", REPO / "shinka" / "champions" / "pool")).resolve()
GAME_LOG = Path(os.environ.get("KAGG_GAME_LOG", HISTORY_DIR / "games.jsonl"))

CROWN_THRESHOLD = float(os.environ.get("KAGG_CROWN_THRESHOLD", "0.75"))
GAMES_PER_SEAT = int(os.environ.get("KAGG_GAMES_PER_SEAT", "20"))
EPISODE_STEPS = int(os.environ.get("KAGG_EPISODE_STEPS", "720"))
MATCH_TIMEOUT = float(os.environ.get("KAGG_MATCH_TIMEOUT", "900"))
NO_INDUCT = os.environ.get("KAGG_NO_INDUCT", "0") == "1"

# Fixed, disjoint seed blocks. Seat0 101..2020, seat1 70102..72021 - the blocks the
# 2026-09-12/14 runs used, so scores stay comparable with the recorded champions.
SEAT0_SEEDS = [101 * i for i in range(1, GAMES_PER_SEAT + 1)]
SEAT1_SEEDS = [70001 + 101 * i for i in range(1, GAMES_PER_SEAT + 1)]
assert len(set(SEAT0_SEEDS)) == GAMES_PER_SEAT and len(set(SEAT1_SEEDS)) == GAMES_PER_SEAT
assert not (set(SEAT0_SEEDS) & set(SEAT1_SEEDS)), "seat seed blocks must be disjoint"

STARTER_SEEDS = [101, 70102]  # sanity: the candidate must beat the built-in starter


# ------------------------------------------------------------------- utilities
def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_agent(path, name):
    """Import a program file and return its `agent` callable."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    agent = getattr(mod, "agent", None)
    if not callable(agent):
        raise AttributeError(f"{path} defines no callable `agent`")
    return agent


def list_champions():
    """Every champion file in the pool, sorted for determinism."""
    if not HISTORY_DIR.is_dir():
        return []
    return sorted(p for p in HISTORY_DIR.glob("*.py") if p.name != "__init__.py")


def purge_agent_modules(paths):
    """Unload policy/oracle/backbone modules before EVERY game.

    Kaggle agents (notably the Mohui v66 backbone) deliberately keep module-level
    state while a game is alive. A pool worker runs multiple games, so without this
    purge its second game inherits the first game's state and is not a fresh,
    reproducible matchup. Do not unload the engine itself.
    """
    roots = [Path(p).resolve() for p in paths]
    for name, module in list(sys.modules.items()):
        # Retaining __main__ is essential: Python's spawn machinery unpickles
        # worker calls through it. Retaining this evaluator module likewise avoids
        # unloading the function currently executing the purge.
        if name == "__main__" or module is sys.modules.get(__name__) \
                or name.startswith(("kaggle_environments", "shinka")):
            continue
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        try:
            path = Path(filename).resolve()
            if any(path.is_relative_to(root) for root in roots):
                del sys.modules[name]
        except (OSError, ValueError, TypeError):
            continue
    importlib.invalidate_caches()


# --------------------------------------------------------------- match workers
def run_single_match_worker(task):
    """One game in a fresh process. Returns (reward0, reward1, crash0, crash1).

    Spawned (never forked) so each worker loads the oracle checkpoint onto the GPU
    exactly once and the backbone modules - which keep per-process state - cannot
    leak between games.
    """
    path0, path1, seed = task
    try:
        # This worker handles many games. Reset all agent-owned module globals before
        # loading either seat so each match starts from a genuine fresh process state.
        purge_agent_modules([
            Path(path0).parent,
            Path(path1).parent,
            REPO / "shinka" / "evolution",
            REPO / "shinka" / "champions" / "dependencies" / "mohui_v66",
            REPO / "research" / "opponent_model",
        ])
        from kaggle_environments import make

        a0 = load_agent(path0, f"p0_{abs(hash((path0, seed))) % 10**8}")
        a1 = load_agent(path1, f"p1_{abs(hash((path1, seed))) % 10**8}")
        env = make("kaggriculture", configuration={"episodeSteps": EPISODE_STEPS, "seed": seed})
        env.run([a0, a1])
        last = env.steps[-1]
        r0 = float(last[0].get("reward") or 0.0)
        r1 = float(last[1].get("reward") or 0.0)
        c0 = last[0].get("status") not in ("DONE", "ACTIVE")
        c1 = last[1].get("status") not in ("DONE", "ACTIVE")
        return r0, r1, bool(c0), bool(c1)
    except Exception:
        sys.stderr.write(traceback.format_exc())
        return 0.0, 0.0, True, True


def run_starter_worker(task):
    """Sanity game vs the engine's built-in starter agent."""
    path, seed = task
    try:
        purge_agent_modules([
            Path(path).parent,
            REPO / "shinka" / "evolution",
            REPO / "shinka" / "champions" / "dependencies" / "mohui_v66",
            REPO / "research" / "opponent_model",
        ])
        from kaggle_environments import make

        a0 = load_agent(path, f"s_{abs(hash((path, seed))) % 10**8}")
        env = make("kaggriculture", configuration={"episodeSteps": EPISODE_STEPS, "seed": seed})
        env.run([a0, "starter"])
        last = env.steps[-1]
        return float(last[0].get("reward") or 0.0), last[0].get("status") not in ("DONE", "ACTIVE")
    except Exception:
        sys.stderr.write(traceback.format_exc())
        return 0.0, True


def play_all(tasks, workers):
    """Run every task; returns results in submission order."""
    results: list = [None] * len(tasks)
    if not tasks:
        return results
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max(1, workers), mp_context=ctx) as pool:
        futs = {pool.submit(run_single_match_worker, t): i for i, t in enumerate(tasks)}
        done = 0
        t0 = time.time()
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                results[i] = fut.result(timeout=MATCH_TIMEOUT)
            except Exception:
                results[i] = (0.0, 0.0, True, True)
            done += 1
            if done % 40 == 0 or done == len(tasks):
                el = time.time() - t0
                eta = el / done * (len(tasks) - done)
                print(f"  {done}/{len(tasks)} games  {el:,.0f}s elapsed  ETA {eta:,.0f}s", flush=True)
    return results


# -------------------------------------------------------------------- crowning
def crown(program_path, win_rate, avg_cash, champions_n):
    """Atomically append a passing candidate to the pool and roster archive.

    A run can evaluate more than one candidate at once. Content-addressing and an
    atomic hard-link make concurrent crowning idempotent: the same bytes can only
    become one champion, and readers never see a partially copied Python file.
    """
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    raw = Path(program_path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    name = f"champ_drq_{digest[:16]}.py"
    dest = HISTORY_DIR / name

    fd, temporary = tempfile.mkstemp(prefix=".crown-", dir=HISTORY_DIR)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(raw)
            fh.flush()
            os.fsync(fh.fileno())
        try:
            os.link(temporary, dest)
        except FileExistsError:
            if _sha256(dest) != digest:
                raise RuntimeError(f"content-address collision at {dest}")
        roster = REPO / "shinka" / "champions" / "roster"
        if roster.is_dir():
            roster_dest = roster / name
            if not roster_dest.exists():
                shutil.copy2(dest, roster_dest)
    finally:
        Path(temporary).unlink(missing_ok=True)

    record = {
        "file": name,
        "crowned": datetime.now(timezone.utc).isoformat(),
        "win_rate": round(win_rate, 4),
        "avg_cash": round(avg_cash, 1),
        "champions_beaten": champions_n,
        "gate": CROWN_THRESHOLD,
        "sha256": digest,
    }
    try:
        with open(HISTORY_DIR / "CROWNED.jsonl", "a") as fh:
            fh.write(json.dumps(record) + "\n")
    except OSError:
        pass
    return name


# ------------------------------------------------------------------ evaluation
def evaluate_candidate(program_path, results_dir, workers=4, no_induct=False):
    program_path = str(Path(program_path).resolve())
    results_dir = Path(results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    champions = list_champions()
    if not champions:
        raise SystemExit(f"no champions found in {HISTORY_DIR}")

    # Import probe: a program that cannot even be imported scores zero. The oracle
    # model is NOT loaded here (the workers each load it once).
    os.environ["KAGG_ORACLE_EAGER"] = "0"
    try:
        load_agent(program_path, "candidate_probe")
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        save_results(results_dir, {"combined_score": 0.0, "public": {"win_rate": 0.0},
                                   "private": {}, "text_feedback": f"IMPORT FAILED: {err}"},
                     correct=False, error=err)
        return 0.0
    os.environ.pop("KAGG_ORACLE_EAGER", None)

    print(f"Candidate : {program_path}")
    print(f"Champions : {len(champions)} in {HISTORY_DIR}")
    print(f"Protocol  : {GAMES_PER_SEAT} seeds seat 0 + {GAMES_PER_SEAT} disjoint seeds seat 1 "
          f"= {2 * GAMES_PER_SEAT} games/champion "
          f"({len(champions) * 2 * GAMES_PER_SEAT} total)")
    print(f"Gate      : {CROWN_THRESHOLD:.0%} overall win rate", flush=True)

    # ---- build the full task list: seat 0 block, then seat 1 block, per champion
    tasks, meta = [], []
    for champ in champions:
        for seed in SEAT0_SEEDS:
            tasks.append((program_path, str(champ), seed))
            meta.append((champ.name, 0, seed))
        for seed in SEAT1_SEEDS:
            tasks.append((str(champ), program_path, seed))
            meta.append((champ.name, 1, seed))

    t_start = time.time()
    raw = play_all(tasks, workers)

    # ---- tally
    per_champ = {c.name: {"w": 0, "l": 0, "t": 0, "s0w": 0, "s1w": 0,
                          "cand_cash": [], "champ_cash": [], "crashes": 0}
                 for c in champions}
    wins = losses = ties = crashes = 0
    cand_cash_all = []
    log_rows = []
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    for (champ_name, seat, seed), res in zip(meta, raw):
        r0, r1, c0, c1 = res
        # map seat-relative results back to candidate / champion
        cand_r, champ_r = (r0, r1) if seat == 0 else (r1, r0)
        cand_c, champ_c = (c0, c1) if seat == 0 else (c1, c0)
        rec = per_champ[champ_name]
        if cand_c or champ_c:
            crashes += 1
            rec["crashes"] += 1
            # a crashed game counts as a loss for the crasher, never as a win
            if cand_c:
                losses += 1
                rec["l"] += 1
            continue
        rec["cand_cash"].append(cand_r)
        rec["champ_cash"].append(champ_r)
        cand_cash_all.append(cand_r)
        if cand_r > champ_r:
            wins += 1
            rec["w"] += 1
            rec["s0w" if seat == 0 else "s1w"] += 1
        elif cand_r < champ_r:
            losses += 1
            rec["l"] += 1
        else:
            ties += 1
            rec["t"] += 1
        log_rows.append({"ts": stamp, "kind": "candidate_vs_champion", "champion": champ_name,
                         "seat": seat, "seed": seed, "cand": cand_r, "champ": champ_r})

    decided = wins + losses + ties
    total = wins + losses + ties  # ties are played games and count in the denominator
    win_rate = wins / total if total else 0.0
    seat0_total = len(champions) * GAMES_PER_SEAT
    seat1_total = len(champions) * GAMES_PER_SEAT
    seat0_wins = sum(r["s0w"] for r in per_champ.values())
    seat1_wins = sum(r["s1w"] for r in per_champ.values())
    avg_cash = sum(cand_cash_all) / len(cand_cash_all) if cand_cash_all else 0.0

    # ---- starter sanity
    starter_cash, starter_crash = 0.0, False
    try:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=min(2, max(1, workers)), mp_context=ctx) as pool:
            outs = list(pool.map(run_starter_worker, [(program_path, s) for s in STARTER_SEEDS]))
        starter_cash = sum(o[0] for o in outs) / len(outs)
        starter_crash = any(o[1] for o in outs)
    except Exception:
        starter_crash = True

    # ---- crowning
    passed = (win_rate >= CROWN_THRESHOLD) and crashes == 0 and not starter_crash
    crowned_as = None
    if passed and not (no_induct or NO_INDUCT):
        try:
            crowned_as = crown(program_path, win_rate, avg_cash, len(champions))
        except Exception as exc:
            print(f"crowning failed: {exc}", file=sys.stderr)

    # ---- game log
    try:
        GAME_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(GAME_LOG, "a") as fh:
            for row in log_rows:
                fh.write(json.dumps(row) + "\n")
    except OSError:
        pass

    # ---- report
    if crowned_as:
        result_label = "PASSED (crowned)"
    elif passed:
        result_label = "PASSED gate (crowning disabled)"
    else:
        result_label = "below gate"
    lines = [
        f"=== DIGITAL RED QUEEN TOURNAMENT ({2 * GAMES_PER_SEAT} MATCHES PER CHAMPION) ===",
        f"Protocol: {GAMES_PER_SEAT} seeds on seat 0 + {GAMES_PER_SEAT} disjoint seeds on seat 1.",
        f"Result: {result_label} (gate >= {CROWN_THRESHOLD:.0%})",
        f"Overall Win Rate vs {len(champions)} Champions: {win_rate:.2%} "
        f"({wins}W - {losses}L - {ties}T across {total} games)",
        f"Average Cash vs Champions: ${avg_cash:,.0f}",
        f"Starter Sanity: ${starter_cash:,.0f}" + ("  [CRASHED]" if starter_crash else ""),
        f"Seat split: seat0 {seat0_wins}/{seat0_total} "
        f"({seat0_wins / seat0_total:.1%}) | seat1 {seat1_wins}/{seat1_total} "
        f"({seat1_wins / seat1_total:.1%})",
        f"Crashes: {crashes}",
    ]
    if crowned_as:
        lines.append(f"CROWNED into the pool as {crowned_as}")
    lines.append("")
    lines.append(f"Breakdown against individual champions ({2 * GAMES_PER_SEAT} matches each):")
    for champ in champions:
        r = per_champ[champ.name]
        cc = sum(r["cand_cash"]) / len(r["cand_cash"]) if r["cand_cash"] else 0.0
        hc = sum(r["champ_cash"]) / len(r["champ_cash"]) if r["champ_cash"] else 0.0
        lines.append(
            f"  - {champ.name:<36}: {r['w']:2d}W - {r['l']:2d}L - {r['t']:2d}T "
            f"(s0 {r['s0w']}/{GAMES_PER_SEAT}, s1 {r['s1w']}/{GAMES_PER_SEAT}) "
            f"| Cand ${cc:,.0f} vs Champ ${hc:,.0f}"
        )
    text_feedback = "\n".join(lines)
    print(text_feedback, flush=True)

    metrics = {
        "combined_score": round(win_rate, 5),
        "public": {
            "win_rate": round(win_rate, 4),
            "seat0_win_rate": round(seat0_wins / seat0_total, 4) if seat0_total else 0.0,
            "seat1_win_rate": round(seat1_wins / seat1_total, 4) if seat1_total else 0.0,
            "avg_cash": round(avg_cash, 1),
            "wins": wins, "losses": losses, "ties": ties,
            "champions_evaluated": len(champions),
            "matches_per_champion": 2 * GAMES_PER_SEAT,
            "total_crashes": crashes,
        },
        "private": {
            "crowned": bool(crowned_as),
            "crowned_as": crowned_as,
            "gate": CROWN_THRESHOLD,
            "starter_cash": round(starter_cash, 1),
            "starter_crash": starter_crash,
            "seat0_seeds": SEAT0_SEEDS,
            "seat1_seeds": SEAT1_SEEDS,
            "wall_seconds": round(time.time() - t_start, 1),
            "per_champion": {k: {kk: vv for kk, vv in v.items()
                                 if kk not in ("cand_cash", "champ_cash")}
                             for k, v in per_champ.items()},
        },
        "text_feedback": text_feedback,
    }
    # A program is "correct" when it ran every game without crashing - that is what
    # Shinka uses to decide whether the program is usable as a parent at all.
    save_results(results_dir, metrics, correct=(crashes == 0 and not starter_crash), error=None)
    return win_rate


def save_results(results_dir, metrics, correct, error=None):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
    with open(results_dir / "correct.json", "w") as fh:
        json.dump({"correct": bool(correct), "error": error}, fh, indent=2)


def main():
    ap = argparse.ArgumentParser(description="Kaggriculture DRQ evaluator")
    ap.add_argument("--program_path", required=True)
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--workers", type=int,
                    default=int(os.environ.get("KAGG_EVAL_WORKERS", "12")))
    ap.add_argument("--no-induct", action="store_true",
                    help="evaluate without crowning into the pool")
    args = ap.parse_args()
    try:
        evaluate_candidate(args.program_path, args.results_dir,
                           workers=args.workers, no_induct=args.no_induct)
    except SystemExit:
        raise
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        save_results(args.results_dir,
                     {"combined_score": 0.0, "public": {"win_rate": 0.0}, "private": {},
                      "text_feedback": f"EVALUATION FAILED: {err}"},
                     correct=False, error=err)


if __name__ == "__main__":
    main()
