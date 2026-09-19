"""Evaluation harness for candidate procedural graphs.

Runs simulation matches against the champions in shinka/champions/pool/.
Captures win/loss rates, average cash, and detailed loss telemetry for LLM mutation prompts.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path
from typing import Any

KAGG_DIR = Path(__file__).resolve().parent.parent.parent
POOL_DIR = KAGG_DIR / "shinka" / "champions" / "pool"
EVO_DIR = KAGG_DIR / "shinka" / "evolution"

if str(EVO_DIR) not in sys.path:
    sys.path.insert(0, str(EVO_DIR))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import agent_graph
from graph_engine import ProceduralGraphEngine
from kaggle_environments import make


def load_module_from_path(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_single_match(args: tuple[str, str, int, int, str]) -> dict[str, Any]:
    """Runs a single simulation game: (candidate_graph_path, champ_path, seed, candidate_seat, champ_name)."""
    cand_graph_path, champ_path, seed, cand_seat, champ_name = args

    try:
        # Load candidate with its specific graph
        engine = ProceduralGraphEngine(Path(cand_graph_path))
        
        # Local agent closure with this engine
        def cand_agent(obs, config=None):
            agent_graph._ENGINE = engine
            return agent_graph.agent(obs, config)

        # Load opponent
        opp_mod = load_module_from_path(Path(champ_path))
        opp_agent = opp_mod.agent

        env = make("kaggriculture", configuration={"randomSeed": seed}, debug=False)
        players = [cand_agent, opp_agent] if cand_seat == 0 else [opp_agent, cand_agent]

        t0 = time.perf_counter()
        env.run(players)
        elapsed = time.perf_counter() - t0

        state = env.state
        cand_state = state[cand_seat]
        opp_state = state[1 - cand_seat]

        cand_reward = float(cand_state.reward or 0.0)
        opp_reward = float(opp_state.reward or 0.0)
        win = cand_reward > opp_reward
        tie = cand_reward == opp_reward

        loss_telemetry = None
        if not win and not tie:
            # Extract high-level loss summary from final step
            last_obs = state[cand_seat].observation or {}
            farm = (last_obs.get("farms") or [{}])[cand_seat] if "farms" in last_obs else {}
            shed = (last_obs.get("private") or {}).get("shed", {})
            loss_telemetry = {
                "champ": champ_name,
                "seed": seed,
                "seat": cand_seat,
                "cand_cash": cand_reward,
                "opp_cash": opp_reward,
                "diff": cand_reward - opp_reward,
                "n_hands": len(farm.get("hands") or []),
                "shed_items": sum(int(v or 0) for v in shed.values())
            }

        return {
            "success": True,
            "win": win,
            "tie": tie,
            "cand_reward": cand_reward,
            "opp_reward": opp_reward,
            "elapsed": elapsed,
            "loss_telemetry": loss_telemetry,
            "status": cand_state.status
        }
    except Exception as exc:
        return {
            "success": False,
            "error": f"{type(exc).__name__}: {exc}",
            "win": False,
            "tie": False,
            "cand_reward": 0.0,
            "opp_reward": 0.0,
            "elapsed": 0.0,
            "loss_telemetry": None,
            "status": "ERROR"
        }


def evaluate_graph(
    graph_path: Path,
    num_seeds_per_champ: int = 2,
    num_workers: int = 8
) -> dict[str, Any]:
    """Evaluates a graph across champions and seeds."""
    champs = sorted(list(POOL_DIR.glob("*.py")))
    if not champs:
        raise RuntimeError(f"No champions found in {POOL_DIR}")

    tasks = []
    base_seed = 1000

    # Pair each champion against both seat 0 and seat 1 on distinct seeds
    for c_idx, champ in enumerate(champs):
        for s_idx in range(num_seeds_per_champ):
            seed = base_seed + c_idx * 100 + s_idx
            # Alternating seats
            for seat in (0, 1):
                tasks.append((str(graph_path), str(champ), seed, seat, champ.name))

    random.seed(42)
    random.shuffle(tasks)

    t0 = time.perf_counter()
    with mp.Pool(processes=min(num_workers, len(tasks))) as pool:
        results = pool.map(run_single_match, tasks)
    total_time = time.perf_counter() - t0

    wins = sum(1 for r in results if r["win"])
    ties = sum(1 for r in results if r["tie"])
    losses = sum(1 for r in results if not r["win"] and not r["tie"] and r["success"])
    crashes = sum(1 for r in results if not r["success"] or r["status"] == "ERROR")
    total_games = len(results)

    cand_rewards = [r["cand_reward"] for r in results if r["success"]]
    opp_rewards = [r["opp_reward"] for r in results if r["success"]]

    avg_cash = sum(cand_rewards) / len(cand_rewards) if cand_rewards else 0.0
    avg_opp_cash = sum(opp_rewards) / len(opp_rewards) if opp_rewards else 0.0
    win_rate = wins / total_games if total_games > 0 else 0.0

    loss_traces = [r["loss_telemetry"] for r in results if r.get("loss_telemetry") is not None]
    loss_traces.sort(key=lambda x: x["diff"])

    return {
        "graph_path": str(graph_path),
        "total_games": total_games,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "crashes": crashes,
        "win_rate": round(win_rate, 4),
        "avg_cash": round(avg_cash, 2),
        "avg_opp_cash": round(avg_opp_cash, 2),
        "cash_diff": round(avg_cash - avg_opp_cash, 2),
        "total_time_sec": round(total_time, 2),
        "worst_losses": loss_traces[:5]
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path, default=Path(__file__).resolve().parent / "policy_graph.json")
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    print(f"Evaluating graph: {args.graph} with {args.seeds} seeds/champ across {args.workers} workers...")
    res = evaluate_graph(args.graph, num_seeds_per_champ=args.seeds, num_workers=args.workers)
    print(json.dumps(res, indent=2))
