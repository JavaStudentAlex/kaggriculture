"""Simulation Match Evaluator for Jev Agent vs Champion Pool.

Executes 720-step Kaggriculture matches against champions, capturing
both seat perspectives, full cash trajectories, and Jev's decision trace.
"""
from __future__ import annotations

import importlib.util
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_DIR = Path(__file__).resolve().parent.parent.parent
EVO_DIR = REPO_DIR / "shinka" / "evolution"
POOL_DIR = REPO_DIR / "shinka" / "champions" / "pool"
PROC_DIR = REPO_DIR / "research" / "procedural_graph"

for p in [str(REPO_DIR), str(EVO_DIR), str(PROC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from kaggle_environments import make
import agent_jev_graph


def load_module_from_path(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_jev_match(
    graph_path: Path,
    champ_path: Optional[Path] = None,
    seed: int = 42,
    cand_seat: int = 0,
    query_interval: int = 3
) -> Dict[str, Any]:
    """Runs a single simulation match between Jev agent and a champion."""
    if champ_path is None:
        champs = sorted(list(POOL_DIR.glob("*.py")))
        if not champs:
            raise FileNotFoundError(f"No champions found in {POOL_DIR}")
        champ_path = champs[0]

    # Initialize Jev actor with candidate graph
    actor = agent_jev_graph.get_actor(graph_path, query_interval=query_interval)
    actor.decision_trace.clear()
    agent_jev_graph.get_fallback_engine(graph_path)

    # Load opponent champion
    opp_mod = load_module_from_path(champ_path)
    opp_agent = opp_mod.agent

    env = make("kaggriculture", configuration={"randomSeed": seed}, debug=False)
    players = [agent_jev_graph.agent, opp_agent] if cand_seat == 0 else [opp_agent, agent_jev_graph.agent]

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

    # Extract final observations
    last_obs = cand_state.observation or {}
    farm = (last_obs.get("farms") or [{}])[cand_seat] if "farms" in last_obs else {}
    shed = (last_obs.get("private") or {}).get("shed", {})

    return {
        "success": True,
        "champ_name": champ_path.name,
        "seed": seed,
        "seat": cand_seat,
        "cand_cash": cand_reward,
        "opp_cash": opp_reward,
        "cash_diff": round(cand_reward - opp_reward, 2),
        "win": win,
        "tie": tie,
        "outcome": "WIN" if win else ("TIE" if tie else "LOSS"),
        "elapsed_sec": round(elapsed, 2),
        "hands_count": len(farm.get("hands") or []),
        "shed_items": sum(int(v or 0) for v in shed.values()),
        "jev_queries_count": len(actor.decision_trace),
        "decision_trace": list(actor.decision_trace)
    }
