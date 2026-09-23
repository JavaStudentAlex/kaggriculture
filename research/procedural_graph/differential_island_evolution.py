"""Differential Multi-Island Procedural Graph Evolutionary Pipeline.

Features:
- High-diversity 100-match differential gauntlet:
  * 50 matches on Seat 0 across 50 disjoint seeds
  * 50 matches on Seat 1 across 50 disjoint seeds
- 4 Parallel Evolutionary Islands:
  * Island 0: Island-Capital (Days 0-10 Workforce Sizing, Day-0 Hires, Livestock Mix)
  * Island 1: Island-Arbitrage (Town Shop Preemption & Scarcity Capture)
  * Island 2: Island-Watering (Late Harvest Freeze & Continuous Hydration Loops)
  * Island 3: Island-Liquidation (Terminal Fire Sale & Midnight Solvency Defense)
- Multi-Armed Bandit (MAB):
  * Dynamic UCB1 selection across 6 frontier LLM mutators
- Two-Level Novelty Filter:
  * Level 1: SentenceTransformers (MiniLM-L6-v2) embedding similarity (<0.985)
  * Level 2: Gemini 3.1 Pro Preview (xhigh) structural judge
- Periodic Cross-Island Synthesis:
  * Meta-Supervisor (gpt-6-astra, xhigh) merges top policy subgraphs every N iterations
- Persistent Telemetry & Watchdog Reporting:
  * Writes real-time stats to differential_evo_status.json
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
from pathlib import Path
import random
import shutil
import signal
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

CURRENT_DIR = Path(__file__).resolve().parent
REPO_DIR = CURRENT_DIR.parent.parent
EVO_DIR = REPO_DIR / "shinka" / "evolution"
POOL_DIR = REPO_DIR / "shinka" / "champions" / "pool"

for p in [str(CURRENT_DIR), str(REPO_DIR), str(EVO_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import evaluate
from bam_graph_mutator import BAMGraphMutator
from kaggriculture_domain_knowledge import build_jev_knowledge_summary
from shinka_graph_mab import UCB1Bandit
from shinka_graph_novelty import cosine_similarity, get_graph_embedding, judge_novelty_with_llm
from shinka_graph_supervisor import run_meta_supervisor

_RUNNING = True


def sigterm_handler(signum, frame):
    global _RUNNING
    print("\n[SIGTERM/SIGINT] Gracefully shutting down after current evaluation...", flush=True)
    _RUNNING = False


# Disjoint seed blocks for differential play
SEEDS_SEAT0 = [101 * (i + 1) for i in range(50)]
SEEDS_SEAT1 = [70001 + 101 * (i + 1) for i in range(50)]


def build_candidate_agent_wrapper(cand_graph_path: Path, out_path: Path) -> Path:
    """Generates an importable agent module bound to the candidate graph."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    code = f'''import os
import sys
from pathlib import Path

REPO = Path("{REPO_DIR}")
sys.path.extend([str(REPO), str(REPO / "shinka" / "evolution"), str(REPO / "research" / "procedural_graph")])

os.environ["KAGG_GRAPH_PATH"] = "{cand_graph_path.resolve()}"
import agent_graph
agent_graph._ENGINE = None
agent = agent_graph.agent
'''
    out_path.write_text(code, encoding="utf-8")
    return out_path


def run_differential_gauntlet(
    cand_graph_path: Path,
    champ_path: Path,
    run_dir: Path,
    workers: int = 4,
    tag: str = "eval",
    status_callback: Optional[Any] = None
) -> Dict[str, Any]:
    """Runs 50 matches on Seat 0 and 50 matches on Seat 1 against champion."""
    wrapper_path = run_dir / f"wrapper_{tag}.py"
    build_candidate_agent_wrapper(cand_graph_path, wrapper_path)

    # 50 matches on Seat 0: candidate is player 0, champion is player 1
    tasks_seat0 = [(str(wrapper_path), str(champ_path), s) for s in SEEDS_SEAT0]
    # 50 matches on Seat 1: champion is player 0, candidate is player 1
    tasks_seat1 = [(str(champ_path), str(wrapper_path), s) for s in SEEDS_SEAT1]

    t0 = time.perf_counter()
    all_tasks = tasks_seat0 + tasks_seat1
    results: List[Any] = [None] * len(all_tasks)

    # Memory-safe ProcessPoolExecutor with max_tasks_per_child=1 to prevent leaks
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing
    ctx = multiprocessing.get_context("spawn")

    done = 0
    with ProcessPoolExecutor(max_workers=max(1, workers), mp_context=ctx, max_tasks_per_child=1) as pool:
        futs = {pool.submit(evaluate.run_single_match_worker, t): idx for idx, t in enumerate(all_tasks)}
        for fut in as_completed(futs):
            idx = futs[fut]
            try:
                results[idx] = fut.result(timeout=600)
            except Exception:
                results[idx] = (0.0, 0.0, True, True)
            done += 1
            if done % 5 == 0 or done == len(all_tasks):
                el = time.perf_counter() - t0
                rate = done / max(0.1, el)
                eta = (len(all_tasks) - done) / max(0.01, rate)
                logging.info(f"   [{done:03d}/100 matches] {el:,.0f}s elapsed | ETA {eta:,.0f}s ({rate*60:.1f} matches/min)")
                if status_callback:
                    status_callback(done, len(all_tasks), el, eta)

    elapsed = time.perf_counter() - t0

    res_seat0 = results[:50]
    res_seat1 = results[50:]

    s0_wins = 0
    s0_cand_cash: List[float] = []
    s0_opp_cash: List[float] = []
    losses: List[Dict[str, Any]] = []

    for i, res in enumerate(res_seat0):
        if res is None:
            r0, r1 = 0.0, 0.0
        else:
            r0, r1, _, _ = res
        s0_cand_cash.append(r0)
        s0_opp_cash.append(r1)
        if r0 > r1:
            s0_wins += 1
        else:
            losses.append({
                "seat": 0,
                "seed": SEEDS_SEAT0[i],
                "cand_cash": r0,
                "opp_cash": r1,
                "cash_deficit": round(r1 - r0, 2)
            })

    s1_wins = 0
    s1_cand_cash: List[float] = []
    s1_opp_cash: List[float] = []

    for i, res in enumerate(res_seat1):
        if res is None:
            r0, r1 = 0.0, 0.0
        else:
            r0, r1, _, _ = res
        # Seat 1: candidate is r1, opp is r0
        s1_cand_cash.append(r1)
        s1_opp_cash.append(r0)
        if r1 > r0:
            s1_wins += 1
        else:
            losses.append({
                "seat": 1,
                "seed": SEEDS_SEAT1[i],
                "cand_cash": r1,
                "opp_cash": r0,
                "cash_deficit": round(r0 - r1, 2)
            })

    losses.sort(key=lambda x: x["cash_deficit"], reverse=True)

    mean_s0_cand = sum(s0_cand_cash) / max(1, len(s0_cand_cash))
    mean_s0_opp = sum(s0_opp_cash) / max(1, len(s0_opp_cash))
    mean_s1_cand = sum(s1_cand_cash) / max(1, len(s1_cand_cash))
    mean_s1_opp = sum(s1_opp_cash) / max(1, len(s1_opp_cash))

    total_wins = s0_wins + s1_wins
    overall_cand_cash = (mean_s0_cand + mean_s1_cand) / 2.0
    overall_opp_cash = (mean_s0_opp + mean_s1_opp) / 2.0
    overall_win_rate = total_wins / 100.0

    return {
        "champ_name": champ_path.name,
        "total_matches": 100,
        "elapsed_sec": round(elapsed, 1),
        "seat0_wins": s0_wins,
        "seat0_win_rate": round(s0_wins / 50.0, 3),
        "seat0_mean_cand_cash": round(mean_s0_cand, 2),
        "seat0_mean_opp_cash": round(mean_s0_opp, 2),
        "seat1_wins": s1_wins,
        "seat1_win_rate": round(s1_wins / 50.0, 3),
        "seat1_mean_cand_cash": round(mean_s1_cand, 2),
        "seat1_mean_opp_cash": round(mean_s1_opp, 2),
        "total_wins": total_wins,
        "overall_win_rate": round(overall_win_rate, 3),
        "overall_cand_cash": round(overall_cand_cash, 2),
        "overall_opp_cash": round(overall_opp_cash, 2),
        "cash_delta": round(overall_cand_cash - overall_opp_cash, 2),
        "worst_losses": losses[:5]
    }


class EvolutionaryIsland:
    def __init__(self, island_id: int, name: str, focus_theme: str, initial_graph: Dict[str, Any]):
        self.island_id = island_id
        self.name = name
        self.focus_theme = focus_theme
        self.champion_graph = copy.deepcopy(initial_graph)
        self.champion_score: Dict[str, Any] = {}
        self.history: List[Dict[str, Any]] = []
        self.rejections: List[Dict[str, Any]] = []
        self.embeddings_archive: List[List[float]] = [get_graph_embedding(initial_graph)]


def main():
    parser = argparse.ArgumentParser(description="Differential Multi-Island Procedural Graph Evolution")
    parser.add_argument("--iterations", type=int, default=200, help="Total evolutionary iterations")
    parser.add_argument("--workers", type=int, default=4, help="Parallel match worker processes")
    parser.add_argument("--supervisor_interval", type=int, default=8, help="Iterations between cross-island merges")
    parser.add_argument("--run_dir", type=Path, default=CURRENT_DIR / "differential_evo_run")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, sigterm_handler)
    signal.signal(signal.SIGTERM, sigterm_handler)

    run_dir = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    status_file = CURRENT_DIR / "differential_evo_status.json"
    log_file = CURRENT_DIR / "differential_evo.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler(sys.stdout)]
    )

    logging.info("=" * 70)
    logging.info("LAUNCHING 200-ITERATION DIFFERENTIAL MULTI-ISLAND EVOLUTION")
    logging.info(f"Workers: {args.workers} CPU cores | Supervisor Interval: {args.supervisor_interval}")
    logging.info("Gauntlet: 50 Matches Seat 0 + 50 Matches Seat 1 (100 Matches/Evaluation)")
    logging.info("=" * 70)

    champs = evaluate.list_champions()
    if not champs:
        raise FileNotFoundError(f"No champions found in {POOL_DIR}")
    primary_champ = champs[0]
    logging.info(f"Primary Benchmark Champion: {primary_champ.name}")

    base_graph_path = CURRENT_DIR / "policy_graph.json"
    master_graph = json.loads(base_graph_path.read_text(encoding="utf-8"))

    def baseline_cb(done, total, el, eta):
        payload = {
            "status": "running",
            "current_iteration": 0,
            "total_iterations": args.iterations,
            "active_island": "Baseline Gauntlet Evaluation",
            "benchmark_champion": primary_champ.name,
            "live_gauntlet_progress": f"{done}/{total} matches ({done/total*100:.1f}%) | {el:.0f}s elapsed | ETA {eta:.0f}s",
            "last_updated_unix": time.time()
        }
        status_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Initial gauntlet on baseline graph
    logging.info("Evaluating baseline graph on 100-match differential gauntlet...")
    base_eval = run_differential_gauntlet(base_graph_path, primary_champ, run_dir, workers=args.workers, tag="baseline", status_callback=baseline_cb)
    logging.info(
        f"Baseline Results -> Overall Win Rate: {base_eval['overall_win_rate']*100:.1f}% "
        f"(Seat 0: {base_eval['seat0_win_rate']*100:.1f}%, Seat 1: {base_eval['seat1_win_rate']*100:.1f}%) | "
        f"Cash: ${base_eval['overall_cand_cash']:,.2f} vs Champ: ${base_eval['overall_opp_cash']:,.2f} "
        f"in {base_eval['elapsed_sec']}s"
    )

    islands_def = [
        (0, "Island-Capital", "Days 0-10 Workforce Sizing, Day-0 Hires, and Initial Seed Ratios"),
        (1, "Island-Arbitrage", "Town Shop Preemption, Contract Fulfillment & Scarcity Capture"),
        (2, "Island-Watering", "Closed-Loop Strawberry Hydration, Moisture Maintenance & Zero Crop Waste"),
        (3, "Island-Liquidation", "Terminal Horizon Fire Sale (Steps 696-720) & Shed Inventory Zeroing"),
        (4, "Island-Solvency", "Midnight Wage Liability Defense & Hour >= 18 Cash Reserve Enforcement"),
        (5, "Island-OracleFrontrun", "Predictive Opponent Counter-Play & TTM 96-Step Glut Frontrunning"),
        (6, "Island-OrderCap", "10-Order Ceiling Dynamic Queue Arbitration & Slot Congestion Defense"),
        (7, "Island-Livestock", "Animal Compounding, 8 Cow / 6 Sheep Acquisition Curves & Feed Balance")
    ]
    islands = [EvolutionaryIsland(i[0], i[1], i[2], master_graph) for i in islands_def]
    for isl in islands:
        isl.champion_score = base_eval

    bandit = UCB1Bandit(CURRENT_DIR / "shinka_evo_run" / "bandit_state.json")
    mutator = BAMGraphMutator()
    domain_knowledge = build_jev_knowledge_summary()

    def write_status(current_iter: int, active_isl: str, extra: Dict[str, Any] | None = None):
        payload = {
            "status": "running" if _RUNNING else "stopped",
            "current_iteration": current_iter,
            "total_iterations": args.iterations,
            "active_island": active_isl,
            "benchmark_champion": primary_champ.name,
            "baseline_score": {
                "win_rate": base_eval["overall_win_rate"],
                "cand_cash": base_eval["overall_cand_cash"],
                "opp_cash": base_eval["overall_opp_cash"],
                "seat0_win_rate": base_eval["seat0_win_rate"],
                "seat1_win_rate": base_eval["seat1_win_rate"]
            },
            "islands": [
                {
                    "name": isl.name,
                    "focus": isl.focus_theme,
                    "champion_win_rate": isl.champion_score.get("overall_win_rate", 0.0),
                    "champion_cash": isl.champion_score.get("overall_cand_cash", 0.0),
                    "accepted_count": len(isl.history),
                    "rejected_count": len(isl.rejections)
                } for isl in islands
            ],
            "bandit_arms": {
                m: {
                    "calls": bandit.state.get(m, {}).get("pulls", 0),
                    "avg_reward": round(bandit.state.get(m, {}).get("total_reward", 0.0) / max(1, bandit.state.get(m, {}).get("pulls", 0)), 3),
                    "crowns": bandit.state.get(m, {}).get("crowns", 0)
                } for m in bandit.arms
            },
            "last_updated_unix": time.time()
        }
        if extra:
            payload["latest_event"] = extra
        status_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    write_status(0, "Initialization", {"event": "Baseline gauntlet evaluated", "baseline": base_eval})

    iteration = 1
    while _RUNNING and iteration <= args.iterations:
        isl_idx = (iteration - 1) % len(islands)
        island = islands[isl_idx]

        logging.info("-" * 65)
        logging.info(f"ITERATION {iteration}/{args.iterations} | {island.name} ({island.focus_theme})")

        # 1. Select MAB reasoning arm
        selected_model = bandit.select_arm()
        logging.info(f"-> UCB1 MAB selected mutator arm: [{selected_model}]")

        # 2. Extract diagnostics from worst losses
        diagnostics = {
            "island_id": island.island_id,
            "island_name": island.name,
            "island_focus": island.focus_theme,
            "current_champ_win_rate": island.champion_score.get("overall_win_rate", 0.0),
            "current_champ_cash": island.champion_score.get("overall_cand_cash", 0.0),
            "worst_differential_losses": island.champion_score.get("worst_losses", [])
        }

        # 3. Generate candidate mutation
        t_mut0 = time.time()
        try:
            cand_graph, rationale = mutator.mutate_graph(
                model_name=selected_model,
                current_graph=island.champion_graph,
                match_diagnostics=diagnostics,
                decision_trace=diagnostics["worst_differential_losses"],
                domain_knowledge=domain_knowledge,
                rejections=[r.get("cause", "") for r in island.rejections[-5:]]
            )
        except Exception as exc:
            logging.error(f"Mutation call failed with {selected_model}: {exc}")
            bandit.update(selected_model, reward=0.0, crowned=False)
            iteration += 1
            continue

        mut_dt = time.time() - t_mut0
        logging.info(f"<- Model [{selected_model}] synthesized candidate in {mut_dt:.1f}s.")
        logging.info(f"   Rationale: {rationale[:140]}...")

        # 4. Two-Level Novelty Check
        cand_emb = get_graph_embedding(cand_graph)
        max_sim = -1.0
        collided_idx = -1
        for idx, p_emb in enumerate(island.embeddings_archive):
            sim = cosine_similarity(cand_emb, p_emb)
            if sim > max_sim:
                max_sim = sim
                collided_idx = idx

        logging.info(f"   [Novelty Level 1] Max Cosine Similarity: {max_sim:.4f}")
        is_novel = True
        novelty_reason = "Embedding structurally unique"

        if max_sim >= 0.985:
            logging.info(f"   [Novelty Level 2] Embedding collision detected ({max_sim:.4f} >= 0.985). Calling Judge...")
            collided_g = island.history[collided_idx]["graph"] if collided_idx < len(island.history) else island.champion_graph
            is_novel, novelty_reason = judge_novelty_with_llm(cand_graph, collided_g, max_sim)
            logging.info(f"   [Novelty Level 2 Verdict]: is_novel={is_novel} | {novelty_reason[:100]}")

        if not is_novel:
            logging.warning(f"   [REJECTED] Mutation lacks strategic novelty. Reason: {novelty_reason}")
            island.rejections.append({"model": selected_model, "cause": novelty_reason, "rationale": rationale})
            bandit.update(selected_model, reward=0.0, crowned=False)
            write_status(iteration, island.name, {"event": "Novelty Rejection", "model": selected_model, "reason": novelty_reason})
            iteration += 1
            continue

        island.embeddings_archive.append(cand_emb)

        # 5. Evaluate novel candidate on 100-match differential gauntlet
        cand_graph_path = run_dir / f"cand_iter_{iteration:03d}_{island.name}.json"
        cand_graph_path.write_text(json.dumps(cand_graph, indent=2), encoding="utf-8")

        logging.info(f"   Running 100-match differential gauntlet (50 Seat 0 + 50 Seat 1) across {args.workers} workers...")
        def cand_cb(done, total, el, eta):
            write_status(iteration, island.name, {
                "event": f"Evaluating candidate: {done}/{total} matches ({done/total*100:.1f}%) | ETA {eta:.0f}s",
                "model": selected_model
            })

        cand_eval = run_differential_gauntlet(cand_graph_path, primary_champ, run_dir, workers=args.workers, tag=f"iter_{iteration}", status_callback=cand_cb)

        win_rate = cand_eval["overall_win_rate"]
        cand_cash = cand_eval["overall_cand_cash"]
        prev_win_rate = island.champion_score.get("overall_win_rate", 0.0)
        prev_cash = island.champion_score.get("overall_cand_cash", 0.0)

        logging.info(
            f"   Gauntlet Result: Overall Win Rate {win_rate*100:.1f}% (Seat 0: {cand_eval['seat0_win_rate']*100:.1f}%, "
            f"Seat 1: {cand_eval['seat1_win_rate']*100:.1f}%) | Cash: ${cand_cash:,.2f} vs Opp: ${cand_eval['overall_opp_cash']:,.2f} "
            f"in {cand_eval['elapsed_sec']}s"
        )

        # 6. Gating & Crowning
        is_promoted = False
        if win_rate > prev_win_rate:
            is_promoted = True
        elif win_rate == prev_win_rate and cand_cash > prev_cash + 500.0:
            is_promoted = True

        if is_promoted:
            logging.info(f"🏆 [CROWNED] Candidate graph PROMOTED as new champion of {island.name}!")
            island.champion_graph = cand_graph
            island.champion_score = cand_eval
            island.history.append({
                "iteration": iteration,
                "model": selected_model,
                "graph": cand_graph,
                "eval": cand_eval,
                "rationale": rationale
            })
            bandit.update(selected_model, reward=1.0, crowned=True)
            event_type = "Promoted"
        else:
            logging.info(f"   [NOT PROMOTED] Candidate did not beat island benchmark ({win_rate*100:.1f}% vs {prev_win_rate*100:.1f}%).")
            # Partial reward if it was positive cash
            reward = 0.2 if cand_cash > prev_cash else 0.05
            bandit.update(selected_model, reward=reward, crowned=False)
            event_type = "Evaluated (Not Promoted)"

        write_status(iteration, island.name, {
            "event": event_type,
            "model": selected_model,
            "win_rate": win_rate,
            "cand_cash": cand_cash,
            "promoted": is_promoted
        })

        # 7. Cross-Island Supervisor Merge Cadence
        if iteration % args.supervisor_interval == 0:
            logging.info("=" * 70)
            logging.info(f"🏛️  INVOKING META-SUPERVISOR (gpt-6-astra) MERGE AT ITERATION {iteration}")
            logging.info("=" * 70)
            try:
                supervisor_recs = run_meta_supervisor(
                    best_graph=islands[0].champion_graph,
                    recent_evals=[{"island": isl.name, "score": isl.champion_score} for isl in islands],
                    recent_rejections=[],
                    current_gen=iteration
                )
                logging.info(f"Meta-Supervisor Guidance: {supervisor_recs[:3]}")

                # Update master policy graph
                best_island = max(islands, key=lambda isl: (isl.champion_score.get("overall_win_rate", 0), isl.champion_score.get("overall_cand_cash", 0)))
                base_graph_path.write_text(json.dumps(best_island.champion_graph, indent=2), encoding="utf-8")
                logging.info(f"Updated global master policy_graph.json from {best_island.name} (Win Rate: {best_island.champion_score.get('overall_win_rate', 0)*100:.1f}%)")
            except Exception as e:
                logging.error(f"Supervisor merge encountered error: {e}")

        iteration += 1

    logging.info("200-iteration differential evolution run finished.")


if __name__ == "__main__":
    main()
