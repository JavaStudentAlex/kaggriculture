"""High-CPU Local Multi-Island Procedural Graph Evolutionary Pipeline.

Optimized for 64-128 core compute instances (e.g. n2d-highcpu-80 on Brev).
Features:
- At least 20 matches per seat against every champion in the growing pool.
- 80 concurrent multiprocessing workers (ProcessPoolExecutor).
- UCB1 Multi-Armed Bandit selecting from frontier LLM mutators via local proxy (:8317).
- Two-stage SIFT tournament; Qwen3-Embedding 8B/Ollama + semantic novelty judge.
- Atomic full-state resume; legacy status-only resumes explicitly reinitialize.
- Periodic cross-island Meta-Supervisor merge (gpt-6-astra).
"""
from __future__ import annotations

import os
import sys

# Critical for high-core CPU multiprocessing: restrict each process to 1 BLAS/OpenMP thread
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import copy
import json
import logging
import math
import multiprocessing
from pathlib import Path
import random
import shutil
import signal
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
from pool_upgrade_state import (
    atomic_json, comparable_score, evaluation_fingerprint, graph_fingerprint,
    legacy_resume, load_checkpoint, rebase_incumbents, save_checkpoint, valid_score,
)
from sift_graph_judge import SIFTGraphJudge
from qwen_novelty_archive import restore_archives, register_graph
from shinka_graph_novelty import EMBEDDING_IDENTITY, OLLAMA_EMBED_MODEL

# SIFT parameters (from MIT+Sakana AI paper arXiv:2609.19526)
SIFT_CANDIDATES_PER_ITERATION = 3   # Generate N candidates, judge pairwise, push only winner
SIFT_MUTATION_RETRIES = 3           # Self-healing: retry failed mutations with error feedback

_RUNNING = True


def sigterm_handler(signum, frame):
    global _RUNNING
    print("\n[SIGTERM/SIGINT] Gracefully shutting down after current evaluation...", flush=True)
    _RUNNING = False


# Disjoint seed blocks
SEEDS_SEAT0 = [101 * (i + 1) for i in range(100)]
SEEDS_SEAT1 = [70001 + 101 * (i + 1) for i in range(100)]


def current_evaluation_fingerprint(champions, seeds_per_champ_seat):
    if not 20 <= seeds_per_champ_seat <= min(len(SEEDS_SEAT0), len(SEEDS_SEAT1)):
        raise ValueError("seeds_per_champ_seat must be between 20 and 100")
    code_files = {Path(__file__), Path(evaluate.__file__),
                  CURRENT_DIR / "pool_upgrade_state.py",
                  CURRENT_DIR / "agent_graph.py", CURRENT_DIR / "graph_engine.py"}
    code_files.update(EVO_DIR.glob("*.py"))
    dependency_dir = REPO_DIR / "shinka" / "champions" / "dependencies"
    code_files.update(dependency_dir.rglob("*.py"))
    code_files.update((REPO_DIR / "research" / "opponent_model").glob("*.py"))
    code_files.update(p for p in (EVO_DIR / "checkpoint").rglob("*") if p.is_file())
    from importlib.metadata import distribution, PackageNotFoundError
    engine_version = "unavailable"
    try:
        engine = distribution("kaggle-environments")
        engine_version = engine.version
        code_files.update(Path(engine.locate_file(p)) for p in (engine.files or [])
                          if "kaggriculture" in str(p) and str(p).endswith((".py", ".json")))
    except PackageNotFoundError:
        pass
    return evaluation_fingerprint(
        champions, SEEDS_SEAT0[:seeds_per_champ_seat], SEEDS_SEAT1[:seeds_per_champ_seat],
        code_files, {"episode_steps": evaluate.EPISODE_STEPS,
                     "engine_version": engine_version,
                     "seeds_per_champ_seat": seeds_per_champ_seat})


def build_candidate_agent_wrapper(cand_graph_path: Path, out_path: Path) -> Path:
    """Generates an importable agent module bound to the candidate graph."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    code = f'''import os
import sys
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

REPO = Path("{REPO_DIR.resolve()}")
sys.path.extend([str(REPO), str(REPO / "shinka" / "evolution"), str(REPO / "research" / "procedural_graph")])

os.environ["KAGG_GRAPH_PATH"] = "{cand_graph_path.resolve()}"
import agent_graph
agent_graph._ENGINE = None
agent = agent_graph.agent
'''
    out_path.write_text(code, encoding="utf-8")
    return out_path


def run_differential_gauntlet_multi_champ(
    cand_graph_path: Path,
    champions: List[Path],
    run_dir: Path,
    workers: int = 80,
    seeds_per_champ_seat: int = 20,
    tag: str = "eval",
    status_callback: Optional[Any] = None,
    expected_fingerprint: Optional[str] = None,
) -> Dict[str, Any]:
    """Runs gauntlet against all benchmark champions across Seat 0 and Seat 1."""
    fingerprint = current_evaluation_fingerprint(champions, seeds_per_champ_seat)
    if expected_fingerprint is not None and fingerprint != expected_fingerprint:
        raise RuntimeError("Evaluation inputs changed before gauntlet; refusing stale comparison")
    graph_digest = graph_fingerprint(json.loads(cand_graph_path.read_text(encoding="utf-8")))
    wrapper_path = run_dir / f"wrapper_{tag}.py"
    build_candidate_agent_wrapper(cand_graph_path, wrapper_path)

    all_tasks = []
    for c in champions:
        for s in SEEDS_SEAT0[:seeds_per_champ_seat]:
            all_tasks.append((str(wrapper_path), str(c), s, 0, c.name))
        for s in SEEDS_SEAT1[:seeds_per_champ_seat]:
            all_tasks.append((str(c), str(wrapper_path), s, 1, c.name))

    t0 = time.perf_counter()
    results: List[Any] = [None] * len(all_tasks)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    ctx = multiprocessing.get_context("spawn")

    done = 0
    with ProcessPoolExecutor(max_workers=max(1, workers), mp_context=ctx, max_tasks_per_child=1) as pool:
        # evaluate.run_single_match_worker takes (p0, p1, seed)
        futs = {pool.submit(evaluate.run_single_match_worker, (t[0], t[1], t[2])): idx for idx, t in enumerate(all_tasks)}
        for fut in as_completed(futs):
            idx = futs[fut]
            try:
                results[idx] = fut.result(timeout=600)
            except Exception:
                results[idx] = (0.0, 0.0, True, True)
            done += 1
            if done % 10 == 0 or done == len(all_tasks):
                el = time.perf_counter() - t0
                rate = done / max(0.1, el)
                eta = (len(all_tasks) - done) / max(0.01, rate)
                logging.info(f"   [{done:03d}/{len(all_tasks)} matches] {el:,.0f}s elapsed | ETA {eta:,.0f}s ({rate*60:.1f} m/min)")
                if status_callback:
                    status_callback(done, len(all_tasks), el, eta)

    elapsed = time.perf_counter() - t0

    total_wins = 0
    total_cand_cash = []
    total_opp_cash = []
    losses: List[Dict[str, Any]] = []
    crash_count = 0

    for idx, t in enumerate(all_tasks):
        r = results[idx] or (0.0, 0.0, True, True)
        r0, r1, crash0, crash1 = r
        if crash0 or crash1 or not (math.isfinite(r0) and math.isfinite(r1)):
            crash_count += 1
            r0 = r1 = 0.0
        seat = t[3]
        champ_name = t[4]
        seed = t[2]
        c_cash = r0 if seat == 0 else r1
        o_cash = r1 if seat == 0 else r0
        total_cand_cash.append(c_cash)
        total_opp_cash.append(o_cash)

        if (seat == 0 and r0 > r1) or (seat == 1 and r1 > r0):
            total_wins += 1
        else:
            losses.append({
                "champ": champ_name,
                "seat": seat,
                "seed": seed,
                "cand_cash": c_cash,
                "opp_cash": o_cash,
                "deficit": round(o_cash - c_cash, 2)
            })

    losses.sort(key=lambda x: x["deficit"], reverse=True)
    n = max(1, len(all_tasks))
    mean_cand = sum(total_cand_cash) / n
    mean_opp = sum(total_opp_cash) / n
    win_rate = total_wins / n

    # Pool membership, opponent bytes, evaluator and graph must remain unchanged.
    if current_evaluation_fingerprint(evaluate.list_champions(), seeds_per_champ_seat) != fingerprint:
        raise RuntimeError("Evaluation inputs changed during gauntlet; results discarded")
    if graph_fingerprint(json.loads(cand_graph_path.read_text(encoding="utf-8"))) != graph_digest:
        raise RuntimeError("Candidate graph changed during gauntlet; results discarded")
    return {
        "valid": crash_count == 0,
        "crash_count": crash_count,
        "evaluation_fingerprint": fingerprint,
        "graph_fingerprint": graph_digest,
        "elapsed_sec": round(elapsed, 1),
        "total_matches": n,
        "total_wins": total_wins,
        "overall_win_rate": round(win_rate, 3),
        "overall_cand_cash": round(mean_cand, 2),
        "overall_opp_cash": round(mean_opp, 2),
        "cash_delta": round(mean_cand - mean_opp, 2),
        "worst_losses": losses[:8]
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
        self.novelty_graphs = [copy.deepcopy(initial_graph)]
        self.embedding_identity = EMBEDDING_IDENTITY


def main():
    parser = argparse.ArgumentParser(description="High-CPU Multi-Island Procedural Graph Evolution")
    parser.add_argument("--iterations", type=int, default=200, help="Total evolutionary iterations")
    parser.add_argument("--workers", type=int, default=80, help="Parallel match worker processes")
    parser.add_argument("--supervisor_interval", type=int, default=8, help="Iterations between cross-island merges")
    parser.add_argument("--seeds_per_champ_seat", type=int, default=20, help="Seeds per champion per seat (default 20 -> 40 matches per champ across 2 seats)")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from differential_evo_status.json")
    parser.add_argument("--run_dir", type=Path, default=CURRENT_DIR / "highcpu_evo_run")
    args = parser.parse_args()
    if not 20 <= args.seeds_per_champ_seat <= 100:
        parser.error("At least 20 and at most 100 seeds per champion per seat are required")

    signal.signal(signal.SIGINT, sigterm_handler)
    signal.signal(signal.SIGTERM, sigterm_handler)

    run_dir = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    status_file = CURRENT_DIR / "differential_evo_status.json"
    log_file = CURRENT_DIR / "highcpu_evo.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler(sys.stdout)]
    )

    champs = evaluate.list_champions()
    if not champs:
        raise FileNotFoundError(f"No champions found in {POOL_DIR}")

    total_gauntlet_matches = len(champs) * args.seeds_per_champ_seat * 2

    logging.info("=" * 70)
    logging.info("LAUNCHING 80-CORE HIGH-CPU MULTI-ISLAND EVOLUTION")
    logging.info(f"Workers: {args.workers} CPU cores | Champions: {len(champs)}")
    logging.info(f"Gauntlet Size: {total_gauntlet_matches} matches per candidate ({args.seeds_per_champ_seat} Seat 0 + {args.seeds_per_champ_seat} Seat 1 across {len(champs)} champions)")
    logging.info("=" * 70)

    base_graph_path = CURRENT_DIR / "policy_graph.json"
    master_graph = json.loads(base_graph_path.read_text(encoding="utf-8"))

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
    checkpoint_file = run_dir / "pool_upgrade_checkpoint.json"
    start_iteration = 1
    base_eval = {}
    provenance: Dict[str, Any] = {"mode": "fresh_master_initialization"}
    if args.resume and checkpoint_file.exists():
        state, islands = load_checkpoint(checkpoint_file, EvolutionaryIsland)
        start_iteration = state["next_iteration"]
        master_graph = state["master_graph"]
        base_eval = state["baseline_score"]
        provenance = state["provenance"]
        logging.info("Restored full island checkpoint; next iteration %s", start_iteration)
        # The checkpoint is authoritative if a crash interrupted master sync.
        atomic_json(base_graph_path, master_graph)
    else:
        islands = [EvolutionaryIsland(i[0], i[1], i[2], master_graph) for i in islands_def]
        if args.resume and status_file.exists():
            start_iteration, provenance = legacy_resume(json.loads(status_file.read_text(encoding="utf-8")))
            logging.warning("%s; next iteration %s (%s)", provenance["warning"],
                            start_iteration, provenance["iteration_policy"])

    migration = restore_archives(islands, run_dir, start_iteration)
    if migration:
        provenance["qwen_archive_migration"] = migration
    logging.info("Advanced SIFT architecture | novelty model=%s | archive=%s",
                 OLLAMA_EMBED_MODEL, EMBEDDING_IDENTITY)
    active_fingerprint = ""

    def persist(next_iteration):
        save_checkpoint(checkpoint_file, islands, next_iteration, master_graph,
                        base_eval, active_fingerprint, provenance)

    def refresh_incumbents(next_iteration):
        nonlocal champs, total_gauntlet_matches, base_eval, active_fingerprint
        champs = evaluate.list_champions()
        fingerprint = current_evaluation_fingerprint(champs, args.seeds_per_champ_seat)
        total_gauntlet_matches = len(champs) * args.seeds_per_champ_seat * 2
        stale = (not comparable_score(base_eval, master_graph, fingerprint)
                 or any(not comparable_score(i.champion_score, i.champion_graph, fingerprint)
                        for i in islands))
        if stale:
            logging.info("Rebaselining all stale incumbent graphs on pool fingerprint %s", fingerprint)
            def evaluate_graph(graph, tag):
                graph_path = run_dir / f"pool_upgrade_{tag}.json"
                atomic_json(graph_path, graph)
                return run_differential_gauntlet_multi_champ(
                    graph_path, champs, run_dir, workers=args.workers,
                    seeds_per_champ_seat=args.seeds_per_champ_seat, tag=tag,
                    expected_fingerprint=fingerprint)
            new_base = rebase_incumbents(islands, master_graph, base_eval, fingerprint, evaluate_graph)
            if current_evaluation_fingerprint(evaluate.list_champions(), args.seeds_per_champ_seat) != fingerprint:
                raise RuntimeError("Pool changed during rebase; refusing comparisons")
            base_eval = new_base
        active_fingerprint = fingerprint
        persist(next_iteration)

    # Preserve explicit legacy provenance even if the first rebase fails.
    persist(start_iteration)
    atomic_json(status_file, {"status": "rebaselining", "current_iteration": start_iteration,
                             "engine": "Advanced multi-island SIFT 2-Stage Tournament",
                             "embedding_model": OLLAMA_EMBED_MODEL,
                             "seeds_per_champ_seat": args.seeds_per_champ_seat,
                             "total_gauntlet_matches": total_gauntlet_matches,
                             "last_updated_unix": time.time()})
    refresh_incumbents(start_iteration)

    bandit = UCB1Bandit(CURRENT_DIR / "shinka_evo_run" / "bandit_state.json")
    mutator = BAMGraphMutator()
    sift_judge = SIFTGraphJudge()
    domain_knowledge = build_jev_knowledge_summary()

    def write_status(current_iter: int, active_isl: str, extra: Dict[str, Any] | None = None):
        payload = {
            "status": "running" if _RUNNING else "stopped",
            "current_iteration": current_iter,
            "total_iterations": args.iterations,
            "active_island": active_isl,
            "engine": f"Dedicated High-CPU Server ({args.workers} cores, {len(champs)} Champions, SIFT 2-Stage Tournament)",
            "champion_pool_size": len(champs),
            "embedding_model": OLLAMA_EMBED_MODEL,
            "seeds_per_champ_seat": args.seeds_per_champ_seat,
            "total_gauntlet_matches": total_gauntlet_matches,
            "baseline_score": {
                "win_rate": base_eval["overall_win_rate"],
                "cand_cash": base_eval["overall_cand_cash"],
                "opp_cash": base_eval["overall_opp_cash"],
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

    iteration = start_iteration
    while _RUNNING and iteration <= args.iterations:
        isl_idx = (iteration - 1) % len(islands)
        island = islands[isl_idx]

        logging.info("-" * 65)
        logging.info(f"ITERATION {iteration}/{args.iterations} | {island.name} ({island.focus_theme})")

        # 1. Extract diagnostics from worst losses
        diagnostics = {
            "island_id": island.island_id,
            "island_name": island.name,
            "island_focus": island.focus_theme,
            "current_champ_win_rate": island.champion_score.get("overall_win_rate", 0.0),
            "current_champ_cash": island.champion_score.get("overall_cand_cash", 0.0),
            "worst_differential_losses": island.champion_score.get("worst_losses", [])
        }

        # 2. SIFT Stage 1: Generate N candidate mutations from diverse MAB arms (with self-healing retry)
        viable_candidates: list = []  # [(graph, rationale, model_name)]
        iteration_novel_entries: list = []  # [(graph, embedding)]
        selected_models: list = []

        logging.info(f"[SIFT Stage 1] Generating {SIFT_CANDIDATES_PER_ITERATION} candidate mutations for pairwise tournament judging...")

        for cand_idx in range(SIFT_CANDIDATES_PER_ITERATION):
            selected_model = bandit.select_arm()
            # Avoid re-selecting same model within same iteration if possible
            safety_counter = 0
            while selected_model in selected_models and safety_counter < 5:
                selected_model = bandit.select_arm()
                safety_counter += 1
            selected_models.append(selected_model)

            logging.info(f"  [Candidate {cand_idx+1}/{SIFT_CANDIDATES_PER_ITERATION}] UCB1 selected: [{selected_model}]")

            # Self-healing retry loop (capture error, feed back to prompt, retry)
            last_error = None
            cand_graph = None
            cand_rationale = ""
            for attempt in range(SIFT_MUTATION_RETRIES):
                t_mut0 = time.time()
                try:
                    error_context = ""
                    if last_error:
                        error_context = f"\n\n### PREVIOUS MUTATION ATTEMPT FAILED:\n{last_error}\nPlease fix the issue and produce a valid graph mutation."

                    cand_graph, cand_rationale = mutator.mutate_graph(
                        model_name=selected_model,
                        current_graph=island.champion_graph,
                        match_diagnostics=diagnostics,
                        decision_trace=diagnostics["worst_differential_losses"],
                        domain_knowledge=domain_knowledge + error_context,
                        rejections=[r.get("cause", "") for r in island.rejections[-5:] if isinstance(r, dict)]
                    )
                    mut_dt = time.time() - t_mut0
                    logging.info(f"    ✓ [{selected_model}] synthesized candidate in {mut_dt:.1f}s (attempt {attempt+1})")
                    logging.info(f"      Rationale: {cand_rationale[:120]}...")
                    last_error = None
                    break
                except Exception as exc:
                    last_error = str(exc)[:500]
                    logging.warning(f"    ✗ [{selected_model}] attempt {attempt+1}/{SIFT_MUTATION_RETRIES} failed: {last_error[:120]}")

            if cand_graph is None:
                logging.error(f"    [EXHAUSTED] All {SIFT_MUTATION_RETRIES} attempts failed for [{selected_model}]")
                bandit.update(selected_model, reward=0.0, crowned=False)
                continue

            # Reject exact canonical duplicates across all islands and current iteration batch
            cand_digest = graph_fingerprint(cand_graph)
            existing_graphs = [g for other in islands for g in other.novelty_graphs] + [g for g, _ in iteration_novel_entries]
            if any(graph_fingerprint(g) == cand_digest for g in existing_graphs):
                island.rejections.append({"model": selected_model, "cause": "Exact graph duplicate"})
                bandit.update(selected_model, reward=0.0, crowned=False)
                logging.info("    Exact graph duplicate rejected")
                continue

            # Two-Level Novelty Check on each candidate, with aligned graph/vector pairs.
            cand_emb = get_graph_embedding(cand_graph)
            max_sim = -1.0
            collided_g = None
            archive = [(g, e) for other in islands
                       for g, e in zip(other.novelty_graphs, other.embeddings_archive)] + iteration_novel_entries
            for prev_graph, p_emb in archive:
                sim = cosine_similarity(cand_emb, p_emb)
                if sim > max_sim:
                    max_sim = sim
                    collided_g = prev_graph

            logging.info(f"    [Candidate {cand_idx+1}] Max Cosine Similarity: {max_sim:.4f}")
            is_novel = True
            novelty_reason = "Below provisional similarity threshold; not proof of uniqueness"

            if max_sim >= 0.985 and collided_g is not None:
                logging.info(f"    [Candidate {cand_idx+1}] Embedding collision ({max_sim:.4f} >= 0.985). Calling Judge...")
                is_novel, novelty_reason = judge_novelty_with_llm(cand_graph, collided_g, max_sim)
                logging.info(f"    [Candidate {cand_idx+1}] Novelty Judge: is_novel={is_novel} | {novelty_reason[:100]}")

            if not is_novel:
                logging.info(f"    [Candidate {cand_idx+1}] Novelty rejected ({max_sim:.4f}): {novelty_reason[:80]}")
                island.rejections.append({"model": selected_model, "cause": novelty_reason, "rationale": cand_rationale})
                bandit.update(selected_model, reward=0.0, crowned=False)
                continue

            iteration_novel_entries.append((cand_graph, cand_emb))
            viable_candidates.append((cand_graph, cand_rationale, selected_model))
            logging.info(f"    [Candidate {cand_idx+1}] ✓ Novel (sim={max_sim:.4f}), enrolled in SIFT tournament pool")

        # 3. SIFT Stage 2: Pairwise LLM-as-a-Judge → Bradley-Terry ranking
        if not viable_candidates:
            logging.warning(f"  [SIFT] No viable candidates produced this iteration. Skipping.")
            write_status(iteration, island.name, {"event": "All candidates failed/rejected"})
            persist(iteration + 1)
            iteration += 1
            continue

        if len(viable_candidates) == 1:
            best_graph, best_rationale, best_model = viable_candidates[0]
            best_bt_score = 1.0
            logging.info(f"  [SIFT Stage 2] Single viable candidate from [{best_model}], skipping pairwise tournament.")
        else:
            logging.info(f"  [SIFT Stage 2] Judging {len(viable_candidates)} candidates pairwise via [{sift_judge.judge_model}]...")
            ranked = sift_judge.rank_candidates(viable_candidates, domain_knowledge, diagnostics)
            best_graph, best_rationale, best_model, best_bt_score = ranked[0]
            logging.info(f"  [SIFT Stage 2] 🏅 Tournament Winner: [{best_model}] (BT θ={best_bt_score:.3f})")
            # Give partial reward to non-winners for producing viable novel mutations
            for _, _, m, _ in ranked[1:]:
                bandit.update(m, reward=0.15, crowned=False)

        # 4. Evaluate SIFT tournament winner across 80 workers
        cand_graph_path = run_dir / f"cand_iter_{iteration:03d}_{island.name}.json"
        cand_graph_path.write_text(json.dumps(best_graph, indent=2), encoding="utf-8")

        def gauntlet_cb(done, total, el, eta):
            write_status(iteration, island.name, {
                "event": f"Gauntlet: {done}/{total} matches ({done/total*100:.1f}%) | {el:.0f}s elapsed | ETA {eta:.0f}s",
                "model": best_model
            })

        # Refresh champion pool dynamically (picks up newly inducted champions)
        refresh_incumbents(iteration)
        champs = evaluate.list_champions()
        total_gauntlet_matches = len(champs) * args.seeds_per_champ_seat * 2

        logging.info(f"   Running {total_gauntlet_matches}-match gauntlet across {args.workers} workers ({len(champs)} champions)...")
        cand_eval = run_differential_gauntlet_multi_champ(
            cand_graph_path, champs, run_dir, workers=args.workers,
            seeds_per_champ_seat=args.seeds_per_champ_seat,
            tag=f"iter_{iteration:03d}", status_callback=gauntlet_cb,
            expected_fingerprint=active_fingerprint
        )

        if not comparable_score(cand_eval, best_graph, active_fingerprint):
            raise RuntimeError("Candidate evaluation invalid; promotion blocked")

        win_rate = cand_eval["overall_win_rate"]
        mean_cash = cand_eval["overall_cand_cash"]
        champ_wr = island.champion_score.get("overall_win_rate", 0.0)
        champ_cash = island.champion_score.get("overall_cand_cash", 0.0)

        logging.info(
            f"   Gauntlet Result: Overall Win Rate {win_rate*100:.1f}% ({cand_eval['total_wins']}/{cand_eval['total_matches']}) | "
            f"Cash: ${mean_cash:,.2f} vs Opp: ${cand_eval['overall_opp_cash']:,.2f} in {cand_eval['elapsed_sec']}s"
        )

        # Differential gating
        improved = (win_rate > champ_wr) or (win_rate == champ_wr and mean_cash > champ_cash + 250.0)
        reward = max(0.0, win_rate - champ_wr) if win_rate > champ_wr else (0.1 if (win_rate == champ_wr and mean_cash > champ_cash) else 0.0)

        if improved:
            logging.info(f"🏆 [CROWNED] Promoted as new champion of {island.name}! (WR: {champ_wr*100:.1f}% -> {win_rate*100:.1f}%)")
            island.champion_graph = best_graph
            island.champion_score = cand_eval
            island.history.append({"iteration": iteration, "graph": best_graph, "eval": cand_eval, "model": best_model, "rationale": best_rationale})
            bandit.update(best_model, reward=reward + 0.5, crowned=True)

            # Master graph sync
            if win_rate >= max((isl.champion_score.get("overall_win_rate", 0.0) for isl in islands), default=0.0):
                logging.info(f"⭐ [GLOBAL RECORD] Updating master policy_graph.json from {island.name} (WR: {win_rate*100:.1f}%)")
                base_graph_path.write_text(json.dumps(best_graph, indent=2), encoding="utf-8")
                master_graph = best_graph

            # --- Red Queen Ratchet: Induct high-performing candidates into opponent pool ---
            if win_rate >= 0.75:
                try:
                    from champion_pool import induct_graph_champion
                    champ_path, induct_status = induct_graph_champion(
                        best_graph, POOL_DIR,
                        iteration=iteration, island=island.name,
                        win_rate=win_rate, cash=mean_cash,
                    )
                    if induct_status == "created" and champ_path:
                        champs = evaluate.list_champions()
                        total_gauntlet_matches = len(champs) * args.seeds_per_champ_seat * 2
                        logging.info(f"⚔️ [INDUCTED] Red Queen Ratchet: Added {champ_path.name} to Champion Pool! Total pool: {len(champs)}")
                        refresh_incumbents(iteration + 1)
                except Exception as exc:
                    logging.error(f"Failed to induct champion: {exc}")

            write_status(iteration, island.name, {"event": "CROWNED", "model": best_model, "win_rate": win_rate, "cash": mean_cash})
        else:
            logging.info(f"❌ [REJECTED] Failed to beat island champion (Cand WR: {win_rate*100:.1f}% <= Champ WR: {champ_wr*100:.1f}%)")
            island.rejections.append({"model": best_model, "cand_eval": cand_eval, "rationale": best_rationale})
            bandit.update(best_model, reward=reward, crowned=False)
            write_status(iteration, island.name, {"event": "REJECTED", "model": best_model, "win_rate": win_rate, "cash": mean_cash})

        # 6. Meta-Supervisor Merge every supervisor_interval
        if iteration % args.supervisor_interval == 0:
            logging.info("=" * 70)
            logging.info(f"🏛️  INVOKING META-SUPERVISOR (gpt-6-astra) MERGE AT ITERATION {iteration}")
            logging.info("=" * 70)
            try:
                recent_evals = [isl.champion_score for isl in islands if isl.champion_score]
                recent_rejections = [{"cause": r.get("cause", "")} for isl in islands for r in isl.rejections[-2:] if isinstance(r, dict)]
                guidance = run_meta_supervisor(
                    best_graph=master_graph,
                    recent_evals=recent_evals,
                    recent_rejections=recent_rejections,
                    current_gen=iteration
                )
                logging.info(f"Meta-Supervisor Guidance: {guidance}")
            except Exception as e:
                logging.error(f"Meta-supervisor merge failed: {e}")

        # Atomically commit all novel candidates produced during this completed iteration
        for g, emb in iteration_novel_entries:
            register_graph(island, g, emb)
        persist(iteration + 1)
        iteration += 1

    logging.info("High-CPU Multi-Island Evolution Run Completed.")


if __name__ == "__main__":
    main()
