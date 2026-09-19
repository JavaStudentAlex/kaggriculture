"""Master Orchestrator for Procedural Graph Evolution.

Autonomous evolutionary search loop running on dev:
1. Gathers recent loss diagnostics from the current best graph.
2. Calls the LLM mutator (gpt-6-astra, gemini-3.1-pro-preview, claude-sonnet-5) to propose Delta G.
3. Evaluates the mutated graph across simulation matches against the champion pool.
4. Gating: Promotes if win rate / cash beats current best; records to rejection memory otherwise.
5. Loops continuously in a background tmux session.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import signal
import sys
import time
from pathlib import Path
from typing import Any

EVO_DIR = Path(__file__).resolve().parent
RUN_DIR = EVO_DIR / "evolution_run"

if str(EVO_DIR) not in sys.path:
    sys.path.insert(0, str(EVO_DIR))

from eval_candidate import evaluate_graph
from mutator import mutate_graph

_RUNNING = True


def signal_handler(signum, frame):
    global _RUNNING
    print("\n[SIGINT/SIGTERM received] Finishing current generation cleanly before exit...", flush=True)
    _RUNNING = False


def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("graph_evo")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def main():
    parser = argparse.ArgumentParser(description="Procedural Graph Evolutionary Search")
    parser.add_argument("--max_generations", type=int, default=100)
    parser.add_argument("--eval_seeds", type=int, default=1, help="Seeds per champion per seat")
    parser.add_argument("--workers", type=int, default=8, help="Parallel match evaluation workers")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    history_dir = RUN_DIR / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    best_dir = RUN_DIR / "best"
    best_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(RUN_DIR / "evolution.log")
    logger.info("=" * 70)
    logger.info("STARTING PROCEDURAL GRAPH EVOLUTION LOOP")
    logger.info(f"Target generations: {args.max_generations} | Eval seeds/champ: {args.eval_seeds} | Workers: {args.workers}")
    logger.info("=" * 70)

    # Initialize best graph from base policy_graph.json if not present
    best_graph_path = best_dir / "policy_graph.json"
    rejections_file = RUN_DIR / "rejections.jsonl"

    if not best_graph_path.exists():
        base_src = EVO_DIR / "policy_graph.json"
        shutil.copy2(base_src, best_graph_path)
        logger.info(f"Initialized best graph from {base_src}")

    # Evaluate base graph if no baseline score exists
    base_metrics_path = best_dir / "best_metrics.json"
    if not base_metrics_path.exists():
        logger.info("Evaluating initial baseline graph...")
        base_metrics = evaluate_graph(best_graph_path, num_seeds_per_champ=args.eval_seeds, num_workers=args.workers)
        base_metrics_path.write_text(json.dumps(base_metrics, indent=2))
        logger.info(f"Baseline Win Rate: {base_metrics['win_rate']*100:.1f}% | Avg Cash: ${base_metrics['avg_cash']:,.2f} | Deficit: ${base_metrics['cash_diff']:,.2f}")
    else:
        base_metrics = json.loads(base_metrics_path.read_text())
        logger.info(f"Loaded existing best metrics: Win Rate: {base_metrics['win_rate']*100:.1f}% | Avg Cash: ${base_metrics['avg_cash']:,.2f}")

    best_win_rate = base_metrics.get("win_rate", 0.0)
    best_avg_cash = base_metrics.get("avg_cash", 0.0)

    # Determine starting generation
    existing_gens = [int(p.name.split("_")[-1]) for p in history_dir.glob("gen_*") if p.is_dir() and p.name.split("_")[-1].isdigit()]
    current_gen = max(existing_gens, default=0) + 1

    while _RUNNING and current_gen <= args.max_generations:
        gen_dir = history_dir / f"gen_{current_gen:03d}"
        gen_dir.mkdir(parents=True, exist_ok=True)

        logger.info("-" * 50)
        logger.info(f"GENERATION {current_gen} / {args.max_generations}")

        # Load recent rejections
        rejections: list[dict[str, Any]] = []
        if rejections_file.exists():
            with open(rejections_file, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            rejections.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass

        # 1. Propose mutation
        logger.info("Requesting graph mutation from LLM...")
        try:
            cand_graph, model_name, rationale = mutate_graph(
                best_graph_path,
                base_metrics,
                rejections[-5:]
            )
            cand_graph_path = gen_dir / "candidate_graph.json"
            cand_graph_path.write_text(json.dumps(cand_graph, indent=2))
            logger.info(f"Mutation generated by [{model_name}]")
            logger.info(f"Rationale: {rationale}")
        except Exception as exc:
            logger.error(f"Mutation generation failed: {exc}. Retrying next cycle in 10s...")
            time.sleep(10)
            continue

        # 2. Evaluate candidate
        logger.info(f"Evaluating candidate graph across {args.eval_seeds * 20} matches with {args.workers} workers...")
        cand_metrics = evaluate_graph(cand_graph_path, num_seeds_per_champ=args.eval_seeds, num_workers=args.workers)
        (gen_dir / "eval_metrics.json").write_text(json.dumps(cand_metrics, indent=2))
        (gen_dir / "meta.json").write_text(json.dumps({
            "model": model_name,
            "rationale": rationale,
            "timestamp": time.time()
        }, indent=2))

        cand_win_rate = cand_metrics["win_rate"]
        cand_cash = cand_metrics["avg_cash"]
        crashes = cand_metrics["crashes"]

        logger.info(f"Gen {current_gen} Results: Win Rate = {cand_win_rate*100:.1f}% (Best: {best_win_rate*100:.1f}%) | Cash = ${cand_cash:,.2f} (Best: ${best_avg_cash:,.2f}) | Crashes = {crashes}")

        # 3. Acceptance / Gating logic
        # Promotes if win rate improves, or if win rate ties and average cash increases by > $500
        is_promoted = False
        if crashes == 0:
            if cand_win_rate > best_win_rate or (cand_win_rate == best_win_rate and cand_cash > (best_avg_cash + 500.0)):
                is_promoted = True

        if is_promoted:
            logger.info(f"🎉 NEW BEST GRAPH CROWNED AT GEN {current_gen}! Win Rate: {cand_win_rate*100:.1f}% | Cash: ${cand_cash:,.2f}")
            shutil.copy2(cand_graph_path, best_graph_path)
            base_metrics_path.write_text(json.dumps(cand_metrics, indent=2))
            base_metrics = cand_metrics
            best_win_rate = cand_win_rate
            best_avg_cash = cand_cash
        else:
            logger.info(f"Candidate rejected (Did not exceed Best: WR {best_win_rate*100:.1f}%, Cash ${best_avg_cash:,.2f}). Recording to rejection memory.")
            rejection_record = {
                "generation": current_gen,
                "model": model_name,
                "rationale": rationale,
                "cand_win_rate": cand_win_rate,
                "cand_cash": cand_cash,
                "cash_diff": cand_metrics["cash_diff"]
            }
            with open(rejections_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(rejection_record) + "\n")

        current_gen += 1
        time.sleep(3)

    logger.info("=" * 70)
    logger.info(f"EVOLUTION LOOP FINISHED. Final Best Win Rate: {best_win_rate*100:.1f}% | Best Cash: ${best_avg_cash:,.2f}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
