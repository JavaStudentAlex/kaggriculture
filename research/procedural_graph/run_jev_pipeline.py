"""Closed-Loop Evolutionary Pipeline for Jev-Powered Procedural Graphs.

Architecture:
1. Environment Match: Jev System One acts in simulation against Kaggle champions.
2. Trace Capture: Turn-by-turn branch choices, confidence, emergency locks, and cash curve.
3. BAM (Multi-Armed Bandit): UCB1 dynamically routes across frontier reasoning models
   (gpt-6-astra, gpt-5.6-luna, gemini-3.1-pro-preview, claude-opus-5, etc.).
4. Strategic Mutation: Selected model analyzes the loss trace and synthesizes a mutated
   Procedural Decision Graph (Delta G) with calibrated edge guidance and conditions.
5. Evaluation & Gating: Mutated graph is evaluated. If cash or win rate improves, it is
   crowned as the new baseline champion graph.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

CURRENT_DIR = Path(__file__).resolve().parent
REPO_DIR = CURRENT_DIR.parent.parent
EVO_DIR = REPO_DIR / "shinka" / "evolution"
POOL_DIR = REPO_DIR / "shinka" / "champions" / "pool"

for p in [str(CURRENT_DIR), str(REPO_DIR), str(EVO_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from bam_graph_mutator import BAMGraphMutator
from jev_match_evaluator import run_jev_match
from kaggriculture_domain_knowledge import build_jev_knowledge_summary
from shinka_graph_mab import MODEL_SPECS, UCB1Bandit

_RUNNING = True


def signal_handler(signum, frame):
    global _RUNNING
    print("\n[SIGINT/SIGTERM] Finishing current match cycle before exiting...", flush=True)
    _RUNNING = False


def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("jev_evo_pipeline")
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
    parser = argparse.ArgumentParser(description="Jev + BAM Procedural Graph Evolutionary Pipeline")
    parser.add_argument("--initial_graph", type=Path, default=CURRENT_DIR / "policy_graph.json")
    parser.add_argument("--generations", type=int, default=10, help="Number of evolutionary cycles")
    parser.add_argument("--query_interval", type=int, default=3, help="Steps between routine Jev queries")
    parser.add_argument("--run_dir", type=Path, default=CURRENT_DIR / "jev_evo_run")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    run_dir = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    best_dir = run_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    history_dir = run_dir / "history"
    history_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(run_dir / "jev_pipeline.log")
    logger.info("=" * 70)
    logger.info("JEV + BAM CLOSED-LOOP PROCEDURAL GRAPH PIPELINE INITIALIZED")
    logger.info(f"Initial Graph: {args.initial_graph}")
    logger.info(f"Target Generations: {args.generations} | Jev Query Interval: {args.query_interval}h")
    logger.info(f"BAM Arms: {list(MODEL_SPECS.keys())}")
    logger.info("=" * 70)

    # Copy initial champion graph
    champ_graph_path = best_dir / "policy_graph.json"
    if not champ_graph_path.exists():
        shutil.copy2(args.initial_graph, champ_graph_path)

    bandit = UCB1Bandit(run_dir / "bandit_state.json")
    mutator = BAMGraphMutator()
    domain_knowledge = build_jev_knowledge_summary()
    rejections_file = run_dir / "rejections.jsonl"

    # 1. Baseline match evaluation
    champs = sorted(list(POOL_DIR.glob("*.py")))
    if not champs:
        raise FileNotFoundError(f"No champions found in {POOL_DIR}")

    baseline_champ = champs[0]
    logger.info(f"Playing baseline match against champion [{baseline_champ.name}]...")
    base_match = run_jev_match(
        graph_path=champ_graph_path,
        champ_path=baseline_champ,
        seed=args.seed,
        cand_seat=0,
        query_interval=args.query_interval
    )

    best_cash = base_match["cand_cash"]
    best_win = base_match["win"]
    logger.info(f"Baseline Outcome: {base_match['outcome']} | Cash: ${best_cash:,.2f} vs Opponent: ${base_match['opp_cash']:,.2f} | Jev Calls: {base_match['jev_queries_count']}")

    last_match_diag = base_match
    current_champ_graph = json.loads(champ_graph_path.read_text(encoding="utf-8"))

    gen = 1
    while _RUNNING and gen <= args.generations:
        gen_dir = history_dir / f"gen_{gen:03d}"
        gen_dir.mkdir(parents=True, exist_ok=True)

        logger.info("-" * 60)
        logger.info(f"GENERATION {gen} / {args.generations}")

        # 1. Multi-Armed Bandit arm selection
        selected_model = bandit.select_arm()
        logger.info(f"🎲 BAM MAB selected reasoning arm: [{selected_model}]")

        # 2. Rejection sample collection
        rejections_sample = []
        if rejections_file.exists():
            lines = [l for l in rejections_file.read_text(encoding="utf-8").splitlines() if l.strip()]
            rejections_sample = [json.loads(l) for l in lines[-3:]]

        # 3. Frontier LLM diagnostic analysis & graph mutation
        try:
            logger.info(f"Querying [{selected_model}] to analyze match trace and propose Procedural Graph mutation...")
            cand_graph, rationale = mutator.mutate_graph(
                model_name=selected_model,
                current_graph=current_champ_graph,
                match_diagnostics=last_match_diag,
                decision_trace=last_match_diag.get("decision_trace", []),
                domain_knowledge=domain_knowledge,
                rejections=rejections_sample
            )
        except Exception as exc:
            logger.error(f"Mutation failed via {selected_model}: {exc}")
            bandit.update(selected_model, reward=0.0, crowned=False)
            gen += 1
            time.sleep(2)
            continue

        cand_graph_path = gen_dir / "candidate_graph.json"
        cand_graph_path.write_text(json.dumps(cand_graph, indent=2), encoding="utf-8")
        logger.info(f"Candidate Graph synthesized (Nodes: {len(cand_graph['nodes'])}, Edges: {len(cand_graph['edges'])}): {rationale[:120]}...")

        # 4. Evaluate candidate graph in simulation match with Jev
        eval_champ = random.choice(champs)
        logger.info(f"Testing Candidate Graph in match vs [{eval_champ.name}]...")
        cand_match = run_jev_match(
            graph_path=cand_graph_path,
            champ_path=eval_champ,
            seed=args.seed + gen,
            cand_seat=gen % 2,
            query_interval=args.query_interval
        )

        cand_cash = cand_match["cand_cash"]
        cand_win = cand_match["win"]
        cash_delta = cand_cash - best_cash
        logger.info(f"Gen {gen} Result: {cand_match['outcome']} | Cash: ${cand_cash:,.2f} (Delta: ${cash_delta:+,.2f}) vs Opp: ${cand_match['opp_cash']:,.2f}")

        (gen_dir / "match_result.json").write_text(json.dumps({
            "generation": gen,
            "model": selected_model,
            "rationale": rationale,
            "outcome": cand_match["outcome"],
            "cand_cash": cand_cash,
            "opp_cash": cand_match["opp_cash"],
            "cash_delta": cash_delta,
            "elapsed_sec": cand_match["elapsed_sec"],
            "jev_queries": cand_match["jev_queries_count"]
        }, indent=2), encoding="utf-8")

        # 5. Gating & Crowning
        is_promoted = False
        if cand_win and not best_win:
            is_promoted = True
        elif cand_cash > best_cash + 1000.0:
            is_promoted = True

        reward = max(0.0, cash_delta / 10000.0)

        if is_promoted:
            logger.info(f"🏆 NEW CHAMPION PROCEDURAL GRAPH CROWNED AT GEN {gen}! Cash: ${cand_cash:,.2f}")
            shutil.copy2(cand_graph_path, champ_graph_path)
            current_champ_graph = cand_graph
            best_cash = cand_cash
            best_win = cand_win
            bandit.update(selected_model, reward=reward, crowned=True)
        else:
            logger.info(f"Candidate did not exceed champion baseline (${best_cash:,.2f}). Recording to rejection memory.")
            bandit.update(selected_model, reward=0.0, crowned=False)
            rej_entry = {
                "generation": gen,
                "model": selected_model,
                "rationale": rationale,
                "cand_cash": cand_cash,
                "best_cash": best_cash
            }
            with open(rejections_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(rej_entry) + "\n")

        last_match_diag = cand_match
        logger.info("BAM Bandit Performance:")
        for arm, stats in bandit.summary().items():
            logger.info(f"  [{arm}]: {stats['pulls']} pulls | avg rwd: {stats['avg_reward']} | {stats['crowns']} crowns")

        gen += 1
        time.sleep(2)

    logger.info("=" * 70)
    logger.info(f"PIPELINE COMPLETE. Final Champion Cash: ${best_cash:,.2f}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
