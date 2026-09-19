"""Complete Shinka-Grade Evolution Engine for Procedural Graphs.

Includes:
1. Multi-Armed Bandit (UCB1) over the 6 Shinka frontier models with advanced reasoning efforts:
   - gpt-6-astra (xhigh)
   - gpt-5.6-luna (xhigh)
   - gemini-3.1-pro-preview (xhigh)
   - gemini-3.8-flash (high)
   - claude-opus-5 (xhigh)
   - claude-sonnet-5 (xhigh)
2. Supervisor (Meta-Review):
   - gpt-6-astra (xhigh) every 5 generations
   - Injects concrete strategic directives into subsequent mutation prompts
3. Two-Level Novelty Pipeline:
   - Level 1: qwen3-embedding:8b via Ollama (4096-dim), similarity threshold 0.985
   - Level 2: gemini-3.1-pro-preview (xhigh) as novelty judge
   - Up to 3 resample attempts on duplicate rejection
4. Fast Match Simulation:
   - 8 parallel workers evaluating candidates against the 10-champion pool
   - Real-time opponent oracle inference on GPU (cuda:0..3)
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import signal
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

EVO_DIR = Path(__file__).resolve().parent
RUN_DIR = EVO_DIR / "shinka_evo_run"

if str(EVO_DIR) not in sys.path:
    sys.path.insert(0, str(EVO_DIR))

from eval_candidate import evaluate_graph
from shinka_graph_mab import MODEL_SPECS, UCB1Bandit
from shinka_graph_novelty import NoveltyChecker
from shinka_graph_supervisor import run_meta_supervisor

_RUNNING = True
LOCAL_PROXY_URL = "http://localhost:8317/v1/chat/completions"


def signal_handler(signum, frame):
    global _RUNNING
    print("\n[SIGINT/SIGTERM] Finishing generation before exiting cleanly...", flush=True)
    _RUNNING = False


def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("shinka_graph_evo")
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


def query_mutator_llm(
    model_name: str,
    parent_graph: dict[str, Any],
    eval_diagnostics: dict[str, Any],
    rejections: list[dict[str, Any]],
    supervisor_recs: list[str]
) -> tuple[dict[str, Any], str]:
    """Queries selected model with full reasoning effort and supervisor context."""
    spec = MODEL_SPECS[model_name]
    model_uri = spec["uri"].split("@")[0].replace("local/", "")
    effort = spec["reasoning_effort"]
    temp = spec["temperature"]

    system_prompt = """You are an elite autonomous systems strategist and Kaggle Grandmaster optimizing a Procedural Execution Graph for Kaggriculture.
You only edit the GRAPH STRUCTURE (nodes, edges, priorities, conditions, guidance, and pitfalls).
You do NOT write python code. You return a strictly validated JSON matching the Procedural Graph schema.
"""

    recs_str = "\n".join(f"- {r}" for r in supervisor_recs) if supervisor_recs else "- Focus on closing the cash deficit."

    prompt = f"""### CONTEXT:
We are evolving the Procedural Graph for our Kaggriculture agent against a pool of 10 champions.
The graph arbitrates decisions across high-priority nodes (Wage Defense, Shed Headroom, Town Shop Preempt, Oracle Frontrun, Farm Execution) within a strict 10-order engine cap.

### CURRENT CHAMPION GRAPH:
```json
{json.dumps(parent_graph, indent=2)}
```

### LATEST EVALUATION METRICS:
- Games: {eval_diagnostics.get('total_games', 0)}
- Win Rate: {eval_diagnostics.get('win_rate', 0.0) * 100:.1f}%
- Average Cash: ${eval_diagnostics.get('avg_cash', 0.0):,.2f}
- Opponent Average Cash: ${eval_diagnostics.get('avg_opp_cash', 0.0):,.2f}
- Cash Deficit: ${eval_diagnostics.get('cash_diff', 0.0):,.2f}
- Worst Defeats:
```json
{json.dumps(eval_diagnostics.get('worst_losses', [])[:3], indent=2)}
```

### STRATEGIC SUPERVISOR DIRECTIVES (From Meta-Review):
{recs_str}

### RECENT REJECTED MUTATIONS (Do not repeat these failures):
```json
{json.dumps(rejections[-3:] if rejections else [], indent=2)}
```

### OBJECTIVE:
Synthesize a strategic graph mutation (Delta G) to eliminate the cash deficit against the champions:
1. Recalibrate condition thresholds (e.g. earlier town shop seed allocation, dynamic feed reserves, or price elasticity limits).
2. Adjust edge priorities (e.g. prioritizing town shop contracts when active).
3. Introduce strategic guards or new nodes.

### OUTPUT FORMAT:
Respond with a strict JSON object:
{{
  "rationale": "<2-3 sentence strategic explanation of why this graph change fixes the cash gap>",
  "graph": <COMPLETE UPDATED GRAPH JSON>
}}
"""
    req_body = {
        "model": model_uri,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": temp,
        "max_tokens": 32768,
        "reasoning_effort": effort
    }
    data = json.dumps(req_body).encode("utf-8")
    req = urllib.request.Request(
        LOCAL_PROXY_URL,
        data=data,
        headers={"Content-Type": "application/json", "Authorization": "Bearer local-key"}
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        content = res["choices"][0]["message"]["content"]

    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
    if m:
        parsed = json.loads(m.group(1))
    else:
        m2 = re.search(r"(\{.*\})", content, re.DOTALL)
        if m2:
            parsed = json.loads(m2.group(1))
        else:
            raise ValueError(f"No JSON found in response from {model_name}")

    rationale = parsed.get("rationale", "Strategic mutation")
    mutated_graph = parsed.get("graph", parsed)

    if "nodes" not in mutated_graph or "edges" not in mutated_graph:
        raise ValueError("Invalid graph: missing 'nodes' or 'edges'")

    for e in mutated_graph["edges"]:
        e["priority"] = int(e.get("priority", 99))

    return mutated_graph, rationale


def main():
    parser = argparse.ArgumentParser(description="Complete Shinka Procedural Graph Evolution")
    parser.add_argument("--max_generations", type=int, default=100)
    parser.add_argument("--eval_seeds", type=int, default=1, help="Seeds per champion per seat")
    parser.add_argument("--workers", type=int, default=20, help="Simulation workers")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    history_dir = RUN_DIR / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    best_dir = RUN_DIR / "best"
    best_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(RUN_DIR / "shinka_evolution.log")
    logger.info("=" * 70)
    logger.info("SHINKA PROCEDURAL GRAPH EVOLUTION ENGINE LAUNCHED")
    logger.info("Active Models (UCB MAB): gpt-6-astra, gpt-5.6-luna, gemini-3.1-pro, gemini-3.8-flash, claude-opus-5, claude-sonnet-5")
    logger.info("Reasoning Efforts: xhigh (Flash: high)")
    logger.info("Supervisor: gpt-6-astra (xhigh) every 5 generations")
    logger.info("Novelty Pipeline: Level 1 (qwen3-embedding:8b, 4096-d) + Level 2 (gemini-3.1-pro, xhigh)")
    logger.info("=" * 70)

    # Initialize MAB & Novelty
    bandit = UCB1Bandit(RUN_DIR / "bandit_state.json")
    novelty_checker = NoveltyChecker(sim_threshold=0.985)

    best_graph_path = best_dir / "policy_graph.json"
    rejections_file = RUN_DIR / "rejections.jsonl"

    if not best_graph_path.exists():
        base_src = EVO_DIR / "policy_graph.json"
        shutil.copy2(base_src, best_graph_path)
        logger.info(f"Initialized champion graph from {base_src}")

    # Register initial graph in novelty archive
    best_graph_data = json.loads(best_graph_path.read_text(encoding="utf-8"))
    novelty_checker.register(best_graph_data)

    # Baseline evaluation
    base_metrics_path = best_dir / "best_metrics.json"
    if not base_metrics_path.exists():
        logger.info("Running baseline evaluation across 20 champion matches...")
        base_metrics = evaluate_graph(best_graph_path, num_seeds_per_champ=args.eval_seeds, num_workers=args.workers)
        base_metrics_path.write_text(json.dumps(base_metrics, indent=2), encoding="utf-8")
    else:
        base_metrics = json.loads(base_metrics_path.read_text(encoding="utf-8"))

    best_win_rate = base_metrics.get("win_rate", 0.0)
    best_avg_cash = base_metrics.get("avg_cash", 0.0)
    logger.info(f"Active Champion Baseline: Win Rate = {best_win_rate*100:.1f}% | Avg Cash = ${best_avg_cash:,.2f}")

    current_gen = 1
    supervisor_directives: list[str] = []
    recent_evals: list[dict[str, Any]] = [base_metrics]

    while _RUNNING and current_gen <= args.max_generations:
        gen_dir = history_dir / f"gen_{current_gen:03d}"
        gen_dir.mkdir(parents=True, exist_ok=True)

        logger.info("-" * 60)
        logger.info(f"GENERATION {current_gen} / {args.max_generations}")

        # 1. Supervisor Meta-Review check (Every 5 generations)
        if current_gen % 5 == 0:
            logger.info("🔍 Invoking Meta-Supervisor [gpt-6-astra (xhigh)] for population review...")
            rejections_list = []
            if rejections_file.exists():
                for line in rejections_file.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        try:
                            rejections_list.append(json.loads(line))
                        except Exception:
                            pass
            try:
                supervisor_directives = run_meta_supervisor(
                    best_graph_data,
                    recent_evals,
                    rejections_list,
                    current_gen
                )
                logger.info(f"Supervisor Directives Issued ({len(supervisor_directives)} recommendations):")
                for rec in supervisor_directives:
                    logger.info(f"  * {rec}")
            except Exception as e:
                logger.warning(f"Supervisor call failed: {e}. Continuing with previous directives.")

        # 2. Multi-Armed Bandit Model Selection
        selected_model = bandit.select_arm()
        effort = MODEL_SPECS[selected_model]["reasoning_effort"]
        logger.info(f"🎲 MAB selected model: [{selected_model}] (effort: {effort})")

        # 3. Mutation Generation with Novelty Inspection (up to 3 attempts)
        cand_graph = None
        cand_rationale = ""
        novelty_passed = False

        for attempt in range(1, 4):
            try:
                rejections_sample = []
                if rejections_file.exists():
                    lines = [l for l in rejections_file.read_text(encoding="utf-8").splitlines() if l.strip()]
                    rejections_sample = [json.loads(l) for l in lines[-4:]]

                raw_graph, raw_rationale = query_mutator_llm(
                    selected_model,
                    best_graph_data,
                    base_metrics,
                    rejections_sample,
                    supervisor_directives
                )
                # Run Two-Level Novelty Check
                is_novel, reason, sim = novelty_checker.check(raw_graph)
                logger.info(f"Novelty Inspection (Attempt {attempt}/3): {reason} (sim: {sim:.4f})")

                if is_novel:
                    cand_graph = raw_graph
                    cand_rationale = raw_rationale
                    novelty_passed = True
                    break
                else:
                    logger.info("Resampling mutation due to duplicate detection...")
            except Exception as exc:
                logger.error(f"Proposal attempt {attempt} failed: {exc}")
                time.sleep(5)

        if not novelty_passed or cand_graph is None:
            logger.warning(f"Generation {current_gen} could not find a novel mutation. Skipping to next cycle.")
            bandit.update(selected_model, reward=0.0, crowned=False)
            current_gen += 1
            continue

        cand_graph_path = gen_dir / "candidate_graph.json"
        cand_graph_path.write_text(json.dumps(cand_graph, indent=2), encoding="utf-8")
        logger.info(f"Mutation accepted for evaluation: {cand_rationale}")

        # 4. Evaluation Gauntlet
        logger.info(f"Running simulation gauntlet across {args.eval_seeds * 20} matches with {args.workers} workers...")
        cand_metrics = evaluate_graph(cand_graph_path, num_seeds_per_champ=args.eval_seeds, num_workers=args.workers)
        
        (gen_dir / "eval_metrics.json").write_text(json.dumps(cand_metrics, indent=2), encoding="utf-8")
        (gen_dir / "meta.json").write_text(json.dumps({
            "generation": current_gen,
            "model": selected_model,
            "effort": effort,
            "rationale": cand_rationale,
            "timestamp": time.time()
        }, indent=2), encoding="utf-8")

        cand_win_rate = cand_metrics["win_rate"]
        cand_cash = cand_metrics["avg_cash"]
        crashes = cand_metrics["crashes"]
        recent_evals.append(cand_metrics)

        logger.info(f"Gen {current_gen} Outcome: WR = {cand_win_rate*100:.1f}% (Best: {best_win_rate*100:.1f}%) | Cash = ${cand_cash:,.2f} (Best: ${best_avg_cash:,.2f}) | Crashes = {crashes}")

        # 5. Gating & Crowning
        is_promoted = False
        cash_delta = cand_cash - best_avg_cash
        reward = max(0.0, (cash_delta / 10000.0) + (cand_win_rate - best_win_rate) * 5.0)

        if crashes == 0:
            if cand_win_rate > best_win_rate or (cand_win_rate == best_win_rate and cash_delta > 500.0):
                is_promoted = True

        if is_promoted:
            logger.info(f"🏆 NEW CHAMPION GRAPH CROWNED AT GEN {current_gen}! Win Rate: {cand_win_rate*100:.1f}% | Cash: ${cand_cash:,.2f}")
            shutil.copy2(cand_graph_path, best_graph_path)
            base_metrics_path.write_text(json.dumps(cand_metrics, indent=2), encoding="utf-8")
            best_graph_data = cand_graph
            base_metrics = cand_metrics
            best_win_rate = cand_win_rate
            best_avg_cash = cand_cash
            bandit.update(selected_model, reward=reward, crowned=True)
        else:
            logger.info("Candidate did not exceed champion baseline. Recording to rejection memory.")
            bandit.update(selected_model, reward=0.0, crowned=False)
            rejection_record = {
                "generation": current_gen,
                "model": selected_model,
                "rationale": cand_rationale,
                "cand_win_rate": cand_win_rate,
                "cand_cash": cand_cash,
                "cash_diff": cand_metrics["cash_diff"]
            }
            with open(rejections_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(rejection_record) + "\n")

        # Log bandit stats
        logger.info("MAB Model Performance Summary:")
        for m_name, m_stat in bandit.summary().items():
            logger.info(f"  [{m_name}]: {m_stat['pulls']} pulls, avg reward {m_stat['avg_reward']}, {m_stat['crowns']} crowns")

        current_gen += 1
        time.sleep(2)

    logger.info("=" * 70)
    logger.info(f"EVOLUTION COMPLETED. Champion Win Rate: {best_win_rate*100:.1f}% | Champion Cash: ${best_avg_cash:,.2f}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
