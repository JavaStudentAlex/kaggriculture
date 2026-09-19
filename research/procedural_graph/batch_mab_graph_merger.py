"""Batch MAB Graph Merger: Parallel Synthesis and Hierarchical Merge.

Dispatches 4 specialized strategic dimensions across the MAB frontier models in parallel:
- Arm 1: gpt-6-astra (Capital Compounding, Day 0 Hires, Livestock Ratio)
- Arm 2: gpt-5.6-luna (Town Shop Arbitrage & Nonlinear Scarcity Capture)
- Arm 3: gemini-3.1-pro-preview (Midnight Liquidity & Shed Capacity Defense)
- Arm 4: claude-opus-5 (Late Harvest Freeze & Closed-Loop Watering)

Then executes a Meta-Supervisor Hierarchical Merge via gpt-6-astra (xhigh)
to produce a single, unified, topologically clean Master Procedural Graph.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import re
import time
import urllib.request
from typing import Any, Dict, List, Tuple

from shinka_graph_mab import UCB1Bandit

LOCAL_PROXY_URL = "http://localhost:8317/v1/chat/completions"
CURRENT_DIR = Path(__file__).resolve().parent
BANDIT_STATE_PATH = CURRENT_DIR / "shinka_evo_run" / "bandit_state.json"


def extract_json_block(text: str) -> Dict[str, Any]:
    """Extracts a valid JSON object from LLM response markdown or raw text."""
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass

    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        try:
            return json.loads(text[first_brace:last_brace + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract valid JSON from response:\n{text[:400]}...")


def query_llm(model: str, system_prompt: str, prompt: str, max_tokens: int = 8192) -> Dict[str, Any]:
    """Queries an LLM arm via the local proxy endpoint."""
    m_id = model.split("@")[0].replace("local/", "") if "@" in model else model
    req_body = {
        "model": m_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": max_tokens
    }
    req = urllib.request.Request(
        LOCAL_PROXY_URL,
        data=json.dumps(req_body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer local-key"}
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        raw = res["choices"][0]["message"]["content"]
        return extract_json_block(raw)


def run_parallel_subgraph_generation(
    current_graph: Dict[str, Any],
    mined_stats: Dict[str, Any],
    bandit: UCB1Bandit | None = None
) -> Dict[str, Dict[str, Any]]:
    """Runs 4 specialized strategic LLM arms in parallel, updating MAB pull counts."""
    elite = mined_stats.get("archetypes", {}).get("elite", {})

    dim_configs = {
        "capital_compounding": {
            "model": "gpt-6-astra",
            "focus": "Early Capital Compounding & Workforce Scaling",
            "instructions": (
                "Optimize the 'capital_compounding' node and edges for Days 0-12.\n"
                f"Empirical Consensus: Day 0 hires median={elite.get('hires_day_0', {}).get('median', 5)}, "
                f"cows bought median={elite.get('cows_bought', {}).get('median', 9)}, "
                f"sheep bought median={elite.get('sheep_bought', {}).get('median', 5)}, "
                f"peak workforce median={elite.get('peak_hands', {}).get('median', 12)}.\n"
                "Formulate exact priority, condition, guidance, and pitfalls."
            )
        },
        "town_shop_arbitrage": {
            "model": "gpt-5.6-luna",
            "focus": "Town Shop Arbitrage & Scarcity Capture",
            "instructions": (
                "Optimize the 'town_shop_preempt' node and edges.\n"
                "Empirical Consensus: Top performers capture 2x-3.5x shop premiums; Crop Dusta captured "
                "nonlinear tomato shortage ($757/unit) by holding inventory until Days 27-29. "
                "keiz supplied 732 milk and 822 strawberries into Ice Cream / Brunch shops.\n"
                "Formulate exact priority, condition, guidance, and pitfalls."
            )
        },
        "liquidity_and_capacity": {
            "model": "gemini-3.1-pro-preview",
            "focus": "Midnight Liquidity & Shed Capacity Defense",
            "instructions": (
                "Optimize the 'wage_defense' and 'shed_headroom' nodes.\n"
                f"Empirical Consensus: Peak shed median={elite.get('peak_shed', {}).get('median', 96)}. "
                "Fatal collapses happen when cash < midnight payroll (n_hands*120+100) at hour >= 18. "
                "Shed clearing must occur before 100/100 discards without starving animals of feed.\n"
                "Formulate exact priority, condition, guidance, and pitfalls."
            )
        },
        "late_harvest_and_liquidation": {
            "model": "claude-opus-5",
            "focus": "Late Harvest Freeze, Watering Loop & Terminal Liquidation",
            "instructions": (
                "Optimize the 'late_harvest_freeze' and a new 'terminal_liquidation' node.\n"
                f"Empirical Consensus: Strawberry seed purchases halt at Day {elite.get('last_strawberry_seed_day', {}).get('median', 11.0)} (p90=13.8). "
                "Continuous watering loop maintains 3-day regrowth; all shed items liquidated before step 720.\n"
                "Formulate exact priority, condition, guidance, and pitfalls."
            )
        }
    }

    results: Dict[str, Dict[str, Any]] = {}

    def worker_fn(task_key: str) -> Tuple[str, Dict[str, Any]]:
        cfg = dim_configs[task_key]
        model_name = cfg["model"]
        sys_prompt = (
            "You are an elite Kaggle Grandmaster specializing in Kaggriculture procedural graphs.\n"
            "Return ONLY a strict JSON object with: {'nodes': [...], 'edges': [...], 'rationale': '...'}"
        )
        prompt = f"""### STRATEGIC DOMAIN: {cfg['focus']}
{cfg['instructions']}

### CURRENT GRAPH STATE:
```json
{json.dumps(current_graph, indent=2)}
```

### OUTPUT SCHEMA:
{{
  "rationale": "<Explanation of changes>",
  "nodes": [ {{ "id": "...", "name": "...", "category": "...", "description": "..." }} ],
  "edges": [
    {{
      "source": "state_audit",
      "target": "...",
      "relation": "TRIGGERS",
      "priority": <int 1-10>,
      "attributes": {{
        "condition": "<python/eval condition>",
        "guidance": "<detailed guidance for Jev Choice>",
        "pitfalls": "<critical pitfalls to steer Jev away from errors>"
      }}
    }}
  ]
}}
"""
        print(f"  -> [MAB Arm Call] Dispatched [{task_key}] to {model_name}...")
        t0 = time.time()
        subgraph = query_llm(model_name, sys_prompt, prompt)
        dt = time.time() - t0
        print(f"  <- [MAB Arm Call] Finished [{task_key}] via {model_name} in {dt:.2f}s")
        return task_key, subgraph

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(worker_fn, k) for k in dim_configs]
        for f in concurrent.futures.as_completed(futures):
            k, sub = f.result()
            results[k] = sub
            if bandit is not None:
                arm_model = dim_configs[k]["model"]
                bandit.update(arm_model, reward=0.5, crowned=False)

    return results


def run_meta_supervisor_merge(
    baseline_graph: Dict[str, Any],
    subgraphs: Dict[str, Dict[str, Any]],
    mined_stats: Dict[str, Any],
    bandit: UCB1Bandit | None = None
) -> Dict[str, Any]:
    """Invokes Meta-Supervisor (gpt-6-astra) to synthesize the unified Master Procedural Graph, tracking MAB count."""
    print("\n🔍 Invoking Meta-Supervisor [gpt-6-astra] to merge all specialized subgraphs...")
    elite = mined_stats.get("archetypes", {}).get("elite", {})

    sys_prompt = (
        "You are the Chief Autonomous Systems Architect and Meta-Supervisor.\n"
        "You unify specialized subgraphs into a single, topologically flawless Master Procedural Graph.\n"
        "You resolve conflicting edge priorities into a strict 1-to-N sequence.\n"
        "Return ONLY a strict JSON object with the final graph."
    )

    prompt = f"""### EMPIRICAL GROUND TRUTH (Mined from 2,674 Matches):
- Elite Reward: Median=${elite.get('final_reward', {}).get('median', 159626):,.1f}
- Day 0 Hires: Median={elite.get('hires_day_0', {}).get('median', 5)}
- Peak Hands: Median={elite.get('peak_hands', {}).get('median', 12)}
- Livestock: Median Cows={elite.get('cows_bought', {}).get('median', 9)}, Sheep={elite.get('sheep_bought', {}).get('median', 5)}
- Strawberry Seed Cutoff: Median Day={elite.get('last_strawberry_seed_day', {}).get('median', 11.0)}
- Peak Shed: Median={elite.get('peak_shed', {}).get('median', 96)}

### BASELINE GRAPH:
```json
{json.dumps(baseline_graph, indent=2)}
```

### SPECIALIZED SUBGRAPHS FROM THE 4 MAB ARMS:
{json.dumps(subgraphs, indent=2)}

### MERGE OBJECTIVES:
1. Synthesize all nodes into a coherent set (state_audit, wage_defense, capital_compounding, town_shop_preempt, shed_headroom, oracle_frontrun, late_harvest_freeze, terminal_liquidation, farm_execution).
2. Assign unique, strictly ordered priorities (1 to N) from highest urgency (wage defense) to routine execution.
3. Ensure edge conditions and guidance incorporate the empirical figures above.
4. Provide comprehensive 'pitfalls' for each edge to prevent Jev System 1 errors.

### OUTPUT SCHEMA (Strict JSON):
{{
  "version": "2.0.0",
  "name": "master_empirical_procedural_graph",
  "description": "Unified Master Procedural Graph synthesized from 2,674 Kaggle matches via 4-arm MAB parallel distillation.",
  "nodes": [ ... ],
  "edges": [ ... ]
}}
"""
    t0 = time.time()
    merged = query_llm("gpt-6-astra", sys_prompt, prompt, max_tokens=16384)
    print(f"Meta-Supervisor synthesis finished in {time.time() - t0:.2f}s!")

    if bandit is not None:
        bandit.update("gpt-6-astra", reward=1.0, crowned=True)

    # Normalize priorities
    edges = merged.get("edges", [])
    for i, e in enumerate(sorted(edges, key=lambda x: int(x.get("priority", 99))), start=1):
        e["priority"] = i

    return merged


def main():
    parser = argparse.ArgumentParser(description="Batch MAB Graph Merger")
    parser.add_argument("--stats", type=Path, default=CURRENT_DIR / "replays_mined_archetypes.json")
    parser.add_argument("--current_graph", type=Path, default=CURRENT_DIR / "policy_graph.json")
    parser.add_argument("--out", type=Path, default=CURRENT_DIR / "policy_graph.json")
    args = parser.parse_args()

    if not args.stats.exists():
        raise FileNotFoundError(f"Stats file {args.stats} does not exist. Run parallel_replay_miner.py first.")

    mined_stats = json.loads(args.stats.read_text(encoding="utf-8"))
    current_graph = json.loads(args.current_graph.read_text(encoding="utf-8"))

    print("=" * 70)
    print("STARTING PARALLEL MAB GRAPH MERGE PIPELINE")
    print(f"Matches analyzed: {mined_stats.get('total_matches_analyzed')}")
    print("MAB Frontier Arms: gpt-6-astra, gpt-5.6-luna, gemini-3.1-pro-preview, claude-opus-5")
    print("Meta-Supervisor: gpt-6-astra (xhigh reasoning)")
    print("=" * 70)

    # Initialize MAB Bandit for call counting and reward tracking
    bandit = UCB1Bandit(BANDIT_STATE_PATH)
    print(f"[MAB Bandit Initialized] Current total calls recorded: {bandit.total_pulls}")

    # 1. Parallel Subgraph Generation
    subgraphs = run_parallel_subgraph_generation(current_graph, mined_stats, bandit=bandit)

    # 2. Meta-Supervisor Hierarchical Merge
    master_graph = run_meta_supervisor_merge(current_graph, subgraphs, mined_stats, bandit=bandit)

    print("\n[MAB Model Call Accounting Summary]:")
    for arm, stats in bandit.summary().items():
        print(f"  - {arm:24}: calls={stats['pulls']:3d} | avg_reward={stats['avg_reward']:.3f} | crowns={stats['crowns']}")
    print(f"Total Cumulative LLM Calls: {bandit.total_pulls}\n")

    # Backup existing
    backup_path = CURRENT_DIR / f"policy_graph_backup_{int(time.time())}.json"
    backup_path.write_text(json.dumps(current_graph, indent=2), encoding="utf-8")
    print(f"Backed up prior graph to {backup_path.name}")

    # Write new master graph
    args.out.write_text(json.dumps(master_graph, indent=2), encoding="utf-8")
    print(f"Successfully wrote Master Procedural Graph v{master_graph.get('version')} to {args.out}")
    print(f"Total Nodes: {len(master_graph.get('nodes', []))} | Total Edges: {len(master_graph.get('edges', []))}")


if __name__ == "__main__":
    main()
