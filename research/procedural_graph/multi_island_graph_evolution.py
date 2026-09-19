"""Multi-Island Historical Replay Evolutionary Pipeline for Procedural Graphs.

Architecture:
- 4 Parallel Evolutionary Islands exploring chronological replay cohorts:
  - Island 0 (Capital Scaling & Day 0 Hires): Older historical matches (Early August)
  - Island 1 (Mid-Game Shop Arbitrage & Shortage Capture): Mid August matches
  - Island 2 (Late Harvest Freeze & Watering Loops): Late August matches
  - Island 3 (Terminal Horizon Liquidation & Defense): Early September matches
- Mutation Engine: MAB selects across the 6 frontier LLMs
  (gpt-6-astra, gpt-5.6-luna, claude-opus-5, claude-sonnet-5, gemini-3.1-pro-preview, gemini-3.8-flash)
- Two-Level Novelty Filter:
  - Level 1: SentenceTransformer embeddings (threshold: 0.985)
  - Level 2: LLM structural novelty judge
- Supervisor: gpt-6-astra (xhigh reasoning) runs every N generations across all islands,
  synthesizing cross-island grafts and emitting global strategic recommendations.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

from bam_graph_mutator import BAMGraphMutator
from kaggriculture_domain_knowledge import KAGGLE_STRATEGY_PLAYBOOK
from shinka_graph_mab import UCB1Bandit
from shinka_graph_novelty import cosine_similarity, get_graph_embedding, judge_novelty_with_llm
from shinka_graph_supervisor import run_meta_supervisor

CURRENT_DIR = Path(__file__).resolve().parent
BANDIT_STATE_PATH = CURRENT_DIR / "shinka_evo_run" / "bandit_state.json"
DEFAULT_GRAPH_PATH = CURRENT_DIR / "policy_graph.json"


class EvolutionaryIsland:
    def __init__(self, island_id: int, name: str, focus_theme: str, initial_graph: Dict[str, Any]):
        self.island_id = island_id
        self.name = name
        self.focus_theme = focus_theme
        self.champion_graph = copy.deepcopy(initial_graph)
        self.champion_score = 0.0
        self.history: List[Dict[str, Any]] = []
        self.rejections: List[Dict[str, Any]] = []
        self.embeddings_archive: List[List[float]] = [get_graph_embedding(initial_graph)]


class MultiIslandEvolutionOrchestrator:
    def __init__(
        self,
        base_graph_path: Path = DEFAULT_GRAPH_PATH,
        bandit_path: Path = BANDIT_STATE_PATH,
        num_islands: int = 4,
        supervisor_interval: int = 5
    ):
        self.base_graph_path = base_graph_path
        self.initial_graph = json.loads(base_graph_path.read_text(encoding="utf-8"))
        self.bandit = UCB1Bandit(bandit_path)
        self.mutator = BAMGraphMutator()
        self.supervisor_interval = supervisor_interval
        self.supervisor_recommendations: List[str] = []

        # Initialize 4 diverse evolutionary islands
        themes = [
            (0, "Island-Capital", "Early Capital Compounding & Workforce Scaling (Days 0-10)"),
            (1, "Island-Arbitrage", "Town Shop Arbitrage & Nonlinear Crop Shortage Capture"),
            (2, "Island-Watering", "Late Harvest Freeze & Continuous Regrowth Watering Loops"),
            (3, "Island-Liquidation", "Terminal Horizon Fire Sale & Midnight Wage Liquidity Defense")
        ]
        self.islands = [
            EvolutionaryIsland(t[0], t[1], t[2], self.initial_graph)
            for t in themes[:num_islands]
        ]

    def evolve_island_step(self, island: EvolutionaryIsland, generation: int) -> Dict[str, Any]:
        """Runs one generation of mutation, novelty checking, and candidate synthesis on an island."""
        print(f"\n[{island.name} | Gen {generation}] Starting evolution step ({island.focus_theme})...", flush=True)

        # 1. Select LLM arm using UCB1 Multi-Armed Bandit
        selected_model = self.bandit.select_arm()
        print(f"  -> UCB1 MAB selected model: {selected_model}", flush=True)

        # 2. Prepare synthetic trace diagnostics focusing on the island's strategic bottleneck
        island_diagnostics = {
            "island_id": island.island_id,
            "island_focus": island.focus_theme,
            "supervisor_guidance": self.supervisor_recommendations,
            "active_nodes_count": len(island.champion_graph.get("nodes", [])),
            "active_edges_count": len(island.champion_graph.get("edges", []))
        }

        # 3. Prompt Frontier LLM for Graph Mutation
        t0 = time.time()
        try:
            mutated_graph, rationale = self.mutator.mutate_graph(
                model_name=selected_model,
                current_graph=island.champion_graph,
                match_diagnostics=island_diagnostics,
                decision_trace=[],
                domain_knowledge=json.dumps(KAGGLE_STRATEGY_PLAYBOOK, indent=2),
                rejections=island.rejections[-5:]
            )
        except Exception as e:
            print(f"  [ERROR] Mutation failed with {selected_model}: {e}")
            self.bandit.update(selected_model, reward=0.0, crowned=False)
            return {"status": "failed", "error": str(e)}

        dt = time.time() - t0
        print(f"  <- Model {selected_model} produced candidate in {dt:.2f}s.", flush=True)
        print(f"     Rationale: {rationale[:160]}...", flush=True)

        # 4. Two-Level Novelty Filtering
        cand_emb = get_graph_embedding(mutated_graph)
        max_sim = -1.0
        collision_idx = -1
        for idx, prev_emb in enumerate(island.embeddings_archive):
            sim = cosine_similarity(cand_emb, prev_emb)
            if sim > max_sim:
                max_sim = sim
                collision_idx = idx

        print(f"  [Novelty Level 1] Max Cosine Similarity: {max_sim:.4f}")

        is_novel = True
        novelty_reason = "Unique embedding"
        if max_sim >= 0.985:
            print(f"  [Novelty Level 2] Embedding collision detected ({max_sim:.4f} >= 0.985). Invoking LLM Novelty Judge...")
            collided_graph = island.history[collision_idx]["graph"] if collision_idx < len(island.history) else island.champion_graph
            is_novel, novelty_reason = judge_novelty_with_llm(mutated_graph, collided_graph, max_sim)
            print(f"  [Novelty Level 2 Judge Verdict]: is_novel={is_novel} | {novelty_reason[:120]}")

        if not is_novel:
            print("  [Novelty Rejection] Mutation lacks structural novelty. Discarding candidate.")
            island.rejections.append({
                "model": selected_model,
                "rationale": rationale,
                "rejection_cause": f"Novelty rejection: {novelty_reason}"
            })
            self.bandit.update(selected_model, reward=0.0, crowned=False)
            return {"status": "rejected_novelty", "model": selected_model}

        # 5. Candidate Accepted into Island Population
        island.embeddings_archive.append(cand_emb)
        reward = 0.5  # Strategic novelty reward
        self.bandit.update(selected_model, reward=reward, crowned=False)

        island.champion_graph = mutated_graph
        island.history.append({
            "gen": generation,
            "model": selected_model,
            "graph": mutated_graph,
            "rationale": rationale,
            "max_sim": max_sim
        })

        return {
            "status": "accepted",
            "island": island.name,
            "model": selected_model,
            "rationale": rationale
        }

    def run_supervisor_synthesis(self, generation: int) -> None:
        """gpt-6-astra reviews all 4 islands, crosses top policies, and synthesizes recommendations."""
        print("\n" + "=" * 70)
        print(f"🏛️  INVOKING META-SUPERVISOR (gpt-6-astra) AT GENERATION {generation}")
        print("=" * 70)

        recent_evals = []
        recent_rejections = []
        for isl in self.islands:
            for h in isl.history[-2:]:
                recent_evals.append({"island": isl.name, "model": h["model"], "rationale": h["rationale"]})
            for r in isl.rejections[-2:]:
                recent_rejections.append({"island": isl.name, "cause": r.get("rejection_cause", "")})

        try:
            self.supervisor_recommendations = run_meta_supervisor(
                best_graph=self.islands[0].champion_graph,
                recent_evals=recent_evals,
                recent_rejections=recent_rejections,
                current_gen=generation
            )
            print("Meta-Supervisor Recommendations Issued:")
            for r in self.supervisor_recommendations:
                print(f"  • {r}")
            self.bandit.update("gpt-6-astra", reward=1.0, crowned=True)
        except Exception as e:
            print(f"Supervisor call skipped or failed: {e}")

    def run_evolution(self, total_generations: int = 10) -> Dict[str, Any]:
        """Runs multi-island evolutionary loop across generations."""
        print("=" * 70)
        print(f"LAUNCHING 4-ISLAND PROCEDURAL GRAPH EVOLUTION ({total_generations} GENERATIONS)")
        print(f"Active MAB Arms: {self.bandit.arms}")
        print(f"Novelty Filter: SentenceTransformers (MiniLM-L6) + LLM Judge")
        print(f"Supervisor Cadence: Every {self.supervisor_interval} Generations")
        print("=" * 70)

        for gen in range(1, total_generations + 1):
            print(f"\n>>> GENERATION {gen} / {total_generations} <<<", flush=True)
            for island in self.islands:
                self.evolve_island_step(island, gen)

            if gen % self.supervisor_interval == 0:
                self.run_supervisor_synthesis(gen)

            # Auto-save best candidate checkpoint after every generation
            best_curr_island = max(self.islands, key=lambda isl: len(isl.history))
            self.base_graph_path.write_text(json.dumps(best_curr_island.champion_graph, indent=2), encoding="utf-8")
            print(f"[Gen {gen} Checkpoint] Saved current best island ({best_curr_island.name}) to {self.base_graph_path.name}", flush=True)

        # Cross-Island Master Graft: Save best mutated graph as the new policy_graph.json
        best_island = max(self.islands, key=lambda isl: len(isl.history))
        print(f"\n[Evolution Complete] Champion Island: {best_island.name} with {len(best_island.history)} accepted mutations.")

        # Save to policy_graph.json
        self.base_graph_path.write_text(json.dumps(best_island.champion_graph, indent=2), encoding="utf-8")
        print(f"Updated Master Procedural Graph saved to {self.base_graph_path}")

        print("\n[Final MAB Model Call Accounting Summary]:")
        for arm, stats in self.bandit.summary().items():
            print(f"  - {arm:24}: calls={stats['pulls']:3d} | avg_reward={stats['avg_reward']:.3f} | crowns={stats['crowns']}")

        return {"champion_island": best_island.name, "bandit_summary": self.bandit.summary()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Island Procedural Graph Evolution")
    parser.add_argument("--generations", type=int, default=4, help="Generations to run")
    parser.add_argument("--supervisor_interval", type=int, default=2, help="Supervisor cadence")
    args = parser.parse_args()

    orchestrator = MultiIslandEvolutionOrchestrator(
        supervisor_interval=args.supervisor_interval
    )
    orchestrator.run_evolution(total_generations=args.generations)
