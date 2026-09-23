"""SIFT-inspired Pairwise LLM Judge + Bradley-Terry Ranking for Graph Evolution.

Based on: "Self-Improvement via Fast Tree-search" (Fu, Kulanthaivelu, Yamada — MIT + Sakana AI, Sep 2026)
arXiv:2609.19526

Key ideas adapted for Kaggriculture procedural graph evolution:
1. Pairwise LLM-as-a-Judge: compare two candidate graphs directly via LLM —
   "which is more likely to improve win rate?" — instead of running 1,000 matches for each.
2. Regularized Bradley-Terry aggregation: pool all pairwise wins/losses into a global
   strength ranking.
3. Rank-based parent sampling: P(i) ∝ exp(-α·r_bt(i) - β·r_acc(i) - η·log(1+v_i))

In our pipeline, the judge acts as a cheap pre-filter: generate N candidate mutations,
judge them pairwise, and push only the BT-top-ranked candidate to the Kaggle gauntlet.
"""
from __future__ import annotations

import json
import math
import os
import urllib.request

from call_telemetry import record_call
from typing import Any, Dict, List, Optional, Tuple

LOCAL_PROXY_URL = "http://localhost:8317/v1/chat/completions"
# Default judge model — strong enough to rank, cheap enough to call frequently
DEFAULT_JUDGE_MODEL = "gpt-6-astra"


def _llm_call(model: str, system_prompt: str, user_prompt: str,
              proxy_url: str = LOCAL_PROXY_URL, proxy_key: str = "local-key",
              temperature: float = 0.15, max_tokens: int = 4096) -> str:
    """Raw LLM call via local proxy."""
    req_body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    data = json.dumps(req_body).encode("utf-8")
    req = urllib.request.Request(
        proxy_url, data=data,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {proxy_key}"}
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            res = json.loads(resp.read().decode("utf-8"))
            content = res["choices"][0]["message"]["content"]
        record_call("sift_judge_llm", "success", model=model)
        return content
    except Exception as exc:
        record_call("sift_judge_llm", "failure", model=model, error=exc)
        raise


class SIFTGraphJudge:
    """Pairwise LLM judge with Bradley-Terry aggregation for procedural graph candidates.

    Usage:
        judge = SIFTGraphJudge()
        candidates = [(graph_A, rationale_A, model_A), (graph_B, rationale_B, model_B), ...]
        ranked = judge.rank_candidates(candidates, domain_knowledge, diagnostics)
        best_graph, best_rationale, best_model = ranked[0]
    """

    def __init__(
        self,
        judge_model: str = DEFAULT_JUDGE_MODEL,
        bt_prior: float = 0.5,
        max_bt_iters: int = 100,
        bt_tol: float = 1e-6,
    ):
        self.judge_model = judge_model
        self.bt_prior = bt_prior          # λ regularization pseudo-count
        self.max_bt_iters = max_bt_iters
        self.bt_tol = bt_tol

        # Persistent archive for cross-iteration BT ranking
        self.win_matrix: Dict[str, Dict[str, int]] = {}  # win_matrix[i][j] = wins of i over j
        self.visit_counts: Dict[str, int] = {}
        self.accuracy_scores: Dict[str, float] = {}  # from gauntlet results

    def _judge_pairwise(
        self,
        graph_a: Dict[str, Any],
        rationale_a: str,
        graph_b: Dict[str, Any],
        rationale_b: str,
        domain_knowledge: str,
        diagnostics: Dict[str, Any],
    ) -> str:
        """Ask the judge LLM: which candidate graph is more promising? Returns 'A' or 'B'."""
        system_prompt = (
            "You are an expert judge evaluating two candidate Procedural Decision Graphs "
            "for an autonomous farming agent in the Kaggriculture Kaggle competition.\n"
            "Your task: determine which candidate graph mutation is MORE LIKELY to improve "
            "the agent's win rate against the champion pool.\n"
            "Consider: economic strategy soundness, edge condition precision, "
            "pitfall/guidance quality, workforce scaling logic, and robustness to failure modes.\n"
            "You MUST respond with ONLY a JSON object: "
            '{"reasoning": "<2-3 sentences>", "winner": "A" or "B"}'
        )

        # Summarize diagnostics compactly
        diag_summary = json.dumps({
            "island": diagnostics.get("island_name", ""),
            "focus": diagnostics.get("island_focus", ""),
            "current_win_rate": diagnostics.get("current_champ_win_rate", 0.0),
            "recent_losses": diagnostics.get("worst_cloud_losses", [])[:3],
        }, indent=2)

        user_prompt = f"""### DOMAIN KNOWLEDGE:
{domain_knowledge[:3000]}

### CURRENT GAME STATE:
{diag_summary}

### CANDIDATE A:
**Rationale:** {rationale_a}
```json
{json.dumps(graph_a, indent=2)[:6000]}
```

### CANDIDATE B:
**Rationale:** {rationale_b}
```json
{json.dumps(graph_b, indent=2)[:6000]}
```

Which candidate is more likely to improve the agent's win rate? Respond with JSON only.
"""
        try:
            response = _llm_call(self.judge_model, system_prompt, user_prompt)
            # Extract winner
            import re
            # Try JSON parse
            m = re.search(r'"winner"\s*:\s*"([AB])"', response)
            if m:
                return m.group(1)
            # Fallback: look for A or B
            if '"A"' in response or "'A'" in response:
                return "A"
            return "B"
        except Exception as e:
            print(f"  [SIFT Judge] Error in pairwise call: {e}")
            return "A"  # Default to first candidate on error

    def _fit_bradley_terry(self, node_ids: List[str]) -> Dict[str, float]:
        """Fit regularized Bradley-Terry model to the win matrix.

        Returns strength scores θ_i normalized so Σθ_i = n.
        """
        n = len(node_ids)
        if n <= 1:
            return {nid: 1.0 for nid in node_ids}

        λ = self.bt_prior
        # Initialize uniform
        θ = {nid: 1.0 for nid in node_ids}

        for _ in range(self.max_bt_iters):
            θ_new = {}
            max_change = 0.0

            for i in node_ids:
                numerator = 0.0
                denominator = 0.0
                for j in node_ids:
                    if i == j:
                        continue
                    w_ij = self.win_matrix.get(i, {}).get(j, 0)
                    w_ji = self.win_matrix.get(j, {}).get(i, 0)
                    total = w_ij + w_ji + 2 * λ
                    numerator += (w_ij + λ)
                    denominator += total / (θ[i] + θ[j])

                θ_new[i] = numerator / max(denominator, 1e-10)
                max_change = max(max_change, abs(θ_new[i] - θ[i]))

            # Normalize: Σθ = n
            total_θ = sum(θ_new.values())
            θ = {k: v * n / total_θ for k, v in θ_new.items()}

            if max_change < self.bt_tol:
                break

        return θ

    def rank_candidates(
        self,
        candidates: List[Tuple[Dict[str, Any], str, str]],
        domain_knowledge: str,
        diagnostics: Dict[str, Any],
    ) -> List[Tuple[Dict[str, Any], str, str, float]]:
        """Rank N candidate graphs using pairwise LLM judging + Bradley-Terry.

        Args:
            candidates: List of (graph, rationale, model_name) tuples
            domain_knowledge: Domain knowledge string for the judge
            diagnostics: Current island diagnostics

        Returns:
            Sorted list of (graph, rationale, model_name, bt_score) from best to worst.
        """
        if len(candidates) <= 1:
            return [(c[0], c[1], c[2], 1.0) for c in candidates]

        # Assign temporary IDs
        node_ids = [f"cand_{i}" for i in range(len(candidates))]

        # Initialize win matrix entries
        for nid in node_ids:
            if nid not in self.win_matrix:
                self.win_matrix[nid] = {}

        # Run all pairwise comparisons (for N=2-3 candidates this is manageable)
        n = len(candidates)
        for i in range(n):
            for j in range(i + 1, n):
                winner = self._judge_pairwise(
                    graph_a=candidates[i][0], rationale_a=candidates[i][1],
                    graph_b=candidates[j][0], rationale_b=candidates[j][1],
                    domain_knowledge=domain_knowledge,
                    diagnostics=diagnostics,
                )

                if winner == "A":
                    self.win_matrix.setdefault(node_ids[i], {})[node_ids[j]] = \
                        self.win_matrix.get(node_ids[i], {}).get(node_ids[j], 0) + 1
                else:
                    self.win_matrix.setdefault(node_ids[j], {})[node_ids[i]] = \
                        self.win_matrix.get(node_ids[j], {}).get(node_ids[i], 0) + 1

                print(f"  [SIFT Judge] {node_ids[i]} ({candidates[i][2]}) vs "
                      f"{node_ids[j]} ({candidates[j][2]}) → Winner: {'A' if winner == 'A' else 'B'} "
                      f"({candidates[i if winner == 'A' else j][2]})")

        # Fit Bradley-Terry
        bt_scores = self._fit_bradley_terry(node_ids)

        # Sort by BT score descending
        indexed = [(candidates[i], node_ids[i], bt_scores.get(node_ids[i], 1.0)) for i in range(n)]
        indexed.sort(key=lambda x: x[2], reverse=True)

        result = [(item[0][0], item[0][1], item[0][2], item[2]) for item in indexed]

        print(f"  [SIFT BT Ranking] " + " > ".join(
            f"{r[2]}(θ={r[3]:.3f})" for r in result
        ))

        # Clean up temporary win matrix entries (these are per-batch)
        for nid in node_ids:
            self.win_matrix.pop(nid, None)

        return result

    def rank_based_parent_sampling_weights(
        self,
        island_ids: List[str],
        alpha: float = 1.0,
        beta: float = 1.0,
        eta: float = 1.0,
    ) -> List[float]:
        """Compute SIFT rank-based parent sampling weights.

        P(i) ∝ exp(-α·r_bt(i) - β·r_acc(i) - η·log(1+v_i))
        For choosing which island to mutate next.
        """
        bt_scores = self._fit_bradley_terry(island_ids)

        # Sort by BT score to get ranks (0 = best)
        bt_sorted = sorted(island_ids, key=lambda x: bt_scores.get(x, 0), reverse=True)
        bt_ranks = {nid: rank for rank, nid in enumerate(bt_sorted)}

        # Sort by accuracy to get accuracy ranks
        acc_sorted = sorted(island_ids, key=lambda x: self.accuracy_scores.get(x, 0), reverse=True)
        acc_ranks = {nid: rank for rank, nid in enumerate(acc_sorted)}

        weights = []
        for nid in island_ids:
            r_b = bt_ranks.get(nid, len(island_ids) - 1)
            r_a = acc_ranks.get(nid, len(island_ids) - 1)
            v = self.visit_counts.get(nid, 0)
            w = math.exp(-alpha * r_b - beta * r_a - eta * math.log(1 + v))
            weights.append(w)

        # Normalize
        total_w = sum(weights)
        if total_w > 0:
            weights = [w / total_w for w in weights]
        else:
            weights = [1.0 / len(island_ids)] * len(island_ids)

        return weights
