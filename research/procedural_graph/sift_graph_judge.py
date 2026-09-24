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
judge them pairwise, and push only the BT-top-ranked candidate to the gauntlet.

The judge must see what differs between two candidates: `rank_edits` shows each edit's
control changes (graph_edits.diff), `rank_candidates` the JSON paths where two full
graphs differ (a 44 kB graph cut at a fixed length hides every mutation). A reply
without a valid verdict is no vote, never a default winner.
"""
from __future__ import annotations

import json
import math
import os
import urllib.request

from call_telemetry import record_call
from bam_graph_mutator import extract_json_block
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


def json_differences(a: Any, b: Any, path: str = "$", out: Optional[list] = None, limit: int = 60) -> list:
    """[(path, value in a, value in b)] for every leaf where two JSON documents differ."""
    out = [] if out is None else out
    if len(out) >= limit:
        return out
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b), key=str):
            json_differences(a.get(key, "<absent>"), b.get(key, "<absent>"), f"{path}.{key}", out, limit)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            json_differences(x, y, f"{path}[{i}]", out, limit)
    elif a != b:
        out.append((path, a, b))
    return out


def _verdict(response: str) -> Optional[str]:
    """'A' or 'B' from a JSON reply; None when the reply carries no valid verdict."""
    try:
        winner = extract_json_block(response).get("winner")
    except Exception:
        return None
    return winner if winner in ("A", "B") else None


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
        text_a: str,
        rationale_a: str,
        text_b: str,
        rationale_b: str,
        knowledge: str,
        context: str,
    ) -> Optional[str]:
        """Ask the judge LLM which candidate is more promising: 'A', 'B' or None (no vote)."""
        system_prompt = (
            "You are an expert judge comparing two candidate changes to a policy graph "
            "for an autonomous farming agent in the Kaggriculture Kaggle competition.\n"
            "Decide which change is MORE LIKELY to increase the agent's cash margin against "
            "the opponent pool. Judge the in-game mechanism each change affects, the "
            "evidence in the measured results, and the risk of hurting other opponents.\n"
            "You MUST respond with ONLY a JSON object: "
            '{"reasoning": "<2-3 sentences>", "winner": "A" or "B"}'
        )
        user_prompt = f"""### VERIFIED KNOWLEDGE AND MEASURED RESULTS:
{knowledge[:6000]}

### CURRENT CHAMPION AND ITS RESULTS:
{context[:4000]}

### CANDIDATE A
**Rationale:** {rationale_a}
**Changes:**
{text_a}

### CANDIDATE B
**Rationale:** {rationale_b}
**Changes:**
{text_b}

Which candidate is more likely to improve the agent's cash margin? Respond with JSON only.
"""
        try:
            response = _llm_call(self.judge_model, system_prompt, user_prompt)
        except Exception as e:
            print(f"  [SIFT Judge] Error in pairwise call (no vote): {e}")
            return None
        verdict = _verdict(response)
        if verdict is None:
            print(f"  [SIFT Judge] Reply without a valid verdict (no vote): {response[:200]!r}")
        return verdict

    @staticmethod
    def _graph_changes(graph: Dict[str, Any], other: Dict[str, Any]) -> str:
        diffs = json_differences(other, graph)
        if not diffs:
            return "(identical to the other candidate)"
        return "\n".join(f"{path}: {json.dumps(mine)[:300]} (other candidate: {json.dumps(theirs)[:300]})"
                         for path, theirs, mine in diffs)

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

        losses = (diagnostics.get("worst_differential_losses")
                  or diagnostics.get("worst_cloud_losses") or [])[:3]
        context = json.dumps({
            "island": diagnostics.get("island_name", ""),
            "focus": diagnostics.get("island_focus", ""),
            "current_win_rate": diagnostics.get("current_champ_win_rate", 0.0),
            "recent_losses": losses,
        }, indent=2)
        # Run all pairwise comparisons (for N=2-3 candidates this is manageable)
        n = len(candidates)
        for i in range(n):
            for j in range(i + 1, n):
                winner = self._judge_pairwise(
                    self._graph_changes(candidates[i][0], candidates[j][0]), candidates[i][1],
                    self._graph_changes(candidates[j][0], candidates[i][0]), candidates[j][1],
                    domain_knowledge, context,
                )
                if winner is None:
                    continue

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

    def rank_edits(
        self,
        proposals: List[Dict[str, Any]],
        knowledge: str,
        context: str,
    ) -> Tuple[List[Tuple[Dict[str, Any], float]], int]:
        """Rank edit proposals ({'changes': [...], 'rationale', 'model', ...}) pairwise.

        Each candidate is shown as its control changes relative to the island champion.
        Returns ([(proposal, BT strength)] best first, number of valid votes); with no
        valid vote every strength is equal and the generation order is kept.
        """
        n = len(proposals)
        if n <= 1:
            return [(p, 1.0) for p in proposals], 0
        ids = [f"edit_{i}" for i in range(n)]
        for nid in ids:
            self.win_matrix[nid] = {}
        votes = 0
        for i in range(n):
            for j in range(i + 1, n):
                winner = self._judge_pairwise(
                    "\n".join(proposals[i]["changes"]), proposals[i]["rationale"],
                    "\n".join(proposals[j]["changes"]), proposals[j]["rationale"],
                    knowledge, context)
                if winner is None:
                    continue
                votes += 1
                w, l = (ids[i], ids[j]) if winner == "A" else (ids[j], ids[i])
                self.win_matrix[w][l] = self.win_matrix[w].get(l, 0) + 1
                print(f"  [SIFT Judge] {proposals[i]['model']} vs {proposals[j]['model']} -> "
                      f"{proposals[i if winner == 'A' else j]['model']}")
        strengths = self._fit_bradley_terry(ids)
        order = sorted(range(n), key=lambda k: -strengths[ids[k]])  # stable: ties keep order
        for nid in ids:
            self.win_matrix.pop(nid, None)
        return [(proposals[k], strengths[ids[k]]) for k in order], votes

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
