"""Two-Level Novelty Inspection Pipeline for Procedural Graphs.

Level 1 (Embeddings):
  Uses local sentence-transformers (all-MiniLM-L6-v2, 384-dim, fast local CPU/GPU).
  Computes cosine similarity against existing evaluated graphs.
  Threshold: 0.985. If >= 0.985, flagged as potential near-duplicate.

Level 2 (LLM Judge):
  Uses gemini-3.1-pro-preview with 'xhigh' reasoning effort.
  Examines structural graph diff against the collision program.
  Rounds decision to True (novel) or False (redundant).
"""
from __future__ import annotations

import json
import math
import urllib.request
from typing import Any

from sentence_transformers import SentenceTransformer

LOCAL_PROXY_URL = "http://localhost:8317/v1/chat/completions"

# Global lazy-loaded embedding model
_EMBED_MODEL: SentenceTransformer | None = None


def get_embed_model() -> SentenceTransformer:
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        _EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBED_MODEL


def get_graph_embedding(graph_dict: dict[str, Any]) -> list[float]:
    """Generates embedding of the graph topology, guidance, and attributes."""
    repr_str = json.dumps(graph_dict, sort_keys=True)
    model = get_embed_model()
    emb = model.encode(repr_str)
    return emb.tolist()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def judge_novelty_with_llm(
    cand_graph: dict[str, Any],
    collided_graph: dict[str, Any],
    sim_score: float
) -> tuple[bool, str]:
    """Level 2 inspection: gemini-3.1-pro-preview rules on substantive novelty."""
    system_prompt = """You are a strict research judge evaluating whether a candidate Procedural Execution Graph represents a genuinely novel strategic mutation or a superficial near-duplicate.
Focus on operational meaning: Do changes alter execution order, conditions, resource buffers, or priority arbitration?
If changes are purely cosmetic text rephrasing without behavioral impact, reject as NOT novel."""

    prompt = f"""### CANDIDATE GRAPH:
```json
{json.dumps(cand_graph, indent=2)}
```

### EXISTING GRAPH (Cosine Similarity: {sim_score:.4f}):
```json
{json.dumps(collided_graph, indent=2)}
```

### TASK:
Determine if the Candidate Graph introduces a substantively novel strategy, decision branch, or constraint compared to the existing graph.
Respond in strict JSON:
{{
  "is_novel": true | false,
  "reason": "<1-2 sentence justification of why this is or is not substantively novel>"
}}
"""
    req_body = {
        "model": "gemini-3.1-pro-preview",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 16384,
        "reasoning_effort": "xhigh"
    }
    data = json.dumps(req_body).encode("utf-8")
    req = urllib.request.Request(
        LOCAL_PROXY_URL,
        data=data,
        headers={"Content-Type": "application/json", "Authorization": "Bearer local-key"}
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        content = res["choices"][0]["message"]["content"]
        
    try:
        import re
        m = re.search(r"(\{.*\})", content, re.DOTALL)
        if m:
            verdict = json.loads(m.group(1))
            return bool(verdict.get("is_novel", False)), str(verdict.get("reason", ""))
    except Exception:
        pass

    # Default to accepting if parser fails
    return True, "Parsed fallback"


class NoveltyChecker:
    def __init__(self, sim_threshold: float = 0.985):
        self.sim_threshold = sim_threshold
        self.archive_embeddings: list[tuple[dict[str, Any], list[float]]] = []

    def register(self, graph_dict: dict[str, Any], embedding: list[float] | None = None) -> None:
        if embedding is None:
            embedding = get_graph_embedding(graph_dict)
        self.archive_embeddings.append((graph_dict, embedding))

    def check(self, cand_graph: dict[str, Any]) -> tuple[bool, str, float]:
        """Runs the two-level novelty check.
        
        Returns: (is_accepted, reason, max_similarity)
        """
        if not self.archive_embeddings:
            cand_emb = get_graph_embedding(cand_graph)
            self.register(cand_graph, cand_emb)
            return True, "Initial seed graph", 0.0

        cand_emb = get_graph_embedding(cand_graph)
        max_sim = -1.0
        most_similar_graph = None

        for prev_graph, prev_emb in self.archive_embeddings:
            sim = cosine_similarity(cand_emb, prev_emb)
            if sim > max_sim:
                max_sim = sim
                most_similar_graph = prev_graph

        # Level 1 check
        if max_sim < self.sim_threshold:
            self.register(cand_graph, cand_emb)
            return True, f"Level 1 Pass: Similarity {max_sim:.4f} < {self.sim_threshold}", max_sim

        if most_similar_graph is None:
            self.register(cand_graph, cand_emb)
            return True, "No comparison graph", 0.0

        # Level 2 check
        is_novel, reason = judge_novelty_with_llm(cand_graph, most_similar_graph, max_sim)
        if is_novel:
            self.register(cand_graph, cand_emb)
            return True, f"Level 2 Pass: LLM judged novel despite similarity {max_sim:.4f} ({reason})", max_sim
        else:
            return False, f"Level 2 Rejection: Duplicate of existing graph (sim {max_sim:.4f}: {reason})", max_sim
