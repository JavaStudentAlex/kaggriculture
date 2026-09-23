"""Two-Level Novelty Inspection Pipeline for Procedural Graphs.

Level 1 (Embeddings):
  Uses Qwen3-Embedding 8B via Ollama (4096 dimensions).
  Computes cosine similarity against existing evaluated graphs.
  Threshold: 0.985 (provisional, not calibrated for Qwen). High similarity triggers
  semantic review; embeddings do not prove behavioral equivalence.

Level 2 (LLM Judge):
  Uses gemini-3.1-pro-preview with 'xhigh' reasoning effort.
  Examines structural graph diff against the collision program.
  Rounds decision to True (novel) or False (redundant).
"""
from __future__ import annotations

import json
import math
import os
import urllib.request
from typing import Any

from call_telemetry import record_call

LOCAL_PROXY_URL = "http://localhost:8317/v1/chat/completions"
OLLAMA_EMBED_URL = os.environ.get("OLLAMA_EMBED_URL", "http://127.0.0.1:11434/api/embed")
OLLAMA_EMBED_MODEL = "qwen3-embedding:8b"
EMBEDDING_IDENTITY = "ollama:qwen3-embedding:8b:4096:canonical-json-v1:full-input"
EMBEDDING_DIM = 4096


def validate_embedding(embedding):
    if (not isinstance(embedding, list) or len(embedding) != EMBEDDING_DIM
            or not all(isinstance(x, (int, float)) and not isinstance(x, bool)
                       and math.isfinite(x) for x in embedding)
            or not any(embedding)):
        raise ValueError("Expected a finite, nonzero 4096-dimensional Qwen embedding")
    return embedding


def get_graph_embedding(graph_dict: dict[str, Any]) -> list[float]:
    """Generates 4,096-dim dense embedding using Qwen3-embedding:8b via Ollama."""
    repr_str = json.dumps(graph_dict, sort_keys=True, separators=(",", ":"), allow_nan=False)
    payload = {
        "model": OLLAMA_EMBED_MODEL,
        "input": repr_str,
        "truncate": False,
        "options": {"num_ctx": 8192},
        "keep_alive": "30m"
    }
    req = urllib.request.Request(
        OLLAMA_EMBED_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        return validate_embedding(res["embeddings"][0])


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a or not b:
        raise ValueError("Cannot compare empty or incompatible embedding spaces")
    if not all(math.isfinite(x) for x in a + b):
        raise ValueError("Non-finite embedding")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        raise ValueError("Zero-norm embedding")
    val = dot / (norm_a * norm_b)
    if not math.isfinite(val):
        raise ValueError("Non-finite cosine similarity")
    return max(-1.0, min(1.0, val))


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
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            res = json.loads(resp.read().decode("utf-8"))
            content = res["choices"][0]["message"]["content"]
        record_call("novelty_judge_llm", "success", model="gemini-3.1-pro-preview")
    except Exception as exc:
        record_call("novelty_judge_llm", "failure", model="gemini-3.1-pro-preview", error=exc)
        raise
        
    try:
        import re
        m = re.search(r"(\{.*\})", content, re.DOTALL)
        if m:
            verdict = json.loads(m.group(1))
            if isinstance(verdict.get("is_novel"), bool):
                return verdict["is_novel"], str(verdict.get("reason", ""))
    except Exception:
        pass

    # An invalid verdict must not certify novelty.
    return False, "Invalid novelty verdict; candidate not certified novel"


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
