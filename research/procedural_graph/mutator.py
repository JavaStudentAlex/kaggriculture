"""LLM-Driven Procedural Graph Mutator.

Reads the current champion graph, the latest evaluation diagnostics (losses, cash deficit),
and rejection memory, then prompts an LLM via the local proxy (http://localhost:8317/v1)
to propose a structured graph modification (tweak edge conditions, priorities, or insert nodes).
"""
from __future__ import annotations

import json
import random
import re
import urllib.request
from pathlib import Path
from typing import Any

LOCAL_PROXY_URL = "http://localhost:8317/v1/chat/completions"
DEFAULT_MODELS = [
    "local/gpt-6-astra@http://localhost:8317/v1",
    "local/gemini-3.1-pro-preview@http://localhost:8317/v1",
    "local/claude-sonnet-5@http://localhost:8317/v1"
]


def query_llm(model: str, prompt: str, system_prompt: str, temperature: float = 0.4) -> str:
    # Model ID extraction
    m_id = model.split("@")[0].replace("local/", "") if "@" in model else model
    req_body = {
        "model": m_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": 4096
    }
    data = json.dumps(req_body).encode("utf-8")
    req = urllib.request.Request(
        LOCAL_PROXY_URL,
        data=data,
        headers={"Content-Type": "application/json", "Authorization": "Bearer local-key"}
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        return res["choices"][0]["message"]["content"]


def extract_json_block(text: str) -> dict[str, Any]:
    """Extracts a JSON dictionary from markdown code blocks or raw text."""
    # Try finding markdown code block
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Fallback to outer brackets
    m2 = re.search(r"(\{.*\})", text, re.DOTALL)
    if m2:
        return json.loads(m2.group(1))
    raise ValueError(f"Could not extract valid JSON from response:\n{text[:300]}...")


def mutate_graph(
    parent_graph_path: Path,
    eval_diagnostics: dict[str, Any],
    rejection_history: list[dict[str, Any]],
    model_override: str | None = None
) -> tuple[dict[str, Any], str, str]:
    """Proposes a mutation to the procedural graph using diagnostic traces.
    
    Returns: (mutated_graph_dict, model_name, rationale)
    """
    with open(parent_graph_path, encoding="utf-8") as f:
        parent_graph = json.load(f)

    model = model_override or random.choice(DEFAULT_MODELS)

    system_prompt = """You are an expert game theory and autonomous systems strategist specializing in simulation agent optimization.
Your role is to optimize a Procedural Execution Graph for an agent playing Kaggriculture.
You only edit the GRAPH STRUCTURE (nodes, edges, priorities, conditions, guidance, and pitfalls).
You do NOT write python code. You return a strictly validated JSON matching the Procedural Graph schema.
"""

    prompt = f"""### CONTEXT:
We are evolving the Procedural Graph for our Kaggriculture agent.
The graph arbitrates decisions across high-priority nodes (Wage Defense, Shed Headroom, Town Shop Preempt, Oracle Frontrun, Farm Execution) within a strict 10-order engine cap.

### CURRENT GRAPH TOPOLOGY:
```json
{json.dumps(parent_graph, indent=2)}
```

### LATEST EVALUATION DIAGNOSTICS:
- Games Played: {eval_diagnostics.get('total_games', 0)}
- Win Rate: {eval_diagnostics.get('win_rate', 0.0) * 100:.1f}%
- Average Cash: ${eval_diagnostics.get('avg_cash', 0.0):,.2f}
- Opponent Average Cash: ${eval_diagnostics.get('avg_opp_cash', 0.0):,.2f}
- Net Cash Deficit: ${eval_diagnostics.get('cash_diff', 0.0):,.2f}
- Worst Losses:
```json
{json.dumps(eval_diagnostics.get('worst_losses', []), indent=2)}
```

### RECENT REJECTED EDITS (Do not repeat these exact strategies):
```json
{json.dumps(rejection_history[-3:] if rejection_history else [], indent=2)}
```

### OBJECTIVE:
Analyze the loss traces (where our agent averaged ~$81k while champions averaged ~$98k+).
Propose a targeted strategic mutation (Delta G) to close the cash gap against the champions:
1. Consider tuning condition thresholds (e.g. earlier town shop diversification, dynamic seed budgets, or pacing sales).
2. Or adjust edge priorities (e.g. ensuring town shop harvest capturing is prioritized when active).
3. Or introduce a new strategic guard/node.

### OUTPUT FORMAT:
Return a JSON object with:
{{
  "rationale": "<2-3 sentence strategic explanation of why this graph change fixes the cash gap>",
  "graph": <THE COMPLETE UPDATED GRAPH JSON with nodes and edges>
}}
"""

    raw_resp = query_llm(model, prompt, system_prompt)
    data = extract_json_block(raw_resp)
    
    rationale = data.get("rationale", "Strategic graph mutation")
    mutated_graph = data.get("graph", data)

    # Basic structural validation
    if "nodes" not in mutated_graph or "edges" not in mutated_graph:
        raise ValueError("Mutated graph is missing required 'nodes' or 'edges' keys.")
    
    # Ensure priorities are integers
    for edge in mutated_graph["edges"]:
        if "priority" not in edge:
            edge["priority"] = 99
        edge["priority"] = int(edge["priority"])

    return mutated_graph, model, rationale
