"""Supervisor / Meta-Reviewer for Procedural Graph Evolution.

Mirrors Shinka's meta-supervisor:
- Model: gpt-6-astra with 'xhigh' reasoning effort
- Cadence: Runs every 5 generations
- Objective: Synthesizes population trends, diagnoses recurring plateaus,
  and emits strategic recommendations injected into subsequent mutation prompts.
"""
from __future__ import annotations

import json
import urllib.request
from typing import Any

LOCAL_PROXY_URL = "http://localhost:8317/v1/chat/completions"


def run_meta_supervisor(
    best_graph: dict[str, Any],
    recent_evals: list[dict[str, Any]],
    recent_rejections: list[dict[str, Any]],
    current_gen: int
) -> list[str]:
    """gpt-6-astra reviews the archive and issues strategic recommendations."""
    system_prompt = """You are the Lead Evolutionary Systems Supervisor and Grandmaster Strategist overseeing a population of self-evolving Procedural Execution Graphs.
Every 5 generations, your job is to analyze what worked, what failed, and where the population is stagnating.
You synthesize 3 to 5 sharp, concrete strategic recommendations for the mutation models to guide their graph edits over the next 5 generations.
Focus on economic fundamentals: market timing, town shop margins, workforce productivity, and resource bottlenecks."""

    prompt = f"""### EVOLUTION CHECKPOINT: Generation {current_gen}

### CURRENT CHAMPION GRAPH:
```json
{json.dumps(best_graph, indent=2)}
```

### RECENT GENERATION RESULTS (Past 5 Gens):
```json
{json.dumps(recent_evals[-5:], indent=2)}
```

### RECENT REJECTED CANDIDATES & FAILURE RATIONALES:
```json
{json.dumps(recent_rejections[-5:], indent=2)}
```

### INSTRUCTIONS:
Analyze the gap between our champion and the opponent pool.
Identify what structural changes or condition calibrations are needed to break through.
Return a JSON array of 3-5 concrete strategic recommendations:
```json
[
  "Recommendation 1: ...",
  "Recommendation 2: ...",
  "Recommendation 3: ..."
]
```
"""
    req_body = {
        "model": "gpt-6-astra",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 16384,
        "reasoning_effort": "xhigh"
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

    try:
        import re
        m = re.search(r"(\[.*\])", content, re.DOTALL)
        if m:
            recs = json.loads(m.group(1))
            if isinstance(recs, list) and all(isinstance(x, str) for x in recs):
                return recs
    except Exception:
        pass

    return [
        "Prioritize capturing unlocked Town Shop demands with dedicated seed allocations.",
        "Calibrate shed headroom dump thresholds to prevent inventory discards on high-yield seeds.",
        "Enforce midnight wage reserve buffer at hour >= 18 to ensure full workforce retention."
    ]
