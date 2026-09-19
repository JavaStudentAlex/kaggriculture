"""BAM (Multi-Armed Bandit) Frontier LLM Procedural Graph Analyzer & Mutator.

Uses UCB1 over the 6 frontier LLM arms to analyze Jev match traces, diagnose
defeat causes, and synthesize strategic mutations (Delta G) to the Procedural Graph:
1. Calibrates edge guidance and pitfalls to steer Jev's Choice criteria.
2. Refines activation conditions and numerical thresholds.
3. Mutates graph topology (adds specialized nodes, prunes unproductive branches).
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

LOCAL_PROXY_URL = "http://localhost:8317/v1/chat/completions"


def extract_json_block(text: str) -> Dict[str, Any]:
    """Extracts a valid JSON object from LLM response markdown or raw text."""
    # 1. Try finding markdown code block
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if m:
        block = m.group(1).strip()
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            pass

    # 2. Balanced brace search from first '{' to last '}'
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidate = text[first_brace:last_brace + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract valid JSON from response:\n{text[:400]}...")



class BAMGraphMutator:
    """Multi-Armed Bandit LLM Mutator for evolving the Procedural Decision Graph."""

    def __init__(self, proxy_url: str = LOCAL_PROXY_URL, proxy_key: str = "local-key"):
        self.proxy_url = proxy_url
        self.proxy_key = proxy_key

    def mutate_graph(
        self,
        model_name: str,
        current_graph: Dict[str, Any],
        match_diagnostics: Dict[str, Any],
        decision_trace: List[Dict[str, Any]],
        domain_knowledge: str,
        rejections: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], str]:
        """Prompts the selected frontier LLM to analyze the Jev match trace and mutate the graph."""
        # Extract clean model identifier
        m_id = model_name.split("@")[0].replace("local/", "") if "@" in model_name else model_name

        system_prompt = (
            "You are an elite autonomous game systems architect and Kaggle Grandmaster.\n"
            "Your mission: evolve and optimize the Procedural Decision Graph for an autonomous agent "
            "playing Kaggriculture, guided at runtime by TypeSafe AI's Jev System One actor.\n"
            "You must return ONLY a strict JSON object containing 'rationale' and 'graph'.\n"
            "Do not output markdown explanations outside the JSON."
        )

        # Filter and sample decision trace for prompt brevity
        sampled_trace = []
        for d in decision_trace:
            if d.get("step", 0) in (0, 6, 12, 18, 24, 48, 72, 96, 120, 240, 360, 480, 600, 719) or d.get("emergency_lock"):
                sampled_trace.append({
                    "step": d.get("step"),
                    "day": d.get("day"),
                    "hour": d.get("hour"),
                    "cash": d.get("state_snapshot", {}).get("cash"),
                    "branch_selected": d.get("selected_branch"),
                    "confidence": d.get("confidence"),
                    "emergency_lock": d.get("emergency_lock"),
                    "expansion_freeze": d.get("expansion_freeze")
                })

        prompt = f"""### COMPETITIVE KAGGLE PLAYBOOK & DOMAIN KNOWLEDGE:
{domain_knowledge}

### CURRENT PROCEDURAL GRAPH:
```json
{json.dumps(current_graph, indent=2)}
```

### LATEST MATCH DIAGNOSTICS:
- Opponent Champion: {match_diagnostics.get('champ_name', 'Unknown')}
- Jev Agent Final Cash: ${match_diagnostics.get('cand_cash', 0.0):,.2f}
- Opponent Final Cash: ${match_diagnostics.get('opp_cash', 0.0):,.2f}
- Net Cash Deficit: ${match_diagnostics.get('cash_diff', 0.0):,.2f}
- Outcome: {match_diagnostics.get('outcome', 'LOSS')}

### JEV ACTOR DECISION TRACE SAMPLE:
```json
{json.dumps(sampled_trace[:20], indent=2)}
```

### RECENT REJECTED MUTATIONS (Do not repeat these exact mistakes):
```json
{json.dumps(rejections[-3:] if rejections else [], indent=2)}
```

### STRATEGIC OBJECTIVE:
Analyze the match diagnostics and Jev's decision trace. Identify why our agent underperformed:
1. Did Jev trigger emergency defense too early (e.g. Day 0-3 when cash was ample and compounding was required)?
2. Is the graph missing essential strategic phases (e.g. 'capital_compounding' for Days 1-12, or 'strawberry_watering_loop')?
3. Are edge conditions, priorities, guidance directives, or pitfall warnings misaligned with winning Kaggle strategies?

Synthesize a targeted Strategic Mutation (Delta G) to the Procedural Graph:
- Add, modify, or prune nodes and edges.
- Refine the 'guidance' and 'pitfalls' on edges so Jev's System 1 Choice criteria explicitly steers it correctly.
- Calibrate conditions (e.g. ensuring wage defense only triggers when hour >= 18 AND day > 1 AND cash < midnight_payroll_due).

### OUTPUT SCHEMA (Strict JSON):
{{
  "rationale": "<2-4 sentence causal explanation of why this graph mutation fixes Jev's decisions and closes the cash deficit>",
  "graph": {{
    "version": "<incremented version, e.g. 1.1.0>",
    "name": "<descriptive name>",
    "description": "<summary of changes>",
    "nodes": [ ... ],
    "edges": [ ... ]
  }}
}}
"""

        req_body = {
            "model": m_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 16384
        }

        data = json.dumps(req_body).encode("utf-8")
        req = urllib.request.Request(
            self.proxy_url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.proxy_key}"
            }
        )

        with urllib.request.urlopen(req, timeout=180) as resp:
            raw_text = resp.read().decode("utf-8")
            response_json = json.loads(raw_text)
            content = response_json["choices"][0]["message"]["content"]

        parsed = extract_json_block(content)
        rationale = parsed.get("rationale", "BAM strategic graph mutation")
        mutated_graph = parsed.get("graph", parsed)

        # Validate schema
        if "nodes" not in mutated_graph or "edges" not in mutated_graph:
            raise ValueError("Mutated graph is missing required 'nodes' or 'edges' keys.")

        for edge in mutated_graph["edges"]:
            edge["priority"] = int(edge.get("priority", 99))
            if "attributes" not in edge:
                edge["attributes"] = {}

        return mutated_graph, rationale
