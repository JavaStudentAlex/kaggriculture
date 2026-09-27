"""BAM (Multi-Armed Bandit) Frontier LLM Procedural Graph Analyzer & Mutator.

`mutate_edit` asks the bandit-selected LLM for an edit of the graph's executable
controls (graph_edits.py): champion parameters on chain nodes, market-stage toggles,
the dispatch order and the surgical switches. That is what the merged runtime
(`runtime: hazel_merged_v1`) executes; its prose does not.

`mutate_graph` is the legacy prose mutation (nodes/edges guidance) for pre-merge graphs.
It refuses merged graphs: its output has no runtime, and agent_graph.py would silently
play the preserved legacy agent instead of the graph being evolved.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
from pathlib import Path

from call_telemetry import record_call
from shinka_graph_mab import MODEL_SPECS
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
        if current_graph.get("runtime") == "hazel_merged_v1":
            raise ValueError("mutate_graph rewrites prose that the merged runtime does not execute; "
                             "use mutate_edit for hazel_merged_v1 graphs")
        # Extract clean model identifier
        m_id = model_name.split("@")[0].replace("local/", "") if "@" in model_name else model_name

        system_prompt = (
            "You are an elite autonomous game systems architect and Kaggle Grandmaster.\n"
            "Your mission: evolve and optimize the Procedural Decision Graph for an autonomous agent "
            "playing Kaggriculture, guided at runtime by TypeSafe AI's Jev System One actor.\n"
            "You must return ONLY a strict JSON object containing 'rationale' and 'graph'.\n"
            "Do not output markdown explanations outside the JSON."
        )

        # Extract mined in-game forensics or sampled Jev traces
        forensic_summary_lines = []
        sampled_trace = []

        for d in decision_trace:
            if isinstance(d, dict) and "primary_failure_mode" in d:
                t_str = " -> ".join([f"D{pt['day']}h{pt['hour']}: ${pt['cash']:,.0f} ({pt['hands']}h/{pt['shed_items']}s)" for pt in d.get("timeline", [])])
                forensic_summary_lines.append(
                    f"[Defeat vs {d.get('champ_name')} (Seed {d.get('seed')}, Seat {d.get('seat')}) | Deficit: -${d.get('deficit', 0):,.2f}]\n"
                    f"• Primary Bottleneck: {d.get('primary_failure_mode')} -> {d.get('failure_diagnosis')}\n"
                    f"• Forensics: Peak Hands: {d.get('peak_hands')}, Peak Shed: {d.get('peak_shed')}/100, "
                    f"Min Midnight Cash: ${d.get('min_midnight_cash')}, Stranded Crops: {d.get('unharvested_ripe_crops')}\n"
                    f"• Trajectory (Day/Hour: Cash (Hands/Shed)): {t_str}"
                )
            elif isinstance(d, dict) and (d.get("step", 0) in (0, 6, 12, 18, 24, 48, 72, 96, 120, 240, 360, 480, 600, 719) or d.get("emergency_lock")):
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

        if forensic_summary_lines:
            forensic_block = "### MINED FORENSIC TRACES FROM IN-GAME LOSSES:\n" + "\n\n".join(forensic_summary_lines[:5]) + "\n"
        elif sampled_trace:
            forensic_block = "### JEV ACTOR DECISION TRACE SAMPLE:\n```json\n" + json.dumps(sampled_trace[:20], indent=2) + "\n```\n"
        else:
            forensic_block = "### LATEST MATCH DIAGNOSTICS:\n" + json.dumps(match_diagnostics, indent=2) + "\n"

        prompt = f"""### COMPETITIVE KAGGLE PLAYBOOK & DOMAIN KNOWLEDGE:
{domain_knowledge}

### CURRENT PROCEDURAL GRAPH:
```json
{json.dumps(current_graph, indent=2)}
```

{forensic_block}
### RECENT REJECTED MUTATIONS (Do not repeat these exact mistakes):
```json
{json.dumps(rejections[-3:] if rejections else [], indent=2)}
```

### STRATEGIC OBJECTIVE:
Analyze the mined game forensics above. Identify the exact causal mechanism behind the deficit:
1. Did the farm suffer a payroll insolvency crisis (cash < payroll at hour >= 18)?
2. Did workers discard crops or stall due to shed capacity choke (shed fullness >= 96)?
3. Did the agent leave unharvested ripe crops on the board at step 720 because terminal liquidation started too late?
4. Did the workforce stall below the elite 12-hand scaling target?

Synthesize a targeted Strategic Mutation (Delta G) to the Procedural Graph to eliminate this exact failure mode:
- Add, modify, or prune nodes and edges.
- Refine the 'guidance' and 'pitfalls' on edges so runtime execution avoids this trap.
- Calibrate conditions, priorities, and numerical thresholds.

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

        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                raw_text = resp.read().decode("utf-8")
                response_json = json.loads(raw_text)
                content = response_json["choices"][0]["message"]["content"]
            record_call("outer_mutator_llm", "success", model=m_id)
        except Exception as exc:
            record_call("outer_mutator_llm", "failure", model=m_id, error=exc)
            raise

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

    def _chat(self, model: str, system_prompt: str, prompt: str, source: str, timeout: float = 600.0) -> str:
        spec = MODEL_SPECS.get(model, {})
        body = {"model": model,
                "messages": [{"role": "system", "content": system_prompt},
                             {"role": "user", "content": prompt}],
                "temperature": spec.get("temperature", 0.3), "max_tokens": 16384}
        if spec.get("reasoning_effort"):
            body["reasoning_effort"] = spec["reasoning_effort"]
        req = urllib.request.Request(
            self.proxy_url, data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.proxy_key}"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                content = json.loads(resp.read().decode("utf-8"))["choices"][0]["message"]["content"]
            record_call(source, "success", model=model)
            return content
        except Exception as exc:
            record_call(source, "failure", model=model, error=exc)
            raise

    def mutate_edit(
        self,
        model_name: str,
        controls: str,
        focus: str,
        results: str,
        history: List[str],
        guidance: List[str],
        knowledge: str,
        error: Optional[str] = None,
        ideas: Optional[List[str]] = None,
    ) -> Tuple[Dict[str, Any], str]:
        """Ask one LLM for an edit of the executable controls; returns (edit, rationale).

        The edit is not checked here: graph_edits.apply_edit and validate_graph do that,
        and their error text comes back through `error` on the next attempt.
        """
        system_prompt = (
            "You tune an executable policy graph for the Kaggriculture Kaggle simulation "
            "(two farms, 720 turns, final cash decides). The graph runs a fixed production engine "
            "and our layers on top of it; you can change only the listed executable controls. Propose "
            "ONE small, mechanism-driven edit and return ONLY a JSON object with keys 'rationale' and 'edit'.")
        history_block = "\n".join(f"- {line}" for line in history[-12:]) or "- none yet"
        guidance_block = "\n".join(f"- {line}" for line in guidance[-5:]) or "- none"
        ideas_block = "\n".join(f"- {line}" for line in (ideas or [])[:20]) or "- none"
        retry = f"\n### YOUR PREVIOUS ATTEMPT WAS REJECTED\n{error}\nFix exactly this problem.\n" if error else ""
        prompt = f"""### HOW A CANDIDATE IS JUDGED
Your edit is applied to the island champion below. The resulting graph plays a paired
gauntlet: the same seeds against every opponent listed under CHAMPION RESULTS (the
knowledge section describes them), plus head-to-head games against the champion itself. Per game, its cash margin is compared with the champion's
margin on the same game; it is promoted only if an exact sign test over the games that
changed is significant (p <= 0.05) with a positive mean change. Edits that change nothing
in play are wasted; edits that help one opponent and hurt the others fail.

### VERIFIED ENGINE FACTS AND MEASURED RESULTS
{knowledge}

### ISLAND FOCUS
{focus}
You may change any control, but prefer this area.

### EXECUTABLE CONTROLS (current values of the island champion)
{controls}

### CHAMPION RESULTS IN THE GAUNTLET
{results}

### EDITS ALREADY PLAYED IN THIS RUN (do not repeat; learn from them. [this island] edits were measured against this champion's line, the others against their own island's champion)
{history_block}

### IDEAS TO EXPLORE (from the research lead; test them, do not assume they work)
{ideas_block}

### SUPERVISOR GUIDANCE
{guidance_block}
{retry}
### OUTPUT (strict JSON, nothing else)
{{"rationale": "<2-4 sentences: the in-game mechanism you expect to change and why cash improves>",
  "edit": {{"parameters": {{"<_CONSTANT>": <value of the same type, or null for the submitted value>}},
            "stages": {{"<market stage id>": true or false}},
            "dispatch_order": "submitted" or "sells_first",
            "surgical": {{"enabled": true or false, "overrides": {{...}}}},
            "engine_parameters": {{"<ENGINE CONSTANT>": <value of the same type, or null for the engine default>}},
            "channels": {{"farmer" | "hands" | "market" | "oracle_guard" | "rival_counter": true or false}},
            "experimental": {{"<switch>": true or false}},
            "counters": {{"<rival class>": {{"<ENGINE CONSTANT or _OG_* parameter>": <value, or null>}} or null}}}}}}
Include only the keys you change (usually 1-3 controls); keys that the CONTROLS section does not
list do not exist for this graph. Tuples and lists are JSON arrays; a table keyed by shop pairs
takes keys "SHOP_A|SHOP_B" and must be given complete (every key of the current value).
"""
        content = self._chat(model_name, system_prompt, prompt, "outer_mutator_llm")
        parsed = extract_json_block(content)
        edit = parsed.get("edit")
        if not isinstance(edit, dict) or not edit:
            raise ValueError("response has no non-empty 'edit' object")
        return edit, str(parsed.get("rationale", "")).strip()
