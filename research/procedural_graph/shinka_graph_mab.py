"""UCB1 Multi-Armed Bandit for LLM Model Selection in Procedural Graph Evolution.

Mirrors Shinka's dynamic model selection:
- Arms: 6 frontier models
- Formula: UCB1 with exploration constant sqrt(2)
- Rewards: Based on candidate cash improvement + crowning bonus
- State persistence: bandit_state.json
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

MODEL_SPECS: dict[str, dict[str, Any]] = {
    "gpt-6-astra": {
        "uri": "local/gpt-6-astra@http://localhost:8317/v1",
        "reasoning_effort": "xhigh",
        "temperature": 0.3
    },
    "gpt-6-sol": {
        "uri": "local/gpt-6-sol@http://localhost:8317/v1",
        "reasoning_effort": "xhigh",
        "temperature": 0.3
    },
    "gpt-6-luna": {
        "uri": "local/gpt-6-luna@http://localhost:8317/v1",
        "reasoning_effort": "xhigh",
        "temperature": 0.3
    },
    "gemini-3.1-pro-preview": {
        "uri": "local/gemini-3.1-pro-preview@http://localhost:8317/v1",
        "reasoning_effort": "xhigh",
        "temperature": 0.3
    },
    "gemini-3.8-flash": {
        "uri": "local/gemini-3.8-flash@http://localhost:8317/v1",
        "reasoning_effort": "high",
        "temperature": 0.3
    },
    "claude-opus-5.5": {
        "uri": "local/claude-opus-5.5@http://localhost:8317/v1",
        "reasoning_effort": "xhigh",
        "temperature": 0.3
    },
    "claude-sonnet-5": {
        "uri": "local/claude-sonnet-5@http://localhost:8317/v1",
        "reasoning_effort": "xhigh",
        "temperature": 0.3
    }
}


class UCB1Bandit:
    def __init__(self, state_file: Path, exploration_c: float = 1.414, arms: list | None = None):
        self.state_file = state_file
        self.c = exploration_c
        self.arms = [a for a in MODEL_SPECS if arms is None or a in arms]
        if not self.arms:
            raise ValueError(f"no bandit arm among {arms}; known models: {list(MODEL_SPECS)}")
        self.state: dict[str, dict[str, Any]] = {
            m: {"pulls": 0, "total_reward": 0.0, "crowns": 0} for m in self.arms
        }
        self.total_pulls = 0
        self.load()

    def load(self) -> None:
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text(encoding="utf-8"))
                for k, v in data.get("arms", {}).items():
                    if k in self.state:
                        self.state[k] = v
                self.total_pulls = sum(v["pulls"] for v in self.state.values())
            except Exception:
                pass

    def save(self) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "total_pulls": self.total_pulls,
            "arms": self.state
        }
        self.state_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def select_arm(self, exclude: tuple | list | set = ()) -> str:
        """Selects the next model to sample using the UCB1 algorithm.

        `exclude` skips arms already used this iteration (UCB1 is deterministic between
        updates, so asking again would return the same arm); if it covers every arm,
        it is ignored.
        """
        arms = [a for a in self.arms if a not in exclude] or list(self.arms)
        # Warmup: ensure every arm is pulled at least once
        for arm in arms:
            if self.state[arm]["pulls"] == 0:
                return arm

        best_score = -float("inf")
        best_arm = arms[0]

        log_total = math.log(max(1, self.total_pulls))

        for arm in arms:
            stats = self.state[arm]
            n = stats["pulls"]
            mean_r = stats["total_reward"] / n
            ucb = mean_r + self.c * math.sqrt(log_total / n)

            if ucb > best_score:
                best_score = ucb
                best_arm = arm

        return best_arm

    def update(self, arm: str, reward: float, crowned: bool = False) -> None:
        """Updates the bandit state with the outcome of the generation."""
        if arm not in self.state:
            return
        self.state[arm]["pulls"] += 1
        self.state[arm]["total_reward"] += max(0.0, reward)
        if crowned:
            self.state[arm]["crowns"] += 1
            self.state[arm]["total_reward"] += 1.0  # Crown bonus
        self.total_pulls += 1
        self.save()

    def summary(self) -> dict[str, Any]:
        return {
            arm: {
                "pulls": self.state[arm]["pulls"],
                "avg_reward": round(self.state[arm]["total_reward"] / max(1, self.state[arm]["pulls"]), 3),
                "crowns": self.state[arm]["crowns"]
            }
            for arm in self.arms
        }
