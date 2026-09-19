#!/usr/bin/env python3
"""Live Monitor for Multi-Island Procedural Graph Evolution."""
import json
import os
from pathlib import Path
import subprocess
import time

LOG_PATH = Path("/home/alex/kaggriculture/research/procedural_graph/multi_island_evolution.log")
BANDIT_PATH = Path("/home/alex/kaggriculture/research/procedural_graph/shinka_evo_run/bandit_state.json")
GRAPH_PATH = Path("/home/alex/kaggriculture/research/procedural_graph/policy_graph.json")


def clear():
    os.system("clear")


def main():
    while True:
        clear()
        print("=" * 70)
        print("  🌾 KAGGRICULTURE MULTI-ISLAND EVOLUTION MONITOR (TMUX)")
        print("=" * 70)

        # 1. Process Status
        res = subprocess.run(["pgrep", "-f", "multi_island_graph_evolution"], capture_output=True, text=True)
        pids = res.stdout.strip().split()
        if pids:
            print(f"  ● Status: RUNNING (PID: {', '.join(pids)})")
        else:
            print("  ○ Status: IDLE / COMPLETED")

        # 2. Bandit Stats
        if BANDIT_PATH.exists():
            try:
                b_data = json.loads(BANDIT_PATH.read_text())
                total_pulls = b_data.get("total_pulls", 0)
                print(f"  ● Total MAB Calls: {total_pulls}")
                print("\n  [MAB Arms Call Accounting & Rewards]:")
                for arm, stats in b_data.get("arms", {}).items():
                    n = stats.get("pulls", 0)
                    r = stats.get("total_reward", 0.0)
                    avg_r = r / max(1, n)
                    c = stats.get("crowns", 0)
                    print(f"    - {arm:24}: calls={n:3d} | avg_reward={avg_r:.3f} | crowns={c}")
            except Exception:
                pass

        # 3. Current Master Graph Check
        if GRAPH_PATH.exists():
            try:
                g_data = json.loads(GRAPH_PATH.read_text())
                v = g_data.get("version", "?")
                n_count = len(g_data.get("nodes", []))
                e_count = len(g_data.get("edges", []))
                print(f"\n  ● Active Graph: v{v} ({n_count} nodes, {e_count} edges)")
            except Exception:
                pass

        # 4. Recent Log Lines
        print("\n" + "-" * 70)
        print("  RECENT ACTIVITY (Last 12 log entries):")
        print("-" * 70)
        if LOG_PATH.exists():
            try:
                lines = LOG_PATH.read_text().splitlines()
                recent = [l for l in lines if l.strip() and not l.startswith("\r")][-12:]
                for l in recent:
                    print(f"  {l}")
            except Exception:
                pass
        print("=" * 70)
        print("  Press Ctrl+C to detach or exit monitor. Auto-refreshing every 5s...")

        time.sleep(5)


if __name__ == "__main__":
    main()
