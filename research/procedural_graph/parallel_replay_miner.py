"""High-Throughput Parallel Replay Miner for Kaggriculture.

Parses official 720-step replay archives from /home/alex/kaggriculture-replays/
across multi-core CPU workers to extract empirical milestone distributions:
1. Elite Champions (>= $150k): Early hires, livestock portfolio, plot expansion cutoffs.
2. Strong Performers ($110k - $150k): Transition boundaries and shop fulfillment ratios.
3. Mid Plateau ($60k - $110k): Premature saturation patterns and order slot waste.
4. Fatal Collapses (< $25k): Exact steps and liquidity deficits causing midnight default.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path
import statistics
import time
from typing import Any, Dict, List, Optional
import zipfile

REPLAY_DIRS = [
    Path("/home/alex/kaggriculture-replays"),
    Path("/home/alex/kaggriculture/replays")
]


def analyze_player_trace(steps: List[Any], seat: int) -> Dict[str, Any]:
    """Extracts tactical and economic milestones for one seat across 720 steps."""
    final_obs = steps[-1][0]["observation"]
    farms = final_obs.get("farms", [])
    if seat >= len(farms):
        return {}

    final_reward = float(steps[-1][seat].get("reward", 0.0) or 0.0)

    hires_day_0 = 0
    peak_hands = 0
    cows_bought = 0
    sheep_bought = 0
    last_expand_step = -1
    last_strawberry_seed_step = -1
    peak_shed = 0
    min_midnight_cash = float("inf")
    bankruptcy_step = -1
    hands_lost = False

    prev_hands = 0
    cash = 0.0

    for t, step_data in enumerate(steps):
        obs = step_data[0]["observation"]
        f = obs.get("farms", [])[seat] if seat < len(obs.get("farms", [])) else {}
        cash = float(f.get("money", 0.0) or 0.0)
        hands = f.get("hands", []) or []
        n_hands = len(hands)

        if n_hands > peak_hands:
            peak_hands = n_hands

        # Check midnight wage liquidity (hour 23)
        hour = int(obs.get("hour", 0) or 0)
        if hour == 23 and t > 0:
            if cash < min_midnight_cash:
                min_midnight_cash = cash

        # Check hand starvation / bankruptcy
        if prev_hands > 0 and n_hands < prev_hands:
            hands_lost = True
            if bankruptcy_step == -1:
                bankruptcy_step = t

        prev_hands = n_hands

        # Shed fullness
        priv = step_data[seat].get("observation", {}).get("private", {}) or {}
        shed = priv.get("shed", {}) or {}
        tot_shed = sum(int(v or 0) for v in shed.values())
        if tot_shed > peak_shed:
            peak_shed = tot_shed

        # Inspect player actions
        act = step_data[seat].get("action", {}) or {}
        market_orders = act.get("market", []) or []

        for o in market_orders:
            if not isinstance(o, (list, tuple)) or len(o) < 1:
                continue
            op = str(o[0])
            item = str(o[1]) if len(o) > 1 else ""
            qty = int(o[2] or 1) if len(o) > 2 else 1

            if op == "HIRE" and t < 24:
                hires_day_0 += 1
            elif op == "BUY_ANIMAL":
                if item == "COW":
                    cows_bought += qty
                elif item == "SHEEP":
                    sheep_bought += qty
            elif op == "EXPAND_LAND":
                last_expand_step = t
            elif op == "BUY_SEED" and item == "STRAWBERRY":
                last_strawberry_seed_step = t

    return {
        "final_reward": final_reward,
        "hires_day_0": hires_day_0,
        "peak_hands": peak_hands,
        "cows_bought": cows_bought,
        "sheep_bought": sheep_bought,
        "last_expand_step": last_expand_step,
        "last_expand_day": round(last_expand_step / 24, 1) if last_expand_step != -1 else -1,
        "last_strawberry_seed_step": last_strawberry_seed_step,
        "last_strawberry_seed_day": round(last_strawberry_seed_step / 24, 1) if last_strawberry_seed_step != -1 else -1,
        "peak_shed": peak_shed,
        "min_midnight_cash": min_midnight_cash if min_midnight_cash != float("inf") else cash,
        "hands_lost": hands_lost,
        "bankruptcy_step": bankruptcy_step
    }


def process_single_episode(args: tuple[str, str]) -> List[Dict[str, Any]]:
    """Worker task to process one episode file from inside a zip archive."""
    zip_path_str, member_name = args
    results = []
    try:
        with zipfile.ZipFile(zip_path_str) as z:
            raw_json = z.read(member_name)
            data = json.loads(raw_json)
            steps = data.get("steps", [])
            if len(steps) < 100:
                return []
            for seat in (0, 1):
                res = analyze_player_trace(steps, seat)
                if res:
                    res["episode_id"] = member_name.replace(".json", "")
                    res["seat"] = seat
                    results.append(res)
    except Exception:
        pass
    return results


def run_mining(max_episodes: Optional[int] = None, workers: int = 16) -> Dict[str, Any]:
    """Dispatches parallel extraction across all available replay archives."""
    all_zips = []
    for d in REPLAY_DIRS:
        if d.exists():
            all_zips.extend(sorted(list(d.glob("**/*.zip"))))

    if not all_zips:
        raise FileNotFoundError(f"No replay archives found in {[str(p) for p in REPLAY_DIRS]}")

    tasks = []
    print(f"Discovered {len(all_zips)} replay archives:")
    for zp in all_zips:
        print(f"  - {zp} ({zp.stat().st_size / (1024*1024):.1f} MB)")
        with zipfile.ZipFile(zp) as z:
            for name in z.namelist():
                if name.endswith(".json") and not name.startswith("__"):
                    tasks.append((str(zp), name))

    if max_episodes and len(tasks) > max_episodes:
        tasks = tasks[:max_episodes]

    total_episodes = len(tasks)
    print(f"Queued {total_episodes} matches ({total_episodes * 2} player trajectories) for mining across {workers} CPU workers...")

    t0 = time.perf_counter()
    with mp.Pool(processes=workers) as pool:
        chunk_results = pool.map(process_single_episode, tasks, chunksize=16)

    all_players: List[Dict[str, Any]] = []
    for chunk in chunk_results:
        all_players.extend(chunk)

    elapsed = time.perf_counter() - t0
    rate = len(tasks) / max(0.01, elapsed)
    print(f"Mining completed in {elapsed:.2f}s ({rate:.1f} matches/sec). Extracted {len(all_players)} player trajectories.")

    # Stratify by performance cohort
    cohorts: Dict[str, List[Dict[str, Any]]] = {
        "elite": [p for p in all_players if p["final_reward"] >= 150000],
        "strong": [p for p in all_players if 110000 <= p["final_reward"] < 150000],
        "mid": [p for p in all_players if 60000 <= p["final_reward"] < 110000],
        "collapsed": [p for p in all_players if p["final_reward"] < 25000]
    }

    summary: Dict[str, Any] = {
        "total_matches_analyzed": total_episodes,
        "total_player_trajectories": len(all_players),
        "cohort_counts": {k: len(v) for k, v in cohorts.items()},
        "mining_rate_matches_per_sec": round(rate, 1),
        "archetypes": {}
    }

    def calc_stats(vals: List[float]) -> Dict[str, float]:
        if not vals:
            return {"median": 0.0, "mean": 0.0, "p10": 0.0, "p90": 0.0}
        s = sorted(vals)
        n = len(s)
        return {
            "median": round(statistics.median(s), 2),
            "mean": round(statistics.mean(s), 2),
            "p10": round(s[int(0.10 * n)], 2),
            "p90": round(s[int(0.90 * n)], 2)
        }

    for name, players in cohorts.items():
        if not players:
            continue
        summary["archetypes"][name] = {
            "count": len(players),
            "final_reward": calc_stats([p["final_reward"] for p in players]),
            "hires_day_0": calc_stats([p["hires_day_0"] for p in players]),
            "peak_hands": calc_stats([p["peak_hands"] for p in players]),
            "cows_bought": calc_stats([p["cows_bought"] for p in players]),
            "sheep_bought": calc_stats([p["sheep_bought"] for p in players]),
            "last_expand_day": calc_stats([p["last_expand_day"] for p in players if p["last_expand_day"] > 0]),
            "last_strawberry_seed_day": calc_stats([p["last_strawberry_seed_day"] for p in players if p["last_strawberry_seed_day"] > 0]),
            "peak_shed": calc_stats([p["peak_shed"] for p in players]),
            "min_midnight_cash": calc_stats([p["min_midnight_cash"] for p in players]),
            "bankruptcy_rate": round(sum(1 for p in players if p["hands_lost"]) / len(players), 3)
        }

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel Replay Miner")
    parser.add_argument("--max_episodes", type=int, default=None, help="Cap episodes to mine")
    parser.add_argument("--workers", type=int, default=16, help="Worker processes")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "replays_mined_archetypes.json")
    args = parser.parse_args()

    results = run_mining(max_episodes=args.max_episodes, workers=args.workers)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Mined archetypes saved to {args.out}")
