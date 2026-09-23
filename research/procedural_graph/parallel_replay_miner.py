"""High-Throughput Parallel Replay Miner for Kaggriculture with Micro-Tactical Forensics.

Parses official 720-step replay archives to extract:
1. Macro Milestones (Day 0 hires, peak hands, livestock counts, land freeze).
2. Seed & Crop Timeline (Carrot/Tomato early cashflow vs. Strawberry compounding vs. late freeze).
3. Product Monetization Matrix (sells by crop, animal products, shed headroom).
4. Physical Execution & Worker Cadence (Water vs. Harvest vs. Well Refills).
5. 10-Order Ceiling Utilization (order slot congestion).
6. Terminal Liquidation Sunset (Steps 672-720 sell volume vs unharvested waste).
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import multiprocessing as mp
from pathlib import Path
import statistics
import time
from typing import Any, Dict, List, Optional
import zipfile


def analyze_player_trace(steps: List[Any], seat: int) -> Dict[str, Any]:
    """Extracts tactical, micro-execution, and economic milestones for one seat across 720 steps."""
    if not steps:
        return {}
    final_obs = steps[-1][0].get("observation", {})
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

    # Micro-tactical counters
    seeds_by_crop: Counter[str] = Counter()
    seeds_phase_0_2: Counter[str] = Counter()   # Days 0-2 (Steps 0-71)
    seeds_phase_3_12: Counter[str] = Counter()  # Days 3-12 (Steps 72-311)
    seeds_phase_13_plus: Counter[str] = Counter() # Days 13+ (Steps 312-719)

    sells_by_product: Counter[str] = Counter()
    worker_action_types: Counter[str] = Counter()

    peak_market_orders = 0
    ticks_with_market_orders = 0
    late_liquidation_units = 0

    prev_hands = 0
    cash = 0.0

    for t, step_data in enumerate(steps):
        obs = step_data[0].get("observation", {})
        f = obs.get("farms", [])[seat] if seat < len(obs.get("farms", [])) else {}
        cash = float(f.get("money", 0.0) or 0.0)
        hands = f.get("hands", []) or []
        n_hands = len(hands)

        if n_hands > peak_hands:
            peak_hands = n_hands

        hour = int(obs.get("hour", 0) or 0)
        if hour == 23 and t > 0:
            if cash < min_midnight_cash:
                min_midnight_cash = cash

        if prev_hands > 0 and n_hands < prev_hands:
            hands_lost = True
            if bankruptcy_step == -1:
                bankruptcy_step = t

        prev_hands = n_hands

        # Private shed state
        priv = step_data[seat].get("observation", {}).get("private", {}) or {}
        shed = priv.get("shed", {}) or {}
        tot_shed = sum(int(v or 0) for v in shed.values())
        if tot_shed > peak_shed:
            peak_shed = tot_shed

        # Actions
        act = step_data[seat].get("action", {}) or {}
        market_orders = act.get("market", []) or []
        n_m_orders = len(market_orders)
        if n_m_orders > peak_market_orders:
            peak_market_orders = n_m_orders
        if n_m_orders > 0:
            ticks_with_market_orders += 1

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
            elif op == "BUY_SEED":
                crop = item.upper()
                seeds_by_crop[crop] += qty
                if crop == "STRAWBERRY":
                    last_strawberry_seed_step = t
                if t < 72:
                    seeds_phase_0_2[crop] += qty
                elif t < 312:
                    seeds_phase_3_12[crop] += qty
                else:
                    seeds_phase_13_plus[crop] += qty
            elif op == "SELL":
                prod = item.upper()
                sells_by_product[prod] += qty
                if t >= 672:
                    late_liquidation_units += qty

        # Worker actions
        hand_acts = act.get("hands", []) or []
        for ha in hand_acts:
            if isinstance(ha, (list, tuple)) and len(ha) > 0:
                worker_action_types[str(ha[0]).upper()] += 1
            elif isinstance(ha, dict) and "type" in ha:
                worker_action_types[str(ha["type"]).upper()] += 1

    # Unharvested units on final board
    final_farm = steps[-1][0].get("observation", {}).get("farms", [])[seat]
    final_tiles = final_farm.get("tiles", [])
    unharvested_ripe_units = 0
    for row in final_tiles:
        for tile in row:
            if isinstance(tile, dict) and tile.get("yield_units", 0) > 0:
                unharvested_ripe_units += int(tile["yield_units"])

    total_waters = worker_action_types.get("WATER", 0)
    total_harvests = worker_action_types.get("HARVEST", 0)
    water_harvest_ratio = round(total_waters / max(1, total_harvests), 2)

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
        "bankruptcy_step": bankruptcy_step,
        # Tactical forensics
        "seeds_by_crop": dict(seeds_by_crop),
        "seeds_phase_0_2": dict(seeds_phase_0_2),
        "seeds_phase_3_12": dict(seeds_phase_3_12),
        "seeds_phase_13_plus": dict(seeds_phase_13_plus),
        "sells_by_product": dict(sells_by_product),
        "worker_action_counts": dict(worker_action_types),
        "water_harvest_ratio": water_harvest_ratio,
        "peak_market_orders": peak_market_orders,
        "late_liquidation_units": late_liquidation_units,
        "unharvested_ripe_units": unharvested_ripe_units,
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


def calc_stats(vals: List[float]) -> Dict[str, float]:
    if not vals:
        return {"median": 0.0, "mean": 0.0, "p10": 0.0, "p90": 0.0}
    s = sorted(vals)
    n = len(s)
    return {
        "median": round(statistics.median(s), 2),
        "mean": round(statistics.mean(s), 2),
        "p10": round(s[int(0.10 * n)], 2),
        "p90": round(s[int(0.90 * n)], 2),
    }


def aggregate_cohort_counters(players: List[Dict[str, Any]], field: str) -> Dict[str, Dict[str, float]]:
    all_keys = set()
    for p in players:
        all_keys.update(p.get(field, {}).keys())
    result = {}
    for k in sorted(all_keys):
        vals = [float(p.get(field, {}).get(k, 0)) for p in players]
        result[k] = calc_stats(vals)
    return result


def run_mining(
    replay_dirs: List[Path],
    max_episodes: Optional[int] = None,
    workers: int = 8,
    status_file: Optional[Path] = None
) -> Dict[str, Any]:
    """Dispatches parallel extraction across all available replay archives."""
    all_zips = []
    for d in replay_dirs:
        if d.exists():
            all_zips.extend(sorted(list(d.glob("**/*.zip"))))

    if not all_zips:
        raise FileNotFoundError(f"No replay archives found in {[str(p) for p in replay_dirs]}")

    tasks = []
    print(f"Discovered {len(all_zips)} replay archives:")
    for zp in all_zips:
        try:
            with zipfile.ZipFile(zp) as z:
                for name in z.namelist():
                    if name.endswith(".json") and not name.startswith("__") and not name.startswith("manifest"):
                        tasks.append((str(zp), name))
        except Exception as e:
            print(f"  Warning: could not read {zp}: {e}")

    if max_episodes and len(tasks) > max_episodes:
        tasks = tasks[:max_episodes]

    total_episodes = len(tasks)
    print(f"Queued {total_episodes} matches ({total_episodes * 2} player trajectories) for mining across {workers} CPU workers...")

    def update_status(status: str, processed: int, elapsed: float, extra: Optional[Dict[str, Any]] = None):
        if not status_file:
            return
        rate = processed / max(0.1, elapsed)
        rem = max(0, total_episodes - processed)
        eta = rem / max(0.1, rate)
        data = {
            "status": status,
            "total_matches": total_episodes,
            "matches_processed": processed,
            "percent_complete": round((processed / max(1, total_episodes)) * 100.0, 1),
            "matches_per_second": round(rate, 1),
            "elapsed_seconds": round(elapsed, 1),
            "eta_seconds": round(eta, 1),
            "last_update_unix": time.time(),
        }
        if extra:
            data.update(extra)
        try:
            status_file.write_text(json.dumps(data, indent=2))
        except Exception:
            pass

    t0 = time.perf_counter()
    update_status("mining", 0, 0.0)

    all_players: List[Dict[str, Any]] = []
    batch_size = 64
    processed_count = 0

    with mp.Pool(processes=workers) as pool:
        for i in range(0, total_episodes, batch_size):
            chunk = tasks[i : i + batch_size]
            chunk_res = pool.map(process_single_episode, chunk)
            for r in chunk_res:
                all_players.extend(r)
            processed_count += len(chunk)
            now = time.perf_counter() - t0
            update_status("mining", processed_count, now)
            if processed_count % 500 == 0 or processed_count == total_episodes:
                rate = processed_count / max(0.1, now)
                print(f"  [{processed_count}/{total_episodes}] {processed_count/total_episodes*100:.1f}% mined ({rate:.1f} m/s)", flush=True)

    elapsed = time.perf_counter() - t0
    rate = len(tasks) / max(0.01, elapsed)
    print(f"Mining completed in {elapsed:.2f}s ({rate:.1f} matches/sec). Extracted {len(all_players)} player trajectories.")

    # Stratify by performance cohort
    cohorts: Dict[str, List[Dict[str, Any]]] = {
        "elite": [p for p in all_players if p["final_reward"] >= 150000],
        "strong": [p for p in all_players if 110000 <= p["final_reward"] < 150000],
        "mid": [p for p in all_players if 60000 <= p["final_reward"] < 110000],
        "collapsed": [p for p in all_players if p["final_reward"] < 25000],
    }

    summary: Dict[str, Any] = {
        "total_matches_analyzed": total_episodes,
        "total_player_trajectories": len(all_players),
        "cohort_counts": {k: len(v) for k, v in cohorts.items()},
        "mining_rate_matches_per_sec": round(rate, 1),
        "archetypes": {},
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
            "bankruptcy_rate": round(sum(1 for p in players if p["hands_lost"]) / len(players), 3),
            "water_harvest_ratio": calc_stats([p["water_harvest_ratio"] for p in players]),
            "peak_market_orders": calc_stats([p["peak_market_orders"] for p in players]),
            "late_liquidation_units": calc_stats([p["late_liquidation_units"] for p in players]),
            "unharvested_ripe_units": calc_stats([p["unharvested_ripe_units"] for p in players]),
            "seeds_phase_0_2": aggregate_cohort_counters(players, "seeds_phase_0_2"),
            "seeds_phase_3_12": aggregate_cohort_counters(players, "seeds_phase_3_12"),
            "seeds_phase_13_plus": aggregate_cohort_counters(players, "seeds_phase_13_plus"),
            "sells_by_product": aggregate_cohort_counters(players, "sells_by_product"),
        }

    update_status("completed", total_episodes, elapsed, extra={"cohort_counts": summary["cohort_counts"]})
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel Replay Miner")
    parser.add_argument("--replays", type=Path, nargs="+", default=[Path("/content/replays"), Path("/home/alex/kaggriculture/replays")])
    parser.add_argument("--max_episodes", type=int, default=None, help="Cap episodes to mine")
    parser.add_argument("--workers", type=int, default=8, help="Worker processes")
    parser.add_argument("--status_file", type=Path, default=Path("/content/status.json"))
    parser.add_argument("--out", type=Path, default=Path("/content/replays_mined_archetypes.json"))
    args = parser.parse_args()

    results = run_mining(replay_dirs=args.replays, max_episodes=args.max_episodes, workers=args.workers, status_file=args.status_file)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Mined archetypes saved to {args.out}")
