#!/usr/bin/env python3
"""Which replay commands the engine skips (encoded as empty slots), who sends them, and what
rejecting them would cost.

    python3 scan_skipped.py kaggriculture-episodes-2026-09-18.zip [...] --workers 4 --out scan.json

Every command of every step goes through data.py's own encoder rules, so the counts are the
encoder's skipped_* stats per seat, plus the commands it still rejects (rejected_*). Until
2026-09-26 each ignored command dropped its whole episode, both seats.
"""
import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data as D   # noqa: E402


def findings(command, market):
    """[(kind, command)] for one command: its skipped_* counters, or rejected_<code>."""
    stats = Counter()
    try:
        D._encode_command(command, market, stats)
    except D.DataError as exc:
        return [("rejected_" + exc.code, repr(command)[:80])]
    return [(key, repr(command)[:80]) for key in stats if key.startswith("skipped_")]


def action_findings(action):
    if not isinstance(action, Mapping):
        return [("skipped_malformed_action", type(action).__name__)]
    found, lists = [], {}
    for key in ("hands", "market"):
        value = action.get(key, [])
        if isinstance(value, (list, tuple)):
            lists[key] = value
        else:
            lists[key] = []
            found.append(("skipped_malformed_action", f"{key} is {type(value).__name__}"))
    found += findings(action.get("farmer"), False)
    for command in lists["hands"]:
        found += findings(command, False)
    for command in lists["market"]:
        found += findings(command, True)
    return found


def scan(task):
    source, member = task
    with zipfile.ZipFile(source) as z:
        doc = json.loads(z.read(member))
    steps = doc.get("steps") or []
    info = doc.get("info") or {}
    teams = info.get("TeamNames") or [None, None]
    seats = []
    for seat in (0, 1):
        found, steps_hit = Counter(), 0
        for t in range(1, len(steps)):
            f = action_findings(steps[t][seat].get("action"))
            steps_hit += bool(f)
            found.update(f)
        last = steps[-1][seat] if steps else {}
        seats.append({"team": teams[seat] if seat < len(teams) else None, "reward": last.get("reward"),
                      "steps_hit": steps_hit, "findings": {f"{k} | {c}": n for (k, c), n in found.items()}})
    return {"source": Path(source).name, "member": member, "steps": len(steps),
            "episode_id": str(info.get("EpisodeId") or Path(member).stem), "seats": seats}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("zips", nargs="+")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="episodes per zip (0: all)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tasks = []
    for source in args.zips:
        with zipfile.ZipFile(source) as z:
            members = sorted(n for n in z.namelist() if n.lower().endswith(".json"))
        tasks += [(source, m) for m in (members[:args.limit] if args.limit else members)]
    with ProcessPoolExecutor(args.workers) as pool:
        episodes = list(pool.map(scan, tasks, chunksize=2))
    Path(args.out).write_text(json.dumps(episodes))

    hit = [e for e in episodes if any(s["steps_hit"] for s in e["seats"])]
    print(f"{len(episodes)} episodes; {len(hit)} ({len(hit) / max(len(episodes), 1):.1%}) hold commands the "
          f"engine skips, which the strict encoder dropped with both seats")
    kinds, kind_episodes = Counter(), Counter()
    senders, clean_seats = defaultdict(lambda: [0, 0, 0]), Counter()
    for e in hit:
        seen = set()
        for s in e["seats"]:
            for key, n in s["findings"].items():
                kinds[key] += n
                seen.add(key)
            if s["steps_hit"]:
                row = senders[s["team"]]
                row[0] += 1
                row[1] += s["steps_hit"]
                row[2] += e["steps"] - 1
            else:
                clean_seats[s["team"]] += 1
        kind_episodes.update(seen)
    print("commands (count | episodes):")
    for key, n in kinds.most_common(25):
        print(f"  {n:8d} | {kind_episodes[key]:5d}  {key}")
    print("teams sending them (episodes, share of their steps with one):")
    for team, (n, steps_hit, total) in sorted(senders.items(), key=lambda kv: -kv[1][0])[:25]:
        print(f"  {n:5d}  {steps_hit / max(total, 1):6.1%}  {team}")
    print("their opponents, whose clean seats were dropped too (episodes):")
    for team, n in clean_seats.most_common(15):
        print(f"  {n:5d}  {team}")


if __name__ == "__main__":
    main()
