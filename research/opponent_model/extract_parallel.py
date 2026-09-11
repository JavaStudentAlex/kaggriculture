"""Parallel driver for extract.py -- one worker per replay archive."""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract import run


def _one(archive, out_dir, max_episodes, stride):
    try:
        return archive, run([archive], out_dir, max_episodes, stride), None
    except Exception as exc:  # a bad archive must not kill the sweep
        return archive, 0, f"{type(exc).__name__}: {exc}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--replays", default="replays", help="directory of .zip archives")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=42)
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args()

    out_dir = Path(args.out)
    archives = sorted(str(p) for p in Path(args.replays).glob("*.zip"))
    # Resumable: an archive whose shard already exists is skipped.
    pending = [a for a in archives if not (out_dir / (Path(a).stem + ".npz")).exists()]
    print(
        f"archives={len(archives)} already_done={len(archives) - len(pending)} "
        f"pending={len(pending)} workers={args.workers} stride={args.stride}",
        flush=True,
    )
    archives = pending
    started = time.time()
    total, failed = 0, []

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_one, a, args.out, args.max_episodes, args.stride) for a in archives]
        for done, fut in enumerate(as_completed(futures), 1):
            archive, rows, err = fut.result()
            total += rows
            if err:
                failed.append((archive, err))
            print(
                f"[{done}/{len(archives)}] {Path(archive).name} rows={rows:,} "
                f"elapsed={time.time() - started:.0f}s" + (f" ERROR {err}" if err else ""),
                flush=True,
            )

    print(f"\nDONE total_rows={total:,} elapsed={time.time() - started:.0f}s")
    for archive, err in failed:
        print(f"FAILED {Path(archive).name}: {err}")


if __name__ == "__main__":
    main()
