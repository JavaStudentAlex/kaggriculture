"""Fetch the official daily replay corpus without extracting it. No credentials required.

Resumable per archive; .part files use HTTP range resume. Valid ZIP structure and
indexed episode count are checked before publication; full member CRC verification
is left to the streaming extraction stage. Public dataset versions may change:
the index snapshot is frozen for each output directory.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import time
import urllib.request
import zipfile
from pathlib import Path

BASE = "https://www.kaggle.com/api/v1/datasets/download/kaggle/"


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def inspect_zip(path, expected):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        episodes = [n for n in names if re.fullmatch(r"[0-9]+\.json", Path(n).name)]
        if len(episodes) != expected:
            raise ValueError(f"episode count mismatch: {len(episodes)} != {expected}")
        manifests = [n for n in names if Path(n).name == "manifest.csv"]
        if not manifests:
            raise ValueError("missing daily manifest")
        # Reading the small manifest also verifies that member's CRC.
        archive.read(manifests[0])
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return {"episodes": len(episodes), "bytes": path.stat().st_size, "sha256": h.hexdigest(),
            "validation": "zip directory + manifest CRC + episode count; episode CRC deferred"}


def fetch(row, out):
    slug = row["daily_dataset_slug"]
    if not re.fullmatch(r"kaggriculture-episodes-\d{4}-\d{2}-\d{2}", slug):
        raise ValueError("Unexpected official dataset slug")
    dst = out / (slug + ".zip")
    receipt = out / (slug + ".receipt.json")
    expected = int(row["episode_count"])
    if dst.exists() and receipt.exists():
        saved = json.loads(receipt.read_text())
        if dst.stat().st_size == saved["bytes"] and saved["episodes"] == expected:
            return {"date": row["date"], "status": "already_downloaded", **saved}
    if shutil.disk_usage(out).free < 50 * 1024**3:
        raise RuntimeError("Less than 50 GiB free; refusing further downloads")
    part = out / (slug + ".zip.part")
    if not dst.exists():
        print(f"START {row['date']}", flush=True)
        for attempt in range(4):
            command = ["curl", "--fail", "--location", "--silent", "--show-error",
                       "--retry", "3", "--retry-delay", "10", "--connect-timeout", "30",
                       "--max-time", "3600", "--speed-time", "120", "--speed-limit", "1024",
                       "--output", str(part)]
            if part.exists() and part.stat().st_size:
                command += ["--continue-at", "-"]
            result = subprocess.run(command + [BASE + slug], capture_output=True, text=True)
            if result.returncode == 0:
                try:
                    info = inspect_zip(part, expected)
                    part.replace(dst)
                    break
                except (zipfile.BadZipFile, ValueError) as exc:
                    print(f"INVALID {slug}: {exc}", flush=True)
                    part.unlink(missing_ok=True)
            elif result.returncode in (33, 36):
                part.unlink(missing_ok=True)  # Server cannot resume; restart this archive only.
            # Never log redirect URLs: they can contain signed access parameters.
            print(f"RETRY {slug} attempt={attempt + 1} curl_exit={result.returncode}", flush=True)
            time.sleep(10 * (attempt + 1))
        else:
            raise RuntimeError(f"Download failed after retries: {slug}")
    else:
        info = inspect_zip(dst, expected)
    saved = {**info, "dataset": "kaggle/" + slug, "date": row["date"], "finished_unix": time.time()}
    atomic_json(receipt, saved)
    print(f"DONE {row['date']} episodes={expected} bytes={saved['bytes']}", flush=True)
    return {"status": "downloaded", **saved}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=2, choices=(1, 2, 3))
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    # Prevent two downloaders from changing receipts or partial files together.
    import fcntl
    lock = (args.out / ".download.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    index = args.out / "official_index.csv"
    if not index.exists():
        with urllib.request.urlopen(BASE + "kaggriculture-episodes-index", timeout=60) as r:
            raw = r.read()
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            data = z.read("manifest.csv")
        tmp = index.with_suffix(".tmp")
        tmp.write_bytes(data)
        tmp.replace(index)
    rows = sorted(csv.DictReader(io.StringIO(index.read_text())), key=lambda r: r["date"], reverse=True)
    if args.limit:
        rows = rows[:args.limit]
    manifest = {"status": "running", "pid": os.getpid(), "started_unix": time.time(),
                "datasets": len(rows), "episodes_expected": sum(int(r["episode_count"]) for r in rows),
                "date_start": min(r["date"] for r in rows), "date_end": max(r["date"] for r in rows),
                "workers": args.workers, "completed": [], "failed": []}
    atomic_json(args.out / "download_status.json", manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("completed", "failed")}), flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(fetch, row, args.out): row for row in rows}
        for job in concurrent.futures.as_completed(jobs):
            try:
                manifest["completed"].append(job.result())
            except Exception as exc:
                manifest["failed"].append({"date": jobs[job]["date"], "error": str(exc)})
                print(f"FAILED {jobs[job]['date']}: {exc}", flush=True)
            manifest["updated_unix"] = time.time()
            atomic_json(args.out / "download_status.json", manifest)
    manifest["status"] = "failed" if manifest["failed"] else "complete"
    atomic_json(args.out / "download_status.json", manifest)
    print("DOWNLOAD_EXIT=" + ("1" if manifest["failed"] else "0"), flush=True)
    raise SystemExit(bool(manifest["failed"]))


if __name__ == "__main__":
    main()
