"""Fetch the official daily replay corpus without extracting it. No credentials required."""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Any, Dict, List, Optional
import urllib.request
import zipfile

BASE = "https://www.kaggle.com/api/v1/datasets/download/kaggle/"


def atomic_json(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def inspect_zip(path: Path, expected: int) -> Dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        episodes = [n for n in names if re.fullmatch(r"[0-9]+\.json", Path(n).name)]
        if len(episodes) != expected:
            raise ValueError(f"episode count mismatch: {len(episodes)} != {expected}")
        manifests = [n for n in names if Path(n).name == "manifest.csv"]
        if not manifests:
            raise ValueError("missing daily manifest")
        archive.read(manifests[0])
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return {
        "episodes": len(episodes),
        "bytes": path.stat().st_size,
        "sha256": h.hexdigest(),
        "validation": "zip directory + manifest CRC + episode count",
    }


def fetch(row: Dict[str, Any], out: Path) -> Dict[str, Any]:
    slug = row["daily_dataset_slug"]
    if not re.fullmatch(r"kaggriculture-episodes-\d{4}-\d{2}-\d{2}", slug):
        raise ValueError("Unexpected official dataset slug")
    dst = out / (slug + ".zip")
    receipt = out / (slug + ".receipt.json")
    expected = int(row["episode_count"])
    if dst.exists() and receipt.exists():
        try:
            saved = json.loads(receipt.read_text())
            if dst.stat().st_size == saved.get("bytes") and saved.get("episodes") == expected:
                return {"date": row["date"], "status": "already_downloaded", **saved}
        except Exception:
            pass

    if shutil.disk_usage(out).free < 10 * 1024**3:
        raise RuntimeError("Less than 10 GiB free; refusing further downloads")
    part = out / (slug + ".zip.part")
    if not dst.exists():
        print(f"START {row['date']} ({slug})", flush=True)
        for attempt in range(4):
            command = [
                "curl",
                "--fail",
                "--location",
                "--silent",
                "--show-error",
                "--retry",
                "3",
                "--retry-delay",
                "5",
                "--connect-timeout",
                "30",
                "--max-time",
                "1800",
                "--speed-time",
                "60",
                "--speed-limit",
                "1024",
                "--output",
                str(part),
            ]
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
                part.unlink(missing_ok=True)
            print(f"RETRY {slug} attempt={attempt + 1} curl_exit={result.returncode}", flush=True)
            time.sleep(5 * (attempt + 1))
        else:
            raise RuntimeError(f"Download failed after retries: {slug}")
    else:
        info = inspect_zip(dst, expected)

    saved = {**info, "dataset": "kaggle/" + slug, "date": row["date"], "finished_unix": time.time()}
    atomic_json(receipt, saved)
    print(f"DONE {row['date']} episodes={expected} bytes={saved['bytes'] / 1e6:.1f}MB", flush=True)
    return {"status": "downloaded", **saved}


def run_download(out: Path, workers: int = 4, limit: int = 0, status_file: Optional[Path] = None) -> List[Dict[str, Any]]:
    out.mkdir(parents=True, exist_ok=True)
    index = out / "official_index.csv"
    if not index.exists():
        print("Fetching official Kaggriculture episodes index...", flush=True)
        with urllib.request.urlopen(BASE + "kaggriculture-episodes-index", timeout=60) as r:
            raw = r.read()
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            data = z.read("manifest.csv")
        tmp = index.with_suffix(".tmp")
        tmp.write_bytes(data)
        tmp.replace(index)

    rows = sorted(csv.DictReader(io.StringIO(index.read_text())), key=lambda r: r["date"], reverse=True)
    if limit:
        rows = rows[:limit]

    total_episodes_expected = sum(int(r["episode_count"]) for r in rows)
    manifest = {
        "status": "downloading",
        "pid": os.getpid(),
        "started_unix": time.time(),
        "total_datasets": len(rows),
        "episodes_expected": total_episodes_expected,
        "completed_datasets": 0,
        "episodes_downloaded": 0,
        "failed": [],
    }

    def update_status():
        if status_file:
            atomic_json(status_file, manifest)

    update_status()

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = {pool.submit(fetch, row, out): row for row in rows}
        for job in concurrent.futures.as_completed(jobs):
            row = jobs[job]
            try:
                res = job.result()
                manifest["completed_datasets"] += 1
                manifest["episodes_downloaded"] += int(row["episode_count"])
            except Exception as exc:
                manifest["failed"].append({"date": row["date"], "error": str(exc)})
                print(f"FAILED {row['date']}: {exc}", flush=True)
            manifest["updated_unix"] = time.time()
            update_status()

    manifest["status"] = "download_complete"
    update_status()
    print("DOWNLOAD_COMPLETE=0" if not manifest["failed"] else "DOWNLOAD_COMPLETE=1", flush=True)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("/home/alex/kaggriculture/replays"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--status_file", type=Path, default=None)
    args = parser.parse_args()

    status_path = args.status_file or (args.out / "download_status.json")
    run_download(out=args.out, workers=args.workers, limit=args.limit, status_file=status_path)
