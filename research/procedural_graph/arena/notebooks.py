#!/usr/bin/env python3
"""Run an arena payload on Kaggle CPU notebooks (private dataset + private script kernels).

    python research/procedural_graph/arena/notebooks.py upload --eval-id ID
    python research/procedural_graph/arena/notebooks.py push   --eval-id ID [--shards 5 --workers 4]
    python research/procedural_graph/arena/notebooks.py wait|status|logs --shards 5
    python research/procedural_graph/arena/notebooks.py fetch  --eval-id ID --shards 5
    python research/procedural_graph/arena/notebooks.py delete --shards 5

The account allows 5 concurrent batch CPU sessions ("Maximum batch CPU session count
of 5 reached" beyond that; pushes are rejected, not queued). A kernel's log and output
are only readable after it finishes. Nothing here submits to the competition.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tarfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / 'runs' / 'arena'
KAGGLE = os.environ.get('KAGGLE_CLI', 'kaggle')
TERMINAL = ('COMPLETE', 'ERROR', 'CANCEL')

KERNEL = r'''#!/usr/bin/env python3
"""Kaggriculture arena shard {shard}/{shards} ({eval_id}). CPU only, no submission."""
import hashlib, json, os, shutil, subprocess, sys, tarfile, time
from pathlib import Path

EVAL_ID, SHARD, SHARDS, WORKERS = {eval_id!r}, {shard}, {shards}, {workers}
os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                  NUMEXPR_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="", KAGG_ORACLE_BACKEND="numpy",
                  KAGG_ORACLE_DEVICE="cpu", PYTHONDONTWRITEBYTECODE="1")
work, root = Path("/kaggle/working"), Path("/tmp/arena/payload")
print(f"shard {{SHARD}}/{{SHARDS}} {{EVAL_ID}} cpus={{os.cpu_count()}}", flush=True)
archives = sorted(Path("/kaggle/input").rglob("payload.tar.gz"))
if archives:  # Kaggle may also auto-extract the archive; accept either form
    with tarfile.open(archives[0]) as tar:
        tar.extractall("/tmp/arena", filter="data")
else:
    found = [p for p in Path("/kaggle/input").rglob("jobs.json")
             if json.loads(p.read_text()).get("evaluation_id") == EVAL_ID]
    if not found:
        raise SystemExit("payload not found")
    shutil.copytree(found[0].parent, root, dirs_exist_ok=True)
manifest = json.loads((root / "jobs.json").read_text())
assert manifest["evaluation_id"] == EVAL_ID, manifest["evaluation_id"]
files = json.loads((root / "files.json").read_text())
bad = [rel for rel, h in files.items() if hashlib.sha256((root / rel).read_bytes()).hexdigest() != h]
assert not bad, f"hash mismatch: {{bad[:5]}}"
print(f"payload verified: {{len(files)}} files, {{len(manifest['jobs'])}} jobs", flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "kaggle-environments==1.32.7"], check=True)
t0 = time.time()
rc = subprocess.run([sys.executable, str(root / "arena.py"), "--jobs", str(root / "jobs.json"),
                     "--root", str(root), "--shard", str(SHARD), "--shards", str(SHARDS),
                     "--workers", str(WORKERS), "--out", str(work / f"results_{{SHARD}}.jsonl"),
                     "--trace-dir", str(work / "traces"), "--deadline", str(10.5 * 3600)]).returncode
print(f"ARENA_EXIT={{rc}} after {{time.time() - t0:.0f}}s", flush=True)
'''


def kaggle(args, check=True):
    r = subprocess.run([KAGGLE, *args], capture_output=True, text=True)
    if check and r.returncode:
        raise SystemExit(f'kaggle {" ".join(args)} failed ({r.returncode}):\n{r.stdout}\n{r.stderr}')
    return (r.stdout or '') + (r.stderr or '')


def slug(eval_id):
    s = re.sub(r'[^a-z0-9-]+', '-', f'kagg-arena-{eval_id}'.lower()).strip('-')
    return s[:50]


def kernel(a, i):
    return f'{a.user}/{a.prefix}-s{i}'


def upload(a):
    payload = Path(a.payload or RUNS / a.eval_id / 'payload')
    stage = RUNS / a.eval_id / 'dataset'
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    with tarfile.open(stage / 'payload.tar.gz', 'w:gz') as tar:
        tar.add(payload, arcname='payload')
    ref = f'{a.user}/{slug(a.eval_id)}'
    (stage / 'dataset-metadata.json').write_text(json.dumps(
        {'title': slug(a.eval_id), 'id': ref, 'licenses': [{'name': 'CC0-1.0'}]}))
    print(kaggle(['datasets', 'create', '-p', str(stage)])[-400:])  # private by default
    for _ in range(90):
        out = kaggle(['datasets', 'status', ref], check=False)
        if 'ready' in out.lower():
            print('dataset ready:', ref)
            return
        time.sleep(20)
    raise SystemExit('dataset not ready in time')


def push(a):
    for i in range(a.shards):
        d = RUNS / a.eval_id / 'kernels' / f's{i}'
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
        (d / 'run.py').write_text(KERNEL.format(eval_id=a.eval_id, shard=i, shards=a.shards, workers=a.workers))
        (d / 'kernel-metadata.json').write_text(json.dumps({
            'id': kernel(a, i), 'title': f'{a.prefix}-s{i}', 'code_file': 'run.py', 'language': 'python',
            'kernel_type': 'script', 'is_private': True, 'enable_gpu': False, 'enable_tpu': False,
            'enable_internet': True, 'dataset_sources': [f'{a.user}/{slug(a.eval_id)}'],
            'competition_sources': []}))
        print(f's{i}:', kaggle(['kernels', 'push', '-p', str(d)]).strip()[-200:])


def states(a):
    out = {}
    for i in range(a.shards):
        s = kaggle(['kernels', 'status', kernel(a, i)], check=False)
        m = re.search(r'KernelWorkerStatus\.([A-Z_]+)', s)
        out[i] = m.group(1) if m else s.strip()[-120:]
    return out


def status(a):
    for i, s in states(a).items():
        print(f's{i}: {s}')


def wait(a):
    last = {}
    while True:
        now = states(a)
        for i, s in now.items():
            if last.get(i) != s:
                print(f's{i}: {s}', flush=True)
        last = now
        if all(any(t in s for t in TERMINAL) for s in now.values()):
            return
        time.sleep(90)


def logs(a):
    for i in range(a.shards):
        out = kaggle(['kernels', 'logs', kernel(a, i)], check=False)
        try:
            lines = [line for e in json.loads(out) for line in str(e.get('data', '')).splitlines()]
        except ValueError:
            lines = out.splitlines()
        print(f'== s{i}', *lines[-a.tail:], sep='\n')


def fetch(a):
    for i, s in states(a).items():
        if not any(t in s for t in TERMINAL):
            print(f's{i}: still {s}')
            continue
        d = RUNS / a.eval_id / 'results' / f's{i}'
        d.mkdir(parents=True, exist_ok=True)
        print(f's{i}:', kaggle(['kernels', 'output', kernel(a, i), '-p', str(d), '-o', '-q'], check=False)[-200:])


def delete(a):
    for i in range(a.shards):
        print(f's{i}:', kaggle(['kernels', 'delete', '-y', kernel(a, i)], check=False).strip()[-200:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=['upload', 'push', 'status', 'wait', 'logs', 'fetch', 'delete'])
    ap.add_argument('--eval-id')
    ap.add_argument('--user', default=os.environ.get('KAGGLE_ARENA_USER', 'sunshinethroughfog'))
    ap.add_argument('--prefix', default='kagg-arena')
    ap.add_argument('--shards', type=int, default=5)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--payload', default=None)
    ap.add_argument('--tail', type=int, default=8)
    a = ap.parse_args()
    if a.command in ('upload', 'push', 'fetch') and not a.eval_id:
        ap.error('--eval-id is required')
    globals()[a.command](a)


if __name__ == '__main__':
    main()
