#!/usr/bin/env python3
"""Build the tape-mining notebook: one Kaggle CPU notebook running tape_mining.py over the attached daily datasets.

    python research/procedural_graph/benchmark/make_tape_mining.py --out DIR --day 2026-09-22 ... --day 2026-09-28
    kaggle kernels push -p DIR            # then: kaggle kernels output <user>/<slug> -p <results dir>

make_plan_mining.py with tape_mining.py as the code file; its output tapes.jsonl.gz feeds the route-set builder.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--day', action='append', required=True, help='a published day, e.g. 2026-09-28')
    ap.add_argument('--user', default='sunshinethroughfog')
    ap.add_argument('--slug', default='kaggriculture-tape-mining')
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(HERE / 'tape_mining.py', out / 'tape_mining.py')
    (out / 'kernel-metadata.json').write_text(json.dumps({
        'id': f'{args.user}/{args.slug}', 'title': args.slug, 'code_file': 'tape_mining.py', 'language': 'python',
        'kernel_type': 'script', 'is_private': 'true', 'enable_gpu': 'false', 'enable_tpu': 'false',
        'enable_internet': 'false', 'dataset_sources': [f'kaggle/kaggriculture-episodes-{d}' for d in args.day],
        'competition_sources': [], 'kernel_sources': []}, indent=2) + '\n')
    print(f'{out}: {len(args.day)} days attached')


if __name__ == '__main__':
    main()
