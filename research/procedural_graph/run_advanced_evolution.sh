#!/usr/bin/env bash
# Sole advanced runner entrypoint: SIFT + islands + Qwen + >=20 matches/seat.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ../../.venv/bin/activate
exec python3 -u highcpu_island_evolution.py --iterations 200 --workers 80 --seeds_per_champ_seat 20 --resume "$@"
