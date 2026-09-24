#!/usr/bin/env bash
# Sole advanced runner entrypoint: edit-based SIFT islands + paired gauntlet (>= 20 games/seat).
# Games on this machine:   bash run_advanced_evolution.sh --workers 60
# Games on a Brev box:     bash run_advanced_evolution.sh --executor ssh --host kagg-arena-80 --workers 60
# Resumes from --run_dir (default runs/evolution) when its checkpoint.json exists.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ../../.venv/bin/activate
exec python3 -u highcpu_island_evolution.py --iterations 200 "$@"
