#!/usr/bin/env bash
# Run the all-champions tournament with the evaluator's environment (same as
# shinka/evolution/eval_once.sh: CUDA venv, MPS, oracle + Mohui paths).
#   bash shinka/champions/evidence/tournament_2026-09-14/run.sh [--out DIR] [--workers N] [--playoff K]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"
EVAL_PY="${EVAL_PY:-/results/kagg/venv-cuda/bin/python}"

export KAGG_REPO="$REPO/shinka"
export KAGG_MOHUI_DIR="$REPO/shinka/champions/dependencies/mohui_v66"
export KAGG_HISTORY_DIR="$REPO/shinka/champions/pool"
export KAGG_GAME_LOG="/results/kagg/tournament_scratch_games.jsonl"
export KAGG_TASK_DIR="$REPO"
export KAGG_ORACLE_SRC="$REPO/shinka/evolution"
export KAGG_OPP_MODEL_SRC="$REPO/research/opponent_model"
export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-/results/kagg/mps/pipe}"

bash "$REPO/shinka/evolution/mps.sh" start
start=$(date +%s)
"$EVAL_PY" "$HERE/tournament.py" "$@" 2>&1 | grep -v -i 'warning\|Loading weights' || true
rc=${PIPESTATUS[0]}
echo "TOURNAMENT_EXIT=$rc  WALL=$(( $(date +%s) - start ))s"
exit "$rc"
