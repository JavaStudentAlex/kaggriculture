#!/usr/bin/env bash
# Launch the Kaggriculture Digital Red Queen evolution (Shinka run 4).
#
#   tmux new -s kagg-shinka \
#     'GENERATIONS=500 bash shinka/evolution/launch_shinka.sh 2>&1 | tee /results/kagg/logs/shinka.log'
#
# Environment knobs:
#   GENERATIONS      number of generations                     (default 500)
#   RESULTS_DIR      run output dir                            (default $REPO/shinka_results)
#   EVAL_WORKERS     parallel games per evaluation job         (default 12)
#   EVAL_JOBS        concurrent candidate evaluations          (default 2)
#   PROPOSAL_JOBS    concurrent LLM proposals                  (default 2)
#   PYTHON           interpreter with kaggle-environments+torch (default: venv, then python3)
#
# Resuming: this shinka_run build has no --resume flag. Point RESULTS_DIR at the
# existing run directory and it continues from the programs already in its database.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"

GENERATIONS="${GENERATIONS:-500}"
RESULTS_DIR="${RESULTS_DIR:-$REPO/shinka_results}"
EVAL_WORKERS="${EVAL_WORKERS:-12}"
EVAL_JOBS="${EVAL_JOBS:-2}"
PROPOSAL_JOBS="${PROPOSAL_JOBS:-2}"

PROXY="${PROXY:-http://localhost:8317/v1}"
OLLAMA="${OLLAMA:-http://localhost:11434}"
EMBED_MODEL="${EMBED_MODEL:-qwen3-embedding:8b}"

# ------------------------------------------------------------------ interpreter
if [[ -n "${PYTHON:-}" ]]; then
  PY="$PYTHON"
elif [[ -x /results/kagg/venv-cuda/bin/python ]]; then
  PY=/results/kagg/venv-cuda/bin/python
elif [[ -x "$REPO/.venv/bin/python" ]]; then
  PY="$REPO/.venv/bin/python"
else
  PY="$(command -v python3)"
fi

# ------------------------------------------------------------- task environment
# The evaluator and every spawned worker resolve the backbone, the oracle source,
# the feature code and the champion pool through these.
export KAGG_TASK_DIR="$REPO"
export KAGG_MOHUI_DIR="$REPO/shinka/champions/dependencies/mohui_v66"
export KAGG_ORACLE_SRC="$HERE"
export KAGG_OPP_MODEL_SRC="$REPO/research/opponent_model"
export KAGG_HISTORY_DIR="${KAGG_HISTORY_DIR:-$REPO/shinka/champions/pool}"
export KAGG_GAME_LOG="${KAGG_GAME_LOG:-$KAGG_HISTORY_DIR/games.jsonl}"
export KAGG_CROWN_THRESHOLD="${KAGG_CROWN_THRESHOLD:-0.75}"
export KAGG_GAMES_PER_SEAT="${KAGG_GAMES_PER_SEAT:-20}"
export KAGG_EVAL_WORKERS="$EVAL_WORKERS"

# The in-play checkpoint is a git-ignored, byte-identical copy of a tracked
# promoted model. Stage it on a fresh clone; refuse an accidental stale copy.
CHECKPOINT_SRC="${KAGG_TTM_SOURCE:-$REPO/models/ttm_c256_h96_ft_2026-09-17}"
CHECKPOINT_DIR="$HERE/checkpoint"
if [[ ! -f "$CHECKPOINT_DIR/model.safetensors" ]]; then
  [[ -f "$CHECKPOINT_SRC/model.safetensors" ]] || {
    echo "MISSING checkpoint source: $CHECKPOINT_SRC"; exit 1;
  }
  echo "checkpoint   : staging $CHECKPOINT_SRC -> $CHECKPOINT_DIR"
  mkdir -p "$CHECKPOINT_DIR"
  cp -a "$CHECKPOINT_SRC/." "$CHECKPOINT_DIR/"
fi
if ! diff -qr "$CHECKPOINT_SRC" "$CHECKPOINT_DIR" >/dev/null; then
  echo "checkpoint   : NOT byte-identical to $CHECKPOINT_SRC"
  echo "               remove $CHECKPOINT_DIR and rerun, or set KAGG_TTM_SOURCE"
  exit 1
fi
export KAGG_TTM_DIR="$CHECKPOINT_DIR"

# GPU inference for the oracle. Workers are spawned, so each loads the checkpoint
# once; MPS lets the many small CUDA clients share the GPUs efficiently.
export KAGG_ORACLE_DEVICE="${KAGG_ORACLE_DEVICE:-auto}"
export KAGG_ORACLE_BACKEND="${KAGG_ORACLE_BACKEND:-auto}"
export TOKENIZERS_PARALLELISM=false

export OPENAI_API_BASE="$PROXY"
export OPENAI_BASE_URL="$PROXY"
export OPENAI_API_KEY="${OPENAI_API_KEY:-local-key}"

# ------------------------------------------------------------------- preflight
echo "== preflight =================================================="
echo "python       : $PY"
"$PY" - <<'PYCHECK'
import sys
ok = True
try:
    import kaggle_environments as ke
    print("kaggle-env   :", getattr(ke, "__version__", "?"))
except Exception as e:
    print("kaggle-env   : MISSING", e); ok = False
try:
    import torch
    print("torch        :", torch.__version__, "cuda", torch.cuda.is_available(),
          "devices", torch.cuda.device_count() if torch.cuda.is_available() else 0)
except Exception:
    print("torch        : not installed -> oracle falls back to the numpy backend (CPU)")
sys.exit(0 if ok else 1)
PYCHECK

for f in initial.py evaluate.py shinka_config.yaml kagg_oracle.py; do
  [[ -f "$HERE/$f" ]] || { echo "MISSING $HERE/$f"; exit 1; }
done
[[ -f "$HERE/checkpoint/model.safetensors" ]] || { echo "MISSING oracle checkpoint"; exit 1; }

n_champs=$(find "$KAGG_HISTORY_DIR" -maxdepth 1 -name '*.py' | wc -l)
echo "champions    : $n_champs in $KAGG_HISTORY_DIR"
[[ "$n_champs" -gt 0 ]] || { echo "empty champion pool"; exit 1; }
echo "protocol     : ${KAGG_GAMES_PER_SEAT} seeds seat0 + ${KAGG_GAMES_PER_SEAT} seeds seat1 per champion"
echo "gate         : ${KAGG_CROWN_THRESHOLD}"

# LLM proxy: every configured worker/supervisor/novelty model must be available.
# Do not start a 500-generation run that silently degenerates because one requested
# provider was absent or cooling down at launch.
models_json="$(curl -sf "$PROXY/models" 2>/dev/null || true)"
[[ -n "$models_json" ]] || { echo "llm proxy    : UNREACHABLE at $PROXY"; exit 1; }
required_models=(
  gpt-6-astra gpt-6-luna gemini-3.1-pro-preview gemini-3.8-flash
  claude-opus-5 claude-sonnet-5
)
missing_models=()
for model in "${required_models[@]}"; do
  if ! grep -Fq "\"$model\"" <<<"$models_json"; then
    missing_models+=("$model")
  fi
done
if ((${#missing_models[@]})); then
  echo "llm proxy    : MISSING required model(s): ${missing_models[*]}"
  exit 1
fi
echo "llm proxy    : all ${#required_models[@]} requested models ready ($PROXY)"

# Embedding model (novelty level 1). It is a mandatory novelty gate, not optional.
ollama_tags="$(curl -sf "$OLLAMA/api/tags" 2>/dev/null || true)"
if [[ -z "$ollama_tags" ]] || ! grep -Fq "\"$EMBED_MODEL\"" <<<"$ollama_tags"; then
  echo "embeddings   : MISSING $EMBED_MODEL at $OLLAMA"
  echo "               Start Ollama and run: ollama pull $EMBED_MODEL"
  exit 1
fi
echo "embeddings   : $EMBED_MODEL ready"

# CUDA MPS, if the helper is present
if [[ -x "$HERE/mps.sh" ]]; then
  bash "$HERE/mps.sh" start || echo "mps          : start failed (continuing)"
fi

# ---------------------------------------------------------------------- launch
mkdir -p "$RESULTS_DIR"

echo "== launching =================================================="
echo "generations  : $GENERATIONS"
echo "results dir  : $RESULTS_DIR"
echo "eval jobs    : $EVAL_JOBS x $EVAL_WORKERS workers | proposal jobs: $PROPOSAL_JOBS"

SHINKA_BIN="${SHINKA_BIN:-$(command -v shinka_run || true)}"
[[ -n "$SHINKA_BIN" ]] || { echo "shinka_run not on PATH"; exit 1; }

start=$(date +%s)
set +e
"$SHINKA_BIN" \
  --task-dir "$HERE" \
  --results_dir "$RESULTS_DIR" \
  --num_generations "$GENERATIONS" \
  --config-fname shinka_config.yaml \
  --max-evaluation-jobs "$EVAL_JOBS" \
  --max-proposal-jobs "$PROPOSAL_JOBS" \
  --verbose
rc=$?
set -e
echo "SHINKA_EXIT=$rc  WALL=$(( $(date +%s) - start ))s"
exit "$rc"
