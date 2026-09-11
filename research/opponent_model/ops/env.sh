#!/usr/bin/env bash
# Rebuild the training + fetch environment on the SSD. /results is an emptyDir
# that vanished on the 2026-09-11 pod restart, so everything uv touches
# (cache, interpreter, venv) is pinned to /results too -- nothing on ceph.
#
# torch MUST come from the cu126 index (driver 535 = CUDA 12.2, cu13x wheels
# fail with "driver too old"). Everything in ONE `uv pip install` call with
# cu126 as the primary index and PyPI as the extra index -- uv re-resolves the
# whole set on every call including already-satisfied deps, so a second call
# without the index pin silently upgraded torch to a cu130 wheel and broke
# CUDA (caught 2026-09-11 by the smoke test below; don't split this call).
set -u
K=/results/kagg
export UV_CACHE_DIR=$K/uv-cache UV_PYTHON_INSTALL_DIR=$K/uv-python
PY=$K/venv-cuda/bin/python
log(){ echo "[$(date -u '+%H:%M:%S')] $*"; }
fail(){ log "$*"; echo ENV_EXIT=1; exit 1; }

log "venv (python 3.11)"
rm -rf "$K/venv-cuda"
uv venv --python 3.11 "$K/venv-cuda" || fail "venv failed"
log "torch cu126 + libraries, one resolve, cu126 primary / PyPI extra"
uv pip install --python "$PY" \
  --index-url https://download.pytorch.org/whl/cu126 \
  --extra-index-url https://pypi.org/simple \
  --index-strategy unsafe-best-match \
  torch granite-tsfm accelerate scikit-learn "numpy<2.4" pandas \
  kaggle==2.2.4 kaggle-environments==1.32.7 || fail "install failed"
log "smoke test: cuda + v3 checkpoint forward pass"
"$PY" - <<'PYEOF' || fail "smoke test failed"
import torch, transformers, kaggle_environments
assert torch.cuda.is_available(), f"CUDA not available (torch {torch.__version__})"
from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(), torch.cuda.device_count(), "gpus",
      "| transformers", transformers.__version__, "| kaggle_environments", kaggle_environments.__version__)
m = TinyTimeMixerForPrediction.from_pretrained(
    "/home/jovyan/kaggriculture/models/ttm_v3_h96").cuda().eval()
with torch.no_grad():
    y = m(past_values=torch.randn(2, 512, 150).cuda()).prediction_outputs
print("v3 checkpoint forward OK:", tuple(y.shape))
PYEOF
log "done"; echo ENV_EXIT=0
