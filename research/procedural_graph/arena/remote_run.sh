#!/usr/bin/env bash
# Run an arena payload on a many-core Linux box (e.g. a Brev n2d-highcpu-80).
#   remote_run.sh <payload.tar.gz> <workers> [run-name]
# Unpacks to ~/arena/<run-name>/payload, installs uv + Python 3.12 + the pinned engine
# once (~/arena/venv), verifies files.json and starts arena.py detached. Progress:
# ~/arena/<run-name>/arena.log; results: results.jsonl and traces/. Marker: ARENA_EXIT=.
# No credentials are needed or copied: the box only plays games.
set -euo pipefail
archive=$1
workers=$2
name=${3:-run}
root=$HOME/arena/$name
mkdir -p "$root"
tar -xzf "$archive" -C "$root"
if [ ! -x "$HOME/arena/venv/bin/python" ]; then
    command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    uv venv --python 3.12 "$HOME/arena/venv"
    VIRTUAL_ENV="$HOME/arena/venv" uv pip install "kaggle-environments==1.32.7"
fi
py=$HOME/arena/venv/bin/python
"$py" - "$root/payload" <<'EOF'
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1])
files = json.loads((root / "files.json").read_text())
bad = [r for r, h in files.items() if hashlib.sha256((root / r).read_bytes()).hexdigest() != h]
assert not bad, f"hash mismatch: {bad[:5]}"
print(f"payload verified: {len(files)} files")
EOF
cd "$root"
nohup bash -c "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 '$py' payload/arena.py \
    --jobs payload/jobs.json --root payload --workers $workers --out results.jsonl --trace-dir traces; \
    echo ARENA_EXIT=\$?" > arena.log 2>&1 &
echo "started $name with $workers workers: $root/arena.log"
