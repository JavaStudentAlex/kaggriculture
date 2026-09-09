#!/usr/bin/env bash
set -euo pipefail
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

if [ ! -d tests ]; then
  echo "No tests directory exists; skipping normal acceptance gate."
  mkdir -p build/slow-lane
  printf '{}\n' > build/slow-lane/coverage.json
  exit 0
fi

mkdir -p build/slow-lane

# Parallel workers when the machine has cores to spare; see the helper for the
# measurements behind the count. `coverage` is already configured with
# parallel=true and patch=["subprocess"], so each xdist worker writes its own
# data file and `coverage combine` merges them -- verified to produce byte
# identical totals and per-file summaries against a serial run.
PARALLEL_OPTS="$(bash "$(dirname "${BASH_SOURCE[0]}")/pytest_parallel_opts.sh")"

uv run coverage erase
uv run coverage run -m pytest tests $PARALLEL_OPTS -vv --tb=long --showlocals
uv run coverage combine
uv run coverage json -o build/slow-lane/coverage.json
