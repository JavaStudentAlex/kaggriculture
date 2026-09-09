#!/usr/bin/env bash
# Print the pytest-xdist options for the current machine on stdout, or nothing
# when the suite should run serially. Callers interpolate the result unquoted:
#
#   uv run coverage run -m pytest tests $(bash .github/scripts/pytest_parallel_opts.sh) -vv
#
# Worker count: every xdist worker re-imports the heavy runtime stack (umap and
# its numba dependency, scipy, matplotlib, pandas), so throughput peaks well
# below the core count. Measured on a 12-core box: 63.4 s serial, 31.4 s at 4
# workers, 27.9 s at 6, and 35.8 s at 12 -- past six the import bill outweighs
# the added parallelism. Hence the cap, and the floor at the cores available so
# CI runners with 2-4 cores do not oversubscribe.
#
# PYTEST_WORKERS overrides the count. PYTEST_WORKERS=1 disables xdist entirely,
# which is what you want when debugging a single failure, attaching a debugger,
# or reproducing a test-ordering effect.
set -euo pipefail

max_workers="${PYTEST_MAX_WORKERS:-6}"

# Deliberately not `nproc`: it honours the affinity mask but ignores a cgroup CPU
# quota, so a container limited with `--cpus=4` on a 12-core host still reports
# 12. cpu_detection.py takes the minimum of os.cpu_count(), the cgroup v1/v2
# quota, and the cpuset -- the same count the language mutation gate sizes its
# worker pool from, so the two gates agree about the machine.
cores="$(python3 "$(dirname "${BASH_SOURCE[0]}")/cpu_detection.py" 2>/dev/null || echo 1)"

default_workers="$cores"
if [ "$default_workers" -gt "$max_workers" ]; then
  default_workers="$max_workers"
fi

workers="${PYTEST_WORKERS:-$default_workers}"

if [ "$workers" -le 1 ]; then
  exit 0
fi

# `-p xdist` is required because the gates export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1,
# which suppresses entry-point plugin discovery. Without it pytest silently runs
# serially and `-n` is rejected as an unknown option.
#
# `--dist load` balances individual tests across workers; it measured faster than
# `--dist loadfile` (27.9 s vs 30.3 s at six workers) and produces identical
# coverage, which `crap_check.py` depends on.
printf -- '-p xdist -n %s --dist load' "$workers"
