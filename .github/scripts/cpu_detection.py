"""Usable CPU count for the repository's parallel gates.

`os.cpu_count()` reports the machine's processors, not the share this process is
allowed to use. Inside a container with `--cpus=4`, a Kubernetes CPU limit, or a
self-hosted runner with a cgroup quota it over-reports, and a gate that sizes its
worker pool from it oversubscribes badly enough to run slower than serial. The
count here is the minimum of every limit the kernel exposes, so both the language
mutation gate and the pytest-xdist gate agree on how big the machine is.

Run as a script to print the count, which is how the shell helpers consume it:

    python .github/scripts/cpu_detection.py
"""

from __future__ import annotations

import math
import os
from pathlib import Path

CGROUP_V2_CPU_MAX = Path("/sys/fs/cgroup/cpu.max")
CGROUP_V1_CPU_QUOTA = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
CGROUP_V1_CPU_PERIOD = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
CGROUP_V2_CPUSET = Path("/sys/fs/cgroup/cpuset.cpus.effective")
CGROUP_V1_CPUSET = Path("/sys/fs/cgroup/cpuset/cpuset.cpus")


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _read_int(path: Path) -> int | None:
    raw = _read_text(path)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _parse_cpu_range_count(raw: str | None) -> int | None:
    if not raw:
        return None

    count = 0
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_raw, end_raw = part.split("-", 1)
            try:
                start = int(start_raw)
                end = int(end_raw)
            except ValueError:
                return None
            if end < start:
                return None
            count += end - start + 1
        else:
            try:
                int(part)
            except ValueError:
                return None
            count += 1

    return count or None


def _cgroup_quota_cpu_count() -> int | None:
    raw_cpu_max = _read_text(CGROUP_V2_CPU_MAX)
    if raw_cpu_max:
        quota_raw, _, period_raw = raw_cpu_max.partition(" ")
        if quota_raw != "max" and period_raw:
            try:
                quota = int(quota_raw)
                period = int(period_raw)
            except ValueError:
                return None
            if quota > 0 and period > 0:
                return max(1, math.ceil(quota / period))

    quota = _read_int(CGROUP_V1_CPU_QUOTA)
    period = _read_int(CGROUP_V1_CPU_PERIOD)
    if quota is not None and period is not None and quota > 0 and period > 0:
        return max(1, math.ceil(quota / period))

    return None


def _cpuset_cpu_count() -> int | None:
    return _parse_cpu_range_count(_read_text(CGROUP_V2_CPUSET)) or _parse_cpu_range_count(_read_text(CGROUP_V1_CPUSET))


def _affinity_cpu_count() -> int | None:
    """CPUs this process may actually be scheduled on.

    `os.cpu_count()` counts the machine's processors and ignores the affinity
    mask, so a `taskset -c 0,1,2` run still reports every core. This is the
    signal `nproc` uses and the cgroup files above do not carry.
    """
    process_cpu_count = getattr(os, "process_cpu_count", None)  # Python 3.13+
    if process_cpu_count is not None:
        return process_cpu_count()
    get_affinity = getattr(os, "sched_getaffinity", None)  # Linux, older versions
    if get_affinity is not None:
        return len(get_affinity(0))
    return None


def available_cpu_count() -> int:
    counts = [os.cpu_count() or 1]
    for limit in (_affinity_cpu_count(), _cgroup_quota_cpu_count(), _cpuset_cpu_count()):
        if limit is not None:
            counts.append(limit)
    return max(1, min(counts))


if __name__ == "__main__":
    print(available_cpu_count())
