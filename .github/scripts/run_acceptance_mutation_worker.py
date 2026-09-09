"""APS Gherkin mutation worker for this repository.

The APS mutator sends mutated feature JSON IR over a persistent newline-delimited
JSON protocol. This worker renders that IR back to a temporary Behave feature,
copies the repository step definitions beside it, runs Behave, and reports
whether the mutated acceptance specification was detected by the tests.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
STEPS_DIR = ROOT / "features" / "steps"


# The APS mutator sends no per-job timeout, so this fallback governs every job. It has to
# clear the slowest feature's full Behave run with room for the CPU contention of a parallel
# gate: `features/unified_dataset.feature` alone takes about 20 seconds because its workflow
# scenarios plan real Snakemake rule graphs in subprocesses. A fallback near that runtime
# turns load spikes into `infrastructure_error` verdicts that say nothing about the mutant.
DEFAULT_JOB_TIMEOUT_SECONDS = 600.0


def _timeout_seconds(raw: str | None) -> float:
    if not raw:
        return DEFAULT_JOB_TIMEOUT_SECONDS
    raw = raw.strip()
    try:
        if raw.endswith("ms"):
            return max(float(raw[:-2]) / 1000.0, 0.1)
        if raw.endswith("s"):
            return max(float(raw[:-1]), 0.1)
        if raw.endswith("m"):
            return max(float(raw[:-1]) * 60.0, 0.1)
        return max(float(raw), 0.1)
    except ValueError:
        return DEFAULT_JOB_TIMEOUT_SECONDS


def _render_feature(feature_ir: dict[str, Any]) -> str:
    lines = [f"Feature: {feature_ir['name']}", ""]
    for scenario in feature_ir.get("scenarios", []):
        examples = scenario.get("examples") or []
        keyword = "Scenario Outline" if examples else "Scenario"
        lines.append(f"  {keyword}: {scenario['name']}")
        for step in scenario.get("steps", []):
            lines.append(f"    {step['keyword']} {step['text']}")
        if examples:
            headers = sorted(examples[0])
            lines.append("")
            lines.append("    Examples:")
            lines.append("      | " + " | ".join(headers) + " |")
            for row in examples:
                lines.append("      | " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
        lines.append("")
    return "\n".join(lines)


def _prepare_behave_tree(feature_json: Path, work_dir: Path) -> Path:
    with feature_json.open(encoding="utf-8") as handle:
        feature_ir = json.load(handle)

    behave_root = work_dir / "behave"
    shutil.rmtree(behave_root, ignore_errors=True)
    behave_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(STEPS_DIR, behave_root / "steps")
    (behave_root / "mutated.feature").write_text(_render_feature(feature_ir), encoding="utf-8")
    return behave_root


def _run_job(request: dict[str, Any]) -> dict[str, Any]:
    started = time.monotonic_ns()
    job_id = str(request.get("id", ""))
    try:
        feature_json = (ROOT / str(request["feature_json"])).resolve()
        work_dir = (ROOT / str(request["work_dir"])).resolve()
        behave_root = _prepare_behave_tree(feature_json, work_dir)
        completed = subprocess.run(
            ["uv", "run", "behave", str(behave_root), "-q"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=_timeout_seconds(request.get("timeout")),
            check=False,
        )
    except Exception as exc:  # noqa: BLE001 - protocol must convert all infra failures to JSON.
        return {
            "id": job_id,
            "outcome": "infrastructure_error",
            "output": "",
            "error": str(exc),
            "duration": time.monotonic_ns() - started,
        }

    output = (completed.stdout + completed.stderr)[-8000:]
    return {
        "id": job_id,
        "outcome": "test_success" if completed.returncode == 0 else "test_failure",
        "output": output,
        "error": "" if completed.returncode in (0, 1) else output,
        "duration": time.monotonic_ns() - started,
    }


def main() -> int:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            response = _run_job(request)
        except json.JSONDecodeError as exc:
            response = {
                "id": "",
                "outcome": "infrastructure_error",
                "output": "",
                "error": f"invalid mutation request JSON: {exc}",
                "duration": 0,
            }
        print(json.dumps(response, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
