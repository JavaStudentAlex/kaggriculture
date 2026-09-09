"""Run mutmut against repository Python source files.

The mutmut native config accepts directories or explicit paths, but a broad
directory path also copies ignored runtime trees into the mutant workspace.
This wrapper keeps the hook generic by discovering Python files from Git and
passing the current source set to mutmut programmatically.
"""

from __future__ import annotations

import inspect
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any

import mutmut.configuration as mutmut_configuration
from cpu_detection import available_cpu_count
from mutmut.configuration import Config
from mutmut.mutation.data import SourceFileMutationData

EXCLUDED_DIRS = {
    ".agents",
    ".codex",
    ".github",
    ".gsd",
    ".skills",
    ".skillss",
    ".venv",
    "features",
    "mutants",
    "skills",
    "tests",
    "tools",
}

ALSO_COPY = [
    Path("tests"),
    Path("test"),
    Path("pyproject.toml"),
    Path("README.md"),
    Path("src"),
    Path("scripts"),
    Path("examples"),
    Path("config.yaml"),
    Path("Snakefile"),
    Path("SAAB timestamps.xlsx"),
    Path("Kraken timestamps.xlsx"),
]

CHUNK_INDEX_ENV = "LANGUAGE_MUTATION_CHUNK_INDEX"
TOTAL_CHUNKS_ENV = "LANGUAGE_MUTATION_TOTAL_CHUNKS"
MAX_CHILDREN_ENV = "LANGUAGE_MUTATION_MAX_CHILDREN"
# Per-mutant wall clock is (estimated_test_seconds + constant) * multiplier. Set it too low
# and a mutant is cut off mid-test, reporting "timeout" without ever being analyzed; set it
# too high and a mutant that hangs the suite holds a worker for that whole time. The whole
# suite runs in about a minute, so this allowance gives any honest test set several times
# the room it needs, which makes a timeout mean the mutant hung the tests -- a detection.
TIMEOUT_CONSTANT_ENV = "LANGUAGE_MUTATION_TIMEOUT_CONSTANT"
TIMEOUT_MULTIPLIER_ENV = "LANGUAGE_MUTATION_TIMEOUT_MULTIPLIER"
DEFAULT_TIMEOUT_CONSTANT = 60.0
DEFAULT_TIMEOUT_MULTIPLIER = 2.0
CPU_COUNT_ALIASES = {"auto", "cpu", "cpus"}


@dataclass(frozen=True)
class MutationGateArgs:
    source_args: list[str]
    chunk_index: int
    total_chunks: int
    max_children: int | None
    print_chunk_matrix: bool


def _git_python_files(root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "*.py"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return [Path(line) for line in result.stdout.splitlines() if line]


def _fallback_python_files(root: Path) -> list[Path]:
    return [path.relative_to(root) for path in root.rglob("*.py") if path.is_file()]


def _is_excluded(path: Path) -> bool:
    return any(part in EXCLUDED_DIRS for part in path.parts[:-1])


def discover_python_sources(root: Path) -> list[Path]:
    try:
        candidates = _git_python_files(root)
    except (FileNotFoundError, subprocess.CalledProcessError):
        candidates = _fallback_python_files(root)

    sources = [path for path in candidates if not _is_excluded(path)]
    return sorted(set(sources), key=lambda path: path.as_posix())


def _positive_int(raw: str, name: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}") from exc
    if value < 1:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}")
    return value


def _non_negative_int(raw: str, name: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a non-negative integer, got {raw!r}") from exc
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {raw!r}")
    return value


def resolve_total_chunks(raw: str | None, *, default: int = 1) -> int:
    if raw is None:
        return default
    if raw.lower() in CPU_COUNT_ALIASES:
        return available_cpu_count()
    return _positive_int(raw, "--total-chunks")


def resolve_max_children(raw: str | None, *, default: int | None = None) -> int | None:
    if raw is None:
        return default
    if raw.lower() in CPU_COUNT_ALIASES:
        return available_cpu_count()
    return _positive_int(raw, "--max-children")


def parse_gate_args(argv: list[str]) -> MutationGateArgs:
    chunk_index_raw = os.environ.get(CHUNK_INDEX_ENV)
    total_chunks_raw = os.environ.get(TOTAL_CHUNKS_ENV)
    max_children_raw = os.environ.get(MAX_CHILDREN_ENV)
    print_chunk_matrix = False

    filtered_argv = []
    i = 0
    while i < len(argv):
        if argv[i] == "--chunk" and i + 1 < len(argv):
            chunk_index_raw = argv[i + 1]
            i += 2
        elif argv[i] == "--total-chunks" and i + 1 < len(argv):
            total_chunks_raw = argv[i + 1]
            i += 2
        elif argv[i] == "--max-children" and i + 1 < len(argv):
            max_children_raw = argv[i + 1]
            i += 2
        elif argv[i] == "--print-chunk-matrix":
            print_chunk_matrix = True
            i += 1
        else:
            filtered_argv.append(argv[i])
            i += 1

    total_chunks_default = available_cpu_count() if print_chunk_matrix or chunk_index_raw is not None else 1
    total_chunks = resolve_total_chunks(total_chunks_raw, default=total_chunks_default)
    max_children_default = 1 if total_chunks > 1 else available_cpu_count()
    max_children = resolve_max_children(max_children_raw, default=max_children_default)

    if print_chunk_matrix:
        chunk_index = 0
    elif chunk_index_raw is None:
        if total_chunks > 1:
            raise ValueError(f"--chunk or {CHUNK_INDEX_ENV} is required when total chunks is greater than 1.")
        chunk_index = 0
    else:
        chunk_index = _non_negative_int(chunk_index_raw, "--chunk")

    if chunk_index >= total_chunks:
        raise ValueError(f"--chunk must be less than --total-chunks; got {chunk_index} of {total_chunks}.")

    return MutationGateArgs(
        source_args=filtered_argv,
        chunk_index=chunk_index,
        total_chunks=total_chunks,
        max_children=max_children,
        print_chunk_matrix=print_chunk_matrix,
    )


def print_chunk_matrix(total_chunks: int, max_children: int | None) -> None:
    matrix = {
        "include": [
            {
                "chunk": chunk,
                "total_chunks": total_chunks,
                **({} if max_children is None else {"max_children": max_children}),
            }
            for chunk in range(total_chunks)
        ]
    }
    print(json.dumps(matrix, separators=(",", ":")))


def _positive_float(raw: str | None, name: str, default: float) -> float:
    """Read a positive float from the environment, falling back to the gate default."""
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive number, got {raw!r}") from exc
    if value <= 0.0:
        raise ValueError(f"{name} must be a positive number, got {raw!r}")
    return value


def resolve_timeouts() -> tuple[float, float]:
    """Return the (constant, multiplier) bounding each mutant's wall clock."""
    return (
        _positive_float(os.environ.get(TIMEOUT_CONSTANT_ENV), TIMEOUT_CONSTANT_ENV, DEFAULT_TIMEOUT_CONSTANT),
        _positive_float(os.environ.get(TIMEOUT_MULTIPLIER_ENV), TIMEOUT_MULTIPLIER_ENV, DEFAULT_TIMEOUT_MULTIPLIER),
    )


def configure_mutmut(source_paths: list[Path]) -> None:
    timeout_constant, timeout_multiplier = resolve_timeouts()
    mutmut_configuration._config = Config(
        also_copy=ALSO_COPY,
        only_mutate=["*.py"],
        do_not_mutate=[
            "tests/*",
            "./tests/*",
            ".github/*",
            "./.github/*",
            ".agents/*",
            "./.agents/*",
            ".codex/*",
            "./.codex/*",
            ".gsd/*",
            "./.gsd/*",
            ".skillss/*",
            "./.skillss/*",
            ".skills/*",
            "./.skills/*",
            ".venv/*",
            "./.venv/*",
            "skills/*",
            "./skills/*",
            "mutants/*",
            "./mutants/*",
            "tools/*",
            "./tools/*",
        ],
        do_not_mutate_patterns=[],
        max_stack_depth=-1,
        debug=True,
        mutate_only_covered_lines=True,
        source_paths=source_paths,
        pytest_add_cli_args=["-x", "-q", "--tb=no"],
        pytest_add_cli_args_test_selection=["tests"],
        timeout_multiplier=timeout_multiplier,
        timeout_constant=timeout_constant,
        type_check_command=[],
        use_setproctitle=platform.system() != "Darwin",
    )


# A mutant that hangs the suite, segfaults it, or is caught by the type check has been
# detected as surely as one that fails an assertion, so those statuses join "killed".
DETECTED_STATUSES = frozenset({"killed", "timeout", "segfault", "caught by type check"})


EQUIVALENTS_FILE = Path(".github/mutation_equivalents.txt")


def load_proven_equivalents(path: Path = EQUIVALENTS_FILE) -> set[str]:
    """Read the mutant keys that carry a written equivalence argument.

    A mutant listed here cannot be killed by any test, so counting it as an
    undetected survivor would leave the gate permanently red and train readers to
    ignore its output. Each entry is justified in
    ``.github/instructions/mutation_equivalents.instructions.md``.
    """
    if not path.is_file():
        return set()
    entries = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.split("#", 1)[0].strip()
        if stripped:
            entries.add(stripped)
    return entries


def print_results() -> None:
    """List every mutant the suite failed to detect, grouped by source file."""
    from mutmut.__main__ import status_by_exit_code, walk_mutatable_files

    equivalents = load_proven_equivalents()
    undetected = 0
    excluded = 0
    for path in walk_mutatable_files():
        mutation_data = SourceFileMutationData(path=path)
        mutation_data.load()
        for mutant_name, exit_code in mutation_data.exit_code_by_key.items():
            status = status_by_exit_code[exit_code]
            if status in DETECTED_STATUSES:
                continue
            if mutant_name in equivalents:
                excluded += 1
                continue
            print(f"    {mutant_name}: {status}")
            undetected += 1
    if excluded:
        print(f"{excluded} proven-equivalent mutant(s) excluded; see {EQUIVALENTS_FILE}.")
    if undetected:
        print(f"{undetected} mutant(s) were not detected; cover them with a test that pins the behaviour.")
    else:
        print("Every mutant was detected or is a proven equivalent.")


def warm_imports() -> None:
    """Pre-load binary dependencies before mutmut snapshots ``sys.modules``."""
    import numpy.testing  # noqa: F401
    import pandas  # noqa: F401
    import scipy  # noqa: F401


def normalize_src_module_name(name: str) -> str:
    """Strip a synthetic ``src.`` prefix from mutmut module names."""
    if name.startswith("src."):
        return name[4:]
    if name == "src":
        return ""
    return name


def patch_mutmut_src_package_trampoline() -> None:  # noqa: C901
    """Let mutmut handle modules stored in this repository's src layout.

    Production files import as top-level modules. The normalization removes the
    synthetic ``src.`` prefix that some mutmut versions expose while building
    trampoline keys.
    """
    import mutmut.mutation.trampoline as trampoline_module
    from mutmut.__main__ import (
        MutmutProgrammaticFailException,
        mangled_name_from_mutant_name,
        record_trampoline_hit,
    )

    def record_normalized_trampoline_hit(name: str) -> None:
        record_trampoline_hit(normalize_src_module_name(name))

    def wrap_in_trampoline(mutants_dict: dict[str, Any], is_classmethod: bool = False) -> Any:  # noqa: C901
        def mutmut_mutated(decorated_func: Any) -> Any:  # noqa: C901
            def trampoline(*args: Any, **kwargs: Any) -> Any:
                orig_func = mutants_dict["_mutmut_orig"]
                call_args = list(args)

                if is_classmethod:
                    call_args = list(args[1:])
                    orig_func = getattr(args[0], orig_func.__name__)

                mutant_under_test = os.environ.get("MUTANT_UNDER_TEST", "")

                if mutant_under_test == "fail":
                    raise MutmutProgrammaticFailException(
                        "Verifying setup. At least one test should fail if mutations cause errors."
                    )

                if mutant_under_test == "stats":
                    record_normalized_trampoline_hit(
                        f"{orig_func.__module__}.{mangled_name_from_mutant_name(orig_func.__name__)}"
                    )
                    return orig_func(*call_args, **kwargs)

                module, _, mutant_name = mutant_under_test.rpartition(".")
                if module != normalize_src_module_name(decorated_func.__module__):
                    return orig_func(*call_args, **kwargs)

                mutated_func = mutants_dict.get(mutant_name)
                if mutated_func is None:
                    return orig_func(*call_args, **kwargs)

                if is_classmethod:
                    mutated_func = getattr(args[0], mutated_func.__name__)
                return mutated_func(*call_args, **kwargs)

            if inspect.isgeneratorfunction(decorated_func):

                @wraps(decorated_func)
                def _trampoline_wrapper(*args: Any, **kwargs: Any) -> Any:
                    yield from trampoline(*args, **kwargs)

            elif inspect.iscoroutinefunction(decorated_func):

                @wraps(decorated_func)
                async def _trampoline_wrapper(*args: Any, **kwargs: Any) -> Any:
                    return await trampoline(*args, **kwargs)

            elif inspect.isasyncgenfunction(decorated_func):

                @wraps(decorated_func)
                async def _trampoline_wrapper(*args: Any, **kwargs: Any) -> Any:
                    async for result in trampoline(*args, **kwargs):
                        yield result

            else:

                @wraps(decorated_func)
                def _trampoline_wrapper(*args: Any, **kwargs: Any) -> Any:
                    return trampoline(*args, **kwargs)

            return _trampoline_wrapper

        return mutmut_mutated

    trampoline_module.record_trampoline_hit = record_normalized_trampoline_hit
    trampoline_module.wrap_in_trampoline = wrap_in_trampoline


def prepare_cached_coverage(root: Path) -> bool:
    """Copy the prior slow-lane coverage database when the matrix artifact has it."""
    coverage_source = root / "build/slow-lane/.coverage"
    coverage_target = root / ".coverage"
    if not coverage_source.exists():
        print("Cached coverage file not found; mutmut will collect coverage in this job.")
        return False
    if coverage_source.stat().st_size == 0:
        print("Cached coverage file is empty; mutmut will collect coverage in this job.")
        return False

    shutil.copy2(coverage_source, coverage_target)
    return True


def _coverage_line_candidates(root: Path, filename: Path) -> tuple[str, ...]:
    """Return coverage.py keys used by local and CI coverage databases."""
    return (
        filename.as_posix(),
        str(filename),
        str((root / filename).resolve()),
        str((root / "mutants" / filename).resolve()),
    )


def _lookup_coverage_lines(coverage_data: Any, root: Path, filename: Path) -> set[int]:
    """Find covered lines for a source file across relative and absolute coverage keys."""
    for candidate in _coverage_line_candidates(root, filename):
        lines = coverage_data.lines(candidate)
        if lines:
            return set(lines)

    suffix = filename.as_posix()
    for measured_file in coverage_data.measured_files():
        if measured_file.endswith(suffix):
            lines = coverage_data.lines(measured_file)
            if lines:
                return set(lines)

    return set()


def patch_mutmut_coverage(coverage_file: Path) -> None:
    """Make mutmut reuse our existing .coverage and handle relative_files=true."""
    import mutmut.__main__
    import mutmut.code_coverage

    original_gather_coverage = mutmut.code_coverage.gather_coverage
    root = Path.cwd()

    def patched_gather_coverage(runner: Any, source_files: list[Path]) -> dict[str, set[int]]:
        import coverage

        cov = coverage.Coverage(data_file=str(coverage_file))
        cov.load()
        coverage_data = cov.get_data()

        covered_lines: dict[str, set[int]] = {}
        mutants_path = Path("mutants")
        for filename in source_files:
            abs_filename = str((mutants_path / filename).absolute())
            covered_lines[abs_filename] = _lookup_coverage_lines(coverage_data, root, filename)

        if not any(covered_lines.values()):
            print("Cached coverage had no lines for selected files; falling back to mutmut coverage collection.")
            return original_gather_coverage(runner, source_files)

        return covered_lines

    mutmut.code_coverage.gather_coverage = patched_gather_coverage
    mutmut.__main__.gather_coverage = patched_gather_coverage


def clean_corrupt_mutmut_metadata(root: Path) -> list[Path]:
    """Remove ignored mutmut metadata files that cannot be decoded as JSON."""
    mutants_dir = root / "mutants"
    if not mutants_dir.exists():
        return []

    removed: list[Path] = []
    for metadata_path in mutants_dir.rglob("*.meta"):
        try:
            with metadata_path.open() as metadata_file:
                json.load(metadata_file)
        except (OSError, json.JSONDecodeError):
            metadata_path.unlink(missing_ok=True)
            removed.append(metadata_path)
    return removed


def reset_mutmut_workspace(root: Path) -> None:
    """Drop mutmut's ignored generated workspace so hooks use current source."""
    shutil.rmtree(root / "mutants", ignore_errors=True)


def _mutmut_status(max_children: int | None) -> int:
    from mutmut.__main__ import _run

    try:
        _run((), max_children)
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


def run_mutmut_once(root: Path, max_children: int | None) -> int:
    try:
        return _mutmut_status(max_children)
    except json.JSONDecodeError:
        removed = clean_corrupt_mutmut_metadata(root)
        if not removed:
            raise
        print(f"Removed {len(removed)} corrupt mutmut metadata files; retrying language mutation.")
        return _mutmut_status(max_children)


def main(argv: list[str] | None = None) -> int:  # noqa: C901
    argv = argv if argv is not None else sys.argv[1:]
    root = Path.cwd()

    try:
        gate_args = parse_gate_args(argv)
    except ValueError as exc:
        print(f"language mutation configuration error: {exc}", file=sys.stderr)
        return 2
    argv = gate_args.source_args

    if gate_args.print_chunk_matrix:
        print_chunk_matrix(gate_args.total_chunks, gate_args.max_children)
        return 0

    if argv == ["--check-config"]:
        source_paths = discover_python_sources(root)
        print(f"language mutation source files: {len(source_paths)}")
        print(f"excluded directories: {', '.join(sorted(EXCLUDED_DIRS))}")
        print(f"default total chunks: {gate_args.total_chunks}")
        print(f"default chunk index: {gate_args.chunk_index}")
        print(f"default max children: {gate_args.max_children or 'mutmut default'}")
        return 0

    if argv:
        candidates = [Path(p) for p in argv]
        source_paths = [path for path in candidates if not _is_excluded(path)]
        source_paths = sorted(set(source_paths), key=lambda path: path.as_posix())
    else:
        source_paths = discover_python_sources(root)

    if gate_args.total_chunks > 1:
        source_paths = [
            path for i, path in enumerate(source_paths) if i % gate_args.total_chunks == gate_args.chunk_index
        ]

    configure_mutmut(source_paths)

    if not source_paths:
        print("No production Python source files found; skipping language mutation gate.")
        return 0
    if gate_args.total_chunks > 1:
        print(
            f"Running language mutation chunk {gate_args.chunk_index + 1}/{gate_args.total_chunks} "
            f"with {len(source_paths)} source files."
        )
    if gate_args.max_children is not None:
        print(f"Using mutmut max children: {gate_args.max_children}.")

    if not (root / "tests").is_dir():
        print("No tests directory found; skipping language mutation gate until tests exist.")
        return 0

    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    os.environ["PYTHONPATH"] = ""

    coverage_target = root / ".coverage"
    copied_coverage = prepare_cached_coverage(root)

    warm_imports()
    patch_mutmut_src_package_trampoline()
    if copied_coverage:
        patch_mutmut_coverage(coverage_target)
    reset_mutmut_workspace(root)

    status = 0
    try:
        status = run_mutmut_once(root, gate_args.max_children)
    finally:
        if copied_coverage:
            coverage_target.unlink(missing_ok=True)

    print_results()
    return status


if __name__ == "__main__":
    raise SystemExit(main())
