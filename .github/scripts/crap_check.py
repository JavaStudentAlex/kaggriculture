#!/usr/bin/env python3
"""Fail when complex functions have weak test coverage.

This implements the CRAP score formula used by the slow-lane CI:

    CRAP = complexity**2 * (1 - coverage)**3 + complexity

Coverage is read from ``coverage json`` output. Cyclomatic complexity is a
small AST-based estimate so the gate stays self-contained for CI.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FunctionRisk:
    path: Path
    qualname: str
    lineno: int
    end_lineno: int
    complexity: int
    covered_lines: int
    executable_lines: int
    coverage: float
    score: float


class ComplexityVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.score = 1
        self._function_depth = 0

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._function_depth += 1
        if self._function_depth == 1:
            self.generic_visit(node)
        self._function_depth -= 1

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._function_depth += 1
        if self._function_depth == 1:
            self.generic_visit(node)
        self._function_depth -= 1

    def visit_If(self, node: ast.If) -> None:
        self.score += 1
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self.score += 1
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.score += 1
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        self.score += 1
        self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        self.score += 1
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        self.score += 1
        self.generic_visit(node)

    def visit_Assert(self, node: ast.Assert) -> None:
        self.score += 1
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        self.score += max(0, len(node.values) - 1)
        self.generic_visit(node)

    def visit_comprehension(self, node: ast.comprehension) -> None:
        self.score += 1 + len(node.ifs)
        self.generic_visit(node)

    def visit_Match(self, node: ast.Match) -> None:
        self.score += len(node.cases)
        self.generic_visit(node)


class FunctionCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.stack: list[str] = []
        self.functions: list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef, int]] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._collect_function(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._collect_function(node)
        self.generic_visit(node)

    def _collect_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        visitor = ComplexityVisitor()
        visitor.visit(node)
        qualname = ".".join([*self.stack, node.name])
        self.functions.append((qualname, node, visitor.score))


def load_coverage(coverage_json: Path, cwd: Path) -> dict[Path, dict[str, set[int]]]:
    with coverage_json.open() as coverage_file:
        payload = json.load(coverage_file)

    coverage_by_path: dict[Path, dict[str, set[int]]] = {}
    for raw_path, file_payload in payload.get("files", {}).items():
        path = Path(raw_path)
        if not path.is_absolute():
            path = cwd / path
        coverage_by_path[path.resolve()] = {
            "executed": set(file_payload.get("executed_lines", [])),
            "missing": set(file_payload.get("missing_lines", [])),
        }
    return coverage_by_path


def function_risks(paths: list[Path], coverage_by_path: dict[Path, dict[str, set[int]]]) -> list[FunctionRisk]:
    risks: list[FunctionRisk] = []
    for path in paths:
        source = path.read_text()
        tree = ast.parse(source, filename=str(path))
        collector = FunctionCollector()
        collector.visit(tree)

        coverage = coverage_by_path.get(path.resolve(), {"executed": set(), "missing": set()})
        executed = coverage["executed"]
        executable = executed | coverage["missing"]

        for qualname, node, complexity in collector.functions:
            end_lineno = getattr(node, "end_lineno", node.lineno)
            function_lines = set(range(node.lineno, end_lineno + 1))
            executable_lines = function_lines & executable
            if executable_lines:
                covered_lines = len(executable_lines & executed)
                coverage_fraction = covered_lines / len(executable_lines)
                executable_count = len(executable_lines)
            else:
                covered_lines = 0
                coverage_fraction = 0.0
                executable_count = max(1, end_lineno - node.lineno + 1)

            score = complexity**2 * (1 - coverage_fraction) ** 3 + complexity
            risks.append(
                FunctionRisk(
                    path=path,
                    qualname=qualname,
                    lineno=node.lineno,
                    end_lineno=end_lineno,
                    complexity=complexity,
                    covered_lines=covered_lines,
                    executable_lines=executable_count,
                    coverage=coverage_fraction,
                    score=score,
                )
            )
    return risks


def print_table(title: str, rows: list[FunctionRisk], limit: int | None = None) -> None:
    if not rows:
        print(f"{title}: none")
        return

    print(title)
    print("| Score | Complexity | Coverage | Location |")
    print("| ---: | ---: | ---: | --- |")
    for risk in rows[:limit]:
        location = f"{risk.path}:{risk.lineno} {risk.qualname}"
        print(
            f"| {risk.score:.2f} | {risk.complexity} | {risk.coverage:.0%} | {location} |"
        )


def existing_paths(raw_paths: list[str]) -> list[Path]:
    paths = [Path(raw_path) for raw_path in raw_paths]
    missing = [path for path in paths if not path.exists()]
    if missing:
        for path in missing:
            print(f"Missing CRAP input path: {path}", file=sys.stderr)
        raise SystemExit(2)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute and enforce CRAP scores from coverage JSON.")
    parser.add_argument("paths", nargs="+", help="Python files to scan.")
    parser.add_argument("--coverage-json", required=True, type=Path, help="Path produced by coverage json.")
    parser.add_argument("--max-score", type=float, default=6.0, help="Maximum allowed CRAP score.")
    parser.add_argument("--top", type=int, default=20, help="Number of highest-risk functions to print.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cwd = Path.cwd()
    coverage_by_path = load_coverage(args.coverage_json, cwd)
    risks = function_risks(existing_paths(args.paths), coverage_by_path)
    risks.sort(key=lambda risk: risk.score, reverse=True)

    print_table("Highest CRAP scores", risks, limit=args.top)
    failures = [risk for risk in risks if risk.score > args.max_score]

    if failures:
        print()
        print_table(f"CRAP failures above {args.max_score:.2f}", failures)
        return 1

    print(f"All CRAP scores are <= {args.max_score:.2f}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
