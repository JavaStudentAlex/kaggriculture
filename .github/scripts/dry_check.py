#!/usr/bin/env python3
"""Find suspicious structural duplication in Python functions.

The check normalizes function ASTs by erasing local naming and literal values,
then reports functions whose structure is identical. It is intentionally
conservative: it catches repeated code shapes without treating every similar
small helper as a CI failure.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FunctionShape:
    path: Path
    qualname: str
    lineno: int
    end_lineno: int
    line_count: int
    digest: str


class Normalizer(ast.NodeTransformer):
    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        self.generic_visit(node)
        node.name = "_function"
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
        self.generic_visit(node)
        node.name = "_function"
        return node

    def visit_Name(self, node: ast.Name) -> ast.Name:
        return ast.copy_location(ast.Name(id="_name", ctx=node.ctx), node)

    def visit_arg(self, node: ast.arg) -> ast.arg:
        return ast.copy_location(ast.arg(arg="_arg", annotation=None, type_comment=None), node)

    def visit_Attribute(self, node: ast.Attribute) -> ast.Attribute:
        self.generic_visit(node)
        node.attr = "_attr"
        return node

    def visit_keyword(self, node: ast.keyword) -> ast.keyword:
        self.generic_visit(node)
        if node.arg is not None:
            node.arg = "_keyword"
        return node

    def visit_Constant(self, node: ast.Constant) -> ast.Constant:
        return ast.copy_location(ast.Constant(value=type(node.value).__name__), node)


class FunctionCollector(ast.NodeVisitor):
    def __init__(self, path: Path, min_lines: int) -> None:
        self.path = path
        self.min_lines = min_lines
        self.stack: list[str] = []
        self.shapes: list[FunctionShape] = []

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
        end_lineno = getattr(node, "end_lineno", node.lineno)
        line_count = end_lineno - node.lineno + 1
        if line_count < self.min_lines:
            return

        clone = ast.fix_missing_locations(Normalizer().visit(ast.parse(ast.unparse(node))))
        normalized = ast.dump(clone, include_attributes=False)
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        qualname = ".".join([*self.stack, node.name])
        self.shapes.append(
            FunctionShape(
                path=self.path,
                qualname=qualname,
                lineno=node.lineno,
                end_lineno=end_lineno,
                line_count=line_count,
                digest=digest,
            )
        )


def collect_shapes(paths: list[Path], min_lines: int) -> list[FunctionShape]:
    shapes: list[FunctionShape] = []
    for path in paths:
        tree = ast.parse(path.read_text(), filename=str(path))
        collector = FunctionCollector(path, min_lines)
        collector.visit(tree)
        shapes.extend(collector.shapes)
    return shapes


def existing_paths(raw_paths: list[str]) -> list[Path]:
    paths = [Path(raw_path) for raw_path in raw_paths]
    missing = [path for path in paths if not path.exists()]
    if missing:
        for path in missing:
            print(f"Missing DRY input path: {path}", file=sys.stderr)
        raise SystemExit(2)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find duplicate Python function structures.")
    parser.add_argument("paths", nargs="+", help="Python files to scan.")
    parser.add_argument("--min-lines", type=int, default=8, help="Ignore functions shorter than this.")
    parser.add_argument("--max-groups", type=int, default=0, help="Maximum allowed duplicate groups.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    shapes = collect_shapes(existing_paths(args.paths), args.min_lines)

    grouped: dict[str, list[FunctionShape]] = {}
    for shape in shapes:
        grouped.setdefault(shape.digest, []).append(shape)

    duplicates = [group for group in grouped.values() if len(group) > 1]
    duplicates.sort(key=lambda group: (-max(shape.line_count for shape in group), group[0].qualname))

    if not duplicates:
        print("DRY structural duplication: none")
        return 0

    print("DRY structural duplication groups")
    for index, group in enumerate(duplicates, start=1):
        print(f"Group {index}:")
        for shape in group:
            print(f"  - {shape.path}:{shape.lineno}-{shape.end_lineno} {shape.qualname}")

    if len(duplicates) > args.max_groups:
        print(f"Found {len(duplicates)} duplicate groups; allowed {args.max_groups}.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
