#!/usr/bin/env python3
"""Build the ladder pool: public Kaggle notebooks whose agents play like the opponents that
beat Linden Brook, as arena bundles.

    python shinka/champions/ladder/build_ladder_pool.py [--cache DIR] [--only NAME,...]

For each member of MEMBERS the notebook is pulled read-only (`kaggle kernels pull -m`) into
--cache and its embedded agent is recovered **statically**: the large string literals of its
code cells are decoded (base64/base85/a85, then zlib/gzip/lzma/bz2) until Python source or a
tar archive appears. No notebook code is executed. The bundle is then

    <name>/agent/<files>   the recovered agent, byte for byte (tar members as shipped)
    <name>/main.py         host_main.py: loads agent/<entry> like kaggle_environments does
    <name>/SOURCE.json     notebook ref and id, pull time, entry, sha256 of every agent file

The recovered sha256 must equal SOURCE.json's when the bundle already exists (a changed
notebook is reported, never silently replaced; pass --update to accept it).
"""
from __future__ import annotations

import argparse
import ast
import base64
import bz2
import datetime as dt
import gzip
import hashlib
import io
import json
import lzma
import os
import re
import shutil
import subprocess
import sys
import tarfile
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent

# name -> (notebook ref, how to pick the agent among the decoded blobs, entry file)
MEMBERS = {
    "abo_v57": ("ahmedberatozer/kaggriculture-v57-funding-order-invariant", "SOURCE_BLOB", "agent.py"),
    "abo_v55": ("ahmedberatozer/kaggriculture-v55-one-turn-market-race-edge", "SOURCE_BLOB", "agent.py"),
    "abo_v43": ("ahmedberatozer/kaggriculture-v43-recovering-lost-harvests", "SOURCE_BYTES", "agent.py"),
    "pilkwang_sep": ("pilkwang/kaggriculture-structured-economic-policy", "archive_bytes", "main.py"),
    "haideptry_shepherd": ("haideptry/the-shepherds-ledger-herd-safe-sovereign", "AGENT_B64", "agent.py"),
    "haideptry_2965": ("haideptry/the-2965-master-hybrid-engine", "AGENT_B64", "agent.py"),
    # the same notebook's version of 2026-09-26 13:54 UTC (haideptry_2965 keeps the one of 09-25, the
    # evidence of Rowan Glen's and Linden Brook's losses); it plays Alder Ford's rival ADRIANO ALMEIDA exactly
    "haideptry_2965_0926": ("haideptry/the-2965-master-hybrid-engine", "AGENT_B64", "agent.py"),
    "robust_economy": ("nihilisticneuralnet/kaggriculture-population-robust-economy", "MAIN_BLOB", "agent.py"),
    "tetsutani_demand": ("tetsutani/demand-preserving-turn-sale-timing", "ARCHIVE_B64", "main.py"),
    # tetsutani's earlier notebook (09-06): the 13-wheat opening; plays three of Rowan Glen's and Linden
    # Brook's 13-wheat rivals move for move (evidence/ladder_pool_20260926/open13_match.jsonl)
    "tetsutani_shape_shop": ("tetsutani/shape-the-shop-work-the-pasture-kaggriculture", "MAIN_B64", "agent.py"),
    "leoprovorov_forecast": ("leoprovorov/four-turn-forecast-notebook-version-2", "SOURCE", "agent.py"),
    "hanifnoerrofiq_pioneers": ("hanifnoerrofiq/pioneers-of-kaggle-town-candidate-2", "content", "agent.py"),
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def const_value(node):
    """Literal str/bytes expressions: constants, +, tuples/lists of parts and sep.join((...))."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes)):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        a, b = const_value(node.left), const_value(node.right)
        if a is not None and b is not None and type(a) is type(b):
            return a + b
    if isinstance(node, (ast.Tuple, ast.List)):
        parts = [const_value(e) for e in node.elts]
        if parts and all(p is not None for p in parts) and len({type(p) for p in parts}) == 1:
            return parts[0][:0].join(parts)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "join":
        sep = const_value(node.func.value)
        if sep is not None and node.args and isinstance(node.args[0], (ast.Tuple, ast.List)):
            parts = [const_value(e) for e in node.args[0].elts]
            if parts and all(p is not None for p in parts):
                return sep.join(parts)
    return None


def blobs(code: str):
    """(label, value) of every literal of at least 5,000 characters; label = the assigned name
    or the dict key it sits under."""
    lines = [l for l in code.splitlines() if not l.lstrip().startswith(("%", "!"))]
    try:
        tree = ast.parse("\n".join(lines))
    except SyntaxError:
        return []
    found = []

    def visit(node, label):
        value = const_value(node) if isinstance(node, ast.expr) else None
        if value is not None and len(value) >= 5000:
            found.append((label, value))
            return
        if isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                key = const_value(k) if k is not None else None
                visit(v, key if isinstance(key, str) else label)
            return
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            label = ast.unparse(node.targets[0])
        for child in ast.iter_child_nodes(node):
            visit(child, label)

    visit(tree, "blob")
    return found


def decode(value):
    """First decoding of a blob that yields Python source (str) or a tar archive ({name: bytes})."""
    raw = value.encode("latin-1", "ignore") if isinstance(value, str) else value
    steps = [("raw", raw)]
    for name, fn in (("b64", base64.b64decode), ("b85", base64.b85decode), ("a85", base64.a85decode)):
        try:
            steps.append((name, fn(raw.strip())))
        except Exception:
            pass
    for how, data in steps:
        for zname, z in (("zlib", zlib.decompress), ("gzip", gzip.decompress), ("lzma", lzma.decompress),
                         ("bz2", bz2.decompress), ("", lambda d: d)):
            try:
                out = z(data)
            except Exception:
                continue
            try:
                with tarfile.open(fileobj=io.BytesIO(out)) as tf:
                    files = {m.name: tf.extractfile(m).read() for m in tf.getmembers()
                             if m.isfile() and not m.name.startswith("/") and ".." not in m.name}
                if files:
                    return f"{how}+{zname}+tar", files
            except Exception:
                pass
            try:
                text = out.decode("utf-8")
            except Exception:
                continue
            if re.search(r"^\s*(def |import |from |class )", text, re.M) and "def " in text:
                return f"{how}+{zname}", text
    return None, None


def recover(notebook: Path, selector: str):
    """The agent files recovered from the notebook's cells: ({relative name: bytes}, how)."""
    nb = json.loads(notebook.read_text())
    cells = ["".join(c.get("source", [])) for c in nb.get("cells", []) if c.get("cell_type") == "code"]
    for code in cells:
        for label, value in blobs(code):
            if selector not in label:
                continue
            how, out = decode(value)
            if isinstance(out, dict):
                return {k: v for k, v in out.items() if "__pycache__" not in k}, how
            if isinstance(out, str):
                return {"agent.py": out.encode("utf-8")}, how
    raise SystemExit(f"{notebook}: no decodable blob labelled {selector!r}")


def pull(ref: str, cache: Path) -> Path:
    dst = cache / ref.replace("/", "__")
    if not list(dst.glob("*.ipynb")):
        dst.mkdir(parents=True, exist_ok=True)
        kaggle = shutil.which("kaggle") or str(Path(sys.executable).with_name("kaggle"))
        subprocess.run([kaggle, "kernels", "pull", ref, "-p", str(dst), "-m"], check=True,
                       stdout=subprocess.DEVNULL)
    return dst


def build(name: str, cache: Path, update: bool) -> dict:
    ref, selector, entry = MEMBERS[name]
    src = pull(ref, cache)
    files, how = recover(next(src.glob("*.ipynb")), selector)
    if entry not in files:
        raise SystemExit(f"{name}: entry {entry} not among recovered files {sorted(files)}")
    meta = json.loads((src / "kernel-metadata.json").read_text())
    record = {"name": name, "notebook": ref, "notebook_id": meta.get("id_no"), "title": meta.get("title"),
              "url": f"https://www.kaggle.com/code/{ref}", "decoded": f"{selector} ({how})", "entry": entry,
              "files": {k: sha(v) for k, v in sorted(files.items())}}
    dst = HERE / name
    old = dst / "SOURCE.json"
    if old.exists():
        previous = json.loads(old.read_text())
        if previous["files"] != record["files"] and not update:
            raise SystemExit(f"{name}: the notebook's agent changed since {previous['pulled_utc']}; "
                             "rerun with --update to replace the bundle")
        record["pulled_utc"] = previous["pulled_utc"] if previous["files"] == record["files"] else None
    record["pulled_utc"] = record.get("pulled_utc") or dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "agent").mkdir(parents=True)
    for rel, data in files.items():
        (dst / "agent" / rel).parent.mkdir(parents=True, exist_ok=True)
        (dst / "agent" / rel).write_bytes(data)
    shutil.copy2(HERE / "host_main.py", dst / "main.py")
    (dst / "SOURCE.json").write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n")
    return record


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, default=Path(os.environ.get("TMPDIR", "/tmp")) / "kagg_ladder_notebooks")
    ap.add_argument("--only", default="", help="comma list of member names")
    ap.add_argument("--update", action="store_true", help="accept a notebook whose agent changed")
    args = ap.parse_args()
    names = [n for n in args.only.split(",") if n] or list(MEMBERS)
    for name in names:
        rec = build(name, args.cache, args.update)
        print(f"{name:20} {rec['notebook']:62} {rec['decoded']:28} files={len(rec['files'])}")


if __name__ == "__main__":
    main()
