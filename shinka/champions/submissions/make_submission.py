#!/usr/bin/env python3
"""Build (and validate) a Kaggle submission bundle for an oracle-based champion.

    python shinka/champions/submissions/make_submission.py \\
        --champion shinka/champions/top/run2_2026-09-12/gen_33/main.py \\
        --name "Orchard Tide" [--message "..."] [--checkpoint shinka/evolution/checkpoint] \\
        [--validate] [--seeds 101,70102]

Produces `submissions/<code_name>/` (staging: the pieces below + validate.py +
MANIFEST.json + README.md) and `submissions/<CodeName>.tar.gz` with `main.py` at
the archive root. It never submits: upload with
    kaggle competitions submission-limits kaggriculture
    kaggle competitions submit -c kaggriculture -f submissions/<CodeName>.tar.gz -m "<message>"
and then watch `kaggle competitions submissions kaggriculture --csv` (PENDING ->
COMPLETE; on ERROR read `episodes <id>` / `logs <episode> <seat>` / `replay`).

What goes in, and why (each rule was learned from a failed run on 2026-09-12,
see orchard_tide/README.md):
  main.py            bootstrap: kaggle_environments exec()s the submitted file with no
                     __file__ and takes the LAST callable, and it APPENDS the bundle dir
                     to sys.path; the champion needs __file__, so main.py locates the
                     bundle, sets the KAGG_* overrides, imports the champion as a module
                     and ends with kaggle_submission_agent().
  champion.py        the champion, byte-identical to the evolved program
  kagg_oracle.py + kagg_ttm_numpy.py
                     the oracle on its numpy backend (KAGG_ORACLE_BACKEND=numpy): the
                     first request to a Kaggle agent has ~60 s and importing torch +
                     transformers in the sandbox blew it (TIMEOUT at step 1); numpy
                     imports in 50 ms and the forecast costs ~320 ms there.
  checkpoint/        model.safetensors, config.json, scaler.npz, labels.json of the
                     checkpoint the champion was evolved with
  opponent_model/    features.py + mechanics.py (the oracle imports them; without them
                     the champion silently plays oracle-less)
  mohui_v66/         the backbone closure with LICENSE + NOTICE
Public names are neutral two-word codenames (no generation, score or exploit in
the message); the generation-to-name mapping stays in MANIFEST.json.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import py_compile
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent            # shinka/champions/submissions
SHINKA = HERE.parent.parent                       # shinka/
REPO = SHINKA.parent
EVOLUTION = SHINKA / "evolution"
CLEAN_VENV_PY = Path("/results/kagg/venv-kaggle-sim/bin/python")

CHECKPOINT_FILES = ("model.safetensors", "config.json", "scaler.npz", "labels.json")

MAIN_PY = '''"""Kaggriculture submission "{name}": an evolved Kaggriculture agent playing with
the opponent order-flow oracle on the CPU (numpy backend).

kaggle_environments loads a submitted main.py with exec() in a bare namespace:
there is no __file__, the directory of main.py is APPENDED to sys.path only while
this module-level code runs, and the LAST callable left in the namespace becomes
the agent. This file therefore only locates the bundle, points the unmodified
champion at the bundled pieces through the env overrides it honours, imports it
as a real module (which restores __file__ for it) and ends with the agent.

Bundle layout (relative to this file): champion.py, kagg_oracle.py,
kagg_ttm_numpy.py, checkpoint/, opponent_model/, mohui_v66/. Only numpy is
needed from the image. Built by shinka/champions/submissions/make_submission.py.
"""
import os
import sys


def _bundle_dir() -> str:
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        pass
    f = sys._getframe(0).f_code.co_filename  # compile(raw, path) kept the path
    if f and f != "<string>":
        d = os.path.dirname(os.path.abspath(f))
        if os.path.isfile(os.path.join(d, "champion.py")):
            return d
    for p in reversed(sys.path):
        if os.path.isfile(os.path.join(p or ".", "champion.py")):
            return os.path.abspath(p or ".")
    return os.getcwd()


HERE = _bundle_dir()

os.environ.setdefault("KAGG_MOHUI_DIR", os.path.join(HERE, "mohui_v66"))
os.environ.setdefault("KAGG_ORACLE_SRC", HERE)
os.environ.setdefault("KAGG_TTM_DIR", os.path.join(HERE, "checkpoint"))
os.environ.setdefault("KAGG_OPP_MODEL_SRC", os.path.join(HERE, "opponent_model"))
os.environ["KAGG_ORACLE_BACKEND"] = "numpy"
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# front of sys.path unconditionally (the loader's own entry is at the END)
sys.path.insert(0, HERE)

import champion as _champion  # noqa: E402  (loads the backbone and the oracle)


def agent(obs, configuration=None):
    return _champion.agent(obs, configuration)


# kaggle_environments takes the LAST callable bound in this file: keep this last.
def kaggle_submission_agent(obs, configuration=None):
    return agent(obs, configuration)
'''

VALIDATE_PY = '''#!/usr/bin/env python3
"""Validate an EXTRACTED submission archive the way Kaggle runs it.

    HOME=<empty dir> python -I validate.py <extracted dir> [seeds]

python -I: no script dir / PYTHONPATH / user site. The repository must be
unreachable (every KAGG_* unset, empty HOME: the champion's last-resort paths go
through Path.home()). Runs complete episodes on every seed in BOTH seats against
the built-in opponents; requires DONE statuses, finite rewards, no traceback in
the agent's stderr; reports the loader's own per-step durations against the 1 s
actTimeout and the 60 s overage bank; proves the oracle is live on the numpy
backend (torch never imported) and that every piece resolved from inside the
extracted directory.
"""
import math
import os
import statistics
import sys
import time

for v in ("KAGG_REPO", "KAGG_MOHUI_DIR", "KAGG_HISTORY_DIR", "KAGG_TASK_DIR", "KAGG_ORACLE_SRC",
          "KAGG_OPP_MODEL_SRC", "KAGG_TTM_DIR", "KAGG_ORACLE_BACKEND"):
    assert v not in os.environ, f"{v} is set; run without the repository environment"
bundle = os.path.abspath(sys.argv[1])
seeds = [int(s) for s in (sys.argv[2] if len(sys.argv) > 2 else "101,70102").split(",")]
opponents = ["starter", "random"]
os.chdir("/")
from kaggle_environments import make  # noqa: E402

main_py = os.path.join(bundle, "main.py")
failures = []
durations = {0: [], 1: []}
print(f"bundle {bundle}\\nHOME={os.environ.get('HOME')} cwd={os.getcwd()}")
for seat in (0, 1):
    for opp in opponents:
        for seed in seeds:
            env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed})
            agents = [main_py, opp] if seat == 0 else [opp, main_py]
            t0 = time.perf_counter()
            env.run(agents)
            wall = time.perf_counter() - t0
            last = env.steps[-1]
            me, other = last[seat], last[1 - seat]
            d = [sl[seat]["duration"] for sl in env.logs if sl and len(sl) > seat and isinstance(sl[seat], dict) and "duration" in sl[seat]]
            errs = [sl[seat].get("stderr", "") for sl in env.logs if sl and len(sl) > seat and isinstance(sl[seat], dict)]
            tracebacks = [e for e in errs if "Traceback" in e or "Error" in e]
            durations[seat] += d
            over = sum(max(0.0, x - env.configuration.actTimeout) for x in d)
            ok = (me.status == "DONE" and other.status == "DONE" and me.reward is not None and math.isfinite(me.reward)
                  and not tracebacks and len(env.steps) >= 720)
            if not ok:
                failures.append((seat, opp, seed, me.status, other.status, me.reward, tracebacks[:2]))
            print(f"seat {seat} vs {opp:7} seed {seed:>6}: {'OK ' if ok else 'BAD'} ${me.reward:>10,.0f} ({me.status}) vs ${other.reward:,.0f} ({other.status}) "
                  f"| {len(env.steps)} states, {wall:.0f} s | steps {len(d)}: mean {1000*statistics.mean(d):.0f} ms, "
                  f"max {1000*max(d):.0f} ms @{d.index(max(d))}, overage {over:.2f} s"
                  + (f" | stderr: {tracebacks[0][:160]!r}" if tracebacks else ""), flush=True)

champ = sys.modules.get("champion")
oracle_mod = sys.modules.get("kagg_oracle")
ttm_np = sys.modules.get("kagg_ttm_numpy")
feat, mech = sys.modules.get("features"), sys.modules.get("mechanics")
stats = getattr(champ, "ORACLE_STATS", None)
inside = lambda p: bool(p) and os.path.abspath(p).startswith(bundle + os.sep)
model = getattr(champ, "_MODEL", None)
model_dir = getattr(model, "model_dir", None)
print("\\nresolution:")
for label, path in (("champion", getattr(champ, "__file__", None)), ("MOHUI_DIR", getattr(champ, "MOHUI_DIR", None)),
                    ("kagg_oracle", getattr(oracle_mod, "__file__", None)), ("checkpoint", model_dir),
                    ("kagg_ttm_numpy", getattr(ttm_np, "__file__", None)), ("features", getattr(feat, "__file__", None)),
                    ("mechanics", getattr(mech, "__file__", None))):
    print(f"  {label:15}: {path}  inside={inside(path or '')}")
print(f"  backend        : {getattr(model, 'backend', None)}  torch imported: {'torch' in sys.modules}")
print(f"  oracle         : device={getattr(champ, 'ORACLE_DEVICE', None)} stats={stats}")
late = [x for s in (0, 1) for x in durations[s][512:]]
allsteps = durations[0] + durations[1]
print(f"\\nsteps {len(allsteps)}: max {1000*max(allsteps):.0f} ms; from turn 512: p99 {1000*sorted(late)[int(0.99*len(late))]:.0f} ms, max {1000*max(late):.0f} ms")
oracle_ok = (stats is not None and stats.get("errors", 1) == 0 and stats.get("forecasts", 0) > 0
             and getattr(model, "backend", None) == "numpy" and "torch" not in sys.modules)
resolved_ok = all(inside(p or "") for p in (getattr(champ, "__file__", ""), getattr(champ, "MOHUI_DIR", ""), getattr(oracle_mod, "__file__", ""),
                                             model_dir, getattr(ttm_np, "__file__", ""), getattr(feat, "__file__", ""), getattr(mech, "__file__", "")))
verdict = "PASS" if (not failures and oracle_ok and resolved_ok) else "FAIL"
print(f"\\nVALIDATION {verdict}: games failed={len(failures)} oracle_live={oracle_ok} all_resolved_inside_bundle={resolved_ok}")
sys.exit(0 if verdict == "PASS" else 1)
'''


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snake(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def camel(name: str) -> str:
    return "".join(w.capitalize() for w in re.split(r"[^A-Za-z0-9]+", name) if w)


def build(args) -> int:
    champion = Path(args.champion).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    name = args.name.strip()
    if not re.fullmatch(r"[A-Za-z]+ [A-Za-z]+", name):
        sys.exit(f"--name must be a neutral two-word codename, got {name!r}")
    if re.search(r"gen|score|win|oracle|shinka", name, re.I):
        sys.exit(f"--name {name!r} leaks internals; pick a neutral codename")
    message = args.message or f"{name} - Adaptive production and trade"
    stage = HERE / snake(name)
    archive = HERE / f"{camel(name)}.tar.gz"
    if stage.exists() and not args.force:
        sys.exit(f"{stage} exists; pass --force to rebuild it")
    for f in CHECKPOINT_FILES:
        if not (checkpoint / f).is_file():
            sys.exit(f"checkpoint incomplete: {checkpoint / f} missing")
    src = champion.read_text()
    for hook in ("KAGG_MOHUI_DIR", "KAGG_ORACLE_SRC", "import kagg_oracle"):
        if hook not in src:
            sys.exit(f"{champion} does not use {hook}: not an oracle-based champion of this lineage")

    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    shutil.copy2(champion, stage / "champion.py")
    for f in ("kagg_oracle.py", "kagg_ttm_numpy.py"):
        shutil.copy2(EVOLUTION / f, stage / f)
    (stage / "checkpoint").mkdir()
    for f in CHECKPOINT_FILES:
        shutil.copy2(checkpoint / f, stage / "checkpoint" / f)
    (stage / "opponent_model").mkdir()
    for f in ("features.py", "mechanics.py"):
        shutil.copy2(REPO / "research" / "opponent_model" / f, stage / "opponent_model" / f)
    shutil.copytree(SHINKA / "champions" / "dependencies" / "mohui_v66", stage / "mohui_v66",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for f in ("LICENSE", "NOTICE"):
        if not (stage / "mohui_v66" / f).is_file():
            sys.exit(f"mohui_v66/{f} missing: the closure must ship its attribution")
    (stage / "main.py").write_text(MAIN_PY.format(name=name))
    (stage / "validate.py").write_text(VALIDATE_PY)

    for py in stage.rglob("*.py"):
        py_compile.compile(str(py), doraise=True)
    for pyc in stage.rglob("__pycache__"):
        shutil.rmtree(pyc)

    excluded = {"validate.py", "README.md", "MANIFEST.json"}
    with tarfile.open(archive, "w:gz") as tar:
        for path in sorted(stage.rglob("*")):
            rel = path.relative_to(stage)
            if path.is_file() and rel.parts[0] not in excluded:
                tar.add(path, arcname=str(rel))
    with tarfile.open(archive) as tar:
        members = sorted(m.name for m in tar.getmembers() if m.isfile())
    assert members.count("main.py") == 1 and "champion.py" in members, members

    manifest = {
        "public_name": name, "message": message, "archive": archive.name, "archive_sha256": sha256(archive),
        "archive_bytes": archive.stat().st_size, "archive_files": members,
        "champion": {"source": str(champion.relative_to(REPO)) if champion.is_relative_to(REPO) else str(champion),
                     "sha256": sha256(champion), "note": args.note or ""},
        "oracle": {"kagg_oracle_sha256": sha256(EVOLUTION / "kagg_oracle.py"),
                   "kagg_ttm_numpy_sha256": sha256(EVOLUTION / "kagg_ttm_numpy.py"),
                   "checkpoint": str(checkpoint), "model_safetensors_sha256": sha256(checkpoint / "model.safetensors"),
                   "labels": json.loads((checkpoint / "labels.json").read_text()), "backend": "numpy"},
        "built": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "submit": f'kaggle competitions submit -c kaggriculture -f {archive.relative_to(REPO)} -m "{message}"',
    }
    (stage / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    (stage / "README.md").write_text(
        f"# {name}\n\nSubmission bundle built {manifest['built']} by `make_submission.py` from "
        f"`{manifest['champion']['source']}` (sha256 `{manifest['champion']['sha256'][:16]}…`) with the checkpoint in "
        f"`{checkpoint}`. Archive `../{archive.name}` ({manifest['archive_bytes']:,} bytes, sha256 "
        f"`{manifest['archive_sha256'][:16]}…`). {args.note or ''}\n\nSee `MANIFEST.json`; re-validate with "
        f"`HOME=<empty> python -I validate.py <extracted dir>` (make_submission.py --validate does this). "
        f"Procedure and the traps it avoids: `.agents/skills/kaggle-simulation-competitions/references/submission-promotion.md`.\n")
    print(f"staged  : {stage}\narchive : {archive} ({manifest['archive_bytes']:,} bytes, {len(members)} files, sha256 {manifest['archive_sha256'][:16]}…)")

    if args.validate:
        rc = validate(stage, archive, args.seeds, manifest)
        (stage / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
        if rc:
            print("VALIDATION FAILED: do not submit")
            return rc
    print(f"\nsubmit  : {manifest['submit']}")
    return 0


def validate(stage: Path, archive: Path, seeds: str, manifest: dict) -> int:
    py = CLEAN_VENV_PY if CLEAN_VENV_PY.exists() else Path(sys.executable)
    with tempfile.TemporaryDirectory(prefix="kagg_submission_") as tmp:
        extracted = Path(tmp) / "x"
        fakehome = Path(tmp) / "home"
        extracted.mkdir()
        fakehome.mkdir()
        with tarfile.open(archive) as tar:
            try:
                tar.extractall(extracted, filter="data")
            except TypeError:  # interpreters without the filter argument
                tar.extractall(extracted)
        env = {k: v for k, v in os.environ.items() if not k.startswith("KAGG_") and k != "PYTHONPATH"}
        env.update({"HOME": str(fakehome), "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
        cmd = [str(py), "-I", str(stage / "validate.py"), str(extracted), seeds]
        if shutil.which("taskset"):
            cmd = ["taskset", "-c", "1"] + cmd
        print(f"\nvalidating the extracted archive with {py} (one core, empty HOME, no repo env)…", flush=True)
        proc = subprocess.run(cmd, cwd="/", env=env, text=True, capture_output=True)
        out = "\n".join(l for l in proc.stdout.splitlines() if l.strip() and "warning" not in l.lower())
        print(out)
        if proc.returncode:
            print(proc.stderr[-2000:])
        manifest["validation"] = {"date": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "python": str(py),
                                  "seeds": seeds, "result": "PASS" if proc.returncode == 0 else "FAIL", "report": out}
        return proc.returncode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--champion", required=True, help="the evolved program (e.g. shinka/champions/top/<run>/gen_N/main.py)")
    ap.add_argument("--name", required=True, help='neutral two-word public codename, e.g. "Orchard Tide"')
    ap.add_argument("--message", default=None, help='submission message; default "<name> - Adaptive production and trade"')
    ap.add_argument("--checkpoint", default=str(EVOLUTION / "checkpoint"), help="the checkpoint the champion was evolved with")
    ap.add_argument("--note", default="", help="private provenance note for MANIFEST.json (run, generation, scores)")
    ap.add_argument("--validate", action="store_true", help="extract the archive and run validate.py in the clean venv")
    ap.add_argument("--seeds", default="101,70102")
    ap.add_argument("--force", action="store_true", help="rebuild an existing staging directory")
    return build(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
