#!/usr/bin/env python3
"""Package a procedural-graph agent (graph + Hazel runtime + predictor [+ calibration]) for Kaggle.

    python3 research/procedural_graph/make_graph_submission.py \\
        --graph research/procedural_graph/evolution_results/feed_fix_2026-09-24/feed15_graph.json \\
        --checkpoint models/ttm_c256_h96_ft_2026-09-24 \\
        --calibration research/procedural_graph/calibration/ttm_c256_h96_ft_2026-09-24/calibration.json \\
        --name "Two Words" --note "<private provenance>" --validate --fidelity 101 --python <clean venv python>

The bundle is the one the arena plays (arena/payload.write_graph_bundle: the graph re-pinned to
the copied runtime, the predictor and its calibration in hazel_runtime/checkpoint/). The
graph's own entrypoint becomes agent_graph.py, byte-identical, so its provenance pin still holds.
A bootstrap main.py goes in front of it. kaggle_environments execs main.py in a bare namespace:
- there is no __file__;
- the bundle dir is APPENDED to sys.path only while the file runs;
- the LAST callable becomes the agent.
The bootstrap therefore:
1. locates the bundle;
2. forces the numpy predictor;
3. imports agent_graph.py as a real module;
4. builds the engine, which verifies every pinned runtime hash;
5. ends with the agent.

Staged in shinka/champions/submissions/<snake name>/ (MANIFEST.json, README.md and validate.py
stay out of the archive), archived as shinka/champions/submissions/<CamelName>.tar.gz.
--validate extracts the archive and plays it the way Kaggle does. It uses python -I, an empty
HOME, no KAGG_* variables, cwd /, one core, and both seats x starter/random x seeds. It requires:
- DONE statuses and no tracebacks;
- the predictor live on numpy, with torch never imported;
- every module resolved inside the extracted directory;
- the engine built, with the calibration present when one was packaged.
--fidelity SEED also plays that seed against starter through the arena harness (process-isolated
BundleAgent on agent_graph.py), which must end with the same cash as through the Kaggle loader.
Submitting stays manual: the command is printed and recorded in the manifest.
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
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

HERE = Path(__file__).resolve().parent            # research/procedural_graph
REPO = HERE.parents[1]
SUBMISSIONS = REPO / 'shinka' / 'champions' / 'submissions'
BUNDLE_AGENT = REPO / 'shinka' / 'evolution' / 'pool_upgrade_bundle_agent.py'
EXCLUDED = {'validate.py', 'README.md', 'MANIFEST.json'}


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PAYLOAD = _module('graph_payload', HERE / 'arena' / 'payload.py')
YARN_FIX = _module('yarn_second_fix', HERE / 'arena' / 'yarn_second_fix.py')
MAKE_SUBMISSION = _module('make_submission', SUBMISSIONS / 'make_submission.py')
sha256, snake, camel = MAKE_SUBMISSION.sha256, MAKE_SUBMISSION.snake, MAKE_SUBMISSION.camel

MAIN_PY = '''"""Kaggriculture submission "{name}": a procedural-graph Kaggriculture agent playing
with the opponent order-flow oracle on the CPU (numpy backend).

kaggle_environments loads a submitted main.py with exec() in a bare namespace:
there is no __file__, the directory of main.py is APPENDED to sys.path only while
this module-level code runs, and the LAST callable left in the namespace becomes
the agent. This file therefore only locates the bundle, imports the unmodified
graph entrypoint agent_graph.py as a real module (which restores __file__ for it
and lets it check its own pinned fingerprint), builds the engine once (it verifies
every runtime file against the graph's pins) and ends with the agent.

Bundle layout (relative to this file): agent_graph.py, policy_graph.json and
hazel_runtime/ (champion.py, graph_runtime.py, kagg_oracle.py, kagg_ttm_numpy.py,
checkpoint/, opponent_model/, mohui_v66/). Only numpy is needed from the image.
Built by research/procedural_graph/make_graph_submission.py.
"""
import importlib.util
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
        if os.path.isfile(os.path.join(d, "agent_graph.py")):
            return d
    for p in reversed(sys.path):
        if os.path.isfile(os.path.join(p or ".", "agent_graph.py")) and \\
                os.path.isfile(os.path.join(p or ".", "policy_graph.json")):
            return os.path.abspath(p or ".")
    return os.getcwd()


HERE = _bundle_dir()

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ["KAGG_ORACLE_BACKEND"] = "numpy"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.pop("KAGG_GRAPH_PATH", None)   # always the bundled policy_graph.json

# front of sys.path unconditionally (the loader's own entry is at the END)
sys.path.insert(0, HERE)

_spec = importlib.util.spec_from_file_location("agent_graph", os.path.join(HERE, "agent_graph.py"))
_graph = importlib.util.module_from_spec(_spec)
sys.modules["agent_graph"] = _graph
_spec.loader.exec_module(_graph)
_graph.get_engine()   # build and verify now, inside the first request


def agent(obs, configuration=None):
    return _graph.agent(obs, configuration)


# kaggle_environments takes the LAST callable bound in this file: keep this last.
def kaggle_submission_agent(obs, configuration=None):
    return agent(obs, configuration)
'''

GRAPH_CHECKS = '''
graph = sys.modules.get("agent_graph")
cal = os.path.join(bundle, "hazel_runtime", "checkpoint", "calibration.json")
engine_built = getattr(graph, "_ENGINE", None) is not None
print(f"  agent_graph    : {getattr(graph, '__file__', None)}  inside={inside(getattr(graph, '__file__', '') or '')}  engine built: {engine_built}")
print(f"  calibration    : {cal if os.path.isfile(cal) else None}  (expected: {EXPECT_CALIBRATION})")
graph_ok = (graph is not None and inside(getattr(graph, "__file__", "") or "") and engine_built
            and inside(model_dir or "") and os.path.isfile(cal) == EXPECT_CALIBRATION)
# a graph whose channels and guard are off never asks the predictor (ladder engines, the rival emulator)
oracle_needed = bool(getattr(getattr(graph, "_ENGINE", None), "oracle_needed", True))
print(f"  oracle needed  : {oracle_needed}")
'''


def validator(expect_calibration: bool) -> str:
    """make_submission.py's validator plus the graph checks."""
    text = MAKE_SUBMISSION.VALIDATE_PY
    verdict = 'verdict = "PASS" if (not failures and oracle_ok and resolved_ok) else "FAIL"'
    if verdict not in text:
        sys.exit('make_submission.VALIDATE_PY changed: update the graph checks')
    text = text.replace(verdict, GRAPH_CHECKS.replace('EXPECT_CALIBRATION', str(expect_calibration)).strip() + '\n'
                        + 'verdict = "PASS" if (not failures and (oracle_ok or not oracle_needed) and resolved_ok '
                          'and graph_ok) else "FAIL"')
    return text.replace('all_resolved_inside_bundle={resolved_ok}"',
                        'all_resolved_inside_bundle={resolved_ok} graph_ok={graph_ok} oracle_needed={oracle_needed}"')


FIDELITY_PY = '''
import json, os, sys
sys.path.insert(0, os.path.dirname(sys.argv[3]))
from kaggle_environments import make
from pool_upgrade_bundle_agent import BundleAgent
bundle, seed = os.path.abspath(sys.argv[1]), int(sys.argv[2])
os.chdir("/")
env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed})
env.run([os.path.join(bundle, "main.py"), "starter"])
kaggle = [s.reward for s in env.steps[-1]]
arena_agent = BundleAgent(bundle, entrypoint="agent_graph.py", timeout=600.0)
env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed})
env.run([arena_agent, "starter"])
arena = [s.reward for s in env.steps[-1]]
arena_agent.close()
print(json.dumps({"seed": seed, "kaggle_loader": kaggle, "arena_harness": arena, "identical": kaggle == arena}))
sys.exit(0 if kaggle == arena else 1)
'''


ENGINE_NOTICE = '''This directory holds the agent of the public Kaggle notebook "{title}"
{url}
pulled {pulled} and shipped unmodified (SOURCE.json lists the sha256 of each file).

The notebook is released under the Apache License, Version 2.0 (LICENSE). The agent file keeps its
upstream copyright and attribution notices, which name the earlier public work it derives from.

The graph policy (policy_graph.json, "engine_parameters") sets some of the file's module-level
constants when the engine loads, and hazel_runtime/ adds its own layers on top of the agent's actions.
'''


EMULATED_NOTICE = '''This directory holds the agent of the public Kaggle notebook "{title}"
{url}
pulled {pulled} and shipped unmodified (SOURCE.json lists the sha256 of each file).

The notebook is released under the Apache License, Version 2.0 (LICENSE). The agent file keeps its
upstream copyright and attribution notices, which name the earlier public work it derives from.

hazel_runtime/rival_emulator.py runs this agent unmodified, from our own observations, to predict the
orders of an opponent that plays it.
'''


def add_engine_notice(stage, graph):
    """Every ladder engine shipped (the backbone's, and those the rival emulator runs) gets the Apache-2.0
    text and a NOTICE naming its notebook (not pinned: the runtime verifies only the pinned files)."""
    backbone = next((n.get('engine') for n in graph['turn']['nodes'] if n['id'] == 'backbone'), None)
    engines_dir = stage / 'hazel_runtime' / 'engines'
    shipped = sorted(d.name for d in engines_dir.iterdir() if (d / 'SOURCE.json').is_file()) if engines_dir.is_dir() else []
    for engine in shipped:
        engine_dir = engines_dir / engine
        source = json.loads((engine_dir / 'SOURCE.json').read_text())
        shutil.copy2(stage / 'hazel_runtime' / 'mohui_v66' / 'LICENSE', engine_dir / 'LICENSE')
        text = ENGINE_NOTICE if engine == backbone else EMULATED_NOTICE
        (engine_dir / 'NOTICE').write_text(text.format(title=source.get('title', engine), url=source.get('url', ''),
                                                       pulled=source.get('pulled_utc', '')))
    return backbone


def clean_run(py, script, args, tmp):
    fakehome = Path(tmp) / 'home'
    fakehome.mkdir(exist_ok=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith('KAGG_') and k != 'PYTHONPATH'}
    env.update({'HOME': str(fakehome), 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
    cmd = [str(py), '-I', str(script), *map(str, args)]
    if shutil.which('taskset'):
        cmd = ['taskset', '-c', '1'] + cmd
    return subprocess.run(cmd, cwd='/', env=env, text=True, capture_output=True)


def build(args) -> int:
    name = args.name.strip()
    if not re.fullmatch(r'[A-Za-z]+ [A-Za-z]+', name):
        sys.exit(f'--name must be a neutral two-word codename, got {name!r}')
    if re.search(r'gen|score|win|oracle|shinka|feed|graph', name, re.I):
        sys.exit(f'--name {name!r} leaks internals; pick a neutral codename')
    message = args.message or f'{name} - Adaptive production and trade'
    graph_file, checkpoint = Path(args.graph).resolve(), Path(args.checkpoint).resolve()
    calibration = Path(args.calibration).resolve() if args.calibration else None
    stage, archive = SUBMISSIONS / snake(name), SUBMISSIONS / f'{camel(name)}.tar.gz'
    if stage.exists() and not args.force:
        sys.exit(f'{stage} exists; pass --force to rebuild it')
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    graph = json.loads(graph_file.read_text())
    PAYLOAD.write_graph_bundle(stage, graph, name, str(checkpoint), calibration)
    add_engine_notice(stage, graph)
    patch = YARN_FIX.apply(stage) if args.yarn_second_fix else None  # re-pins the graph itself
    (stage / 'main.py').rename(stage / 'agent_graph.py')
    (stage / 'main.py').write_text(MAIN_PY.format(name=name))
    (stage / 'validate.py').write_text(validator(calibration is not None))
    for f in ('LICENSE', 'NOTICE'):
        if not (stage / 'hazel_runtime' / 'mohui_v66' / f).is_file():
            sys.exit(f'hazel_runtime/mohui_v66/{f} missing: the closure must ship its attribution')
    for py in stage.rglob('*.py'):
        py_compile.compile(str(py), doraise=True)
    for pyc in list(stage.rglob('__pycache__')):
        shutil.rmtree(pyc)
    packed = json.loads((stage / 'policy_graph.json').read_text())
    bad = [r for r, h in packed['provenance']['runtime_bundle_hashes'].items()
           if sha256(stage / 'hazel_runtime' / r) != h]
    if bad or sha256(stage / 'agent_graph.py') != packed['provenance']['entrypoint_sha256']:
        sys.exit(f'pins do not match the staged files: {bad or "agent_graph.py"}')

    with tarfile.open(archive, 'w:gz') as tar:
        for path in sorted(stage.rglob('*')):
            rel = path.relative_to(stage)
            if path.is_file() and rel.parts[0] not in EXCLUDED:
                tar.add(path, arcname=str(rel))
    with tarfile.open(archive) as tar:
        members = sorted(m.name for m in tar.getmembers() if m.isfile())
    assert members.count('main.py') == 1 and 'agent_graph.py' in members and 'policy_graph.json' in members, members

    rel = lambda p: str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p)
    manifest = {
        'public_name': name, 'message': message, 'archive': archive.name, 'archive_sha256': sha256(archive),
        'archive_bytes': archive.stat().st_size, 'archive_files': members,
        'graph': {'source': rel(graph_file), 'sha256': sha256(graph_file),
                  'packed_sha256': sha256(stage / 'policy_graph.json'), 'runtime': packed.get('runtime'),
                  'parameters': {n['id']: n['parameters'] for c in ('turn', 'market') for n in packed[c]['nodes']
                                 if n.get('parameters')},
                  'note': args.note or ''},
        'predictor': {'checkpoint': rel(checkpoint), 'model_sha256': sha256(checkpoint / 'model.safetensors'),
                      'labels': json.loads((checkpoint / 'labels.json').read_text()), 'backend': 'numpy'},
        'calibration': ({'source': rel(calibration), 'sha256': sha256(calibration),
                         **{k: v for k, v in json.loads(calibration.read_text()).items() if k != 'factors'}}
                        if calibration else None),
        'runtime': {f: sha256(stage / 'hazel_runtime' / f) for f in ('graph_runtime.py', 'champion.py', 'kagg_oracle.py',
                                                                      'kagg_ttm_numpy.py')},
        'backbone_patch': patch,
        'built': dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'submit': f'kaggle competitions submit -c kaggriculture -f {rel(archive)} -m "{message}"',
    }
    print(f'staged  : {stage}\narchive : {archive} ({manifest["archive_bytes"]:,} bytes, {len(members)} files, '
          f'sha256 {manifest["archive_sha256"][:16]}…)', flush=True)

    rc = 0
    py = Path(args.python) if args.python else Path(sys.executable)
    if args.validate or args.fidelity is not None:
        with tempfile.TemporaryDirectory(prefix='kagg_graph_submission_') as tmp:
            extracted = Path(tmp) / 'x'
            extracted.mkdir()
            with tarfile.open(archive) as tar:
                tar.extractall(extracted, filter='data')
            if args.validate:
                print(f'\nvalidating the extracted archive with {py} (python -I, one core, empty HOME, no repo env)…',
                      flush=True)
                proc = clean_run(py, stage / 'validate.py', [extracted, args.seeds], tmp)
                out = '\n'.join(ln for ln in proc.stdout.splitlines() if ln.strip() and 'warning' not in ln.lower())
                print(out, flush=True)
                if proc.returncode:
                    print(proc.stderr[-2000:])
                manifest['validation'] = {'date': dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
                                          'python': str(py), 'seeds': args.seeds,
                                          'result': 'PASS' if proc.returncode == 0 else 'FAIL', 'report': out}
                rc = rc or proc.returncode
            if args.fidelity is not None:
                script = Path(tmp) / 'fidelity.py'
                script.write_text(FIDELITY_PY)
                shutil.copy2(BUNDLE_AGENT, Path(tmp) / 'pool_upgrade_bundle_agent.py')
                print(f'\nfidelity: seed {args.fidelity} vs starter, Kaggle loader vs arena harness…', flush=True)
                proc = clean_run(py, script, [extracted, args.fidelity, Path(tmp) / 'pool_upgrade_bundle_agent.py'], tmp)
                last = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else proc.stderr[-800:]
                print(last, flush=True)
                manifest['fidelity'] = {'seed': args.fidelity, 'result': 'PASS' if proc.returncode == 0 else 'FAIL',
                                        'report': last}
                rc = rc or proc.returncode
    (stage / 'MANIFEST.json').write_text(json.dumps(manifest, indent=1) + '\n')
    cal_line = (f"calibration `{manifest['calibration']['source']}` (sha256 `{manifest['calibration']['sha256'][:16]}…`)"
                if calibration else 'no calibration')
    patch_line = (f"- backbone patch: `{patch['script']}` (`{patch['file']}` sha256 `{patch['sha256'][:16]}…`)\n"
                  if patch else '')
    (stage / 'README.md').write_text(
        f"# {name}\n\nSubmission bundle built {manifest['built']} by `research/procedural_graph/make_graph_submission.py`.\n\n"
        f"- graph: `{manifest['graph']['source']}` (sha256 `{manifest['graph']['sha256'][:16]}…`), runtime "
        f"`{manifest['graph']['runtime']}`\n- predictor: `{manifest['predictor']['checkpoint']}` (model sha256 "
        f"`{manifest['predictor']['model_sha256'][:16]}…`, numpy backend)\n- {cal_line}\n{patch_line}"
        f"- archive: `../{archive.name}` ({manifest['archive_bytes']:,} bytes, sha256 `{manifest['archive_sha256'][:16]}…`)\n"
        f"- validation: {manifest.get('validation', {}).get('result', 'not run')}; fidelity: "
        f"{manifest.get('fidelity', {}).get('result', 'not run')}\n\n{args.note or ''}\n\n"
        f"See `MANIFEST.json`. Re-validate with `HOME=<empty> python -I validate.py <extracted dir>`. Procedure: "
        f"`.agents/skills/kaggle-simulation-competitions/references/submission-promotion.md`.\n")
    if rc:
        print('\nVALIDATION FAILED: do not submit')
        return rc
    print(f"\nsubmit  : {manifest['submit']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--graph', required=True, help='a hazel_merged_v1 graph file (e.g. an evolution best graph)')
    ap.add_argument('--checkpoint', default=str(HERE / 'hazel_runtime' / 'checkpoint'), help='predictor checkpoint dir')
    ap.add_argument('--calibration', help='calibration.json for that predictor (arena/calib_fit.py)')
    ap.add_argument('--name', required=True, help='neutral two-word public codename')
    ap.add_argument('--message', default=None, help='default "<name> - Adaptive production and trade"')
    ap.add_argument('--note', default='', help='private provenance note for MANIFEST.json')
    ap.add_argument('--validate', action='store_true', help='play the extracted archive the way Kaggle does')
    ap.add_argument('--seeds', default='101,70102')
    ap.add_argument('--fidelity', type=int, default=None, metavar='SEED',
                    help='also play SEED vs starter via the arena harness; the cash must match the Kaggle loader')
    ap.add_argument('--python', help='clean interpreter with kaggle-environments 1.32.7 and numpy (default: this one)')
    ap.add_argument('--force', action='store_true', help='rebuild an existing staging directory')
    ap.add_argument('--yarn-second-fix', action='store_true',
                    help='apply arena/yarn_second_fix.py to the staged backbone (yarn store as second shop -> bakery_yarn)')
    return build(ap.parse_args())


if __name__ == '__main__':
    raise SystemExit(main())
