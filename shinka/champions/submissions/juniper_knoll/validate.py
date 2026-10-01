#!/usr/bin/env python3
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
print(f"bundle {bundle}\nHOME={os.environ.get('HOME')} cwd={os.getcwd()}")
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
print("\nresolution:")
for label, path in (("champion", getattr(champ, "__file__", None)), ("MOHUI_DIR", getattr(champ, "MOHUI_DIR", None)),
                    ("kagg_oracle", getattr(oracle_mod, "__file__", None)), ("checkpoint", model_dir),
                    ("kagg_ttm_numpy", getattr(ttm_np, "__file__", None)), ("features", getattr(feat, "__file__", None)),
                    ("mechanics", getattr(mech, "__file__", None))):
    print(f"  {label:15}: {path}  inside={inside(path or '')}")
print(f"  backend        : {getattr(model, 'backend', None)}  torch imported: {'torch' in sys.modules}")
print(f"  oracle         : device={getattr(champ, 'ORACLE_DEVICE', None)} stats={stats}")
late = [x for s in (0, 1) for x in durations[s][512:]]
allsteps = durations[0] + durations[1]
print(f"\nsteps {len(allsteps)}: max {1000*max(allsteps):.0f} ms; from turn 512: p99 {1000*sorted(late)[int(0.99*len(late))]:.0f} ms, max {1000*max(late):.0f} ms")
oracle_ok = (stats is not None and stats.get("errors", 1) == 0 and stats.get("forecasts", 0) > 0
             and getattr(model, "backend", None) == "numpy" and "torch" not in sys.modules)
resolved_ok = all(inside(p or "") for p in (getattr(champ, "__file__", ""), getattr(champ, "MOHUI_DIR", ""), getattr(oracle_mod, "__file__", ""),
                                             model_dir, getattr(ttm_np, "__file__", ""), getattr(feat, "__file__", ""), getattr(mech, "__file__", "")))
graph = sys.modules.get("agent_graph")
cal = os.path.join(bundle, "hazel_runtime", "checkpoint", "calibration.json")
engine_built = getattr(graph, "_ENGINE", None) is not None
print(f"  agent_graph    : {getattr(graph, '__file__', None)}  inside={inside(getattr(graph, '__file__', '') or '')}  engine built: {engine_built}")
print(f"  calibration    : {cal if os.path.isfile(cal) else None}  (expected: {False})")
graph_ok = (graph is not None and inside(getattr(graph, "__file__", "") or "") and engine_built
            and inside(model_dir or "") and os.path.isfile(cal) == False)
# a graph whose channels and guard are off never asks the predictor (ladder engines, the rival emulator)
oracle_needed = bool(getattr(getattr(graph, "_ENGINE", None), "oracle_needed", True))
print(f"  oracle needed  : {oracle_needed}")
verdict = "PASS" if (not failures and (oracle_ok or not oracle_needed) and resolved_ok and graph_ok) else "FAIL"
print(f"\nVALIDATION {verdict}: games failed={len(failures)} oracle_live={oracle_ok} all_resolved_inside_bundle={resolved_ok} graph_ok={graph_ok} oracle_needed={oracle_needed}")
sys.exit(0 if verdict == "PASS" else 1)
