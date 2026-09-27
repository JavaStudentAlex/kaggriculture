"""Kaggriculture submission "Birch Hollow": a procedural-graph Kaggriculture agent playing
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
        if os.path.isfile(os.path.join(p or ".", "agent_graph.py")) and \
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
