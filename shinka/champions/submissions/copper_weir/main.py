"""Kaggriculture submission "Copper Weir": an evolved Kaggriculture agent playing with
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
