"""Kaggriculture submission: Shinka run-2 gen 33 (`dynamic_shop_batch_sizing`,
crowned 2026-09-12) playing with the opponent order-flow oracle on the CPU.

kaggle_environments loads a submitted main.py with exec() in a bare namespace:
there is no __file__, the directory of main.py is on sys.path only while this
module-level code runs, and the LAST callable left in the namespace becomes the
agent. This file therefore does nothing but locate the bundle, point the
unmodified champion at the bundled pieces through the env overrides it already
honours, import it as a real module (which restores __file__ for it) and expose
agent() as the last callable.

Bundle layout (all relative to this file):
  champion_gen33.py   the champion, byte-identical to shinka_results gen_33/main.py
  kagg_oracle.py      the oracle module, byte-identical to shinka/evolution/kagg_oracle.py
  kagg_ttm_numpy.py   the checkpoint's network in numpy (KAGG_ORACLE_BACKEND=numpy below):
                      no torch / transformers import -- the first request to a Kaggle
                      agent has ~60 s and importing torch there blew it on 2026-09-12
  checkpoint/         the TTM checkpoint it serves (models/ttm_v3_h96_ft_2026-09-11)
  mohui_v66/          the Mohui v66 backbone closure (Apache-2.0, LICENSE/NOTICE inside)
  opponent_model/     features.py + mechanics.py from research/opponent_model (the oracle's input builder)
Only numpy is needed from the image.
"""
import os
import sys


def _bundle_dir() -> str:
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        pass
    # exec'd by kaggle_environments: compile(raw, path) kept the path of this file
    # in the code object, so every frame of this file carries it.
    f = sys._getframe(0).f_code.co_filename
    if f and f != "<string>":
        d = os.path.dirname(os.path.abspath(f))
        if os.path.isfile(os.path.join(d, "champion_gen33.py")):
            return d
    # last resort: the directory the loader appended for sibling imports, then cwd
    for p in reversed(sys.path):
        if os.path.isfile(os.path.join(p or ".", "champion_gen33.py")):
            return os.path.abspath(p or ".")
    return os.getcwd()


HERE = _bundle_dir()

# The champion and the oracle look at these before any path guessing.
os.environ.setdefault("KAGG_MOHUI_DIR", os.path.join(HERE, "mohui_v66"))
os.environ.setdefault("KAGG_ORACLE_SRC", HERE)
os.environ.setdefault("KAGG_TTM_DIR", os.path.join(HERE, "checkpoint"))
os.environ.setdefault("KAGG_OPP_MODEL_SRC", os.path.join(HERE, "opponent_model"))  # features.py, mechanics.py
# numpy inference (validated: ~120-150 ms per forecast on one core from turn 512
# against the 1000 ms step limit; identical games to the torch checkpoint). One
# BLAS thread: the GEMMs are small and a sandbox core count is unknown.
os.environ["KAGG_ORACLE_BACKEND"] = "numpy"
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Front of sys.path, unconditionally: the loader has already APPENDED this
# directory (and pops that entry again after this file runs), so a "not already
# present" guard would leave the bundle behind whatever sys.path[0] happens to be.
sys.path.insert(0, HERE)

import champion_gen33 as _champion  # noqa: E402  (loads the backbone and the oracle)


def agent(obs, configuration=None):
    return _champion.agent(obs, configuration)


# kaggle_environments takes the LAST callable bound in this file: keep this the
# final definition.
def kaggle_submission_agent(obs, configuration=None):
    return agent(obs, configuration)
