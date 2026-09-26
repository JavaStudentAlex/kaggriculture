"""Host for a public ladder agent recovered by build_ladder_pool.py (exact files in agent/).

The agent file is loaded the way kaggle_environments loads a submission: its code is exec'd
in a fresh namespace (no __file__) with its directory on sys.path, and the last callable it
defines is the agent, called with (observation, configuration) cut to its argument count, on
structified inputs. `agent` below is what the arena's bundle host calls.
"""
import json
import os
import sys
from pathlib import Path

from kaggle_environments.agent import get_last_callable
from kaggle_environments.utils import structify

_HERE = Path(__file__).resolve().parent
_ENTRY = _HERE / "agent" / json.loads((_HERE / "SOURCE.json").read_text())["entry"]
sys.path.insert(0, str(_ENTRY.parent))
_CWD = os.getcwd()
os.chdir(_ENTRY.parent)
try:
    _POLICY = get_last_callable(_ENTRY.read_text(encoding="utf-8"), path=str(_ENTRY))
finally:
    os.chdir(_CWD)
_NARGS = _POLICY.__code__.co_argcount if hasattr(_POLICY, "__code__") else 2


def agent(observation, configuration=None):
    return _POLICY(*[structify(observation), structify(configuration)][:_NARGS])
