"""V57 (agent.py, unchanged) with the ladder's 13-wheat opening: buy 13 wheat at step 0 and
sell 9 at step 1 in place of V57's own step-0/1 wheat orders; every other order and every
later turn is V57's. The ladder variants of the V57 family that beat Rowan Glen most often
(16 of its first 28 losses, 2026-09-26) open this way; no public notebook does."""
import inspect
from pathlib import Path

from kaggle_environments.agent import get_last_callable

_HERE = Path(inspect.currentframe().f_code.co_filename).resolve().parent
_PARENT = get_last_callable((_HERE / "agent.py").read_text(encoding="utf-8"), path=str(_HERE / "agent.py"))
_OPENING = {0: [["BUY_PRODUCT", "WHEAT", 13]], 1: [["SELL", "WHEAT", 9]]}


def agent(observation, configuration=None):
    action = _PARENT(observation, configuration)
    step = int(observation["step"])
    if step in _OPENING and isinstance(action, dict):
        rest = [o for o in action.get("market") or []
                if not (isinstance(o, (list, tuple)) and len(o) >= 2 and o[1] == "WHEAT"
                        and o[0] in ("BUY_PRODUCT", "SELL"))]
        action = dict(action, market=[list(o) for o in _OPENING[step]] + rest)
    return action
