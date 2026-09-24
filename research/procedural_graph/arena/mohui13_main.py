"""Ladder-proxy opponent: the bundled Mohui v66 backbone with a 13/9 opening wheat scalp.

The opening (buy 13 wheat at step 0, sell 9 at step 1, drop the backbone's own step-1
wheat buy) is what the top Mohui-based opponents played in Hazel Weir's largest
ladder losses (replays 109194836, 109205361, 109577172, 109776371). Everything else
is the unmodified backbone, called every turn so its closed-loop state stays intact.
"""
from candidate_v66_meta_closed_loop import agent as _raw_agent
from kaggriculture_agent.env.adapter import sanitize_action

_OBSERVATION_ERRORS = (AttributeError, IndexError, KeyError, TypeError, ValueError)


def agent(obs, configuration=None):
    try:
        action = _raw_agent(obs, configuration)
        step = int(obs.get("step", 0) or 0)
        if step == 0:
            action["market"] = [["BUY_PRODUCT", "WHEAT", 13]]
        elif step == 1:
            action["market"] = [["SELL", "WHEAT", 9]] + [
                o for o in action.get("market") or []
                if not (isinstance(o, (list, tuple)) and len(o) >= 2 and o[0] == "BUY_PRODUCT" and o[1] == "WHEAT")]
        return sanitize_action(action, obs, configuration)
    except _OBSERVATION_ERRORS:
        return {"farmer": ["PASS"], "hands": [], "market": []}
