"""Replay opponent built by make_replay_opponents.py: plays one seat of a recorded ladder game.

At observation step t it returns the action the recorded player took from that observation. It does not
react to the other seat; SOURCE.json names the game.
"""
import gzip
import json
from pathlib import Path

_ACTIONS = json.loads(gzip.decompress((Path(__file__).resolve().parent / 'actions.json.gz').read_bytes()))


def agent(observation, configuration=None):
    step = int(observation['step'])
    return (_ACTIONS[step] if 0 <= step < len(_ACTIONS) else None) or {}
