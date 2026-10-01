"""main.py of a KAD-MD-24 / KAD-HP-1 arena bundle (make_arena.py, kad_rl.py): `agent` plays the model
(play.load_player) with the checkpoint and settings in this bundle's settings.json (the checkpoint
path is relative to the bundle). With "device": "cuda..." the bundle's process gets the GPU back
(the arena's bundle_agent hides it from every agent) before torch is imported."""
import json
import os
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SETTINGS = json.loads((_HERE / 'settings.json').read_text())
if str(_SETTINGS.get('device', 'cpu')).startswith('cuda'):
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('KAD_GPU', '0')

from play import load_player  # noqa: E402  (after the GPU is made visible)

_PLAYER = load_player(_HERE / _SETTINGS.pop('checkpoint', 'best.pt'), **_SETTINGS)


def agent(observation, configuration=None):
    return _PLAYER(observation)
