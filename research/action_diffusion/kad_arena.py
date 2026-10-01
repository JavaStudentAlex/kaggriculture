#!/usr/bin/env python3
"""arena.py plus the full replay of every game, in the Kaggle replay format data.prepare encodes.

Runs from an arena payload (arena.py and bundle_agent.py next to it; make_arena.py adds this file)
with arena.py's arguments. Each game's env.toJSON() goes gzipped to $KAD_EPISODE_DIR as
<tag>_seed<seed>_aseat<seat>.json.gz, with info.EpisodeId (that name) and info.TeamNames (the
bundle of each seat); the result line gets `episode`. A job with "episode": false is not saved.
Every result line also gets `farm`: each seat's farm at the start of the CENSUS_DAYS (farm_census).
The games themselves are arena.py's: its play() and main loop run unchanged, and the env they
create is only kept to be saved and counted.
"""
import gzip
import json
import os
from pathlib import Path

import kaggle_environments
from kaggle_environments.envs.kaggriculture.kaggriculture import ANIMALS, CROPS, LAND_PRICES

import arena

EPISODES = os.environ.get('KAD_EPISODE_DIR')
CENSUS_DAYS = (5, 10, 15, 20, 25)
TURNS_PER_DAY = 24
_make, _play = kaggle_environments.make, arena.play
_games = {}


def farm_census(steps, days=CENSUS_DAYS):
    """{seat: {day: counts}}, the farms in the observation at the first step of each day (the farm
    after `day` whole days). value is what the farm cost: its bought land, the animals placed on it
    and the seeds in the ground (coops and pastures cost nothing)."""
    census = {}
    for day in days:
        t = day * TURNS_PER_DAY
        if t >= len(steps):
            break
        farms = (steps[t][0].get('observation') or {}).get('farms') or []
        for seat, farm in enumerate(farms):
            animals = {name: 0 for name in ANIMALS}
            crops = {name: 0 for name in CROPS}
            for row in farm.get('tiles') or []:
                for tile in row:
                    if isinstance(tile, dict):
                        if tile.get('animal') in animals:
                            animals[tile['animal']] += 1
                        elif tile.get('kind') == 'PLANT' and tile.get('crop') in crops:
                            crops[tile['crop']] += 1
            quadrants = len(farm.get('unlocked_quadrants') or ['NW'])
            value = (sum(LAND_PRICES[:quadrants - 1]) + sum(ANIMALS[a]['cost'] * n for a, n in animals.items())
                     + sum(CROPS[c]['seed'] * n for c, n in crops.items()))
            census.setdefault(str(seat), {})[str(day)] = dict(
                value=value, quadrants=quadrants, animals=animals, crops=crops, hands=len(farm.get('hands') or []),
                money=round(float(farm.get('money') or 0)))
    return census


def _capture(*args, **kwargs):
    _games['env'] = env = _make(*args, **kwargs)
    return env


def play(job, bundle_class, opts):
    _games.clear()
    rec = _play(job, bundle_class, opts)
    env = _games.get('env')
    if env is not None and env.steps:
        try:
            rec['farm'] = farm_census(env.steps)
        except Exception as exc:   # the game still counts; it is scored by its margin alone
            rec['farm_error'] = repr(exc)[:2000]
    if EPISODES and job.get('episode', True) and env is not None and env.steps:
        try:
            doc = env.toJSON()
            names = [Path(job['b']).name] * 2
            names[job['a_seat']] = Path(job['a']).name
            name = f"{job['tag']}_seed{job['seed']}_aseat{job['a_seat']}"
            doc['info'] = dict(doc.get('info') or {}, EpisodeId=name, TeamNames=names)
            path = Path(EPISODES) / f'{name}.json.gz'
            path.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(path, 'wt', compresslevel=6) as fh:
                json.dump(doc, fh, separators=(',', ':'))
            rec['episode'] = path.name
        except Exception as exc:
            rec['errors'].append(dict(phase='episode', message=repr(exc)[:2000]))
    return rec


# At module level, so that arena's spawned workers (which re-import this script) get them too.
kaggle_environments.make = _capture
arena.play = play

if __name__ == '__main__':
    arena.main()
