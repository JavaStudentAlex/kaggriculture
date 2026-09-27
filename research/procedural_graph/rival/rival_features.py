"""The rival's public state per step, from recorded ladder games (the input of the rival classifier).

    python rival_features.py <index.json> <replay dir> <out dir>

For each game of the index whose replay is in the replay dir (episode-<id>-replay.json), writes
<out dir>/<id>.json: our seat, the town's shops, and per step (as our agent saw it) the rival's
vector (RIVAL_FIELDS), our own vector, the market stock and prices. Only public observation
fields are read, so a live agent can build the same vectors (hazel_runtime/rival_model.py).
"""
import json
import sys
from pathlib import Path

CROPS = ('WHEAT', 'CARROT', 'TOMATO', 'STRAWBERRY', 'MELON')
ANIMALS = ('COW', 'SHEEP', 'GOOSE')
PRODUCTS = ('WHEAT', 'CARROT', 'TOMATO', 'STRAWBERRY', 'MELON', 'EGG', 'MILK', 'WOOL', 'FERTILIZER')
RIVAL_FIELDS = ('money', 'hands', 'land', 'hires_today') + tuple(f'crop_{c}' for c in CROPS) + \
    tuple(f'animal_{a}' for a in ANIMALS) + ('empty_pasture', 'empty_coop', 'farmer_x', 'farmer_y')


def farm_vector(farm):
    """One farm's public state as numbers (RIVAL_FIELDS order)."""
    crops = dict.fromkeys(CROPS, 0)
    animals = dict.fromkeys(ANIMALS, 0)
    empty_pasture = empty_coop = 0
    for row in farm.get('tiles') or []:
        for tile in row or []:
            if not isinstance(tile, dict):
                continue
            kind = tile.get('kind')
            if kind == 'PLANT' and tile.get('crop') in crops:
                crops[tile['crop']] += 1
            elif kind in ('PASTURE', 'COOP'):
                animal = tile.get('animal')
                if animal in animals:
                    animals[animal] += 1
                elif kind == 'PASTURE':
                    empty_pasture += 1
                else:
                    empty_coop += 1
    farmer = farm.get('farmer') or [-1, -1]
    return [float(farm.get('money') or 0), len(farm.get('hands') or []), len(farm.get('unlocked_quadrants') or []),
            int(farm.get('hires_today') or 0)] + [crops[c] for c in CROPS] + [animals[a] for a in ANIMALS] + \
        [empty_pasture, empty_coop, int(farmer[0]), int(farmer[1])]


def game_vectors(replay, our_seat):
    out = {'our_seat': our_seat, 'fields': RIVAL_FIELDS, 'products': PRODUCTS, 'shops': None,
           'rival': [], 'ours': [], 'stock': [], 'prices': []}
    for t, step in enumerate(replay['steps']):
        obs = step[0].get('observation') or {}
        farms = obs.get('farms')
        if not farms:
            continue
        if out['shops'] is None and t >= 144:
            out['shops'] = ((obs.get('town') or {}).get('unlocked_shops') or [])[:2]
        out['rival'].append(farm_vector(farms[1 - our_seat]))
        out['ours'].append(farm_vector(farms[our_seat]))
        market = obs.get('market') or {}
        out['stock'].append([int((market.get('inventory') or {}).get(p, 0)) for p in PRODUCTS])
        out['prices'].append([float((market.get('prices') or {}).get(p, 0)) for p in PRODUCTS])
    return out


def main():
    index, replays, out = json.load(open(sys.argv[1])), Path(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    done = 0
    for g in index:
        path = replays / f"episode-{g['id']}-replay.json"
        dst = out / f"{g['id']}.json"
        if not path.exists() or dst.exists():
            continue
        vec = game_vectors(json.load(open(path)), int(g['seat']))
        vec.update(id=g['id'], res=g['res'], diff=g['diff'], team=g['team'], orate=g.get('orate'))
        dst.write_text(json.dumps(vec, separators=(',', ':')))
        done += 1
    print(f'{done} games written to {out}')


if __name__ == '__main__':
    main()
