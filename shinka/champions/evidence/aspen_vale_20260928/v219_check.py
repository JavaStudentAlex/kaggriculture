"""Why the 09-27 engine's tomato investment (layer V219, main.py:1265) did or did not fire in recorded games
(cliproxyapi). It fires on day 18 only, when the farm has exactly the quadrants NW, NE and SW, at least $12,000,
a tomato price of at least CROP_MIN_PRICE (70), at least three pizza shops or farmers' markets in town and no tomato
yet. Per game: those conditions at the start of day 18 from our seat, both farms' tomato tiles and quadrants on
day 20, and when the rival planted its first tomato.

    python v219_check.py <index.json> <replay dir>
"""
import json
import sys


def tomato_tiles(farm):
    return sum(1 for row in farm['tiles'] for t in row if isinstance(t, dict) and t.get('crop') == 'TOMATO')


index, replays = json.load(open(sys.argv[1])), sys.argv[2]
print('episode   margin  team            | day 18: quads money  tomato$ pizza+FM | tomatoes day 20 ours/rival, quads | rival first tomato')
for g in sorted(index, key=lambda g: g['diff']):
    steps = json.load(open(f"{replays}/episode-{g['id']}-replay.json"))['steps']
    ours = g['seat']
    o18 = steps[432][ours]['observation']
    farm = o18['farms'][ours]
    shops = o18['town']['unlocked_shops']
    tomato_shops = sum(s in ('PIZZA_SHOP', 'FARMERS_MARKET') for s in shops)
    quads = sorted(farm['unlocked_quadrants'])
    o20 = steps[480][0]['observation']
    first = next((t for t in range(1, len(steps)) if tomato_tiles(steps[t][0]['observation']['farms'][1 - ours])), None)
    blocked = [why for why, bad in (('quadrants', set(quads) != {'NW', 'NE', 'SW'}), ('money', farm['money'] < 12000),
                                    ('price', o18['market']['prices']['TOMATO'] < 70), ('shops', tomato_shops < 3)) if bad]
    print(f"{g['id']} {g['diff']:+7,.0f} {g['team'][:15]:15s} | {len(quads)} {farm['money']:7,.0f} {o18['market']['prices']['TOMATO']:5d} "
          f"{tomato_shops} ({','.join(s[:5] for s in shops)}) | {tomato_tiles(o20['farms'][ours])}/{tomato_tiles(o20['farms'][1 - ours])}, "
          f"{len(o20['farms'][ours]['unlocked_quadrants'])}/{len(o20['farms'][1 - ours]['unlocked_quadrants'])} | "
          f"{'-' if first is None else f'day {first // 24} h{first % 24}'} | blocked: {','.join(blocked) or 'nothing'}", flush=True)
