"""Where each of Aspen Vale's lost games was lost (cliproxyapi, on the kept loss replays): both seats' sales by product,
their purchases, the money gap by game day and the farms at days 10 and 20.

    python story_jk.py <index.json> <replay dir> <out.json>

The action stored at step t was chosen from the observation at t-1 (AGENTS.md 4.3). A SELL order sells at most what
the seat's shed held at t-1, at that step's market price, so a product's income here is an estimate (the price falls
as units are sold); both seats are estimated the same way. Purchases are counted by order (what was asked for).
"""
import collections
import json
import sys

PRODUCTS = ('WHEAT', 'CARROT', 'TOMATO', 'STRAWBERRY', 'MELON', 'EGG', 'MILK', 'WOOL', 'FERTILIZER')



def load_replay(replays, eid):
    """episode-<id>-replay.json, or its gzipped copy (.json.gz)."""
    import gzip
    import os
    p = f"{replays}/episode-{eid}-replay.json"
    if os.path.exists(p):
        return json.load(open(p))
    with gzip.open(p + ".gz", "rt") as fh:
        return json.load(fh)

def farm_census(farm):
    """Crops and animals on one farm's tiles (rival_features.farm_vector's reading), plus hands and quadrants."""
    c = collections.Counter()
    for row in farm.get('tiles') or []:
        for tile in row or []:
            if not isinstance(tile, dict):
                continue
            if tile.get('kind') == 'PLANT' and tile.get('crop'):
                c[tile['crop']] += 1
            elif tile.get('kind') in ('PASTURE', 'COOP'):
                c[tile.get('animal') or f"empty_{tile['kind'].lower()}"] += 1
    c['hands'] = len(farm.get('hands') or [])
    c['quadrants'] = len(farm.get('unlocked_quadrants') or [])
    return dict(c)


def seat_story(steps, seat):
    sold = collections.Counter()
    income = collections.Counter()
    bought = collections.Counter()
    for t in range(1, len(steps)):
        obs = steps[t - 1][seat]['observation']
        shed = dict((obs.get('private') or {}).get('shed') or {})
        prices = obs['market']['prices']
        for o in (steps[t][seat].get('action') or {}).get('market') or []:
            if not o:
                continue
            if o[0] == 'SELL' and len(o) > 2 and o[1] in prices:
                n = max(0, min(int(o[2]), int(shed.get(o[1], 0))))
                shed[o[1]] = shed.get(o[1], 0) - n
                sold[o[1]] += n
                income[o[1]] += n * prices[o[1]]
            elif o[0] == 'HIRE':
                bought['HIRE'] += 1
            elif len(o) > 2:
                bought[f'{o[0]}:{o[1]}'] += int(o[2])
    return sold, income, bought


def main():
    index, replays, out = json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3]
    stories = {}
    for g in sorted(index, key=lambda g: g['diff']):
        r = load_replay(replays, g['id'])
        steps, ours = r['steps'], g['seat']
        money = lambda t, s: steps[t][0]['observation']['farms'][s]['money']
        per = {}
        for who, s in (('ours', ours), ('rival', 1 - ours)):
            sold, income, bought = seat_story(steps, s)
            per[who] = {'sold': dict(sold), 'income': dict(income), 'bought': dict(bought),
                        'farm_d10': farm_census(steps[240][0]['observation']['farms'][s]),
                        'farm_d20': farm_census(steps[480][0]['observation']['farms'][s])}
        gap = {d: money(min(24 * d, len(steps) - 1), ours) - money(min(24 * d, len(steps) - 1), 1 - ours)
               for d in (5, 10, 15, 20, 25, 30)}
        last_ahead = max((t for t in range(len(steps)) if money(t, ours) > money(t, 1 - ours)), default=None)
        stories[g['id']] = {'team': g['team'], 'orate': g['orate'], 'diff': g['diff'], 'seat': ours, 'gap_by_day': gap,
                            'last_step_ahead': last_ahead, **per}
        inc = {p: per['ours']['income'].get(p, 0) - per['rival']['income'].get(p, 0) for p in PRODUCTS}
        worst = sorted(inc.items(), key=lambda kv: kv[1])[:3]
        print(f"{g['id']} {g['diff']:+8,.0f} {g['team'][:18]:18s} gap d10 {gap[10]:+7,.0f} d20 {gap[20]:+7,.0f} "
              f"d25 {gap[25]:+7,.0f} | last ahead step {last_ahead} | income gaps "
              + ', '.join(f'{p} {v:+,.0f}' for p, v in worst), flush=True)
    json.dump(stories, open(out, 'w'), indent=1, ensure_ascii=False)


if __name__ == '__main__':
    main()
