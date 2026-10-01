#!/usr/bin/env python3
"""The production plan of every route tape of our ladder engines, and the tape each world gets (run on cliproxyapi).

    cd research/procedural_graph && python benchmark/route_plans.py --out route_plans.json [--engine NAME ...]

The public engine plays pre-recorded 719-step route tapes (yhay81's shop router): route 0 until step 144, then
the route its tables give the first two shops (`_R108_SHOP_ROUTES` for worlds without a yarn store,
`_R110_OLD_SHOPS` with one, `_V92_TABLE` overriding both), and route 2 from step 648 (day 27). For each engine
this writes every route's plan as that effective tape (routes[0][:144] + routes[r][144:648] + routes[2][648:]),
counted like plan_mining.py counts a top player's actions, and the route of each of the 64 worlds. The engine's
layers change some of these orders in play (HERD2 turns a goose purchase into cows in about 6 of 48 games), so a
tape's plan is the plan before them.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PG = HERE.parent
sys.path.insert(0, str(PG))
sys.path.insert(0, str(HERE))

from hazel_runtime import engines  # noqa: E402
from plan_mining import plan_of  # noqa: E402

SHOPS = ('BAKERY', 'BRUNCH_SPOT', 'FARMERS_MARKET', 'ICE_CREAM_SHOP', 'PET_CAFE', 'PIZZA_SHOP', 'SMOOTHIE_SHOP',
         'YARN_STORE')


def world_routes(ns):
    """{'A|B': route} for every ordered pair of first two shops, as the engine's _router picks it."""
    new, old, override = ns.get('_R108_SHOP_ROUTES', {}), ns.get('_R110_OLD_SHOPS', {}), ns.get('_V92_TABLE', {})
    out = {}
    for pair in itertools.product(SHOPS, repeat=2):
        route = new.get(pair, 100) if 'YARN_STORE' not in pair else old.get(pair, 0)
        out['|'.join(pair)] = override.get(pair, route)
    return out


def effective(routes, route):
    return list(routes[0][:144]) + list(routes[route][144:648]) + list(routes[2][648:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--engine', action='append', default=None)
    args = ap.parse_args()
    report = {}
    for name in args.engine or ['tetsutani_demand', 'tetsutani_demand_0927']:
        ns = engines.Engine(str(PG / 'hazel_runtime' / 'engines' / name)).namespace
        routes = ns['_ROUTES']
        plans = {}
        for route in sorted(routes):
            buys, sells = plan_of(effective(routes, route))
            plans[str(route)] = {'buys': buys, 'sells': sells, 'opening': (routes[route][0] or {}).get('market')}
        report[name] = {'routes': plans, 'world_route': world_routes(ns),
                        'v92_table': engines.to_json(ns.get('_V92_TABLE', {}))}
        print(f"{name}: {len(plans)} routes, worlds by route: "
              f"{sorted(set(report[name]['world_route'].values()))}", flush=True)
    Path(args.out).write_text(json.dumps(report, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
