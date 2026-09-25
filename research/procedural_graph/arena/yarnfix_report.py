#!/usr/bin/env python3
"""Fixed (yarn-second route) vs old (Linden Brook) on the ladder seeds: the result table.

    python research/procedural_graph/arena/yarnfix_report.py <run dir with results.jsonl and traces/>

The two agents play identically until step 144, so both games of a seed (fixed in seat 0, fixed
in seat 1) see the same town up to day 6. The fix acts only if that town opened a YARN_STORE
second after a shop other than YARN_STORE / PET_CAFE (the trace's shops at step 145). In every
other seed the agents never differ: both games of the seed are the same game, and the cash is
decided by the seat alone (the engine settles HIRE / BUY_LAND in seat order, so the seats can end
a few dollars apart); such a seed must give identical cash per seat in both games and nets to zero.
Checks: every game DONE without errors, repeats of a (seed, seat) identical, the mirror property.
"""
import collections
import gzip
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
rows = [json.loads(line) for line in (run / 'results.jsonl').read_text().splitlines() if line.strip()]
games = {(r['tag'], r['seed'], r['a_seat']): r for r in rows}  # the last line of a key wins (reruns)
bad = [k for k, r in games.items() if r.get('errors') or r.get('statuses') != ['DONE', 'DONE']]
print(f"{len(games)} games; not DONE or with errors: {bad or 'none'}")


def shops_at_145(r):
    with gzip.open(run / 'traces' / r['trace'], 'rt') as fh:
        return json.load(fh)[145]['shops']


pairs = collections.defaultdict(dict)  # (seed, fixed seat) -> tag -> rewards by seat
town = {}
for (tag, seed, seat), r in games.items():
    pairs[seed, seat][tag] = tuple(r['rewards'])
    town.setdefault(seed, shops_at_145(r))
nondet = {k: v for k, v in pairs.items() if len(set(v.values())) > 1}
print(f"(seed, seat) pairs: {len(pairs)}; played more than once: {sum(len(v) > 1 for v in pairs.values())}; "
      f"repeats that differ: {len(nondet)} {list(nondet.items())[:3]}")

rewards = {k: next(iter(v.values())) for k, v in pairs.items()}  # repeats are identical (checked above)
seeds = sorted({s for s, _ in rewards})
trigger = {s: len(town[s]) >= 2 and town[s][1] == 'YARN_STORE' and town[s][0] not in ('YARN_STORE', 'PET_CAFE')
           for s in seeds}


def margin(seed, seat):
    rw = rewards[seed, seat]
    return rw[seat] - rw[1 - seat]


mirror_ok, mirror_bad, exact_ties = 0, [], 0
for s in seeds:
    if trigger[s] or (s, 0) not in rewards or (s, 1) not in rewards:
        continue
    if rewards[s, 0] == rewards[s, 1]:  # same cash per seat whichever agent sat there
        mirror_ok += 1
        exact_ties += rewards[s, 0][0] == rewards[s, 0][1]
    else:
        mirror_bad.append((s, rewards[s, 0], rewards[s, 1]))
n_plain = sum(1 for s in seeds if not trigger[s])
print(f"\nseeds without the trigger: {n_plain}; identical cash per seat in both games: {mirror_ok} "
      f"(exact ties: {exact_ties}; the rest differ only by seat); violations: {mirror_bad or 'none'}")

t_seeds = [s for s in seeds if trigger[s]]
print(f"seeds where the yarn store opened second (fix active): {len(t_seeds)} of {len(seeds)}")
res = collections.Counter()
per_seat = {0: collections.Counter(), 1: collections.Counter()}
total = 0.0
for s in t_seeds:
    for seat in (0, 1):
        if (s, seat) not in rewards:
            continue
        m = margin(s, seat)
        r = 'W' if m > 0 else 'L' if m < 0 else 'T'
        res[r] += 1
        per_seat[seat][r] += 1
        total += m
        rw = rewards[s, seat]
        print(f"  seed {s:>10}  fixed in seat {seat}: fixed {rw[seat]:>9,.0f}  old {rw[1 - seat]:>9,.0f}  {m:+9,.0f}  {r}"
              f"   shops by day 6: {town[s][:2]}")
n = sum(res.values())
if n:
    print(f"fix active: fixed {res['W']}W-{res['L']}L-{res['T']}T in {n} games (seat 0: {per_seat[0]['W']}W-{per_seat[0]['L']}L, "
          f"seat 1: {per_seat[1]['W']}W-{per_seat[1]['L']}L), mean margin {total / n:+,.0f}")
all_res = collections.Counter('W' if margin(*k) > 0 else 'L' if margin(*k) < 0 else 'T' for k in rewards)
print(f"all (seed, seat) pairs: fixed {all_res['W']}W-{all_res['L']}L-{all_res['T']}T; mean margin "
      f"{sum(margin(*k) for k in rewards) / len(rewards):+,.1f} (non-trigger seeds net to zero seed by seed)")
