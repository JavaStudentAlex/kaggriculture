"""Alder Ford's ladder record from index.json: overall, by block of games, by rival rating, and the new
lost and tied games (those after the previous analysis's 100 games).

usage: python summary.py [index.json] [previous index.json]
"""
import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
games = json.load(open(sys.argv[1] if len(sys.argv) > 1 else HERE / 'index.json'))
old = {g['id'] for g in json.load(open(sys.argv[2] if len(sys.argv) > 2 else HERE.parent / 'alder_ford_20260927' / 'index.json'))}


def rec(gs):
    c = collections.Counter(g['res'] for g in gs)
    return f"{c['W']}-{c['L']}-{c['T']}"


def band(r):
    if r is None:
        return 'unlisted'
    for lo in (2300, 2200, 2100, 2000, 1900):
        if r >= lo:
            return f'{lo}+' if lo == 2300 else f'{lo}-{lo + 99}'
    return '< 1900'


new = [g for g in games if g['id'] not in old]
print(f"{len(games)} games {rec(games)}: the previous analysis's {len(games) - len(new)} {rec([g for g in games if g['id'] in old])}, "
      f"new {len(new)} {rec(new)} ({new[0]['t'][:16] if new else ''} to {games[-1]['t'][:16]})")
for i in range(0, len(games), 50):
    block = games[i:i + 50]
    print(f"  games {i + 1:3d}-{i + len(block):3d} {rec(block):9s} {block[0]['t'][5:16]} to {block[-1]['t'][5:16]}")
print('by rival rating (rating now):          all          new')
bands = ['2300+', '2200-2299', '2100-2199', '2000-2099', '1900-1999', '< 1900', 'unlisted']
for b in bands:
    a = [g for g in games if band(g['orate']) == b]
    n = [g for g in a if g['id'] not in old]
    print(f"  {b:10s} {len(a):3d} {rec(a):10s} {len(n):3d} {rec(n)}")
strong = [g for g in new if g['orate'] is None or g['orate'] >= 1900]
print(f"new games against rivals rated 1,900+ or unlisted: {len(strong)} {rec(strong)}")
print('new lost and tied games:')
for g in new:
    if g['res'] != 'W':
        print(f"  {g['id']} {g['t'][5:16]} {g['res']} {g['diff']:+8.0f}  {(g['team'] or '')[:28]:28s} {g['orate'] or '':>7}")
