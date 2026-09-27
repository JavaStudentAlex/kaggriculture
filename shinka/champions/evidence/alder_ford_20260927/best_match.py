"""Per game: the public bundles that follow the rival's recorded moves longest (match_ladder_games.py output)."""
import json, sys, collections
from pathlib import Path
rows = collections.defaultdict(list)
for f in sorted(Path(sys.argv[1]).glob('*.jsonl')):
    for line in open(f):
        r = json.loads(line)
        rows[(r['episode'], r.get('opponent'))].append((r.get('equal_steps', 0), r['bundle'], r.get('of')))
for (ep, opp), found in sorted(rows.items()):
    found.sort(reverse=True)
    best = found[0]
    tag = 'EXACT' if best[0] == best[2] else ''
    print(ep, f"{(opp or '')[:22]:22s}", tag.ljust(5), ', '.join(f"{b} {n}" for n, b, _ in found[:3]))
