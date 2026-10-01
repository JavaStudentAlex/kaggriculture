"""How much of the 60 s overage bank each seat used in our recorded games, and the final statuses (cliproxyapi).

    python overage.py index_all.json replays
"""
import gzip
import json
import statistics
import sys

idx = json.load(open(sys.argv[1]))
out = {}
for g in idx:
    with gzip.open(f"{sys.argv[2]}/episode-{g['id']}-replay.json.gz", 'rt') as f:
        r = json.load(f)
    steps = r['steps']
    ours = g['seat']
    rem = [s[ours]['observation'].get('remainingOverageTime') for s in steps]
    rem_o = [s[1 - ours]['observation'].get('remainingOverageTime') for s in steps]
    first = rem[0]
    # the bank spent at step 0 (imports, engine build) and afterwards
    out[g['id']] = {'sub': g['sub'], 'res': g['res'], 'statuses': r.get('statuses'), 'start': first,
                    'after_step1': rem[2] if len(rem) > 2 else None, 'end': rem[-1], 'rival_end': rem_o[-1],
                    'spent_after_step1': (rem[2] - rem[-1]) if len(rem) > 2 else None}
json.dump(out, open('overage.json', 'w'), indent=1)
for sub in ('juniper', 'aspen'):
    v = [x for x in out.values() if x['sub'] == sub]
    if not v:
        continue
    st = {tuple(x['statuses']) for x in v}
    print(f"{sub}: {len(v)} games, statuses {st}; bank at step 2 median {statistics.median(x['after_step1'] for x in v):.1f} s, "
          f"at the end median {statistics.median(x['end'] for x in v):.1f} s, min {min(x['end'] for x in v):.1f} s; "
          f"spent after step 1: median {statistics.median(x['spent_after_step1'] for x in v):.2f} s, "
          f"max {max(x['spent_after_step1'] for x in v):.2f} s")
