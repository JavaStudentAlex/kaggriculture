"""The first move each single-change variant plays differently from the recorded agent (cliproxyapi, after
counterfactual.py): per game, the step and the two actions where they differ, and the result the variant reaches.

    python first_moves.py [bundle ...]
"""
import json
import sys

for b in sys.argv[1:] or ('j_noguard', 'j_look3', 'j_ca15', 'j_sr12'):
    print('==', b)
    for line in open(f'cf/{b}.jsonl'):
        d = json.loads(line)
        fd = d['first_difference']
        if not fd:
            continue
        s = fd['step']
        print(f"  {d['episode']} {d['team'][:12]:12s} {d['res']} {d['diff']:+6d} step {s} (day {s // 24} h{s % 24}) "
              f"-> {d['cf_res']} {d['cf_diff']:+6.0f}")
        for k in ('farmer', 'hands', 'market'):
            if fd['bundle'].get(k) != fd['recorded'].get(k):
                print(f"      {k}: variant {json.dumps(fd['bundle'].get(k))[:160]}")
                print(f"      {'':{len(k)}s}  played  {json.dumps(fd['recorded'].get(k))[:160]}")
