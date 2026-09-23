"""Compare matching fully executed games, keeping cash and win metrics separate."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean


def load(path):
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    keyed = {(r['opponent_id'], r['seat'], r['seed']): r for r in records}
    if len(keyed) != len(records):
        raise ValueError('Duplicate opponent/seat/seed')
    return keyed


def compare(baseline, candidate):
    b, c = load(baseline), load(candidate)
    if set(b) != set(c):
        raise ValueError('Game sets differ; cannot claim complete comparison')
    if not b or any(not r['valid'] for r in [*b.values(), *c.values()]):
        raise ValueError('Empty or invalid game set')
    groups = {}
    rows = []
    for key in sorted(b):
        old, new = b[key], c[key]
        seat = key[1]
        old_cash = old['rewards'][seat]
        new_cash = new['rewards'][seat]
        old_margin = old_cash - old['rewards'][1-seat]
        new_margin = new_cash - new['rewards'][1-seat]
        row = dict(opponent=key[0], seat=seat, seed=key[2],
                   baseline_outcome=old['outcome'], candidate_outcome=new['outcome'],
                   baseline_cash=old_cash, candidate_cash=new_cash,
                   baseline_margin=old_margin, candidate_margin=new_margin,
                   margin_change=new_margin-old_margin,
                   cash_change=new_cash-old_cash)
        rows.append(row)
        groups.setdefault(key[0], []).append(row)
    summaries = {}
    for name, records in {'overall': rows, **groups}.items():
        summaries[name] = {
            'games': len(records),
            'baseline_WLD': dict(Counter(r['baseline_outcome'] for r in records)),
            'candidate_WLD': dict(Counter(r['candidate_outcome'] for r in records)),
            'baseline_draw_half_score': mean({'W': 1, 'D': .5, 'L': 0}[r['baseline_outcome']] for r in records),
            'candidate_draw_half_score': mean({'W': 1, 'D': .5, 'L': 0}[r['candidate_outcome']] for r in records),
            'mean_cash_change': mean(r['cash_change'] for r in records),
            'mean_margin_change': mean(r['margin_change'] for r in records),
            'margin_improvements': sum(r['margin_change'] > 0 for r in records),
            'margin_regressions': sum(r['margin_change'] < 0 for r in records),
        }
    return {'summaries': summaries, 'games': rows,
            'scope': 'Matched-seed local regression check. Not proof of general superiority; repeated seeds are not independent trials.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = compare(args.baseline, args.candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['summaries'], indent=2))
