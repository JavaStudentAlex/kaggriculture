"""Compare complete graph/submission actions on the same 720-observation stream."""
import gzip
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE / 'merged_payload'
sys.path.insert(0, str(ROOT))
from pool_upgrade_bundle_agent import BundleAgent


def main():
    trace = HERE.parents[1] / 'frozen_eval/runs/iter27_trace_mining/local_smoke/traces/submission_hazel_weir_seat0_seed1205926567.json.gz'
    # Compare Hazel's own observed history, not the paralyzed agent's state.
    with gzip.open(trace, 'rt') as handle:
        steps = json.load(handle)['steps']
    graph = BundleAgent(ROOT / 'agents/candidate_merged_graph')
    source = BundleAgent(ROOT / 'agents/submission_hazel_weir')
    started = time.monotonic()
    count = 0
    mismatches = []
    times = {'graph': [], 'source': []}
    try:
        graph.start(); source.start()
        for i, step in enumerate(steps):
            obs = step[1]['observation']
            begin = time.monotonic(); actual = graph(obs, {})
            times['graph'].append(time.monotonic() - begin)
            begin = time.monotonic(); expected = source(obs, {})
            times['source'].append(time.monotonic() - begin)
            count += 1
            if actual != expected:
                mismatches.append({'step': i, 'graph': actual, 'source': expected})
            if i % 120 == 0:
                print(json.dumps({'compared': count, 'mismatches': len(mismatches)}), flush=True)
        report = {'observations': count, 'mismatches': mismatches,
                  'pass': not mismatches and count == 720,
                  'source_trace': str(trace), 'observed_seat': 1,
                  'elapsed_seconds': time.monotonic() - started,
                  'timing_including_rpc': {k: {'mean_seconds': sum(v)/len(v), 'max_seconds': max(v)} for k,v in times.items()},
                  'caveat': 'Behavior equivalence on this replay plus synthetic tests; not a proof over every possible game state.'}
        (HERE / 'replay_equivalence.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))
        if not report['pass']: raise SystemExit(1)
    finally:
        graph.close(); source.close()


if __name__ == '__main__':
    main()
