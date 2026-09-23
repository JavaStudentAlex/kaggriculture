"""Replay real observations to verify runtime health, not counterfactual match strength."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import agent_graph


def check(trace_path, output):
    with gzip.open(trace_path, 'rt') as stream:
        replay = json.load(stream)
    engine = agent_graph.get_engine()
    calls = 0
    events = Counter()
    failures = []
    for step in replay['steps']:
        obs = step[0].get('observation')
        if not obs or 'farms' not in obs:
            continue
        engine.agent(obs, replay.get('configuration', {}))
        calls += 1
        for event in engine.last_telemetry:
            events[event.get('reason', str(event))] += 1
            if 'failed' in event.get('reason', ''):
                failures.append(event)
    stats = engine.champion.ORACLE_STATS
    result = {'scope': 'Observation replay runtime-health check only; not an on-policy performance evaluation',
              'trace': str(trace_path), 'calls': calls, 'fallback_count': engine.fallback_count,
              'last_error': engine.last_error, 'oracle_stats': stats,
              'extension_failures': failures, 'events': dict(events)}
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    assert calls >= 700, 'Not a full episode trace'
    assert engine.fallback_count == 0 and not failures, 'Runtime fallback detected'
    assert stats['forecasts'] > 0 and stats['errors'] == 0, 'Predictor not working'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    check(args.trace, args.output)
