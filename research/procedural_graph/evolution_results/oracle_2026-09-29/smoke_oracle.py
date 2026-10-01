"""One full game per fresh process; verify the required predictor with an exact-engine rival.

Run on cliproxyapi's existing venv, in tmux. The rival runs in its own RPC process.
This is a runtime/timing check, not a competitive promotion test.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--opponent', type=Path, required=True)
    parser.add_argument('--rpc', type=Path, required=True)
    parser.add_argument('--seat', type=int, choices=(0, 1), required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    from kaggle_environments import make

    os.environ.pop('KAGG_GRAPH_PATH', None)
    started = time.monotonic()
    candidate = load('oracle_smoke_candidate', args.bundle.resolve() / 'main.py')
    engine = candidate.get_engine()
    startup = time.monotonic() - started
    assert engine.oracle_needed and engine.oracle_required and engine.guard_enabled
    rpc = load('oracle_smoke_rpc', args.rpc.resolve())
    rival = rpc.BundleAgent(args.opponent, startup_timeout=180, timeout=60)
    times, predicted, actual, guard_changes = [], {}, {}, []
    guard = sys.modules['oracle_guard']
    original = guard.apply

    def record_guard(obs, orders, state, params):
        result = original(obs, orders, state, params)
        if result != orders:
            guard_changes.append(int(obs['step']))
        return result

    guard.apply = record_guard

    def ours(obs, config):
        t0 = time.monotonic()
        action = candidate.agent(obs, config)
        times.append((int(obs['step']), (time.monotonic() - t0) * 1000))
        prediction = engine.emulator.prediction()
        if prediction is not None:
            predicted[int(obs['step'])] = prediction
        return action

    def theirs(obs, config):
        action = rival(obs, config)
        actual[int(obs['step'])] = action
        return action

    env = make('kaggriculture', configuration={'seed': 101}, debug=False)
    try:
        env.run([ours, theirs] if args.seat == 0 else [theirs, ours])
    finally:
        rival.close()
    final = env.steps[-1]
    canon = lambda a: json.dumps({k: a.get(k) for k in ('farmer', 'hands', 'market')}, sort_keys=True)
    wrong = [step for step, value in predicted.items() if canon(value) != canon(actual.get(step, {}))]
    late = sorted(ms for step, ms in times if step >= 256)
    stats = dict(engine.champion.ORACLE_STATS)
    report = {
        'seat': args.seat, 'seed': 101, 'states': len(env.steps),
        'statuses': [s['status'] for s in final], 'rewards': [s['reward'] for s in final],
        'oracle': stats, 'guard_changed_turns': len(guard_changes),
        'emulator_predictions': len(predicted), 'wrong_predictions': wrong,
        'emulator_identity': engine.emulator.identity(), 'races': engine.emulator.races,
        'fallbacks': engine.fallback_count, 'startup_s': startup,
        'late_p99_ms': late[min(len(late) - 1, int(len(late) * .99))] if late else None,
        'max_turn_ms': max((ms for _, ms in times), default=0),
        'model_sha256': hashlib.sha256((args.bundle / 'hazel_runtime/checkpoint/model.safetensors').read_bytes()).hexdigest(),
    }
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)
    assert report['statuses'] == ['DONE', 'DONE'], report
    assert stats['forecasts'] == 463 and stats['errors'] == 0, report
    assert not wrong and not engine.fallback_count and len(predicted) > 600, report


if __name__ == '__main__':
    main()
