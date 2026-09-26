#!/usr/bin/env python3
"""Which layers of a ladder engine change its actions in play, and which engine parameters each
layer reads: the map the mutation prompts use to aim edits at levers that act.

    python research/procedural_graph/engine_activity.py --engine tetsutani_demand \
        --plan evolution_results/ladder_2026-09-26/plan.json --lost 12 --random 12 --workers 4 \
        --out evolution_results/ladder_2026-09-26/engine_activity.json

A ladder engine is a stack of layers: a layer saves the previous `agent` under a global name
(`_PARENT = agent; del agent`) and defines a new `agent` that calls it and edits its action. The
engine is loaded in-process (a fresh load per game, as Kaggle does) and every saved parent is
wrapped: each turn records the action every layer returned and the action it got from its parent.
A layer *acts* on a turn when the two differ. The wrappers only record (--check plays one job
unwrapped too and compares the final cash). The opponent runs in its own process (BundleAgent).

A layer's parameters are the engine catalog constants its own code reads, directly or through the
module's helper functions and classes, not through its parent. Parameters that no layer reads are
read while the module loads (tables, routes, the chassis settings) or only by the base chassis.
"""
from __future__ import annotations

import argparse
import collections
import json
import multiprocessing as mp
import os
import sys
import time
import types
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / 'hazel_runtime'))

ENGINES = HERE / 'hazel_runtime' / 'engines'


def _norm(action):
    """An action as plain JSON (tuples -> lists), copied before a caller can edit it in place."""
    return json.loads(json.dumps(action, default=str))


def _code_names(code):
    names = set(code.co_names)
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            names |= _code_names(const)
    return names


class Recorder:
    """Wraps an engine's saved parents; tallies per layer the turns on which it changed the action."""

    def __init__(self, policy):
        self.policy = policy
        self.ns = policy.__globals__
        self.pointers = {name: fn for name, fn in self.ns.items() if self._saved_agent(name, fn)}
        self.layers = {}
        for name, fn in sorted(self.pointers.items(), key=lambda kv: kv[1].__code__.co_firstlineno):
            self.layers.setdefault(fn.__code__.co_firstlineno, {'pointer': name, 'fn': fn})
        self.layers[policy.__code__.co_firstlineno] = {'pointer': '(final agent)', 'fn': policy}
        for line, layer in self.layers.items():
            refs = _code_names(layer['fn'].__code__) & set(self.pointers)
            layer['parents'] = sorted(refs)
        self.tally = {line: collections.Counter() for line in self.layers}
        self.stack = []
        for name, fn in self.pointers.items():
            self.ns[name] = self._wrap(fn)
        self.top = self._wrap(policy)

    def _saved_agent(self, name, fn):
        """A policy function kept under a global name other than its own: a layer's saved parent
        (`_PARENT = agent`), whatever the function is called (`agent`, `cha20_entry_agent`, ...)."""
        if not (isinstance(fn, types.FunctionType) and fn.__globals__ is self.ns and fn is not self.policy):
            return False
        code = fn.__code__
        first = code.co_varnames[:1]
        return code.co_argcount >= 1 and first in (('observation',), ('obs',)) and (
            name != fn.__name__ or fn.__name__ == 'agent')

    def _wrap(self, fn):
        line = fn.__code__.co_firstlineno

        def recorder(*args, **kwargs):
            frame = {'line': line, 'children': []}
            if self.stack:
                self.stack[-1]['children'].append(frame)
            self.stack.append(frame)
            try:
                out = fn(*args, **kwargs)
            finally:
                self.stack.pop()
            frame['out'] = _norm(out)
            return out
        recorder.__name__ = fn.__name__
        recorder.__dict__ = fn.__dict__   # the same attribute dict: engines keep state on them (_IMPL.chassis)
        return recorder

    def __call__(self, *args):
        root = {'line': None, 'children': []}
        self.stack = [root]
        out = self.top(*args)
        self.stack = []
        frame = root['children'][0] if root['children'] else None
        while frame is not None:
            counts = self.tally[frame['line']]
            counts['turns'] += 1
            children = frame['children']
            if not children:
                if self.layers[frame['line']]['parents']:
                    counts['replaced'] += 1     # answered without calling its parent
                break
            if len(children) > 1:
                counts['multi_calls'] += 1
            if frame['out'] != children[0]['out']:
                counts['acted'] += 1
            frame = children[0]
        return out


def layer_parameters(recorder, constants):
    """{layer line: sorted catalog constants its own code reads (not through its parent)}."""
    ns = recorder.ns
    out = {}
    for line, layer in recorder.layers.items():
        reads, seen, todo = set(), set(), [layer['fn'].__code__]
        while todo:
            code = todo.pop()
            if id(code) in seen:
                continue
            seen.add(id(code))
            for name in _code_names(code):
                if name in constants:
                    reads.add(name)
                if name in recorder.pointers or name == 'agent':
                    continue
                value = ns.get(name)
                if isinstance(value, types.FunctionType) and value.__globals__ is ns:
                    todo.append(value.__code__)
                    continue
                cls = value if isinstance(value, type) else type(value)
                for attr in vars(cls).values() if hasattr(cls, '__dict__') else ():
                    if isinstance(attr, (staticmethod, classmethod)):
                        attr = attr.__func__
                    if isinstance(attr, types.FunctionType) and attr.__globals__ is ns:
                        todo.append(attr.__code__)
        out[line] = sorted(reads)
    return out


def layer_comment(lines, line):
    """The comment block just above a layer (above its `def agent` or its parent-saving lines)."""
    i, found = line - 2, []
    skipped = 0
    while i >= 0 and skipped < 12 and len(found) < 3:
        text = lines[i].strip()
        if text.startswith('#'):
            found.insert(0, text.lstrip('#').strip())
        elif found:
            break
        else:
            skipped += 1
        i -= 1
    return ' '.join(found)[:240]


def play(job):
    """One full game: the instrumented engine in seat a_seat against a pool bundle."""
    from kaggle_environments import make
    from kaggle_environments.utils import structify
    from hazel_runtime import engines
    sys.path.insert(0, str(REPO / 'shinka' / 'evolution'))
    from pool_upgrade_bundle_agent import BundleAgent

    t0 = time.monotonic()
    engine = engines.Engine(ENGINES / job['engine'], job.get('parameters') or None)
    recorder = Recorder(engine.policy) if job.get('record', True) else None
    call = recorder if recorder else engine.policy
    nargs = engine.nargs
    opponent = BundleAgent(Path(job['opponent_dir']), entrypoint='main.py', timeout=60, startup_timeout=300)
    opponent.start()
    try:
        def ours(obs, config):
            return call(*[structify(obs), structify(config)][:nargs])

        def theirs(obs, config):
            return opponent(obs, config)

        seats = [theirs, theirs]
        seats[job['a_seat']] = ours
        env = make('kaggriculture', configuration={'seed': job['seed']}, debug=False)
        env.run(seats)
    finally:
        opponent.close()
    final = env.steps[-1]
    rewards = [s.get('reward') for s in final]
    result = {k: job[k] for k in ('tag', 'seed', 'a_seat', 'set') if k in job}
    result.update(rewards=rewards, statuses=[s.get('status') for s in final], seconds=round(time.monotonic() - t0, 1))
    if recorder:
        result['tally'] = {str(line): dict(c) for line, c in recorder.tally.items()}
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--engine', default='tetsutani_demand')
    ap.add_argument('--parameters', type=Path, default=None, help='JSON {name: value} of engine parameters')
    ap.add_argument('--plan', type=Path, required=True, help='gauntlet plan (ladder_seed_plan.py) to draw jobs from')
    ap.add_argument('--lost', type=int, default=12, help='lost-seed jobs to play (spread over the plan)')
    ap.add_argument('--random', type=int, default=12, help='random-seed jobs to play (spread over the plan)')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--bundles', type=Path, default=Path('/tmp/engine_activity_bundles'),
                    help='where opponent bundles are built (payload.build_opponent)')
    ap.add_argument('--check', action='store_true', help='also play the first job unwrapped and compare the cash')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()

    sys.path.insert(0, str(HERE / 'arena'))
    import payload
    from hazel_runtime import engines

    plan = json.loads(args.plan.read_text())
    picked = []
    for kind, count in (('lost', args.lost), ('random', args.random)):
        entries = [e for e in plan if e.get('set') == kind]
        step = max(1, len(entries) // max(1, count))
        picked += entries[::step][:count]
    parameters = json.loads(args.parameters.read_text()) if args.parameters else {}
    jobs = []
    for entry in picked:
        name = entry['tag']
        if not (args.bundles / 'bundles' / name).exists():
            payload.build_opponent(args.bundles, name)
        jobs.append(dict(entry, engine=args.engine, parameters=parameters,
                         opponent_dir=str(args.bundles / 'bundles' / name)))
    if args.check and jobs:
        jobs.append(dict(jobs[0], record=False, check=True))
    t0 = time.monotonic()
    with mp.get_context('spawn').Pool(args.workers) as pool:
        results = []
        for i, res in enumerate(pool.imap_unordered(play, jobs), 1):
            results.append(res)
            print(f'[{i}/{len(jobs)}] {res.get("tag")} seed {res.get("seed")} seat {res.get("a_seat")}: '
                  f'{res["rewards"]} ({res["seconds"]} s, {time.monotonic() - t0:.0f} s total)', flush=True)
    checked = [r for r in results if 'tally' not in r]
    if checked:
        twin = next(r for r in results if 'tally' in r and (r['tag'], r['seed'], r['a_seat']) ==
                    (checked[0]['tag'], checked[0]['seed'], checked[0]['a_seat']))
        if twin['rewards'] != checked[0]['rewards']:
            raise SystemExit(f'recording changed the game: {twin["rewards"]} vs {checked[0]["rewards"]}')
        print('check: the recorded game equals the unwrapped one', twin['rewards'])
    played = [r for r in results if 'tally' in r]

    source_dir = ENGINES / args.engine
    entry = json.loads((source_dir / 'SOURCE.json').read_text())['entry']
    source = (source_dir / 'agent' / entry).read_text(encoding='utf-8')
    lines = source.splitlines()
    constants = engines.catalog(source)
    recorder = Recorder(engines.Engine(source_dir).policy)
    reads = layer_parameters(recorder, constants)
    layers = []
    for line, layer in sorted(recorder.layers.items()):
        per_game = [collections.Counter(r['tally'].get(str(line), {})) for r in played]
        acted = [c['acted'] for c in per_game]
        layers.append({
            'line': line, 'saved_as': layer['pointer'], 'comment': layer_comment(lines, line),
            'turns': sum(c['turns'] for c in per_game), 'acted_turns': sum(acted),
            'acted_per_game': round(sum(acted) / max(1, len(played)), 1),
            'games_acted': sum(1 for a in acted if a), 'games': len(played),
            'replaced_turns': sum(c['replaced'] for c in per_game),
            'multi_call_turns': sum(c['multi_calls'] for c in per_game),
            'parameters': reads[line]})
    attributed = {p for layer in layers for p in layer['parameters']}
    report = {
        'engine': args.engine, 'parameters': parameters, 'games': [{k: v for k, v in r.items() if k != 'tally'}
                                                                    for r in played],
        'layers': layers,
        'load_time_or_chassis_parameters': sorted(p for p, spec in constants.items()
                                                  if spec['used'] and p not in attributed),
        'unused_parameters': sorted(p for p, spec in constants.items() if not spec['used'])}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + '\n')
    active = [l for l in layers if l['acted_turns']]
    print(f'{len(played)} games; {len(active)} of {len(layers)} layers acted; report {args.out}')
    for layer in layers:
        print(f"  line {layer['line']:>5} {layer['acted_per_game']:>7} turns/game, {layer['games_acted']}/"
              f"{layer['games']} games | {layer['comment'][:90]} | {', '.join(layer['parameters'])[:160]}")


if __name__ == '__main__':
    main()
