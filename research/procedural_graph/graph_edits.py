"""Executable controls of the policy graph and edits to them.

The graph's prose (concept descriptions, edge guidance) is documentation; the runtime
(`hazel_runtime/graph_runtime.py`) executes only these controls, so they are the
mutation surface of the evolution:

- `parameters` on turn/market chain nodes: the champion's EVOLVE-block constants
  (same JSON type as the submitted value) and the runtime's RUNTIME_PARAMETERS;
- `enabled: false` on a market stage (market_setup and routine_dispatch always run);
- `order` on routine_dispatch: 'submitted' or 'sells_first';
- the top-level `surgical` config (`hazel_runtime/surgical.py` OVERRIDES).

An edit is a JSON object with any of these keys, applied to a full graph:

    {"parameters": {"_NAME": value or null},   # null: back to the submitted value
     "stages": {"stage_id": true or false},
     "dispatch_order": "submitted" or "sells_first",
     "surgical": {"enabled": bool, "overrides": {...}}}

`apply_edit` checks names and types offline; `validate_graph` then builds the runtime
and plays the opening turns in a child process, which is what catches the rest.
"""
from __future__ import annotations

import ast
import copy
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from hazel_runtime import graph_runtime, surgical  # noqa: E402

CHAMPION = HERE / 'hazel_runtime' / 'champion.py'
EDIT_KEYS = ('parameters', 'stages', 'dispatch_order', 'surgical')
NODE_FEATURES = ('parameters', 'enabled', 'order')
ALWAYS_ON = ('market_setup', 'routine_dispatch')
MARKET_IDS = [stage[0] for stage in graph_runtime.MARKET_STAGES]
TURN_IDS = [stage[0] for stage in graph_runtime.TURN_STAGES]
# Line ranges of the champion's turn-level channels (derived state, farmer, hands);
# market stages take their ranges from graph_runtime.MARKET_STAGES.
_TURN_RANGES = (('state', 208, 282), ('farmer', 752, 787), ('hands', 788, 853))
# RUNTIME_PARAMETERS belong to the stage whose statement they parameterize.
_RUNTIME_HOME = {'_TOWN_CADENCE_PHASE': 'town_and_fertilizer', '_FEED_RESERVE_LOOKAHEAD': 'feed_reserve'}
_RUNTIME_NOTES = {'_TOWN_CADENCE_PHASE': 'shop-cadence phase: town sales on steps with '
                  '(step - phase) % 4 == 0; the engine consumes town stock after the market '
                  'on step % 4 == 0',
                  '_FEED_RESERVE_LOOKAHEAD': 'the two-day wheat feed reserve also counts animals bought, '
                  'carried, ordered this turn and empty pastures/coops (an animal escapes after two '
                  'unfed days; without it the day-6 herd expansion starved a cow in every game vs Mohui)'}


def _stage_ranges():
    starts = [(key, line) for key, line, _ in graph_runtime.MARKET_STAGES]
    ranges = [(key, line, (starts[i + 1][1] - 1) if i + 1 < len(starts) else line)
              for i, (key, line) in enumerate(starts)]
    return list(_TURN_RANGES[:1]) + ranges + list(_TURN_RANGES[1:])


def _comment(lines, lineno):
    """Inline comment of an assignment, else the comment block right above it."""
    line = lines[lineno - 1]
    if '#' in line:
        return line.split('#', 1)[1].strip()
    above = []
    i = lineno - 2
    while i >= 0 and lines[i].strip().startswith('#') and len(above) < 3:
        text = re.sub(r'[-=]{5,}', '', lines[i].strip().lstrip('#')).strip()
        if text:
            above.insert(0, text)
        i -= 1
    return ' '.join(above)


def catalog(source_path=CHAMPION):
    """{name: {default, type, stages, home, note, used}} for every graph parameter.

    `used` is False for a constant the champion defines but never reads: setting it
    cannot change play, so edits of it are refused.
    """
    source = Path(source_path).read_text()
    tree = ast.parse(source)
    lines = source.splitlines()
    names = graph_runtime.evolvable_constants(tree, source)
    ranges = _stage_ranges()
    out = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not (isinstance(target, ast.Name) and target.id in names):
                continue
            pattern = re.compile(rf'\b{re.escape(target.id)}\b')
            used = [key for key, lo, hi in ranges
                    if any(pattern.search(lines[i - 1]) for i in range(lo, hi + 1))]
            read = any(pattern.search(text) for i, text in enumerate(lines, 1) if i != node.lineno)
            home = next((key for key in used if key in MARKET_IDS), used[0] if used else 'market')
            default = ast.literal_eval(node.value)
            out[target.id] = {'default': default, 'type': type(default).__name__,
                              'stages': used, 'home': home, 'note': _comment(lines, node.lineno),
                              'used': read}
    for name, default in graph_runtime.RUNTIME_PARAMETERS.items():
        out[name] = {'default': default, 'type': type(default).__name__,
                     'stages': [_RUNTIME_HOME[name]], 'home': _RUNTIME_HOME[name],
                     'note': _RUNTIME_NOTES.get(name, 'runtime parameter'), 'used': True}
    return out


def _chain_nodes(graph):
    return {node['id']: node for chain in ('turn', 'market') for node in graph[chain]['nodes']}


def _json_value(value):
    return list(value) if isinstance(value, tuple) else value


def settings(graph, constants=None):
    """Normalized executable settings: identical settings mean identical play.

    Parameters equal to the submitted value, enabled stages, the submitted dispatch
    order and the overrides of a disabled surgical config are dropped.
    """
    constants = constants or catalog()
    params, stages = {}, {}
    order = 'submitted'
    for node_id, node in _chain_nodes(graph).items():
        for name, value in (node.get('parameters') or {}).items():
            spec = constants.get(name)
            if spec is not None and not spec['used']:
                continue
            if spec is None or _json_value(graph_runtime._like(value, spec['default'], name)) != _json_value(spec['default']):
                params[name] = value
        if node.get('enabled') is False:
            stages[node_id] = False
        if node_id == 'routine_dispatch':
            order = node.get('order', 'submitted')
    config = graph.get('surgical') or {'enabled': False}
    enabled = surgical.enabled(config)
    surgical_settings = ({'enabled': True, 'overrides': surgical.overrides(config)}
                         if enabled else {'enabled': False})
    return {'parameters': dict(sorted(params.items())), 'stages': dict(sorted(stages.items())),
            'dispatch_order': order, 'surgical': surgical_settings}


def settings_key(graph, constants=None):
    return json.dumps(settings(graph, constants), sort_keys=True, separators=(',', ':'))


def apply_edit(graph, edit, constants=None):
    """New graph with `edit` applied; unknown names, types and values raise ValueError."""
    constants = constants or catalog()
    if not isinstance(edit, dict) or not edit:
        raise ValueError('edit must be a non-empty JSON object')
    unknown = set(edit) - set(EDIT_KEYS)
    if unknown:
        raise ValueError(f'unknown edit keys {sorted(unknown)}; allowed: {list(EDIT_KEYS)}')
    out = copy.deepcopy(graph)
    nodes = _chain_nodes(out)
    params = edit.get('parameters') or {}
    if not isinstance(params, dict):
        raise ValueError('edit.parameters must be an object {name: value}')
    for name, value in params.items():
        spec = constants.get(name)
        if spec is None:
            raise ValueError(f'unknown parameter {name}; editable parameters are the catalog names')
        if not spec['used'] and value is not None:
            raise ValueError(f'{name} has no effect: the champion code never reads it')
        holders = [n for n in nodes.values() if name in (n.get('parameters') or {})]
        if value is None:
            for node in holders:
                del node['parameters'][name]
                if not node['parameters']:
                    del node['parameters']
            continue
        value = _json_value(graph_runtime._like(value, spec['default'], name))
        target = holders[0] if holders else nodes[spec['home']]
        target.setdefault('parameters', {})[name] = value
        for node in holders[1:]:
            del node['parameters'][name]
            if not node['parameters']:
                del node['parameters']
    stage_flags = edit.get('stages') or {}
    if not isinstance(stage_flags, dict):
        raise ValueError('edit.stages must be an object {market stage id: bool}')
    for stage, flag in stage_flags.items():
        if stage not in MARKET_IDS:
            raise ValueError(f'unknown market stage {stage}; stages are {MARKET_IDS}')
        if type(flag) is not bool:
            raise ValueError(f'stage flag for {stage} must be true or false')
        if stage in ALWAYS_ON and not flag:
            raise ValueError(f'{stage} cannot be disabled')
        if flag:
            nodes[stage].pop('enabled', None)
        else:
            nodes[stage]['enabled'] = False
    if 'dispatch_order' in edit:
        order = edit['dispatch_order']
        if order not in graph_runtime.ORDER_POLICIES:
            raise ValueError(f'dispatch_order must be one of {list(graph_runtime.ORDER_POLICIES)}')
        if order == 'submitted':
            nodes['routine_dispatch'].pop('order', None)
        else:
            nodes['routine_dispatch']['order'] = order
    if 'surgical' in edit:
        config = edit['surgical']
        surgical.enabled(config)  # fails closed on unknown names/values
        out['surgical'] = copy.deepcopy(config)
    if settings_key(out, constants) == settings_key(graph, constants):
        raise ValueError('edit changes nothing: every value equals the current setting')
    return out


def diff(before, after, constants=None):
    """Human-readable control changes from `before` to `after` (for prompts and logs)."""
    constants = constants or catalog()
    a, b = settings(before, constants), settings(after, constants)
    lines = []
    for name in sorted(set(a['parameters']) | set(b['parameters'])):
        old = a['parameters'].get(name, constants[name]['default'])
        new = b['parameters'].get(name, constants[name]['default'])
        if _json_value(old) != _json_value(new):
            lines.append(f"{name} ({constants[name]['home']}): {json.dumps(_json_value(old))} -> "
                         f"{json.dumps(_json_value(new))}")
    for stage in MARKET_IDS:
        old, new = a['stages'].get(stage, True), b['stages'].get(stage, True)
        if old != new:
            lines.append(f"stage {stage}: {'enabled' if old else 'disabled'} -> {'enabled' if new else 'disabled'}")
    if a['dispatch_order'] != b['dispatch_order']:
        lines.append(f"routine_dispatch order: {a['dispatch_order']} -> {b['dispatch_order']}")
    if a['surgical'] != b['surgical']:
        lines.append(f"surgical: {json.dumps(a['surgical'])} -> {json.dumps(b['surgical'])}")
    return lines


def describe_controls(graph, constants=None):
    """The mutation surface with the graph's current values, as prompt text."""
    constants = constants or catalog()
    current = settings(graph, constants)
    rows = ['PARAMETERS (name | type | current | submitted | stage(s) | note):']
    order = MARKET_IDS + TURN_IDS
    for name, spec in sorted(constants.items(), key=lambda kv: order.index(kv[1]['home'])
                             if kv[1]['home'] in order else 99):
        if not spec['used']:
            continue
        value = current['parameters'].get(name, spec['default'])
        rows.append(f"{name} | {spec['type']} | {json.dumps(_json_value(value))} | "
                    f"{json.dumps(_json_value(spec['default']))} | {','.join(spec['stages']) or '-'} | {spec['note']}")
    inert = sorted(name for name, spec in constants.items() if not spec['used'])
    if inert:
        rows.append(f"(defined but never read by the champion, so not editable: {', '.join(inert)})")
    rows.append('')
    rows.append('MARKET STAGES in execution order (id: summary [state]); '
                f"{' and '.join(ALWAYS_ON)} always run:")
    for key, _, summary in graph_runtime.MARKET_STAGES:
        state = 'disabled' if current['stages'].get(key) is False else 'enabled'
        rows.append(f'{key}: {summary} [{state}]')
    rows.append('')
    rows.append(f"DISPATCH ORDER (routine_dispatch): {current['dispatch_order']}; options "
                f"{list(graph_runtime.ORDER_POLICIES)}")
    rows.append(f"SURGICAL: {json.dumps(current['surgical'])}; overrides and allowed values "
                f"{json.dumps({k: list(v) for k, v in surgical.OVERRIDES.items()})}")
    return '\n'.join(rows)


_VALIDATE = r'''
import json, os, sys
graph_path, steps, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
os.environ['KAGG_GRAPH_PATH'] = graph_path
sys.path.insert(0, os.getcwd())
import agent_graph
engine = agent_graph.get_engine()
if agent_graph._LEGACY is not None:
    raise SystemExit('graph does not declare the hazel_merged_v1 runtime')
from kaggle_environments import make
env = make('kaggriculture', configuration={'seed': seed, 'episodeSteps': steps}, debug=False)
env.run([agent_graph.agent, lambda obs, config: {}])
statuses = [s.get('status') for s in env.steps[-1]]
print(json.dumps({'statuses': statuses, 'fallbacks': engine.fallback_count,
                  'last_error': engine.last_error, 'steps': len(env.steps)}))
'''


def validate_graph(graph_path, steps=30, seed=20260925, python=None, timeout=600):
    """Build the runtime for a graph and play its first turns in a fresh interpreter.

    Raises ValueError with the child's error text (fed back to the mutating LLM) when
    the runtime rejects the graph, a stage falls back to the backbone, or play fails.
    """
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               CUDA_VISIBLE_DEVICES='', KAGG_ORACLE_BACKEND='numpy', KAGG_ORACLE_DEVICE='cpu',
               PYTHONDONTWRITEBYTECODE='1')
    env.pop('KAGG_GRAPH_EXPERIMENTS', None)
    proc = subprocess.run([python or sys.executable, '-c', _VALIDATE, str(Path(graph_path).resolve()), str(steps), str(seed)],
                          cwd=HERE, env=env, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-6:]
        raise ValueError('runtime rejected the graph: ' + ' | '.join(tail)[:1500])
    report = json.loads(proc.stdout.strip().splitlines()[-1])
    if report['fallbacks']:
        raise ValueError(f"a graph stage failed and fell back to the backbone {report['fallbacks']} time(s) "
                         f"in the first {steps} turns: {report['last_error']}")
    if report['statuses'] != ['DONE', 'DONE']:
        raise ValueError(f"validation game ended with statuses {report['statuses']}")
    return report
