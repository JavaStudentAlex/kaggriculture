"""Executable controls of the policy graph and edits to them.

The graph's prose (concept descriptions, edge guidance) is documentation; the runtime
(`hazel_runtime/graph_runtime.py`) executes only these controls, so they are the
mutation surface of the evolution:

- `parameters` on turn/market chain nodes: the champion's EVOLVE-block constants
  (same JSON type as the submitted value) and the runtime's RUNTIME_PARAMETERS;
- `enabled: false` on a market stage (market_setup and routine_dispatch always run);
- `order` on routine_dispatch: 'submitted' or 'sells_first';
- the top-level `surgical` config (`hazel_runtime/surgical.py` OVERRIDES).

On a graph whose backbone runs a ladder engine (make_ladder_graph.py) there are also:

- `engine_parameters` on the backbone node: the engine's top-level constants (engines.py);
- the channels: `enabled` on the farmer, hands and market turn nodes (false = the engine's
  action passes through) and on the optional oracle_guard node (the predictor's front-run);
- `experimental.switches`: the extension stages of hazel_runtime/experimental.py;
- the optional rival_counter node (channel `rival_counter`; an edit that switches it on inserts it) and its
  `counters`: per rival family (hazel_runtime/rival_model.py), values for switchable engine constants and
  `_OG_*` guard parameters that apply while the rival looks like that family;
- the optional rival_emulator node (channel `rival_emulator`, inserted the same way) with its `_EM_*`
  parameters: the public engines it runs in the rival's place (hazel_runtime/rival_emulator.py). While one
  of them matches the rival exactly, the counter family `engine:<name>` applies and, with `_EM_RACE`, our
  market orders are rearranged against the rival's known orders;
- the optional tactic node (channel `tactic`, inserted the same way) and its `code`: a Python function
  `tactic(obs, action, memory, info)` the evolution writes, run in a sandbox on our action every turn
  (hazel_runtime/tactic.py);
- the optional land_plot node (channel `land_plot`, inserted the same way) with its `_LP_*` parameters: the
  fourth quadrant bought from an evolved day and farmed with a crop or geese by hands hired for it
  (hazel_runtime/land_plot.py).

An edit is a JSON object with any of these keys, applied to a full graph:

    {"parameters": {"_NAME": value or null},   # null: back to the submitted value
     "stages": {"stage_id": true or false},
     "dispatch_order": "submitted" or "sells_first",
     "surgical": {"enabled": bool, "overrides": {...}},
     "engine_parameters": {"NAME": value or null},
     "channels": {"farmer" | "hands" | "market" | "oracle_guard" | "rival_counter" | "rival_emulator" |
                  "land_plot" | "tactic": bool},
     "experimental": {"<switch>": true or false},
     "counters": {"<family>": {"NAME": value or null} or null},
     "tactic": "<python source defining tactic(obs, action, memory, info)>" or null}

A setting that cannot change play (a market parameter while the market channel is off, say)
is refused, so no gauntlet is spent on it.

`apply_edit` checks names and types offline; `validate_graph` then builds the runtime
and plays the opening turns in a child process, which is what catches the rest.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from hazel_runtime import (engines, experimental, graph_runtime, land_plot, oracle_guard, rival_emulator,  # noqa: E402
                            rival_model, surgical, tactic)
from hazel_runtime import kad_copilot  # noqa: E402

CHAMPION = HERE / 'hazel_runtime' / 'champion.py'
EDIT_KEYS = ('parameters', 'stages', 'dispatch_order', 'surgical', 'engine_parameters', 'channels',
             'experimental', 'counters', 'tactic')
CHANNEL_DEFAULTS = {'farmer': True, 'hands': True, 'market': True, 'oracle_guard': False, 'rival_counter': False,
                    'rival_emulator': False, 'tactic': False, 'land_plot': False, 'kad_copilot': False}
RIVAL_NODE = {'id': 'rival_counter', 'binding': 'rival_counter', 'enabled': True,
              'summary': "The rival's family from its public farm; that family's counter-settings"}
EMULATOR_NODE = {'id': 'rival_emulator', 'binding': 'rival_emulator', 'enabled': True,
                 'summary': "The rival's public engine run in its place from our view; its known orders"}
TACTIC_NODE = {'id': 'tactic', 'binding': 'tactic', 'enabled': True,
               'summary': 'The evolved Python tactic, run in a sandbox on our action'}
LAND_NODE = {'id': 'land_plot', 'binding': 'land_plot', 'enabled': True,
             'summary': 'The fourth quadrant bought from an evolved day and farmed by hired hands'}
KAD_NODE = {'id': 'kad_copilot', 'binding': 'kad_copilot', 'enabled': True,
            'summary': 'KAD-HP-1 (top-player replay policy) adds confident sales and idle-hand jobs'}
KAD_FILES = ('kad_copilot.py', 'kad/__init__.py', 'kad/kad_numpy.py', 'kad/kad_data.py', 'kad/weights.npz')
_SWITCHABLE = {}
ENGINES_DIR = HERE / 'hazel_runtime' / 'engines'
_ENGINE_CATALOGS = {}
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
    for name, default in oracle_guard.PARAMETERS.items():
        out[name] = {'default': default, 'type': type(default).__name__, 'stages': ['oracle_guard'],
                     'home': 'oracle_guard', 'note': oracle_guard.NOTES.get(name, ''), 'used': True}
    for name, default in rival_model.PARAMETERS.items():
        out[name] = {'default': default, 'type': type(default).__name__, 'stages': ['rival_counter'],
                     'home': 'rival_counter', 'note': rival_model.NOTES.get(name, ''), 'used': True}
    for name, default in rival_emulator.PARAMETERS.items():
        out[name] = {'default': default, 'type': type(default).__name__, 'stages': ['rival_emulator'],
                     'home': 'rival_emulator', 'note': rival_emulator.NOTES.get(name, ''), 'used': True}
    for name, default in land_plot.PARAMETERS.items():
        out[name] = {'default': default, 'type': type(default).__name__, 'stages': ['land_plot'],
                     'home': 'land_plot', 'note': land_plot.NOTES.get(name, ''), 'used': True}
    for name, default in kad_copilot.PARAMETERS.items():
        out[name] = {'default': default, 'type': type(default).__name__, 'stages': ['kad_copilot'],
                     'home': 'kad_copilot', 'note': kad_copilot.NOTES.get(name, ''), 'used': True}
    return out


def emulated_engines(graph):
    """The engines the graph's rival_emulator node runs ([] without the node or with it off)."""
    node = _chain_nodes(graph).get('rival_emulator')
    if not node or not node.get('enabled', False):
        return []
    return list((node.get('parameters') or {}).get('_EM_ENGINES', rival_emulator.PARAMETERS['_EM_ENGINES']))


def rival_families(graph=None):
    """{class: description} of the rival_counter stage (rival_model.CLASSES), plus `engine:<name>` for every
    engine the graph's rival_emulator runs."""
    out = dict(rival_model.CLASSES)
    for name in (emulated_engines(graph) if graph is not None else []):
        out[f'engine:{name}'] = (f'plays the public engine {name} exactly (the rival_emulator matched it move for '
                                 f'move); its orders of each turn are known before ours')
    return out


def switchable_engine_parameters(graph):
    """Engine parameters of the graph's engine that a rival counter may set during a game."""
    name = engine_name(graph)
    if name == graph_runtime.DEFAULT_ENGINE:
        return set()
    if name not in _SWITCHABLE:
        meta = json.loads((ENGINES_DIR / name / 'SOURCE.json').read_text())
        source = (ENGINES_DIR / name / 'agent' / meta['entry']).read_text(encoding='utf-8')
        _SWITCHABLE[name] = engines.switchable(source)
    return _SWITCHABLE[name]


def _insert_before(turn, node, before):
    """Put `node` into the turn chain right before the node `before`, keeping the edges in chain order."""
    ids = [n['id'] for n in turn['nodes']]
    i = ids.index(before)
    turn['nodes'].insert(i, copy.deepcopy(node))
    edge = {'source': node['id'], 'target': before, 'relation': 'NEXT'}
    if i == 0:
        turn['entry'] = node['id']
        turn['edges'].insert(0, edge)
        return
    k = next(j for j, e in enumerate(turn['edges']) if e.get('source') == ids[i - 1] and e.get('target') == before)
    turn['edges'][k] = dict(turn['edges'][k], target=node['id'])
    turn['edges'].insert(k + 1, edge)


def _insert_rival_node(graph):
    _insert_before(graph['turn'], RIVAL_NODE, 'backbone')
    if isinstance(graph.get('nodes'), list) and not any(n.get('id') == 'rival_counter' for n in graph['nodes']):
        graph['nodes'].append({
            'id': 'rival_counter', 'name': 'rival counter',
            'description': ("Compare the rival's public farm with ours every turn; by step 93 the rival is a mirror of "
                            'our engine, a buy-N/sell-N-5 opener, a wheat-92 seller, another opening or other '
                            "(rival_model.MirrorTracker), and that class's counters (engine constants, _OG_* guard "
                            'parameters) apply for the rest of the game.'),
            'bindings': ['rival_counter'], 'binding_semantics': 'Executable stage'})
        graph.setdefault('edges', []).append({'source': 'rival_counter', 'target': 'backbone',
                                              'relation': 'CONFIGURES', 'scope': 'turn'})
    pins = (graph.get('provenance') or {}).get('runtime_bundle_hashes')
    if isinstance(pins, dict):
        pins['rival_model.py'] = _sha(HERE / 'hazel_runtime' / 'rival_model.py')


def _insert_emulator_node(graph):
    _insert_before(graph['turn'], EMULATOR_NODE, graph['turn']['nodes'][0]['id'])
    if isinstance(graph.get('nodes'), list) and not any(n.get('id') == 'rival_emulator' for n in graph['nodes']):
        graph['nodes'].append({
            'id': 'rival_emulator', 'name': 'rival emulator',
            'description': ("Run the public engines of _EM_ENGINES in the rival's place from our own view of the game: "
                            'every observation shows both farms, and the rival\'s private stock is rebuilt with the '
                            'environment\'s own rules; an engine that disagrees with an observation is dropped. While one '
                            'matches, the rival\'s orders of each turn are known before ours (rival_emulator.py).'),
            'bindings': ['rival_emulator'], 'binding_semantics': 'Executable stage'})
        graph.setdefault('edges', []).append({'source': 'rival_emulator', 'target': 'backbone',
                                              'relation': 'CONFIGURES', 'scope': 'turn'})
    _pin_emulator(graph)


def _insert_tactic_node(graph):
    _insert_before(graph['turn'], TACTIC_NODE, 'sanitize')
    if isinstance(graph.get('nodes'), list) and not any(n.get('id') == 'tactic' for n in graph['nodes']):
        graph['nodes'].append({
            'id': 'tactic', 'name': 'evolved tactic',
            'description': ('Python code written by the evolution: tactic(obs, action, memory, info) returns our '
                            'action for the turn; it runs in a sandbox with a per-turn budget (tactic.py).'),
            'bindings': ['tactic'], 'binding_semantics': 'Executable stage'})
        graph.setdefault('edges', []).append({'source': 'tactic', 'target': 'sanitize', 'relation': 'CONFIGURES',
                                              'scope': 'turn'})
    pins = (graph.get('provenance') or {}).get('runtime_bundle_hashes')
    if isinstance(pins, dict):
        pins['tactic.py'] = _sha(HERE / 'hazel_runtime' / 'tactic.py')


def _insert_land_node(graph):
    ids = [n['id'] for n in graph['turn']['nodes']]
    _insert_before(graph['turn'], LAND_NODE, 'tactic' if 'tactic' in ids else 'sanitize')
    if isinstance(graph.get('nodes'), list) and not any(n.get('id') == 'land_plot' for n in graph['nodes']):
        graph['nodes'].append({
            'id': 'land_plot', 'name': 'land plot',
            'description': ('From day _LP_DAY buy the fourth quadrant (SE; SW first when the tape has not bought it), '
                            'hire _LP_WORKERS hands a day after the engine\'s own hires and farm _LP_TILES SE tiles with '
                            '_LP_USE: a crop or geese; the plot buys its inputs and sells its produce (land_plot.py).'),
            'bindings': ['land_plot'], 'binding_semantics': 'Executable stage'})
        graph.setdefault('edges', []).append({'source': 'land_plot', 'target': 'sanitize', 'relation': 'CONFIGURES',
                                              'scope': 'turn'})
    pins = (graph.get('provenance') or {}).get('runtime_bundle_hashes')
    if isinstance(pins, dict):
        pins['land_plot.py'] = _sha(HERE / 'hazel_runtime' / 'land_plot.py')


def _insert_kad_node(graph):
    ids = [n['id'] for n in graph['turn']['nodes']]
    _insert_before(graph['turn'], KAD_NODE, next(i for i in ('land_plot', 'tactic', 'sanitize') if i in ids))
    if isinstance(graph.get('nodes'), list) and not any(n.get('id') == 'kad_copilot' for n in graph['nodes']):
        graph['nodes'].append({
            'id': 'kad_copilot', 'name': 'KAD copilot',
            'description': ('KAD-HP-1, the policy trained on the published top-player replays (numpy, kad/), runs every '
                            '_KC_EVERY turns from _KC_FROM_STEP to _KC_TO_STEP: its confident SELLs are appended after '
                            'the engine\'s orders and idle hands do its job where they stand (kad_copilot.py).'),
            'bindings': ['kad_copilot'], 'binding_semantics': 'Executable stage'})
        graph.setdefault('edges', []).append({'source': 'kad_copilot', 'target': 'sanitize', 'relation': 'CONFIGURES',
                                              'scope': 'turn'})
    pins = (graph.get('provenance') or {}).get('runtime_bundle_hashes')
    if isinstance(pins, dict):
        for name in KAD_FILES:
            pins[name] = _sha(HERE / 'hazel_runtime' / name)


def tactic_code(graph):
    """The code of the graph's tactic node, or None without the node or with it off."""
    node = _chain_nodes(graph).get('tactic')
    return node.get('code') if node and node.get('enabled', False) else None


def _pin_emulator(graph):
    """Pin rival_emulator.py and the files of every engine the emulator runs (the bundle then carries them)."""
    pins = (graph.get('provenance') or {}).get('runtime_bundle_hashes')
    if not isinstance(pins, dict):
        return
    pins['rival_emulator.py'] = _sha(HERE / 'hazel_runtime' / 'rival_emulator.py')
    for name in emulated_engines(graph):
        meta = json.loads((ENGINES_DIR / name / 'SOURCE.json').read_text())
        pins[f'engines/{name}/SOURCE.json'] = _sha(ENGINES_DIR / name / 'SOURCE.json')
        pins[f"engines/{name}/agent/{meta['entry']}"] = _sha(ENGINES_DIR / name / 'agent' / meta['entry'])


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def repinned(graph):
    """The graph with its pinned runtime files re-hashed as they are on disk (write_graph_bundle does
    the same inside every bundle), so a runtime change does not invalidate the graphs being evolved."""
    out = copy.deepcopy(graph)
    pins = (out.get('provenance') or {}).get('runtime_bundle_hashes') or {}
    for relative in list(pins):
        path = HERE / 'hazel_runtime' / relative
        if path.exists():
            pins[relative] = _sha(path)
    if 'entrypoint_sha256' in (out.get('provenance') or {}):
        out['provenance']['entrypoint_sha256'] = _sha(HERE / 'agent_graph.py')
    return out


def counter_settings(graph):
    """{family: {NAME: JSON value}} of the rival_counter node ({} when absent or off)."""
    node = _chain_nodes(graph).get('rival_counter')
    if not node or not node.get('enabled', False):
        return {}
    return {f: dict(sorted(v.items())) for f, v in sorted((node.get('counters') or {}).items()) if v}


def engine_name(graph):
    backbone = next((n for n in graph['turn']['nodes'] if n['id'] == 'backbone'), {})
    return backbone.get('engine', graph_runtime.DEFAULT_ENGINE)


def engine_catalog(graph):
    """{name: {default, type, note}} of the graph engine's editable constants ({} for Mohui)."""
    name = engine_name(graph)
    if name == graph_runtime.DEFAULT_ENGINE:
        return {}
    if name not in _ENGINE_CATALOGS:
        meta = json.loads((ENGINES_DIR / name / 'SOURCE.json').read_text())
        source = (ENGINES_DIR / name / 'agent' / meta['entry']).read_text(encoding='utf-8')
        _ENGINE_CATALOGS[name] = {k: v for k, v in engines.catalog(source).items() if v['used']}
    return _ENGINE_CATALOGS[name]


def channels(graph):
    """Effective channel flags (a graph without the oracle_guard node cannot enable it)."""
    nodes = {n['id']: n for n in graph['turn']['nodes']}
    out = {key: nodes[key].get('enabled', default) for key, default in CHANNEL_DEFAULTS.items() if key in nodes}
    return out


def _inert(graph, spec_stages):
    """Whether settings read only in `spec_stages` cannot change play under the graph's channels."""
    on = channels(graph)
    for stage in spec_stages:
        if stage in MARKET_IDS and on.get('market'):
            return False
        if stage in ('farmer', 'hands') and on.get(stage):
            return False
        if stage == 'state' and (on.get('farmer') or on.get('hands') or on.get('market')):
            return False
        if stage == 'oracle_guard' and on.get('oracle_guard'):
            return False
        if stage == 'rival_counter' and on.get('rival_counter'):
            return False
        if stage == 'rival_emulator' and on.get('rival_emulator'):
            return False
        if stage == 'tactic' and on.get('tactic'):
            return False
        if stage == 'land_plot' and on.get('land_plot'):
            return False
        if stage == 'kad_copilot' and on.get('kad_copilot'):
            return False
    return True


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
    out = {'parameters': dict(sorted(params.items())), 'stages': dict(sorted(stages.items())),
           'dispatch_order': order, 'surgical': surgical_settings}
    engine = engine_name(graph)
    if engine != graph_runtime.DEFAULT_ENGINE:
        spec = engine_catalog(graph)
        backbone = _chain_nodes(graph)['backbone']
        values = {}
        for name, value in (backbone.get('engine_parameters') or {}).items():
            if name in spec and engines.to_json(engines.from_json(value, spec[name]['default'], name)) != \
                    engines.to_json(spec[name]['default']):
                values[name] = engines.to_json(engines.from_json(value, spec[name]['default'], name))
        flags = {k: v for k, v in channels(graph).items() if v != CHANNEL_DEFAULTS[k]}
        switches = experimental.settings((graph.get('experimental') or {}).get('switches'))
        out.update(engine=engine, engine_parameters=dict(sorted(values.items())), channels=dict(sorted(flags.items())),
                   experimental={k: v for k, v in sorted(switches.items()) if v})
        out['stages'] = {k: v for k, v in out['stages'].items() if k in MARKET_IDS}
        counters = counter_settings(graph)
        if counters:            # only graphs that use the stage get the key (earlier settings keys stay valid)
            out['counters'] = counters
        code = tactic_code(graph)
        if code:
            out['tactic'] = code
    return out


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
    flags = edit.get('channels') or {}
    if not isinstance(flags, dict):
        raise ValueError('edit.channels must be an object {channel: bool}')
    if engine_name(out) != graph_runtime.DEFAULT_ENGINE:   # optional stages first: their parameters live on them
        if flags.get('rival_counter') is True and 'rival_counter' not in nodes:
            _insert_rival_node(out)
        if flags.get('rival_emulator') is True and 'rival_emulator' not in nodes:
            _insert_emulator_node(out)
        if flags.get('tactic') is True and 'tactic' not in nodes:
            _insert_tactic_node(out)
        if flags.get('land_plot') is True and 'land_plot' not in nodes:
            _insert_land_node(out)
        if flags.get('kad_copilot') is True and 'kad_copilot' not in nodes:
            _insert_kad_node(out)
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
        if not holders and spec['home'] not in nodes:
            raise ValueError(f"no effect with the current channels: {name} belongs to the {spec['home']} stage, which this "
                             f'graph does not have; switch it on in the same edit with "channels"')
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
    ladder = engine_name(out) != graph_runtime.DEFAULT_ENGINE
    if any(k in edit for k in ('engine_parameters', 'channels', 'experimental', 'counters')) and not ladder:
        raise ValueError('engine_parameters, channels, experimental and counters need a graph with a ladder engine')
    engine_params = edit.get('engine_parameters') or {}
    if not isinstance(engine_params, dict):
        raise ValueError('edit.engine_parameters must be an object {NAME: value}')
    if engine_params:
        spec = engine_catalog(out)
        holder = nodes['backbone'].setdefault('engine_parameters', {})
        for name, value in engine_params.items():
            if name not in spec:
                raise ValueError(f'unknown engine parameter {name}; editable ones are listed under ENGINE PARAMETERS')
            if value is None:
                holder.pop(name, None)
            else:
                holder[name] = engines.to_json(engines.from_json(value, spec[name]['default'], name))
        if not holder:
            nodes['backbone'].pop('engine_parameters')
    for channel, flag in flags.items():
        if channel not in CHANNEL_DEFAULTS or channel not in nodes:
            raise ValueError(f'unknown channel {channel}; channels are {sorted(set(CHANNEL_DEFAULTS) & set(nodes))}')
        if type(flag) is not bool:
            raise ValueError(f'channel {channel} must be true or false')
        if flag == CHANNEL_DEFAULTS[channel]:
            nodes[channel].pop('enabled', None)
        else:
            nodes[channel]['enabled'] = flag
    switches = edit.get('experimental') or {}
    if not isinstance(switches, dict):
        raise ValueError('edit.experimental must be an object {switch: bool}')
    if switches:
        current = dict((out.get('experimental') or {}).get('switches') or {})
        current.update(switches)
        experimental.settings(current)   # unknown switches, types and dependencies fail here
        out.setdefault('experimental', {})['switches'] = {k: v for k, v in current.items() if v}
    counters = edit.get('counters') or {}
    if not isinstance(counters, dict):
        raise ValueError('edit.counters must be an object {family: {NAME: value or null} or null}')
    if counters:
        node = _chain_nodes(out).get('rival_counter')
        if node is None:
            raise ValueError('counters need the rival_counter stage: add "channels": {"rival_counter": true} '
                             'to the same edit')
        families = rival_families(out)
        switchable = switchable_engine_parameters(out)
        spec = engine_catalog(out)
        table = copy.deepcopy(node.get('counters') or {})
        for family, values in counters.items():
            if family not in families:
                raise ValueError(f'unknown rival family {family!r}; families are {sorted(families)}')
            if values is None:
                table.pop(family, None)
                continue
            if not isinstance(values, dict) or not values:
                raise ValueError(f'counters of {family} must be an object {{NAME: value or null}}')
            entry = table.setdefault(family, {})
            for name, value in values.items():
                if name in oracle_guard.PARAMETERS:
                    typed = None if value is None else _json_value(
                        graph_runtime._like(value, oracle_guard.PARAMETERS[name], name))
                elif name in switchable and name in spec:
                    typed = None if value is None else engines.to_json(
                        engines.from_json(value, spec[name]['default'], name))
                else:
                    raise ValueError(f'{name} cannot be set per rival family; allowed are the _OG_* guard parameters '
                                     f'and these engine parameters: {sorted(switchable & set(spec))}')
                if typed is None:
                    entry.pop(name, None)
                else:
                    entry[name] = typed
            if not entry:
                table.pop(family)
        if table:
            node['counters'] = table
        else:
            node.pop('counters', None)
    if 'tactic' in edit:
        node = _chain_nodes(out).get('tactic')
        if node is None:
            raise ValueError('a tactic needs the tactic stage: add "channels": {"tactic": true} to the same edit')
        if edit['tactic'] is None:
            node.pop('code', None)
            node['enabled'] = False
        else:
            try:
                tactic.check(edit['tactic'])
            except tactic.TacticError as exc:
                raise ValueError(f'tactic refused: {exc}') from None
            node['code'] = edit['tactic']
    if (_chain_nodes(out).get('tactic') or {}).get('enabled', False) and not tactic_code(out):
        raise ValueError('the tactic stage needs code: give it under "tactic" in the same edit')
    for name in emulated_engines(out):
        if not (ENGINES_DIR / name / 'SOURCE.json').exists():
            raise ValueError(f'_EM_ENGINES: no engine {name!r}; engines are '
                             f"{sorted(p.name for p in ENGINES_DIR.iterdir() if (p / 'SOURCE.json').exists())}")
    if 'rival_emulator' in _chain_nodes(out):
        _pin_emulator(out)
    kad = _chain_nodes(out).get('kad_copilot')
    if kad is not None and kad.get('enabled', False):
        try:
            kad_copilot.check({k: v for k, v in (kad.get('parameters') or {}).items() if k in kad_copilot.PARAMETERS})
        except ValueError as exc:
            raise ValueError(f'kad_copilot refused: {exc}') from None
    plot = _chain_nodes(out).get('land_plot')
    if plot is not None and plot.get('enabled', False):
        try:
            land_plot.check({k: v for k, v in (plot.get('parameters') or {}).items() if k in land_plot.PARAMETERS})
        except ValueError as exc:
            raise ValueError(f'land_plot refused: {exc}') from None
    if ladder:
        inert = [name for name in params if params[name] is not None
                 and _inert(out, constants[name]['stages'])]
        inert += [f'stage {k}' for k in stage_flags if _inert(out, [k])]
        if 'dispatch_order' in edit and _inert(out, ['routine_dispatch']):
            inert.append('dispatch_order')
        if 'surgical' in edit and _inert(out, ['market_setup', 'farmer']):
            inert.append('surgical')
        if counters and _inert(out, ['rival_counter']):
            inert.append('counters')
        if inert:
            raise ValueError(f"no effect with the current channels: {', '.join(inert)} (their channel is off; "
                             'switch it on in the same edit with "channels", or edit engine_parameters)')
    if settings_key(out, constants) == settings_key(graph, constants):
        raise ValueError('edit changes nothing: every value equals the current setting')
    return out


MIGRATION_KEYS = ('parameters', 'stages', 'engine_parameters', 'channels', 'experimental', 'counters')
_RESET = {'parameters': None, 'engine_parameters': None, 'stages': True, 'experimental': False, 'counters': None}
_ABSENT = object()


def migration_edit(recipient, donor, seed, constants=None):
    """The edit that gives `recipient` what `donor` changed relative to `seed` (island mixing).

    A setting the recipient changed itself keeps the recipient's value; a setting the donor put
    back to its default is reset. None when the donor has nothing the recipient lacks.
    """
    constants = constants or catalog()
    start, given, own = (settings(g, constants) for g in (seed, donor, recipient))
    edit = {}
    for key in MIGRATION_KEYS:
        s, d, r = (x.get(key) or {} for x in (start, given, own))
        changed = {}
        for name in sorted(set(s) | set(d)):
            before = s.get(name, _ABSENT)
            if d.get(name, _ABSENT) == before or r.get(name, _ABSENT) != before:
                continue
            if name in d:
                changed[name] = d[name]
            else:
                changed[name] = CHANNEL_DEFAULTS[name] if key == 'channels' else _RESET[key]
        if changed:
            edit[key] = changed
    if given['dispatch_order'] != start['dispatch_order'] and own['dispatch_order'] == start['dispatch_order']:
        edit['dispatch_order'] = given['dispatch_order']
    if given.get('tactic') != start.get('tactic') and own.get('tactic') == start.get('tactic'):
        edit['tactic'] = given.get('tactic')
        if given.get('tactic'):
            edit.setdefault('channels', {})['tactic'] = True
    return edit or None


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
    if 'engine' in a or 'engine' in b:
        spec = engine_catalog(after)
        ea, eb = a.get('engine_parameters', {}), b.get('engine_parameters', {})
        for name in sorted(set(ea) | set(eb)):
            old = ea.get(name, engines.to_json(spec[name]['default']) if name in spec else None)
            new = eb.get(name, engines.to_json(spec[name]['default']) if name in spec else None)
            if old != new:
                lines.append(f"engine {name}: {json.dumps(old)[:200]} -> {json.dumps(new)[:200]}")
        ca, cb = channels(before), channels(after)
        for channel in CHANNEL_DEFAULTS:
            if ca.get(channel) != cb.get(channel):
                lines.append(f"channel {channel}: {'on' if ca.get(channel) else 'off'} -> {'on' if cb.get(channel) else 'off'}")
        xa, xb = a.get('experimental', {}), b.get('experimental', {})
        for switch in sorted(set(xa) | set(xb)):
            if xa.get(switch, False) != xb.get(switch, False):
                lines.append(f"experimental {switch}: {xa.get(switch, False)} -> {xb.get(switch, False)}")
        if a.get('tactic') != b.get('tactic'):
            def summary(code):
                if not code:
                    return 'none'
                return f"{len(code.splitlines())} lines, sha {hashlib.sha256(code.encode()).hexdigest()[:8]}"
            lines.append(f"tactic: {summary(a.get('tactic'))} -> {summary(b.get('tactic'))}")
        ra, rb = a.get('counters', {}), b.get('counters', {})
        for family in sorted(set(ra) | set(rb)):
            fa, fb = ra.get(family, {}), rb.get(family, {})
            for name in sorted(set(fa) | set(fb)):
                if fa.get(name) != fb.get(name):
                    lines.append(f"counter vs {family}: {name} {json.dumps(fa.get(name, 'default'))} -> "
                                 f"{json.dumps(fb.get(name, 'default'))}")
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
    hazel_rows = [r for r in rows if not r.startswith(('_OG_', '_EM_', '_LP_'))]
    if 'engine' in current:
        spec = engine_catalog(graph)
        on = channels(graph)
        rows = [f"PRODUCTION ENGINE: {current['engine']} (a public ladder agent; everything it does is set by "
                'the ENGINE PARAMETERS below, and our layers act on its action only where a channel is on).',
                '',
                'CHANNELS (on/off): ' + ', '.join(f'{k}={"on" if v else "off"}' for k, v in on.items()),
                '  farmer/hands on = our idle-unit rescues edit the engine\'s unit actions; market on = our full '
                'market pipeline (the PARAMETERS and MARKET STAGES below) rewrites the engine\'s orders; '
                'oracle_guard on = only the predictor front-run (the _OG_* parameters) adds sells into the '
                "engine's empty order slots. Off = the engine's own action passes through.",
                '',
                'ENGINE PARAMETERS (name | type | current | engine default | note):']
        values = current.get('engine_parameters', {})
        for name, item in sorted(spec.items(), key=lambda kv: kv[1]['lineno']):
            value = values.get(name, engines.to_json(item['default']))
            rows.append(f"{name} | {item['type']} | {json.dumps(value)[:300]} | "
                        f"{json.dumps(engines.to_json(item['default']))[:300]} | {item['note'][:160]}")
        rows += ['', 'EXPERIMENTAL SWITCHES (need a channel on to matter; capacity_liquidation and '
                 'opponent_pressure need order_arbitration): ' + ', '.join(
                     f"{k}={current['experimental'].get(k, False)}" for k in experimental.SWITCHES), '']
        guard = [(n, c) for n, c in constants.items() if c['home'] == 'oracle_guard']
        rows.append('ORACLE GUARD PARAMETERS (edit them under the key "parameters", e.g. {"parameters": '
                    '{"_OG_SCORE": 0.55}}; they are not engine parameters) (name | type | current | default | note):')
        for name, c in guard:
            rows.append(f"{name} | {c['type']} | {json.dumps(_json_value(current['parameters'].get(name, c['default'])))} | "
                        f"{json.dumps(_json_value(c['default']))} | {c['note']}")
        families = rival_families(graph)
        if families:
            switch = sorted(switchable_engine_parameters(graph) & set(spec))
            rows += ['', f"RIVAL COUNTER (channel rival_counter={'on' if on.get('rival_counter') else 'off'}; "
                     'an edit that switches it on inserts the stage). Every turn it compares the rival\'s public farm '
                     '(money, hands, land, crops, herd) with ours: on the same seed a copy of our engine plays exactly '
                     'as we do. By step 93 it puts the rival in one class below; from then on that class\'s counters '
                     'apply (engine constants and _OG_* guard parameters). Classes without counters, and every rival '
                     'before its class is known, play the graph\'s own settings.',
                     'RIVAL CLASSES (decided by step 93, then fixed for the game):'] + [f'  {f}: {d}' for f, d in families.items()] + [
                     'ENGINE PARAMETERS A COUNTER MAY SET (read at call time): ' + ', '.join(switch),
                     'CURRENT COUNTERS: ' + json.dumps(current.get('counters', {})),
                     'Edit example: {"channels": {"rival_counter": true}, "counters": {"mirror": {"_EV_H": 12, '
                     '"_DP_H": 12}}}; {"counters": {"nsell_opener": null}} removes a class\'s counters.']
        emulator = [(n, c) for n, c in constants.items() if c['home'] == 'rival_emulator']
        rows += ['', f"RIVAL EMULATOR (channel rival_emulator={'on' if on.get('rival_emulator') else 'off'}; an edit "
                 'that switches it on inserts the stage). It runs the public engines of _EM_ENGINES in the rival\'s place '
                 'from our own view: every observation shows both farms in full, the rival\'s private stock is rebuilt '
                 'with the environment\'s rules, and an engine that disagrees with an observation is dropped. Five ladder '
                 'rivals that ran a public engine were emulated exactly for all 719 moves. While one engine matches '
                 '(after _EM_LOCK steps) the rival counter uses the family engine:<name> (e.g. {"channels": '
                 '{"rival_emulator": true, "rival_counter": true}, "counters": {"engine:tetsutani_demand": {"_EV_H": 12}}}), '
                 'and with _EM_RACE our market orders are rearranged against the rival\'s known orders of the turn (same '
                 'orders, the slots that earn us the most; nothing bought or sold changes).',
                 'RIVAL EMULATOR PARAMETERS (edit them under "parameters") (name | type | current | default | note):']
        for name, c in emulator:
            rows.append(f"{name} | {c['type']} | {json.dumps(_json_value(current['parameters'].get(name, c['default'])))} | "
                        f"{json.dumps(_json_value(c['default']))} | {c['note']}")
        plot = [(n, c) for n, c in constants.items() if c['home'] == 'land_plot']
        rows += ['', f"LAND PLOT (channel land_plot={'on' if on.get('land_plot') else 'off'}; an edit that switches it on "
                 'inserts the stage). The engine\'s tapes buy the second quadrant on day 6 and the third on day 11 in every '
                 'game and never the fourth; no engine parameter moves them. This stage is the lever on land: from day '
                 '_LP_DAY it buys the fourth quadrant (SE, $4,000; SW first, +$2,000, when the tape has not bought it yet), '
                 'hires _LP_WORKERS hands every day after the engine\'s own hires (the k-th extra hire of a day costs '
                 'fib(hires so far): with the tape\'s 9-11 hands, $55-233 each) and farms _LP_TILES SE tiles with _LP_USE: '
                 'a crop (planted, watered daily, harvested at peak, replanted until _LP_LAST_PLANT_DAY) or GOOSE (one coop '
                 'and one $300 goose per tile, fed a wheat a day, cared for: about 2 eggs a day each from 4 days after '
                 'placing, plus 1 fertilizer a day). It buys its own seeds, geese and feed and sells its produce. Edit '
                 'example: {"channels": {"land_plot": true}, "parameters": {"_LP_USE": "GOOSE", "_LP_TILES": 8, '
                 '"_LP_DAY": 12}}.',
                 'LAND PLOT PARAMETERS (edit them under "parameters") (name | type | current | default | note):']
        for name, c in plot:
            rows.append(f"{name} | {c['type']} | {json.dumps(_json_value(current['parameters'].get(name, c['default'])))} | "
                        f"{json.dumps(_json_value(c['default']))} | {c['note']}")
        kad = [(n, c) for n, c in constants.items() if c['home'] == 'kad_copilot']
        rows += ['', f"KAD COPILOT (channel kad_copilot={'on' if on.get('kad_copilot') else 'off'}). KAD-HP-1 is a "
                 'policy network trained on tens of thousands of top-player replays from Kaggle\'s daily datasets (it '
                 'predicts, for our exact state, the market orders and each unit\'s next job that a top team would '
                 'choose). Alone it loses to the ladder bots (its whole-market-order accuracy is 40%, per-slot order 89%, '
                 'job 85%); as a copilot on top of our engine it is a second opinion. Every _KC_EVERY turns from '
                 '_KC_FROM_STEP to _KC_TO_STEP it runs once (about 0.15 s on a core, 0.4 s on Kaggle, whose budget is 1 s '
                 'a turn) and two levers act on its advice: _KC_SELL appends its confident SELL orders (probability >= '
                 '_KC_SELL_P at its slot, products in _KC_SELL_ITEMS, at most _KC_MAX_ORDERS a turn, never into the '
                 'engine\'s deliberate empty [] slots), and _KC_HANDS lets a hand the engine left on PASS do KAD\'s job '
                 'where it stands (WATER an unwatered crop, HARVEST a ripe tomato/strawberry or an animal). A tactic '
                 'reads the same advice as info[\'kad\'] and can use it in any way you write.',
                 'KAD COPILOT PARAMETERS (edit them under "parameters") (name | type | current | default | note):']
        for name, c in kad:
            rows.append(f"{name} | {c['type']} | {json.dumps(_json_value(current['parameters'].get(name, c['default'])))} | "
                        f"{json.dumps(_json_value(c['default']))} | {c['note']}")
        code = current.get('tactic')
        rows += ['', f"TACTIC (channel tactic={'on' if on.get('tactic') else 'off'}; an edit with \"channels\": {{\"tactic\": "
                 'true} and "tactic": "<python source>" inserts the stage; "tactic": null removes it). Python you write, run '
                 'in a sandbox on our action every turn after every other stage and before sanitize (which enforces '
                 'legality; the rival emulator may then reorder our market orders). Define exactly '
                 'def tactic(obs, action, memory, info) and return the action (or None to keep it).',
                 "  obs: the observation as a plain dict: obs['step'] (0-719, 24 steps a day, obs['day'], obs['hour']), "
                 "obs['player'] (our seat; the rival is 1 - player), obs['farms'][seat] with money, tiles[y][x] (None, "
                 "'LOCKED', or a dict: kind PLANT/COOP/PASTURE/WEED, crop, animal, yield_units, watered_today, fed_today, "
                 "planted_day, ...), farmer [x, y], hands [[x, y], ...], unlocked_quadrants, hires_today; obs['market'] "
                 "('prices' and 'inventory' per product), obs['town'] ('unlocked_shops'), obs['private'] (our 'shed' "
                 "{item: n}, 'seeds' {crop: n}, 'inventories' [one {item: n} per worker]). Both farms are public.",
                 "  action: {'farmer': [op, ...], 'hands': [[op, ...], ...], 'market': [order, ...]} as the engine and "
                 "our layers left it. Orders: ['SELL', item, n], ['BUY_PRODUCT', 'WHEAT' or 'FERTILIZER', n], "
                 "['BUY_SEED', crop, n], ['BUY_ANIMAL', animal, n], ['HIRE'], ['BUY_LAND']; [] keeps a slot empty; at "
                 'most 10 slots. Both seats\' lists clear index by index and the two orders at one index unit by unit, so '
                 'a sale in an earlier slot than the rival\'s sale of the same product gets the better prices.',
                 "  memory: a dict kept for the whole game (empty at step 0). info: {'rival_family': the rival "
                 "counter's class or None, 'rival_engine': the public engine the rival emulator matched or None, "
                 "'rival_action': the rival's action for this very turn when the emulator knows it or None, "
                 "'forecast': the predictor's forecast, {'score_k' and 'units_k': {product: value}} for k in 1, 4, 24, "
                 "96, when our channels run it, or None, 'kad': KAD's latest advice when the kad_copilot stage is on, "
                 "else None: {'step': the turn it was made (it runs every _KC_EVERY turns), 'orders': [[op, item, "
                 "amount, probability], ...] the market orders a top player would place now (amount 101 = all), "
                 "'buy_land_p': P(BUY_LAND), 'units': [{'unit': 0 farmer / k hand k-1, 'job', 'item', 'target': [x, y], "
                 "'now': NORTH/SOUTH/EAST/WEST/PASS/DO, 'p'}, ...]}}.",
                 '  Sandbox: no imports (math\'s float functions are there as math: sqrt, log, exp, floor, ceil, ...; no pow, factorial or comb), no classes, no global/nonlocal, no attributes '
                 'starting with _, no str.format, no print/open/eval/getattr; loops, comprehensions and calls spend '
                 'ticks, at most 200,000 per turn and 0.25 s. A turn that raises or runs out of budget keeps the action; '
                 'after 20 of them the tactic is off for the game. Keep it small and targeted (a few dozen lines): act '
                 'only in the situations you mean to change and leave the action alone otherwise, so the gauntlet\'s '
                 'changed games show the effect. The engine\'s own logic cannot be edited, only the action it produced.',
                 'CURRENT TACTIC CODE: ' + ('none' if not code else '\n' + code)]
        hazel = [r for r in hazel_rows if r and not r.startswith('_RC_')]
        rows += ['', 'OUR MARKET-PIPELINE CONTROLS (only matter with the market channel on; the farmer/hands '
                 'rescue constants only with those channels on):'] + hazel
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
last = {}
def agent(obs, config):
    last['obs'] = obs
    last['action'] = agent_graph.agent(obs, config)
    return last['action']
env = make('kaggriculture', configuration={'seed': seed, 'episodeSteps': steps}, debug=False)
env.run([agent, lambda obs, config: {}])
statuses = [s.get('status') for s in env.steps[-1]]
report = {'statuses': statuses, 'fallbacks': engine.fallback_count, 'last_error': engine.last_error,
          'steps': len(env.steps)}
t = getattr(engine, 'tactic', None)
if t is not None and last:
    report.update(tactic_calls=t.calls, tactic_errors=t.errors, tactic_error=t.last_error, tactic_probes=[])
    base = json.loads(json.dumps(last['obs']))
    for i, step in enumerate(PROBE_STEPS):
        info = engine.tactic_info(None)
        if i % 2:
            info['rival_action'] = last['action']
        before = t.errors
        t.apply(dict(base, step=step, day=step // 24, hour=step % 24), last['action'], info)
        if t.errors > before:
            report['tactic_probes'].append({'step': step, 'error': t.last_error})
print(json.dumps(report))
'''
PROBE_STEPS = (96, 240, 480, 600, 700, 712, 719)


def validate_graph(graph_path, steps=30, seed=20260925, python=None, timeout=600):
    """Build the runtime for a graph and play its first turns in a fresh interpreter.

    Raises ValueError with the child's error text (fed back to the mutating LLM) when
    the runtime rejects the graph, a stage falls back to the backbone, or play fails. A
    tactic must also run without an error on every turn of that game and when called at
    the PROBE_STEPS on the game's last observation (moved to those steps), so a tactic's
    late-game code is exercised too.
    """
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               CUDA_VISIBLE_DEVICES='', KAGG_ORACLE_BACKEND='numpy', KAGG_ORACLE_DEVICE='cpu',
               PYTHONDONTWRITEBYTECODE='1')
    env.pop('KAGG_GRAPH_EXPERIMENTS', None)
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as tmp:
        # the pins re-hashed as write_graph_bundle does, so the graph is checked against this runtime
        tmp.write(json.dumps(repinned(json.loads(Path(graph_path).read_text()))))
    try:
        proc = subprocess.run([python or sys.executable, '-c', _VALIDATE.replace('PROBE_STEPS', repr(PROBE_STEPS)),
                               tmp.name, str(steps), str(seed)],
                              cwd=HERE, env=env, capture_output=True, text=True, timeout=timeout)
    finally:
        os.unlink(tmp.name)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-6:]
        raise ValueError('runtime rejected the graph: ' + ' | '.join(tail)[:1500])
    report = json.loads(proc.stdout.strip().splitlines()[-1])
    if report['fallbacks']:
        raise ValueError(f"a graph stage failed and fell back to the backbone {report['fallbacks']} time(s) "
                         f"in the first {steps} turns: {report['last_error']}")
    if report['statuses'] != ['DONE', 'DONE']:
        raise ValueError(f"validation game ended with statuses {report['statuses']}")
    if report.get('tactic_errors'):
        raise ValueError(f"the tactic failed on {report['tactic_errors']} of the {report['tactic_calls']} turns of "
                         f"the validation game (each such turn keeps the action): {report['tactic_error']}")
    if report.get('tactic_probes'):
        probe = report['tactic_probes'][0]
        raise ValueError(f"the tactic failed when called at step {probe['step']} (the validation game's last "
                         f"observation with its step, day and hour moved there; the odd probes also pass our "
                         f"action as info['rival_action']). Check late-game branches, missing keys and the "
                         f"budget: {probe['error']}")
    return report
