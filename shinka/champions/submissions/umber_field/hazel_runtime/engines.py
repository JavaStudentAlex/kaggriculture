"""Public ladder agents as the graph's production engine (the `backbone` turn node's `engine`).

An engine lives in hazel_runtime/engines/<name>/: the agent's files exactly as recovered from
its public notebook (shinka/champions/ladder/<name>/agent/, Apache-2.0 notices inside) under
agent/, and SOURCE.json (notebook, entry file, sha256 of every file). It is loaded the way
kaggle_environments loads a submission: the entry file's code runs in a fresh namespace and
the last callable it defines is the agent, called with (observation, configuration) cut to
its argument count.

Graph `engine_parameters` give new values to the engine's top-level constants: every
`NAME = <literal>` assignment at module level, assigned once, whose name is constant-like and
whose value is a bool, number, string or a small list/tuple/dict of them (state holders such
as *_REPORT, *_STATES or empty containers are not parameters). The value is written into the
source before it runs, so tables and objects built while the module loads (a chassis made
from _SETTINGS, routes keyed by shops) see it. JSON cannot key an object by a tuple: a
tuple-keyed table (e.g. shop pair -> route) takes keys "SHOP_A|SHOP_B".
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path

from kaggle_environments.agent import get_last_callable
from kaggle_environments.utils import structify

_CONSTANT = re.compile(r'_?[A-Z][A-Z0-9_]*')
# Constants of the V5x ladder lineage that restate the game (prices, costs, moves, what each shop
# buys, animal cycles, tile coordinates) or hold encoded data: editing them breaks the agent's model
# of the engine instead of changing its policy, so they are not parameters.
FACTS = frozenset({
    'PRODUCTS', 'SEED_PRICE', 'ANIMAL_COST', 'ANIMAL_STRUCTURE', 'LAND_PRICES', 'MOVES', 'PASS_ACTION',
    'LAST_ACT_STEP', 'DEFAULT_SETTINGS', 'MAX_ORDERS', '_UNIT_NS', '_R37_MARKET_PARAMS', '_R37_PRICE_FLOOR',
    '_V92_P_BLOB', '_V92_P_INDEX', '_SL_TILES', '_VT_TILES', '_VE_SEED', '_VE_ANIMAL', '_CA_MOVES', '_CH_MOVES',
    '_CS_MOVES', '_OR2_ANIMAL', '_OR2_SHOPS', '_OR2_ITEMS', '_OR2_ONGOING', '_V9_SHOP_ITEMS', '_RACE_SHOPS',
    '_HD2_SPEC', '_HD2_SHOP_TYPES', '_Y_PRODUCT', '_Y_STRUCT', '_Y_COST', '_Y_SEED', '_V9_HERD_PRODUCT',
    '_V9_COURIER_ACCESS', '_V93_ROUTE_BY_RIVAL', '_CH_ANIMALS', '_R88_ANIMAL_DAYS', '_SR_PRODUCTS',
    '_V7_SM_PRODUCTS', '_V7_SM_ANIMAL', '_V7_SM_LAND', '_CA_CROP', '_SM_PRODUCTS', '_SM_ANIMAL', '_SM_LAND',
})
_STATE = re.compile(r'(REPORT|STATS|STATES|STATE|CACHE|PROBES|PREVIOUS|PLANS|LAST|COMBINED|ERRORS)$')
_MAX_REPR = 4000


def _literal(node):
    try:
        return True, ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return False, None


def _parameter_value(value):
    """Whether a literal can be a graph parameter (and is not a state container)."""
    if isinstance(value, bool) or isinstance(value, int):
        return True
    if isinstance(value, str):
        return len(value) <= 200
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, (list, tuple, dict)):
        return 0 < len(value) and len(repr(value)) <= _MAX_REPR
    return False


def _comment(lines, lineno):
    line = lines[lineno - 1]
    if '#' in line and not line.lstrip().startswith('#'):
        return line.split('#', 1)[1].strip()[:200]
    above, i = [], lineno - 2
    while i >= 0 and lines[i].strip().startswith('#') and len(above) < 2:
        above.insert(0, lines[i].strip().lstrip('#').strip())
        i -= 1
    return ' '.join(above)[:200]


def catalog(source):
    """{name: {default, type, note, used, lineno}} of an engine source's parameters."""
    tree = ast.parse(source)
    lines = source.splitlines()
    counts = {}
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(
            node, (ast.AugAssign, ast.AnnAssign)) else []
        for target in targets:
            for name in ast.walk(target):
                if isinstance(name, ast.Name):
                    counts[name.id] = counts.get(name.id, 0) + 1
    rebound = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Global):
            rebound |= set(node.names)
    shared = set()   # lines holding more than one top-level statement (a = 1; b = 2)
    seen = {}
    for node in tree.body:
        for line in range(node.lineno, node.end_lineno + 1):
            if line in seen:
                shared.add(line)
            seen[line] = node
    out = {}
    for node in tree.body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)):
            continue
        name = node.targets[0].id
        if (not _CONSTANT.fullmatch(name) or name in FACTS or _STATE.search(name) or counts.get(name) != 1
                or name in rebound or any(l in shared for l in range(node.lineno, node.end_lineno + 1))):
            continue
        ok, value = _literal(node.value)
        if not ok or not _parameter_value(value):
            continue
        pattern = re.compile(rf'\b{re.escape(name)}\b')
        used = any(pattern.search(text) for i, text in enumerate(lines, 1) if i != node.lineno)
        out[name] = {'default': value, 'type': type(value).__name__, 'note': _comment(lines, node.lineno),
                     'used': used, 'lineno': node.lineno}
    return out


def switchable(source):
    """Parameters that can change during a game (the rival_counter stage sets them per rival family):
    read only inside function bodies, at call time. A name also read at module level, in a default
    argument, a decorator or a class body takes its value when the module loads, so it is left out."""
    constants = catalog(source)
    load_time, call_time = set(), set()

    def visit(node, in_body):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            (call_time if in_body else load_time).add(node.id)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            for d in getattr(node, 'decorator_list', []) + node.args.defaults + node.args.kw_defaults:
                if d is not None:
                    visit(d, in_body)
            for stmt in (node.body if isinstance(node.body, list) else [node.body]):
                visit(stmt, True)
            return
        for child in ast.iter_child_nodes(node):
            visit(child, in_body)

    for node in ast.parse(source).body:
        visit(node, False)
    return {n for n, c in constants.items() if c['used'] and n in call_time and n not in load_time}


def to_json(value):
    """A parameter value as JSON (tuples -> lists, tuple keys -> 'A|B')."""
    if isinstance(value, dict):
        return {('|'.join(map(str, k)) if isinstance(k, tuple) else k): to_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json(v) for v in value]
    return value


def from_json(value, template, name):
    """A JSON value in the type of the engine constant `template`; ValueError otherwise."""
    def fail():
        raise ValueError(f'engine parameter {name}: {json.dumps(value)[:120]} does not match the type of '
                         f'{json.dumps(to_json(template))[:120]}')
    if isinstance(template, bool):
        if type(value) is not bool:
            fail()
        return value
    if isinstance(template, int):
        if type(value) is not int:
            fail()
        return value
    if isinstance(template, float):
        if type(value) not in (int, float) or not math.isfinite(value):
            fail()
        return float(value)
    if isinstance(template, str):
        if not isinstance(value, str):
            fail()
        return value
    if isinstance(template, (list, tuple)):
        if not isinstance(value, list):
            fail()
        items = [from_json(v, template[0], name) for v in value] if template else list(value)
        return tuple(items) if isinstance(template, tuple) else items
    if isinstance(template, dict):
        if not isinstance(value, dict) or not template:
            fail()
        key0, val0 = next(iter(template.items()))
        out = {}
        for key, item in value.items():
            if isinstance(key0, tuple):
                parts = tuple(key.split('|'))
                if len(parts) != len(key0):
                    fail()
                k = parts
            elif isinstance(key0, int) and not isinstance(key0, bool):
                try:
                    k = int(key)
                except ValueError:
                    fail()
            else:
                k = key
            out[k] = from_json(item, val0, name)
        return out
    fail()


def rewrite(source, parameters, constants):
    """The engine source with each parameter's literal replaced (one line per replacement kept
    at its place, so line numbers in tracebacks stay meaningful)."""
    lines = source.splitlines(keepends=True)
    tree = ast.parse(source)
    spans = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in parameters:
                spans[node.targets[0].id] = (node.lineno, node.end_lineno)
    for name, value in sorted(parameters.items(), key=lambda kv: -spans[kv[0]][0]):
        if name not in constants or name not in spans:
            raise ValueError(f'unknown engine parameter {name}')
        start, end = spans[name]
        indent = re.match(r'\s*', lines[start - 1]).group(0)
        text = f'{indent}{name} = {repr(from_json(to_json(value), constants[name]["default"], name))}\n'
        lines[start - 1:end] = [text] + ['\n'] * (end - start)
    return ''.join(lines)


class Engine:
    """One loaded engine: `agent(obs, configuration)` and its effective parameters."""

    def __init__(self, directory, parameters=None):
        self.directory = Path(directory).resolve()
        meta = json.loads((self.directory / 'SOURCE.json').read_text())
        self.name, self.entry = meta['name'], self.directory / 'agent' / meta['entry']
        for rel, digest in meta['files'].items():
            if hashlib.sha256((self.directory / 'agent' / rel).read_bytes()).hexdigest() != digest:
                raise ValueError(f'engine {self.name}: {rel} differs from SOURCE.json')
        source = self.entry.read_text(encoding='utf-8')
        self.source = source
        self.constants = catalog(source)
        self.parameters = {}
        for name, value in (parameters or {}).items():
            spec = self.constants.get(name)
            if spec is None:
                raise ValueError(f'unknown engine parameter {name} for {self.name}')
            if not spec['used']:
                raise ValueError(f'engine parameter {name} is never read by {self.name}')
            self.parameters[name] = from_json(value, spec['default'], name)
        if self.parameters:
            source = rewrite(source, self.parameters, self.constants)
        sys.path.insert(0, str(self.entry.parent))
        cwd = os.getcwd()
        os.chdir(self.entry.parent)
        try:
            self.policy = get_last_callable(source, path=str(self.entry))
        finally:
            os.chdir(cwd)
        self.nargs = self.policy.__code__.co_argcount if hasattr(self.policy, '__code__') else 2
        # the module namespace the engine's functions read their constants from (rival_counter)
        self.namespace = getattr(self.policy, '__globals__', None)

    def agent(self, obs, configuration=None):
        return self.policy(*[structify(obs), structify(configuration)][:self.nargs])
