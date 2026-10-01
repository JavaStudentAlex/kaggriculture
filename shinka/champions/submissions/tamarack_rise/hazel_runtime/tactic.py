"""The tactic stage: Python code the evolution writes, run on our action every turn.

The graph's optional `tactic` turn node (right before `sanitize`; off unless `enabled: true`) carries `code`, the
source of

    def tactic(obs, action, memory, info):
        ...
        return action          # or None to keep the action as it is

- obs: this turn's observation as a plain dict (step, player, farms, market, town, day, hour, private);
- action: our action so far, {'farmer': [...], 'hands': [[...], ...], 'market': [[...] or [], ...]} (a copy);
- memory: a dict kept for the whole game (empty at step 0);
- info: {'rival_family': the rival counter's class or None, 'rival_engine': the public engine the rival emulator
  matched or None, 'rival_action': the rival's action for this turn when the emulator knows it or None,
  'forecast': the predictor's forecast when our channels computed it or None}.

The code runs in a sandbox. No imports, no class definitions, no global/nonlocal, no attributes that start with an
underscore, format strings (str.format) or padding methods, no names that start with two, none of FORBIDDEN (open,
eval, exec, getattr, print, ...); the builtins in BUILTINS and the float functions of `math` (MATH) are there,
`range` is limited to RANGE_MAX values. Every loop iteration, comprehension element and function call spends a
tick; a turn may spend TICKS ticks and BUDGET_S seconds. The budget is checked at ticks, so the operations that can
run long in one step are checked before they run: `**`, `*` and `<<` go through _pow, _mul and _lshift (at most
MAX_BITS-bit integers, MAX_ITEMS-long repeated sequences). A tactic that raises, runs out of budget or returns
something that is not an action leaves the action unchanged; after MAX_ERRORS such turns in a game it stays off for
the rest of that game. `sanitize` then enforces legality as for any action.
"""
from __future__ import annotations

import ast
import builtins
import copy
import json
import math
import time
import types

BUILTINS = ('abs', 'all', 'any', 'bool', 'dict', 'divmod', 'enumerate', 'filter', 'float', 'frozenset', 'int',
            'isinstance', 'len', 'list', 'map', 'max', 'min', 'reversed', 'round', 'set', 'sorted', 'str',
            'sum', 'tuple', 'zip', 'Exception', 'ValueError', 'KeyError', 'IndexError', 'TypeError',
            'ZeroDivisionError', 'ArithmeticError', 'LookupError')
FORBIDDEN = {'open', 'eval', 'exec', 'compile', 'getattr', 'setattr', 'delattr', 'hasattr', 'globals', 'locals',
             'vars', 'input', 'print', 'breakpoint', 'help', 'dir', 'type', 'object', 'super', 'memoryview',
             'bytearray', 'bytes', 'classmethod', 'staticmethod', 'property', 'id', 'hash', 'iter', 'next',
             'callable', 'issubclass', 'format', 'repr', 'ascii', 'chr', 'ord', 'bin', 'hex', 'oct', 'exit', 'quit',
             'pow'}
MATH = ('ceil', 'floor', 'trunc', 'fabs', 'sqrt', 'exp', 'log', 'log2', 'log10', 'log1p', 'expm1', 'sin', 'cos',
        'tan', 'atan', 'atan2', 'tanh', 'hypot', 'isfinite', 'isinf', 'isnan', 'isclose', 'copysign', 'fmod',
        'inf', 'nan', 'pi', 'e')
TICKS = 200_000
BUDGET_S = 0.25
RANGE_MAX = 100_000
MAX_ERRORS = 20
MAX_CHARS = 20_000
ACTION_KEYS = ('farmer', 'hands', 'market')
ENTRY = 'tactic'
MAX_BITS = 4096
MAX_ITEMS = 100_000
HOOKS = ('_tick', '_tick_true', '_pow', '_mul', '_lshift')          # the budget's own names in the sandbox
BAD_ATTRIBUTES = ('format', 'format_map', 'ljust', 'rjust', 'center', 'zfill', 'expandtabs')


def _bad_name(name):
    return name.startswith('__') or name in HOOKS


class TacticError(ValueError):
    """Code the sandbox refuses."""


class TacticBudget(Exception):
    """A turn ran out of ticks or time."""


def check(source):
    """The code's syntax tree, or TacticError naming what is not allowed."""
    if not isinstance(source, str) or not source.strip():
        raise TacticError('the tactic code must be a non-empty string')
    if len(source) > MAX_CHARS:
        raise TacticError(f'the tactic code is longer than {MAX_CHARS} characters')
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise TacticError(f'syntax error in the tactic code: {exc}') from None
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise TacticError('imports are not allowed (math.sqrt, math.log, math.floor and the other float functions are there as `math`)')
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            raise TacticError('global and nonlocal are not allowed: keep state in `memory`')
        if isinstance(node, (ast.ClassDef, ast.AsyncFunctionDef, ast.Await, ast.AsyncFor, ast.AsyncWith, ast.Yield,
                             ast.YieldFrom, ast.With)):
            raise TacticError(f'{type(node).__name__} is not allowed in a tactic')
        if isinstance(node, ast.Attribute) and (node.attr.startswith('_') or node.attr in BAD_ATTRIBUTES):
            raise TacticError(f'attribute {node.attr} is not allowed (no attributes starting with an underscore, '
                              'no str.format)')
        if isinstance(node, ast.Name) and (_bad_name(node.id) or node.id in FORBIDDEN):
            raise TacticError(f'{node.id} is not allowed in a tactic')
        if isinstance(node, ast.FunctionDef) and _bad_name(node.name):
            raise TacticError(f'function name {node.name} is not allowed')
        if isinstance(node, ast.arg) and _bad_name(node.arg):
            raise TacticError(f'argument name {node.arg} is not allowed')
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            continue
        if isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) for t in node.targets):
            try:
                ast.literal_eval(node.value)
            except ValueError:
                raise TacticError('top-level assignments must be literal constants') from None
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            continue
        raise TacticError(f'only functions, literal constants and docstrings at the top level (line {node.lineno})')
    entry = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == ENTRY]
    if not entry:
        raise TacticError('the code must define def tactic(obs, action, memory, info)')
    if [a.arg for a in entry[0].args.args] != ['obs', 'action', 'memory', 'info'] or entry[0].args.vararg \
            or entry[0].args.kwarg or entry[0].args.kwonlyargs or entry[0].args.defaults:
        raise TacticError('the entry must be exactly def tactic(obs, action, memory, info)')
    return tree


_CHECKED = {ast.Pow: '_pow', ast.Mult: '_mul', ast.LShift: '_lshift'}


class _Ticker(ast.NodeTransformer):
    """Spend a tick at every function entry, loop iteration and comprehension element, and send `**`, `*` and
    `<<` through their size checks."""

    @staticmethod
    def _tick():
        return ast.Expr(value=ast.Call(func=ast.Name(id='_tick', ctx=ast.Load()), args=[], keywords=[]))

    def _prepend(self, node):
        self.generic_visit(node)
        node.body.insert(0, self._tick())
        return node

    visit_FunctionDef = visit_For = visit_While = _prepend

    def visit_comprehension(self, node):
        self.generic_visit(node)
        node.ifs.append(ast.Call(func=ast.Name(id='_tick_true', ctx=ast.Load()), args=[], keywords=[]))
        return node

    def visit_BinOp(self, node):
        self.generic_visit(node)
        hook = _CHECKED.get(type(node.op))
        if hook is None:
            return node
        return ast.copy_location(ast.Call(func=ast.Name(id=hook, ctx=ast.Load()), args=[node.left, node.right],
                                          keywords=[]), node)

    def visit_AugAssign(self, node):
        self.generic_visit(node)
        hook = _CHECKED.get(type(node.op))
        if hook is None:
            return node
        load = copy.deepcopy(node.target)
        for sub in ast.walk(load):
            if hasattr(sub, 'ctx'):
                sub.ctx = ast.Load()
        call = ast.Call(func=ast.Name(id=hook, ctx=ast.Load()), args=[load, node.value], keywords=[])
        return ast.copy_location(ast.Assign(targets=[node.target], value=call), node)


def _int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def _pow(a, b):
    if _int(a) and _int(b) and b > 0 and max(1, abs(a).bit_length()) * b > MAX_BITS:
        raise TacticBudget(f'an integer power of more than {MAX_BITS} bits')
    return a ** b


def _mul(a, b):
    if _int(a) and _int(b) and abs(a).bit_length() + abs(b).bit_length() > MAX_BITS:
        raise TacticBudget(f'an integer product of more than {MAX_BITS} bits')
    for seq, n in ((a, b), (b, a)):
        if isinstance(seq, (str, list, tuple)) and _int(n) and len(seq) * n > MAX_ITEMS:
            raise TacticBudget(f'a repeated sequence of more than {MAX_ITEMS} items')
    return a * b


def _lshift(a, b):
    if _int(a) and _int(b) and abs(a).bit_length() + b > MAX_BITS:
        raise TacticBudget(f'a shift to more than {MAX_BITS} bits')
    return a << b


def _plain(value):
    return json.loads(json.dumps(value))


def valid_action(action):
    """Whether a tactic's return value has the shape of an action."""
    if not isinstance(action, dict) or not set(action) <= set(ACTION_KEYS):
        return False
    farmer, hands, market = action.get('farmer', ['PASS']), action.get('hands', []), action.get('market', [])
    return (isinstance(farmer, list) and bool(farmer) and isinstance(hands, list)
            and all(isinstance(h, list) for h in hands) and isinstance(market, list)
            and all(isinstance(o, list) for o in market) and len(market) <= 20)


class Tactic:
    def __init__(self, source):
        tree = _Ticker().visit(check(source))
        ast.fix_missing_locations(tree)
        self._count = 0
        self._deadline = 0.0
        safe = {name: getattr(builtins, name) for name in BUILTINS}

        def limited_range(*args):
            r = range(*args)
            if len(r) > RANGE_MAX:
                raise TacticBudget(f'range of {len(r)} values (at most {RANGE_MAX})')
            return r
        safe['range'] = limited_range
        env = {'__builtins__': safe, 'math': types.SimpleNamespace(**{n: getattr(math, n) for n in MATH}),
               '_tick': self._tick, '_tick_true': self._tick_true, '_pow': _pow, '_mul': _mul, '_lshift': _lshift}
        exec(compile(tree, '<tactic>', 'exec'), env)
        self.function = env[ENTRY]
        self.memory = {}
        self.errors = 0
        self.calls = 0
        self.changed = 0
        self.last_step = -1
        self.last_error = None

    def _tick(self):
        self._count += 1
        if self._count > TICKS:
            raise TacticBudget(f'more than {TICKS} ticks in one turn')
        if self._count % 1000 == 0 and time.perf_counter() > self._deadline:
            raise TacticBudget(f'more than {BUDGET_S} s in one turn')

    def _tick_true(self):
        self._tick()
        return True

    def apply(self, obs, action, info):
        """The action the tactic returns for this turn, or `action` unchanged."""
        step = int(obs.get('step', 0) or 0)
        if step == 0 or step <= self.last_step:      # a new game
            self.memory, self.errors = {}, 0
        self.last_step = step
        if self.errors >= MAX_ERRORS or not isinstance(action, dict):
            return action
        self.calls += 1
        self._count, self._deadline = 0, time.perf_counter() + BUDGET_S
        try:
            out = self.function(_plain(obs), copy.deepcopy(action), self.memory, _plain(info))
        except Exception as exc:      # includes TacticBudget, RecursionError, MemoryError
            self.errors += 1
            self.last_error = repr(exc)[:300]
            return action
        if out is None:
            return action
        if not valid_action(out):
            self.errors += 1
            self.last_error = f'not an action: {repr(out)[:200]}'
            return action
        out = {key: copy.deepcopy(out.get(key, action.get(key))) for key in ACTION_KEYS}
        if out != {key: action.get(key) for key in ACTION_KEYS}:
            self.changed += 1
        return out
