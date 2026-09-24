"""Executable, source-pinned Hazel policy graph. No LLM or network at runtime.

The market subgraph is compiled by adding yield checkpoints to the original
function AST. With surgical overrides disabled and no graph parameters or stage
toggles its expressions, closures, side effects, and early returns are unchanged.
Graph text is documentation; bindings below are the executable API.

Behaviour the graph itself controls (policy changes are graph edits, not code):
- `parameters` on any turn/market chain node: values for the champion's
  EVOLVE-block constants (same JSON-compatible type), plus RUNTIME_PARAMETERS;
- `enabled: false` on a market chain node: that stage's statements are not
  compiled (its checkpoint stays) unless a later stage reads a name it defines;
- `order` on the routine_dispatch node: 'submitted' or 'sells_first'.
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

try:
    from experimental import Extensions, STAGES as EXTENSION_STAGES
    import surgical
except ImportError:
    from hazel_runtime.experimental import Extensions, STAGES as EXTENSION_STAGES
    from hazel_runtime import surgical

# Statement boundaries in the hash-pinned submitted champion.py (1-indexed).
MARKET_STAGES = (
    ('market_setup', 288, 'Initialize orders, sell consolidation, protected items and deferred queue'),
    ('opening_scalp', 351, 'Step 0 wheat purchase / step 1 sale; early returns are preserved'),
    ('deferred_sales', 362, 'Reissue evicted sells with stock and expiry checks'),
    ('wage_liquidity', 375, 'Narrow midnight product-purchase reserve guard'),
    ('investment_freeze', 392, 'Cancel non-paying late investments'),
    ('optional_cash_floor', 403, 'Preserve submitted disabled cash-floor flag'),
    ('optional_livestock', 417, 'Preserve submitted disabled livestock-topup flag'),
    ('feed_reserve', 428, 'Endgame-aware feed reserve including day-28 pickup slack'),
    ('shed_pressure', 434, 'Tiered shed overflow relief'),
    ('predrop_headroom', 455, 'Protected cheap-stock clearance accounting for carried harvest'),
    ('oracle_frontrun', 475, 'Forecast-driven sales, price floors and feed protection'),
    ('town_and_fertilizer', 516, 'Adaptive shop cadence, inventory-aware batches, fertilizer sales'),
    ('early_liquidation', 631, 'Forecast-led early and phased endgame liquidation'),
    ('final_liquidation', 702, 'Final eight-step liquidation and final full-stock sale'),
    ('routine_dispatch', 750, 'Return all remaining routine orders, retaining submitted order and cap'),
)
TURN_STAGES = (
    ('backbone', 'Submitted Mohui v66 production, routing, hiring and base trading'),
    ('oracle_observe', 'Observe once per turn using the bundled CPU oracle'),
    ('state', 'Derive farm, inventory, carried-stock and market state'),
    ('farmer', 'Preserve backbone action; idle-only weed/water/feed rescue'),
    ('hands', 'Preserve backbone hand actions; idle-only coordinated maintenance'),
    ('market', 'Execute the complete source-pinned market subgraph'),
    *EXTENSION_STAGES,
    ('sanitize', 'Preserve legal actions including DROP; enforce hand count and order cap'),
    ('oracle_record', 'Record the final action actually dispatched'),
)


def validate_chain(graph, expected):
    nodes = graph.get('nodes', [])
    ids = [n['id'] for n in nodes]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError('Missing, duplicate, or unknown executable node')
    if any(n.get('binding') != n['id'] for n in nodes):
        raise ValueError('Unbound graph node')
    edges = graph.get('edges', [])
    pairs = [(e.get('source'), e.get('target')) for e in edges]
    if pairs != list(zip(expected, expected[1:])):
        raise ValueError('Disconnected/reordered graph: source dependencies must be retained')
    if graph.get('entry') != expected[0] or graph.get('exit') != expected[-1]:
        raise ValueError('Invalid entry/exit')
    return {n['id']: n for n in nodes}


# Parameters the runtime introduces for the graph (default = submitted behaviour).
# _TOWN_CADENCE_PHASE: the town block's shop-cadence steps are those with
# (step - phase) % 4 == 0 (and % 2 under shed pressure / in the endgame).
RUNTIME_PARAMETERS = {'_TOWN_CADENCE_PHASE': 0}
_PARAMETERIZED = {
    541: ('_TOWN_CADENCE_PHASE',
          'cadence = oracle_cadence_bypass or step % 4 == 0 or (shed_used >= 60 and step % 2 == 0) '
          'or (step >= 672 and step % 2 == 0)',
          'cadence = oracle_cadence_bypass or (step - _TOWN_CADENCE_PHASE) % 4 == 0 or '
          '(shed_used >= 60 and (step - _TOWN_CADENCE_PHASE) % 2 == 0) or '
          '(step >= 672 and (step - _TOWN_CADENCE_PHASE) % 2 == 0)'),
}
ORDER_POLICIES = ('submitted', 'sells_first')


def evolvable_constants(tree, source):
    """Module-level constants assigned between the champion's EVOLVE-BLOCK markers."""
    lines = source.splitlines()
    start = next(i for i, line in enumerate(lines, 1) if 'EVOLVE-BLOCK-START' in line)
    end = next(i for i, line in enumerate(lines, 1) if 'EVOLVE-BLOCK-END' in line)
    names = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and start < node.lineno < end:
            names |= {t.id for t in node.targets
                      if isinstance(t, ast.Name) and re.fullmatch(r'_[A-Z][A-Z0-9_]*', t.id)}
    if not names:
        raise ValueError('No evolvable constants found')
    return names


def _like(value, template, name):
    """JSON value -> the champion constant's type; fail closed on anything else."""
    if isinstance(template, bool):
        ok = type(value) is bool
    elif isinstance(template, int):
        ok = type(value) is int
    elif isinstance(template, float):
        ok = type(value) in (int, float) and math.isfinite(value)
        value = float(value) if ok else value
    elif isinstance(template, str):
        ok = isinstance(value, str)
    elif isinstance(template, (tuple, list)):
        ok = isinstance(value, list)
        if ok and template:
            value = [_like(v, template[0], name) for v in value]
        value = tuple(value) if ok and isinstance(template, tuple) else value
    elif isinstance(template, dict):
        ok = isinstance(value, dict) and all(isinstance(k, str) for k in value)
        if ok and template:
            first = next(iter(template.values()))
            value = {k: _like(v, first, name) for k, v in value.items()}
    else:
        ok = False
    if not ok:
        raise ValueError(f'Graph parameter {name}: value does not match the champion constant type')
    return value


def graph_parameters(graph, champion, tree, source):
    """Merge node-level `parameters` of both chains; conflicting duplicates fail."""
    allowed = evolvable_constants(tree, source) | set(RUNTIME_PARAMETERS)
    merged, owner = {}, {}
    for chain in ('turn', 'market'):
        for node in graph[chain]['nodes']:
            params = node.get('parameters') or {}
            if not isinstance(params, dict):
                raise ValueError(f'parameters of {node["id"]} must be an object')
            for name, value in params.items():
                if name not in allowed:
                    raise ValueError(f'Unknown graph parameter {name} on {node["id"]}')
                template = champion.__dict__.get(name, RUNTIME_PARAMETERS.get(name))
                value = _like(value, template, name)
                if name in merged and merged[name] != value:
                    raise ValueError(f'Graph parameter {name} set differently by {owner[name]} and {node["id"]}')
                merged[name], owner[name] = value, node['id']
    return merged


def _names(statements, store):
    out = set()
    for statement in statements:
        for node in ast.walk(statement):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store if store else ast.Load):
                out.add(node.id)
            elif store and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out.add(node.name)
    return out


def check_disabled_stages(stages, enabled, arguments):
    """A disabled stage may not be the first definer of a name a later enabled stage reads."""
    for i, (key, statements) in enumerate(stages):
        if enabled[key]:
            continue
        defined_before = set(arguments)
        for earlier, earlier_statements in stages[:i]:
            if enabled[earlier]:
                defined_before |= _names(earlier_statements, True)
        read_later = set()
        for later, later_statements in stages[i + 1:]:
            if enabled[later]:
                read_later |= _names(later_statements, False)
        missing = (_names(statements, True) & read_later) - defined_before
        if missing:
            raise ValueError(f'Disabled stage {key} defines {sorted(missing)} read by later stages')


class HazelGraph:
    def __init__(self, champion, graph_path):
        self.champion = champion
        self.graph = json.loads(Path(graph_path).read_text())
        source_path = Path(champion.__file__)
        source = source_path.read_bytes()
        actual = hashlib.sha256(source).hexdigest()
        if actual != self.graph['source']['champion_sha256']:
            raise ValueError('Submitted source fingerprint mismatch')
        self.turn_ids = [s[0] for s in TURN_STAGES]
        self.market_ids = [s[0] for s in MARKET_STAGES]
        validate_chain(self.graph['turn'], self.turn_ids)
        market_nodes = validate_chain(self.graph['market'], self.market_ids)
        text = source.decode()
        tree = ast.parse(text)
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == 'evolve_market_orders')
        # Graph-declared constants replace the champion's before anything is compiled;
        # the compiled market stages and the farmer/hand channels read these globals.
        self.parameters = graph_parameters(self.graph, champion, tree, text)
        champion.__dict__.update(self.parameters)
        for line, (name, original, parameterized) in _PARAMETERIZED.items():
            if name not in self.parameters:
                continue
            index = next(i for i, s in enumerate(function.body) if s.lineno == line)
            if ast.unparse(function.body[index]) != original:
                raise ValueError(f'Parameterized statement mismatch at {line}')
            replacement = ast.parse(parameterized).body[0]
            for node in ast.walk(replacement):
                if hasattr(node, 'lineno'):
                    node.lineno, node.end_lineno = line, line
                    node.col_offset, node.end_col_offset = 0, 0
            function.body[index] = replacement
        self.surgical_enabled = surgical.enabled(self.graph.get('surgical'))
        self.surgical_overrides = surgical.overrides(self.graph.get('surgical'))
        if self.surgical_enabled:
            function = surgical.rewrite_market(function, self.surgical_overrides)
        # 'off' keeps the champion's own farmer/hand rescue channels.
        self.idle_version = (self.surgical_overrides['idle_dispatch']
                             if self.surgical_enabled else 'off')
        self.dispatch_order = market_nodes['routine_dispatch'].get('order', 'submitted')
        if self.dispatch_order not in ORDER_POLICIES:
            raise ValueError(f'routine_dispatch order must be one of {ORDER_POLICIES}')
        enabled = {key: market_nodes[key].get('enabled', True) for key in self.market_ids}
        if any(type(value) is not bool for value in enabled.values()):
            raise ValueError('market stage enabled flags must be booleans')
        if not all(enabled[key] for key in ('market_setup', 'routine_dispatch')):
            raise ValueError('market_setup and routine_dispatch cannot be disabled')
        self.disabled_stages = [key for key in self.market_ids if not enabled[key]]
        function.name = '_graph_market_generator'
        body, stage, stages = [], None, []
        for statement in function.body:
            if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant) and isinstance(statement.value.value, str):
                body.append(statement)
                continue
            idx = max(i for i, (_, start, _) in enumerate(MARKET_STAGES)
                      if start <= statement.lineno)
            key = self.market_ids[idx]
            if key != stage:
                start = MARKET_STAGES[idx][1]
                end = MARKET_STAGES[idx + 1][1] - 1 if idx + 1 < len(MARKET_STAGES) else function.end_lineno
                if market_nodes[key].get('source_lines') != [start, end]:
                    raise ValueError('Graph source boundary mismatch')
                # A disabled stage keeps its checkpoint so traces stay comparable.
                body.append(ast.Expr(value=ast.Yield(value=ast.Constant(key))))
                stage = key
                stages.append((key, []))
            stages[-1][1].append(statement)
            if enabled[key]:
                body.append(statement)
        check_disabled_stages(stages, enabled, [a.arg for a in function.args.args])
        function.body = body
        module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
        namespace = {}
        # Real champion globals preserve deferred state and submitted constants.
        exec(compile(module, str(source_path) + ':graph', 'exec'), champion.__dict__, namespace)
        self.market_generator = namespace['_graph_market_generator']
        self.last_trace = []
        self.fallback_count = 0
        self.last_error = None
        switches = self.graph.get('experimental', {}).get('switches', {})
        override = os.environ.get('KAGG_GRAPH_EXPERIMENTS')
        if override is not None:
            switches = {**switches, **json.loads(override)}
        self.extensions = Extensions(switches)
        self.extension_ids = {stage[0] for stage in EXTENSION_STAGES}
        self.last_telemetry = []

    def market(self, obs, player, base, state):
        gen = self.market_generator(obs, player, base, state)
        try:
            expected = 0
            while True:
                key = next(gen)
                if key != self.market_ids[expected]:
                    raise ValueError('Unexpected executable market transition')
                self.last_trace.append(key)
                expected += 1
        except StopIteration as done:
            orders = done.value
        if self.dispatch_order == 'sells_first' and isinstance(orders, list):
            # Stable partition. Both seats' orders at one index clear unit by unit in
            # lockstep and index i+1 starts only after both index-i orders finish, so
            # a SELL queued behind HIRE/BUY clears after an opponent's earlier SELL of
            # the same item. Selling first only adds cash and shed room for purchases.
            def is_sell(order):
                return isinstance(order, (list, tuple)) and bool(order) and order[0] == 'SELL'
            orders = [o for o in orders if is_sell(o)] + [o for o in orders if not is_sell(o)]
        return orders

    def _fallback(self, stage, obs, exc):
        # A silent fallback would make a broken variant look like the backbone;
        # report the first few on stderr (never stdout: the RPC protocol owns it),
        # naming the graph: both seats of a game share the harness's stderr.
        self.last_error = repr(exc)
        self.fallback_count += 1
        if self.fallback_count <= 5:
            print(f'graph fallback #{self.fallback_count} [{self.graph.get("name")}] at step '
                  f'{obs.get("step")} stage {stage}: {self.last_error[:300]}', file=sys.stderr, flush=True)

    def agent(self, obs, configuration=None):
        c = self.champion
        self.last_trace = []
        self.last_error = None
        self.extensions.start()
        self.last_telemetry = self.extensions.telemetry
        player = int(obs.get('player', 0) or 0)
        base = None
        evolved = None
        n_hands = 0
        forecast = None
        state = None
        rescued_hands = None
        failed = False
        for key in self.turn_ids:
            if failed and key in ('state', 'farmer', 'hands', 'market'):
                continue
            self.last_trace.append(key)
            if key == 'backbone':
                try:
                    base = c._mohui.kaggle_agent_v66_meta_closed_loop(obs, configuration)
                except Exception as exc:
                    self._fallback(key, obs, exc)
                if not isinstance(base, dict):
                    base = {'farmer': ['PASS'], 'hands': [], 'market': []}
                farms = obs.get('farms', []) or []
                farm = farms[player] if player < len(farms) else {}
                n_hands = len(farm.get('hands') or [])
            elif key == 'oracle_observe':
                forecast = c._oracle_observe(obs, configuration)
            elif key in ('state', 'farmer', 'hands', 'market'):
                try:
                    if key == 'state':
                        state = c.farm_state(obs, player, forecast)
                        evolved = {}
                    elif key == 'farmer':
                        if self.idle_version != 'off':
                            evolved['farmer'], rescued_hands = surgical.idle_dispatch(obs, player, base, state, c)
                        else:
                            evolved['farmer'] = c.evolve_farmer_action(obs, player, base.get('farmer'), state)
                    elif key == 'hands':
                        if self.idle_version != 'off':
                            evolved['hands'] = rescued_hands
                        else:
                            evolved['hands'] = c.evolve_hand_actions(obs, player, base.get('hands'), state)
                    else:
                        evolved['market'] = self.market(obs, player, base.get('market'), state)
                except Exception as exc:
                    evolved = base
                    failed = True
                    self._fallback(key, obs, exc)
            elif key in self.extension_ids:
                if failed:
                    self.extensions.event(key, 'baseline_failed_extension_skipped')
                    continue
                try:
                    evolved = self.extensions.run(key, obs, player, evolved, state, configuration)
                except Exception as exc:
                    self.extensions.event(key, 'extension_failed_closed', error=repr(exc))
                    if key == 'experimental_state':
                        self.extensions.ctx = None
            elif key == 'sanitize':
                final = c._sanitize(evolved, base, n_hands)
            elif key == 'oracle_record':
                c._oracle_record(final)
        return final
