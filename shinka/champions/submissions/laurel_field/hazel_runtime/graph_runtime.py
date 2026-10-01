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
- `order` on the routine_dispatch node: 'submitted' or 'sells_first';
- `engine` on the backbone node: the production engine, 'mohui_v66' (default) or a public
  ladder agent bundled in engines/<name>/ (engines.py), with `engine_parameters` for its
  top-level constants;
- `enabled: false` on the farmer, hands or market turn node: that channel of the backbone's
  action is passed through unchanged (a ladder engine with all three off plays exactly as
  the public agent does);
- the optional `oracle_guard` turn node (after `market`; off unless `enabled: true`): the
  predictor's front-run sells added to the market orders (oracle_guard.py), with its `_OG_*`
  `parameters`;
- the optional `rival_counter` turn node (first; off unless `enabled: true`): classifies the rival
  relative to our own play (rival_model.MirrorTracker: mirror, nsell_opener, wheat92_seller,
  other_opening, other; decided by step 93) and from then on applies that class's `counters`
  ({class: {NAME: value}}: switchable engine constants and `_OG_*` guard parameters);
- the optional `rival_emulator` turn node (first of all; off unless `enabled: true`): runs the public engines
  of `_EM_ENGINES` in the rival's place from our view of the game (rival_emulator.RivalEmulator). Once one
  of them has matched `_EM_LOCK` steps, the rival_counter stage uses the family `engine:<name>` if it has
  counters for it, and with `_EM_RACE` our final market orders are rearranged against the rival's known
  orders of the turn;
- the optional `land_plot` turn node (after the extension stages; off unless `enabled: true`): the fourth
  quadrant bought from day `_LP_DAY` and farmed by hands hired for it, with a crop or geese (land_plot.py), with
  its `_LP_*` `parameters`;
- the optional `tactic` turn node (right before `sanitize`; off unless `enabled: true`): Python `code` the
  evolution writes, run in a sandbox on our action every turn (tactic.py).
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
    import engines
    import oracle_guard
    import rival_model
    import rival_emulator
    import tactic
    import land_plot
except ImportError:
    from hazel_runtime.experimental import Extensions, STAGES as EXTENSION_STAGES
    from hazel_runtime import surgical, engines, oracle_guard, rival_model, rival_emulator, tactic, land_plot


def _kad_module():
    """The KAD copilot module (numpy KAD-HP-1), or None: a bundle without it still runs every graph that lacks the
    node. A plain import must find the module itself, not a same-named directory elsewhere on sys.path."""
    try:
        import kad_copilot as module
        if hasattr(module, 'KadCopilot'):
            return module
    except ImportError:
        pass
    try:
        from hazel_runtime import kad_copilot as module
        return module
    except ImportError:
        return None


kad_copilot = _kad_module()
_KAD_PARAMETERS = kad_copilot.PARAMETERS if kad_copilot is not None else {}

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
    ('rival_emulator', "Optional: the rival's public engine run in its place from our view; its known orders"),
    ('rival_counter', "Optional: the rival's family from its public farm; that family's counter-settings"),
    ('backbone', 'Submitted Mohui v66 production, routing, hiring and base trading'),
    ('oracle_observe', 'Observe once per turn using the bundled CPU oracle'),
    ('state', 'Derive farm, inventory, carried-stock and market state'),
    ('farmer', 'Preserve backbone action; idle-only weed/water/feed rescue'),
    ('hands', 'Preserve backbone hand actions; idle-only coordinated maintenance'),
    ('market', 'Execute the complete source-pinned market subgraph'),
    ('oracle_guard', 'Optional: predictor front-run sells added to the engine market orders'),
    *EXTENSION_STAGES,
    ('kad_copilot', 'Optional: KAD-HP-1 (top-player replay policy) adds confident sales and idle-hand jobs'),
    ('land_plot', 'Optional: the fourth quadrant bought from an evolved day and farmed by hired hands'),
    ('tactic', 'Optional: the evolved Python tactic, run in a sandbox on our action'),
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
# _FEED_RESERVE_LOOKAHEAD: the wheat kept back from every sale (two days of feed) also counts
# the animals about to join the herd: bought and still in the shed or a worker's hands, ordered
# this turn, and empty pastures/coops. The submitted reserve counts only placed animals, so on
# 2026-09-24 400 of 400 games against plain Mohui sold the feed of the day-6 herd expansion
# and lost a cow on day 8.
RUNTIME_PARAMETERS = {'_TOWN_CADENCE_PHASE': 0, '_FEED_RESERVE_LOOKAHEAD': False}
_ANIMALS = "('COW', 'SHEEP', 'GOOSE')"
_PARAMETERIZED = {
    432: ('_FEED_RESERVE_LOOKAHEAD',
          "feed_reserve = 0 if step >= 696 else max(2, st['n_animals'] + 6) if step >= 672 else "
          "max(4, st['n_animals'] * 2)",
          "feed_reserve = 0 if step >= 696 else max(2, st['n_animals'] + 6) if step >= 672 else "
          "max(4, 2 * (st['n_animals'] + ((sum(int(shed.get(a, 0) or 0) for a in " + _ANIMALS + ") "
          "+ sum(int(inv.get(a, 0) or 0) for inv in (obs.get('private') or {}).get('inventories', []) or [] "
          "if isinstance(inv, dict) for a in " + _ANIMALS + ") "
          "+ sum(int(o[2] or 0) for o in mkt if isinstance(o, (list, tuple)) and len(o) >= 3 "
          "and o[0] == 'BUY_ANIMAL') "
          "+ sum(1 for row in st['tiles'] for cell in row or [] if isinstance(cell, dict) "
          "and cell.get('kind') in ('PASTURE', 'COOP') and not cell.get('animal'))) "
          "if _FEED_RESERVE_LOOKAHEAD else 0)))"),
    541: ('_TOWN_CADENCE_PHASE',
          'cadence = oracle_cadence_bypass or step % 4 == 0 or (shed_used >= 60 and step % 2 == 0) '
          'or (step >= 672 and step % 2 == 0)',
          'cadence = oracle_cadence_bypass or (step - _TOWN_CADENCE_PHASE) % 4 == 0 or '
          '(shed_used >= 60 and (step - _TOWN_CADENCE_PHASE) % 2 == 0) or '
          '(step >= 672 and (step - _TOWN_CADENCE_PHASE) % 2 == 0)'),
}
ORDER_POLICIES = ('submitted', 'sells_first')
DEFAULT_ENGINE = 'mohui_v66'
CHANNELS = ('farmer', 'hands', 'market')
# The engine walks both seats' order lists index by index (lockstep) and reads at most 10 entries.
# Ladder engines place deliberate empty entries ([]) to delay later orders by a slot; a passed-through
# market keeps them, which the champion's sanitizer (it drops empty entries) would not.
MARKET_SLOTS = 10
OPTIONAL_TURN_STAGES = ('rival_emulator', 'rival_counter', 'oracle_guard', 'kad_copilot', 'land_plot', 'tactic')   # without the node, no stage


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
    allowed = (evolvable_constants(tree, source) | set(RUNTIME_PARAMETERS) | set(oracle_guard.PARAMETERS)
               | set(rival_model.PARAMETERS) | set(rival_emulator.PARAMETERS) | set(land_plot.PARAMETERS)
               | set(_KAD_PARAMETERS))
    merged, owner = {}, {}
    for chain in ('turn', 'market'):
        for node in graph[chain]['nodes']:
            params = node.get('parameters') or {}
            if not isinstance(params, dict):
                raise ValueError(f'parameters of {node["id"]} must be an object')
            for name, value in params.items():
                if name not in allowed:
                    raise ValueError(f'Unknown graph parameter {name} on {node["id"]}')
                template = (oracle_guard.PARAMETERS[name] if name in oracle_guard.PARAMETERS
                            else rival_model.PARAMETERS[name] if name in rival_model.PARAMETERS
                            else rival_emulator.PARAMETERS[name] if name in rival_emulator.PARAMETERS
                            else land_plot.PARAMETERS[name] if name in land_plot.PARAMETERS
                            else _KAD_PARAMETERS[name] if name in _KAD_PARAMETERS
                            else champion.__dict__.get(name, RUNTIME_PARAMETERS.get(name)))
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
        present = {n['id'] for n in self.graph['turn'].get('nodes', [])}
        self.turn_ids = [s[0] for s in TURN_STAGES if s[0] not in OPTIONAL_TURN_STAGES or s[0] in present]
        self.market_ids = [s[0] for s in MARKET_STAGES]
        turn_nodes = validate_chain(self.graph['turn'], self.turn_ids)
        market_nodes = validate_chain(self.graph['market'], self.market_ids)
        backbone = turn_nodes['backbone']
        self.engine_name = backbone.get('engine', DEFAULT_ENGINE)
        if self.engine_name == DEFAULT_ENGINE:
            if backbone.get('engine_parameters'):
                raise ValueError('engine_parameters need a ladder engine on the backbone node')
            self.engine = None
        else:
            if not isinstance(self.engine_name, str) or not re.fullmatch(r'[a-z0-9_]+', self.engine_name):
                raise ValueError(f'invalid engine name {self.engine_name!r}')
            self.engine = engines.Engine(Path(__file__).resolve().parent / 'engines' / self.engine_name,
                                         backbone.get('engine_parameters') or {})
        self.channels = {key: turn_nodes[key].get('enabled', True) for key in CHANNELS}
        if any(type(value) is not bool for value in self.channels.values()):
            raise ValueError('farmer/hands/market enabled flags must be booleans')
        guard = turn_nodes.get('oracle_guard', {})
        self.guard_enabled = guard.get('enabled', False)
        if type(self.guard_enabled) is not bool:
            raise ValueError('oracle_guard enabled flag must be a boolean')
        # The forecast is read only by our channels and the guard; with all of them off it is not
        # computed at all (the oracle costs most of a turn's time from step 256 on).
        self.oracle_needed = self.guard_enabled or any(self.channels.values())
        self.oracle_required = self.graph.get('require_oracle', False)
        if type(self.oracle_required) is not bool or (self.oracle_required and not self.oracle_needed):
            raise ValueError('required oracle needs an enabled forecast consumer')
        text = source.decode()
        tree = ast.parse(text)
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == 'evolve_market_orders')
        # Graph-declared constants replace the champion's before anything is compiled;
        # the compiled market stages and the farmer/hand channels read these globals.
        parameters = graph_parameters(self.graph, champion, tree, text)
        self.guard_parameters = {k: v for k, v in parameters.items() if k in oracle_guard.PARAMETERS}
        self.parameters = {k: v for k, v in parameters.items() if k not in oracle_guard.PARAMETERS
                           and k not in rival_model.PARAMETERS and k not in rival_emulator.PARAMETERS
                           and k not in land_plot.PARAMETERS and k not in _KAD_PARAMETERS}
        self.guard_active = self.guard_parameters
        self._init_rival_emulator(turn_nodes.get('rival_emulator', {}), parameters)
        self._init_rival_counter(turn_nodes.get('rival_counter', {}), parameters)
        self._init_tactic(turn_nodes.get('tactic', {}))
        self._init_land_plot(turn_nodes.get('land_plot', {}), parameters)
        self._init_kad_copilot(turn_nodes.get('kad_copilot', {}), parameters)
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

    def _init_rival_emulator(self, node, parameters):
        self.emulator = None
        enabled = node.get('enabled', False)
        if type(enabled) is not bool:
            raise ValueError('rival_emulator enabled flag must be a boolean')
        if not enabled:
            return
        if self.engine is None:
            raise ValueError('rival_emulator needs a ladder engine on the backbone node')
        settings = {**rival_emulator.PARAMETERS,
                    **{k: v for k, v in parameters.items() if k in rival_emulator.PARAMETERS}}
        names = settings['_EM_ENGINES']
        if not isinstance(names, list) or not all(isinstance(n, str) and re.fullmatch(r'[a-z0-9_]+', n)
                                                   for n in names):
            raise ValueError('_EM_ENGINES must be a list of engine names')
        self.emulator = rival_emulator.RivalEmulator(Path(__file__).resolve().parent / 'engines', names,
                                                     settings['_EM_LOCK'], settings['_EM_RACE'])

    def _init_tactic(self, node):
        self.tactic = None
        enabled = node.get('enabled', False)
        if type(enabled) is not bool:
            raise ValueError('tactic enabled flag must be a boolean')
        if enabled:
            self.tactic = tactic.Tactic(node.get('code'))   # TacticError (a ValueError) on code the sandbox refuses

    def _init_land_plot(self, node, parameters):
        self.plot = None
        enabled = node.get('enabled', False)
        if type(enabled) is not bool:
            raise ValueError('land_plot enabled flag must be a boolean')
        if not enabled:
            return
        if self.engine is None:
            raise ValueError('land_plot needs a ladder engine on the backbone node')
        self.plot = land_plot.LandPlot({k: v for k, v in parameters.items() if k in land_plot.PARAMETERS})

    def _init_kad_copilot(self, node, parameters):
        self.kad = None
        enabled = node.get('enabled', False)
        if type(enabled) is not bool:
            raise ValueError('kad_copilot enabled flag must be a boolean')
        self.kad_required = self.graph.get('require_kad', False)
        if type(self.kad_required) is not bool or (self.kad_required and not enabled):
            raise ValueError('require_kad needs an enabled kad_copilot node')
        if not enabled:
            return
        if kad_copilot is None:
            raise ValueError('kad_copilot is enabled but hazel_runtime/kad_copilot.py (or kad/) is missing')
        self.kad = kad_copilot.KadCopilot({k: v for k, v in parameters.items() if k in _KAD_PARAMETERS},
                                          required=self.kad_required)

    def tape_rest(self, obs):
        """The engine's remaining tape actions of the current day (a route-tape chassis), or None."""
        try:
            chassis = self.engine.namespace['_IMPL'].chassis
            route = chassis.players[int(obs.get('player', 0) or 0)]['route']
            step = int(obs.get('step', 0) or 0)
            return chassis.routes[route][step + 1:(step // 24 + 1) * 24]
        except Exception:
            return None

    def tactic_info(self, forecast):
        """What the tactic sees besides the observation and our action (plain JSON values)."""
        summary = None
        if isinstance(forecast, dict):
            summary = {k: v for k, v in forecast.items() if k == 'step' or k.startswith(('score_', 'units_'))}
        return {'rival_family': self.rival.decision if self.rival is not None else None,
                'rival_engine': self.emulator.identity() if self.emulator is not None else None,
                'rival_action': self.emulator.prediction() if self.emulator is not None else None,
                'forecast': summary,
                'kad': self.kad.last if getattr(self, 'kad', None) is not None else None}

    def _init_rival_counter(self, node, parameters):
        self.rival = None
        enabled = node.get('enabled', False)
        if type(enabled) is not bool:
            raise ValueError('rival_counter enabled flag must be a boolean')
        if not enabled:
            return
        if self.engine is None or self.engine.namespace is None:
            raise ValueError('rival_counter needs a ladder engine on the backbone node')
        self.rival = rival_model.MirrorTracker()
        switchable = engines.switchable(self.engine.source)
        families = list(self.rival.clusters) + ([f'engine:{h.name}' for h in self.emulator.hypotheses]
                                                if self.emulator is not None else [])
        counters = rival_model.check_counters(node.get('counters') or {}, families, switchable,
                                              oracle_guard.PARAMETERS)
        self.rival_counters = {}
        for family, settings in counters.items():
            typed = {}
            for name, value in settings.items():
                if name in oracle_guard.PARAMETERS:
                    typed[name] = _like(value, oracle_guard.PARAMETERS[name], name)
                else:
                    typed[name] = engines.from_json(value, self.engine.constants[name]['default'], name)
            self.rival_counters[family] = typed
        names = {n for settings in self.rival_counters.values() for n in settings if n not in oracle_guard.PARAMETERS}
        self.engine_base = {n: self.engine.namespace[n] for n in sorted(names)}
        self.rival_active = None

    def _apply_counter(self, family):
        """Engine constants and guard parameters of `family`'s counter (None: the graph's own)."""
        settings = self.rival_counters.get(family, {}) if family else {}
        for name, base in self.engine_base.items():
            self.engine.namespace[name] = settings.get(name, base)
        self.guard_active = {**self.guard_parameters,
                             **{k: v for k, v in settings.items() if k in oracle_guard.PARAMETERS}}
        self.rival_active = family

    def rival_counter(self, obs):
        step = int(obs.get('step', 0) or 0)
        if step == 0 and self.rival_active is not None:
            self._apply_counter(None)
        family = self.rival.update(obs)
        engine = self.emulator.identity() if self.emulator is not None else None
        if engine is not None and f'engine:{engine}' in self.rival_counters:
            family = f'engine:{engine}'   # the rival runs this engine exactly: its own counter
        chosen = family if family in self.rival_counters else None
        if chosen != self.rival_active:
            self._apply_counter(chosen)

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

    def observe_oracle(self, obs, configuration):
        forecast = self.champion._oracle_observe(obs, configuration) if self.oracle_needed else None
        if self.oracle_required:
            tracker = getattr(self.champion, '_TRACKER', None)
            stats = getattr(self.champion, 'ORACLE_STATS', {})
            if tracker is None or tracker.model is None or stats.get('errors'):
                raise RuntimeError(f"required oracle unavailable: {stats.get('last_error', 'tracker/model missing')}")
            if int(obs.get('step', 0) or 0) >= tracker.min_context and forecast is None:
                raise RuntimeError('required oracle did not forecast after warmup')
        return forecast

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
            if key == 'rival_emulator':
                if self.emulator is not None:
                    try:
                        self.emulator.begin(obs, configuration)
                    except Exception as exc:
                        self._fallback(key, obs, exc)
            elif key == 'rival_counter':
                if self.rival is not None:
                    try:
                        self.rival_counter(obs)
                    except Exception as exc:
                        self._fallback(key, obs, exc)
            elif key == 'backbone':
                try:
                    base = (self.engine.agent(obs, configuration) if self.engine is not None
                            else c._mohui.kaggle_agent_v66_meta_closed_loop(obs, configuration))
                except Exception as exc:
                    self._fallback(key, obs, exc)
                if not isinstance(base, dict):
                    base = {'farmer': ['PASS'], 'hands': [], 'market': []}
                farms = obs.get('farms', []) or []
                farm = farms[player] if player < len(farms) else {}
                n_hands = len(farm.get('hands') or [])
            elif key == 'oracle_observe':
                forecast = self.observe_oracle(obs, configuration)
            elif key in CHANNELS and not self.channels[key]:
                if not failed:
                    evolved[key] = base.get(key)
            elif key == 'oracle_guard':
                if self.guard_enabled and not failed:
                    try:
                        evolved['market'] = oracle_guard.apply(obs, evolved.get('market'), state, self.guard_active)
                    except Exception as exc:
                        self._fallback(key, obs, exc)
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
            elif key == 'kad_copilot':
                if self.kad is not None and not failed and isinstance(evolved, dict):
                    try:
                        evolved = self.kad.apply(obs, evolved)
                    except Exception as exc:
                        if self.kad_required:
                            raise
                        self._fallback(key, obs, exc)
            elif key == 'land_plot':
                if self.plot is not None and not failed and isinstance(evolved, dict):
                    try:
                        evolved = self.plot.apply(obs, evolved, self.tape_rest(obs))
                    except Exception as exc:
                        self._fallback(key, obs, exc)
            elif key == 'tactic':
                if self.tactic is not None and isinstance(evolved, dict):
                    try:
                        evolved = self.tactic.apply(obs, evolved, self.tactic_info(forecast))
                    except Exception as exc:
                        self._fallback(key, obs, exc)
            elif key == 'sanitize':
                final = c._sanitize(evolved, base, n_hands)
                orders = evolved.get('market') if isinstance(evolved, dict) else None
                if not self.channels['market'] and isinstance(orders, list):
                    # the engine's list with its empty slots (the champion's sanitizer drops them)
                    final['market'] = [list(o) if isinstance(o, (list, tuple)) else [] for o in orders][:MARKET_SLOTS]
                if self.emulator is not None:
                    try:
                        final = self.emulator.race(final, obs)
                    except Exception as exc:
                        self._fallback('rival_emulator', obs, exc)
                    self.emulator.finish(final)
            elif key == 'oracle_record':
                if self.oracle_needed:
                    c._oracle_record(final)
        return final
