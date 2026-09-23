"""Executable, source-pinned Hazel policy graph. No LLM or network at runtime.

The market subgraph is compiled by adding yield checkpoints to the original
function AST. Its expressions, closures, side effects, and early returns are
unchanged. Graph text is documentation; bindings below are the executable API.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

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
        tree = ast.parse(source.decode())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == 'evolve_market_orders')
        function.name = '_graph_market_generator'
        body, stage = [], None
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
                body.append(ast.Expr(value=ast.Yield(value=ast.Constant(key))))
                stage = key
            body.append(statement)
        function.body = body
        module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
        namespace = {}
        # Real champion globals preserve deferred state and submitted constants.
        exec(compile(module, str(source_path) + ':graph', 'exec'), champion.__dict__, namespace)
        self.market_generator = namespace['_graph_market_generator']
        self.last_trace = []
        self.fallback_count = 0
        self.last_error = None

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
            return done.value

    def agent(self, obs, configuration=None):
        c = self.champion
        self.last_trace = []
        self.last_error = None
        player = int(obs.get('player', 0) or 0)
        base = None
        evolved = None
        n_hands = 0
        forecast = None
        state = None
        failed = False
        for key in self.turn_ids:
            if failed and key in ('state', 'farmer', 'hands', 'market'):
                continue
            self.last_trace.append(key)
            if key == 'backbone':
                try:
                    base = c._mohui.kaggle_agent_v66_meta_closed_loop(obs, configuration)
                except Exception as exc:
                    self.last_error = repr(exc)
                    self.fallback_count += 1
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
                        evolved['farmer'] = c.evolve_farmer_action(obs, player, base.get('farmer'), state)
                    elif key == 'hands':
                        evolved['hands'] = c.evolve_hand_actions(obs, player, base.get('hands'), state)
                    else:
                        evolved['market'] = self.market(obs, player, base.get('market'), state)
                except Exception as exc:
                    evolved = base
                    failed = True
                    self.last_error = repr(exc)
                    self.fallback_count += 1
            elif key == 'sanitize':
                final = c._sanitize(evolved, base, n_hands)
            elif key == 'oracle_record':
                c._oracle_record(final)
        return final
