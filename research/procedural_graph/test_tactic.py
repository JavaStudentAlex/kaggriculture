"""The tactic stage: the sandbox, the budget, the edits that carry code, and a tactic in a real game."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import graph_edits
from hazel_runtime import tactic

HERE = Path(__file__).resolve().parent
SEED_GRAPH = HERE / 'evolution_results' / 'ladder_2026-09-26' / 'seed_graph.json'
SELL_MILK = '''
def tactic(obs, action, memory, info):
    """From day 20 on, sell the milk in the shed first."""
    memory['turns'] = memory.get('turns', 0) + 1
    milk = obs['private']['shed'].get('MILK', 0)
    if obs['step'] < 480 or milk <= 0:
        return None
    market = [o for o in action['market'] if not (o and o[0] == 'SELL' and o[1] == 'MILK')]
    return dict(action, market=[['SELL', 'MILK', milk]] + market[:9])
'''
OBS = {'step': 500, 'player': 0, 'private': {'shed': {'MILK': 4}, 'seeds': {}, 'inventories': [{}]}}
ACTION = {'farmer': ['PASS'], 'hands': [], 'market': [['SELL', 'WOOL', 2], []]}


class SandboxTest(unittest.TestCase):
    def refused(self, code, message):
        with self.assertRaisesRegex(tactic.TacticError, message):
            tactic.check(code)

    def test_refuses_what_reaches_outside(self):
        self.refused('import os\ndef tactic(obs, action, memory, info):\n    return None', 'imports')
        self.refused('def tactic(obs, action, memory, info):\n    return ().__class__', 'attribute __class__')
        self.refused('def tactic(obs, action, memory, info):\n    print(1)', 'print is not allowed')
        self.refused('def tactic(obs, action, memory, info):\n    return "{0.x}".format(obs)', 'format')
        self.refused('def tactic(obs, action, memory, info):\n    return "x".ljust(10 ** 9)', 'ljust')
        self.refused('def tactic(obs, action, memory, info):\n    return pow(2, 3)', 'pow is not allowed')
        self.refused('def tactic(obs, action, memory, info):\n    return _mul(2, 3)', '_mul is not allowed')
        self.refused('def tactic(obs, action, memory, info):\n    global x', 'global')
        self.refused('x = len([])\ndef tactic(obs, action, memory, info):\n    return None', 'literal constants')
        self.refused('def other(obs):\n    return None', 'must define')
        self.refused('def tactic(obs, action):\n    return None', 'exactly')

    def test_accepts_ordinary_code(self):
        tactic.check(SELL_MILK)
        tactic.check('LIMIT = 3\ndef helper(n):\n    return [i for i in range(n) if i % 2]\n'
                     'def tactic(obs, action, memory, info):\n    for _ in helper(LIMIT):\n        pass\n    return None')

    def test_applies_keeps_memory_and_resets_per_game(self):
        t = tactic.Tactic(SELL_MILK)
        out = t.apply(OBS, ACTION, {})
        self.assertEqual(out['market'], [['SELL', 'MILK', 4], ['SELL', 'WOOL', 2], []])
        self.assertEqual(ACTION['market'], [['SELL', 'WOOL', 2], []])      # the input is not modified
        t.apply(dict(OBS, step=501), ACTION, {})
        self.assertEqual(t.memory['turns'], 2)
        self.assertEqual(t.apply(dict(OBS, step=100), ACTION, {}), ACTION)  # an earlier step: a new game
        self.assertEqual(t.memory['turns'], 1)
        self.assertEqual(t.changed, 2)

    def test_budget_and_errors_leave_the_action(self):
        t = tactic.Tactic('def tactic(obs, action, memory, info):\n    while True:\n        pass')
        self.assertEqual(t.apply(OBS, ACTION, {}), ACTION)
        self.assertIn('ticks', t.last_error)
        big = tactic.Tactic('def tactic(obs, action, memory, info):\n    return sum(range(10 ** 9))')
        self.assertEqual(big.apply(OBS, ACTION, {}), ACTION)
        self.assertIn('range', big.last_error)
        for code, message in (('return 10 ** 10 ** 8', 'power'), ('return [0] * 10 ** 9', 'repeated'),
                              ('x = 3\n    x **= 10 ** 6\n    return None', 'power'), ('return 1 << 10 ** 9', 'shift'),
                              ('return "ab" * 10 ** 7', 'repeated'), ('return math.factorial(10 ** 6)', 'factorial')):
            slow = tactic.Tactic('def tactic(obs, action, memory, info):\n    ' + code)
            self.assertEqual(slow.apply(OBS, ACTION, {}), ACTION)
            self.assertIn(message, slow.last_error)
        ok = tactic.Tactic('def tactic(obs, action, memory, info):\n    x = 2\n    x *= 3\n    memory["v"] = '
                           '[x ** 2, 2 ** 0.5, [1] * 3, 1 << 4, math.sqrt(16)]\n    return None')
        ok.apply(OBS, ACTION, {})
        self.assertEqual((ok.errors, ok.memory['v']), (0, [36, 2 ** 0.5, [1, 1, 1], 16, 4.0]))
        bad = tactic.Tactic('def tactic(obs, action, memory, info):\n    return {"market": "x"}')
        for step in range(1, tactic.MAX_ERRORS + 5):
            self.assertEqual(bad.apply(dict(OBS, step=step), ACTION, {}), ACTION)
        self.assertEqual((bad.errors, bad.calls), (tactic.MAX_ERRORS, tactic.MAX_ERRORS))   # then off for the game


class TacticEditTest(unittest.TestCase):
    def setUp(self):
        self.graph = json.loads(SEED_GRAPH.read_text())

    def test_the_edit_inserts_the_stage_before_sanitize(self):
        out = graph_edits.apply_edit(self.graph, {'channels': {'tactic': True}, 'tactic': SELL_MILK})
        ids = [n['id'] for n in out['turn']['nodes']]
        self.assertEqual(ids[ids.index('tactic') + 1], 'sanitize')
        self.assertIn('tactic.py', out['provenance']['runtime_bundle_hashes'])
        self.assertEqual(graph_edits.settings(out)['tactic'], SELL_MILK)
        self.assertTrue(any(line.startswith('tactic: none -> ') for line in graph_edits.diff(self.graph, out)))
        removed = graph_edits.apply_edit(out, {'tactic': None})
        self.assertNotIn('tactic', graph_edits.settings(removed))

    def test_bad_code_and_missing_stage_are_refused(self):
        with self.assertRaisesRegex(ValueError, 'tactic refused: imports'):
            graph_edits.apply_edit(self.graph, {'channels': {'tactic': True}, 'tactic': 'import os'})
        with self.assertRaisesRegex(ValueError, 'needs the tactic stage'):
            graph_edits.apply_edit(self.graph, {'tactic': SELL_MILK})
        with self.assertRaisesRegex(ValueError, 'needs code'):
            graph_edits.apply_edit(self.graph, {'channels': {'tactic': True}})

    def test_mixing_carries_a_tactic(self):
        donor = graph_edits.apply_edit(self.graph, {'channels': {'tactic': True}, 'tactic': SELL_MILK})
        edit = graph_edits.migration_edit(self.graph, donor, self.graph)
        self.assertEqual((edit['tactic'], edit['channels']['tactic']), (SELL_MILK, True))
        self.assertEqual(graph_edits.settings(graph_edits.apply_edit(self.graph, edit))['tactic'], SELL_MILK)


class TacticGameTest(unittest.TestCase):
    def test_validation_refuses_a_tactic_that_fails_late(self):
        code = ('def tactic(obs, action, memory, info):\n    if obs["step"] >= 600:\n'
                '        return memory["plan"][0]\n    return None')
        graph = graph_edits.apply_edit(json.loads(SEED_GRAPH.read_text()), {'channels': {'tactic': True}, 'tactic': code})
        with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as tmp:
            tmp.write(json.dumps(graph))
            path = tmp.name
        try:
            with self.assertRaisesRegex(ValueError, "called at step 600.*KeyError\\('plan'\\)"):
                graph_edits.validate_graph(path, steps=12)
        finally:
            os.unlink(path)

    def test_a_tactic_plays_a_real_game(self):
        code = SELL_MILK.replace('obs[\'step\'] < 480', 'obs[\'step\'] < 20')
        graph = graph_edits.apply_edit(json.loads(SEED_GRAPH.read_text()), {'channels': {'tactic': True}, 'tactic': code})
        with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as tmp:
            tmp.write(json.dumps(graph))
            path = tmp.name
        try:
            report = graph_edits.validate_graph(path, steps=40)
        finally:
            os.unlink(path)
        self.assertEqual(report['fallbacks'], 0)
        script = ('import json, os, sys\nos.environ["KAGG_GRAPH_PATH"] = sys.argv[1]\nsys.path.insert(0, os.getcwd())\n'
                  'import agent_graph\nfrom kaggle_environments import make\ng = agent_graph.get_engine()\n'
                  'env = make("kaggriculture", configuration={"seed": 7, "episodeSteps": 40}, debug=False)\n'
                  'env.run([agent_graph.agent, lambda obs, config: {}])\n'
                  'print(json.dumps({"calls": g.tactic.calls, "errors": g.tactic.errors, "last": g.tactic.last_error}))')
        with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as tmp:
            tmp.write(json.dumps(graph_edits.repinned(graph)))
            path = tmp.name
        try:
            proc = subprocess.run([sys.executable, '-c', script, path], cwd=HERE, capture_output=True, text=True,
                                  timeout=900, env=dict(os.environ, KAGG_ORACLE_BACKEND='numpy', CUDA_VISIBLE_DEVICES=''))
        finally:
            os.unlink(path)
        self.assertEqual(proc.returncode, 0, proc.stderr[-1500:])
        report = json.loads(proc.stdout.strip().splitlines()[-1])
        self.assertEqual((report['calls'], report['errors']), (39, 0), report['last'])


if __name__ == '__main__':
    unittest.main()
