"""The rival_emulator stage: the edits that switch it on, the order race, and emulation in real games."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import graph_edits
from hazel_runtime import rival_emulator

HERE = Path(__file__).resolve().parent
SEED_GRAPH = HERE / 'evolution_results' / 'ladder_2026-09-26' / 'seed_graph.json'
CONFIGURATION = {'boardSize': 10, 'episodeSteps': 720, 'turnsPerDay': 24, 'shedCapacity': 100,
                 'maxMarketOrdersPerTurn': 10, 'farmHandCostMult': 1, 'townShopSellInterval': 4,
                 'townCenterSellInterval': 24, 'townShopUnlockInterval': 3, 'weedSpawnChance': 0.005,
                 'startingMoney': 3000}


class EmulatorEditTest(unittest.TestCase):
    def setUp(self):
        self.graph = json.loads(SEED_GRAPH.read_text())

    def test_switching_on_inserts_the_stage_first_and_pins_the_engines(self):
        out = graph_edits.apply_edit(self.graph, {'channels': {'rival_emulator': True, 'rival_counter': True}})
        self.assertEqual(out['turn']['entry'], 'rival_emulator')
        self.assertEqual([n['id'] for n in out['turn']['nodes']][:3], ['rival_emulator', 'rival_counter', 'backbone'])
        pins = out['provenance']['runtime_bundle_hashes']
        self.assertIn('rival_emulator.py', pins)
        for name in rival_emulator.PARAMETERS['_EM_ENGINES']:
            self.assertIn(f'engines/{name}/SOURCE.json', pins)
        channels = graph_edits.settings(out)['channels']
        self.assertTrue(channels['rival_counter'] and channels['rival_emulator'])

    def test_engine_families_become_counter_classes(self):
        edit = {'channels': {'rival_emulator': True, 'rival_counter': True},
                'counters': {'engine:tetsutani_demand': {'_EV_H': 12}}}
        out = graph_edits.apply_edit(self.graph, edit)
        self.assertEqual(graph_edits.settings(out)['counters'], {'engine:tetsutani_demand': {'_EV_H': 12}})
        with self.assertRaisesRegex(ValueError, 'unknown rival family'):
            graph_edits.apply_edit(self.graph, {'channels': {'rival_counter': True},
                                                'counters': {'engine:tetsutani_demand': {'_EV_H': 12}}})

    def test_unknown_engine_and_lock_parameter(self):
        with self.assertRaisesRegex(ValueError, 'no engine'):
            graph_edits.apply_edit(self.graph, {'channels': {'rival_emulator': True},
                                                'parameters': {'_EM_ENGINES': ['no_such_engine']}})
        out = graph_edits.apply_edit(self.graph, {'channels': {'rival_emulator': True}, 'parameters': {'_EM_LOCK': 6}})
        self.assertEqual(graph_edits.settings(out)['parameters']['_EM_LOCK'], 6)
        with self.assertRaisesRegex(ValueError, 'no effect'):
            graph_edits.apply_edit(self.graph, {'parameters': {'_EM_LOCK': 6}})


class RaceTest(unittest.TestCase):
    def test_a_sale_behind_the_rivals_moves_ahead(self):
        env = rival_emulator.environment()
        public = {'farms': [env._new_farm(10, 3000), env._new_farm(10, 3000)], 'market': env._new_market(),
                  'town': env._new_town(), 'day': 1, 'hour': 0, 'step': 24}
        private = env._new_private()
        private['shed']['MILK'] = 10
        emulator = rival_emulator.RivalEmulator.__new__(rival_emulator.RivalEmulator)
        emulator.race_enabled, emulator.lock, emulator.races, emulator.race_gain = True, 1, 0, 0.0
        rival = rival_emulator.Hypothesis('tetsutani_demand', engine=None)
        rival.private, rival.matched = json.loads(json.dumps(private)), 5
        rival.predicted = {'farmer': ['PASS'], 'hands': [], 'market': [['SELL', 'MILK', 5]]}
        emulator.hypotheses = [rival]
        emulator.pending = {'step': 24, 'public': public, 'private': private, 'configuration': CONFIGURATION}
        ours = {'farmer': ['PASS'], 'hands': [], 'market': [['BUY_SEED', 'WHEAT', 1], ['SELL', 'MILK', 5]]}
        raced = emulator.race(ours, {'player': 0})
        self.assertEqual(raced['market'], [['SELL', 'MILK', 5], ['BUY_SEED', 'WHEAT', 1]])
        self.assertEqual(emulator.races, 1)
        self.assertGreater(emulator.race_gain, 0)
        # with no product in common nothing moves
        rival.predicted = {'farmer': ['PASS'], 'hands': [], 'market': [['SELL', 'WOOL', 5]]}
        self.assertIs(emulator.race(ours, {'player': 0}), ours)


GAME = r'''
import json, os, sys
graph_path, rival_name, steps, seed, seat = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
os.environ['KAGG_GRAPH_PATH'] = graph_path
sys.path.insert(0, os.getcwd())
import agent_graph
from hazel_runtime import engines
from kaggle_environments import make
graph = agent_graph.get_engine()
emu = graph.emulator
rival = engines.Engine(os.path.join('hazel_runtime', 'engines', rival_name))
predicted, played = {}, {}
def ours(obs, config):
    action = agent_graph.agent(obs, config)
    p = emu.prediction()
    if p is not None:
        predicted[int(obs['step'])] = p
    return action
def theirs(obs, config):
    action = rival.agent(obs, config)
    played[int(obs['step'])] = json.loads(json.dumps(action))
    return action
env = make('kaggriculture', configuration={'seed': seed, 'episodeSteps': steps}, debug=False)
env.run([ours, theirs] if seat == 0 else [theirs, ours])
canon = lambda a: json.dumps({k: a.get(k) for k in ('farmer', 'hands', 'market')}, sort_keys=True)
wrong = [s for s, p in predicted.items() if canon(p) != canon(played.get(s, {}))]
print(json.dumps({'hypotheses': {h.name: {'alive': h.alive, 'matched': h.matched, 'dropped_at': h.dropped_at}
                                 for h in emu.hypotheses},
                  'predicted': len(predicted), 'wrong': wrong[:5], 'identity': emu.identity(),
                  'races': emu.races, 'fallbacks': graph.fallback_count, 'last_error': graph.last_error,
                  'statuses': [s.get('status') for s in env.steps[-1]]}))
'''


class EmulationGameTest(unittest.TestCase):
    """120 steps of a real game against a public engine: the emulator must lock onto it and predict every move."""

    def play(self, rival, seat):
        graph = json.loads(SEED_GRAPH.read_text())
        graph = graph_edits.apply_edit(graph, {'channels': {'rival_emulator': True}, 'parameters': {'_EM_LOCK': 4}})
        env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', KAGG_ORACLE_BACKEND='numpy',
                   KAGG_ORACLE_DEVICE='cpu', CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
        with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as tmp:
            tmp.write(json.dumps(graph_edits.repinned(graph)))
        try:
            proc = subprocess.run([sys.executable, '-c', GAME, tmp.name, rival, '120', '20260928', str(seat)], cwd=HERE,
                                  env=env, capture_output=True, text=True, timeout=1200)
        finally:
            os.unlink(tmp.name)
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        return json.loads(proc.stdout.strip().splitlines()[-1])

    def check(self, report, rival, other):
        print(json.dumps(report), file=sys.stderr)
        self.assertEqual(report['statuses'], ['DONE', 'DONE'])
        self.assertEqual(report['fallbacks'], 0, report['last_error'])
        self.assertTrue(report['hypotheses'][rival]['alive'], report)
        self.assertFalse(report['hypotheses'][other]['alive'], report)
        self.assertEqual(report['identity'], rival)
        self.assertGreater(report['predicted'], 50)
        self.assertEqual(report['wrong'], [])

    def test_a_09_27_engine_rival_from_seat_0(self):
        self.check(self.play('tetsutani_demand_0927', 0), 'tetsutani_demand_0927', 'tetsutani_demand')

    def test_an_old_engine_rival_from_seat_1(self):
        self.check(self.play('tetsutani_demand', 1), 'tetsutani_demand', 'tetsutani_demand_0927')


if __name__ == '__main__':
    unittest.main()
