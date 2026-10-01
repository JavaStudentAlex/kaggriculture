"""The kad_copilot stage: the numpy KAD advisor's invariants on our action, fail-closed / required behaviour, the graph
edit, and the evolution's KAD requirement (2026-09-29; run on cliproxyapi, not on the laptop)."""
from __future__ import annotations

import copy
import json
import os
import unittest
from pathlib import Path

import graph_edits
import highcpu_island_evolution as evo
from hazel_runtime import kad_copilot
from test_land_plot import SEED_GRAPH, action, obs

ALLOWED = set(kad_copilot.PARAMETERS['_KC_SELL_ITEMS'])


def game(steps, **kw):
    """Synthetic observations of one game (the land-plot tests' fixture) with stock in the shed."""
    shed = {'MILK': 30, 'WOOL': 12, 'STRAWBERRY': 20, 'EGG': 8, 'WHEAT': 40}
    return [obs(t, shed=dict(shed), **kw) for t in range(steps)]


class CopilotTests(unittest.TestCase):
    def run_turns(self, params, steps=14, market=None):
        copilot = kad_copilot.KadCopilot(params)
        pairs = []
        for o in game(steps):
            before = action(market=copy.deepcopy(market) if market is not None else None)
            after = copilot.apply(o, copy.deepcopy(before))
            pairs.append((o, before, after))
        return copilot, pairs

    def test_edits_only_append_allowed_sales_and_idle_hand_jobs(self):
        params = {'_KC_EVERY': 2, '_KC_FROM_STEP': 8, '_KC_TO_STEP': 719, '_KC_SELL_P': 0.0, '_KC_JOB_P': 0.0}
        copilot, pairs = self.run_turns(params, market=[['SELL', 'MILK', 2], [], ['SELL', 'WOOL', 1]])
        self.assertEqual(copilot.report.get('calls'), 3)          # steps 8, 10 and 12
        self.assertNotIn('failures', copilot.report)
        for o, before, after in pairs:
            n = len(before['market'])
            self.assertEqual(after['market'][:n], before['market'])  # the engine's slots, [] included, untouched
            shed = o['private']['shed']
            for order in after['market'][n:]:
                self.assertEqual(order[0], 'SELL')
                self.assertIn(order[1], ALLOWED)
                sold = sum(x[2] for x in after['market'] if x and x[0] == 'SELL' and x[1] == order[1])
                self.assertLessEqual(sold, shed.get(order[1], 0))
            self.assertLessEqual(len(after['market']), 10)
            for b, a in zip(before['hands'], after['hands']):
                self.assertTrue(a == b or (b == ['PASS'] and a[0] in ('WATER', 'HARVEST')))
            if o['step'] < 8 or o['step'] % 2:
                self.assertEqual(after, before)                    # no advice on these turns: action unchanged
        self.assertEqual(copilot.last['step'], 12)
        json.dumps(copilot.last)                                   # plain JSON for the tactic's info['kad']

    def test_rl_player_settings_reach_the_torch_advisor(self):
        """main.py maps a bundle's settings.json to KAGG_KAD_* variables; the 09-29 residual RL sampled at 1.0 in name
        only, because the temperature and seed never reached the advisor (every sample replayed one game)."""
        made = []

        class Stub:
            TORCH_AVAILABLE = True

            class TorchKadAdvisor:
                def __init__(self, checkpoint, device=None, temperature=0.0, seed=None):
                    made.append(dict(checkpoint=checkpoint, device=device, temperature=temperature, seed=seed))

        names = ('KAGG_KAD_BACKEND', 'KAGG_KAD_DEVICE', 'KAGG_KAD_CHECKPOINT', 'KAGG_KAD_TEMPERATURE', 'KAGG_KAD_SEED')
        saved_env = {k: os.environ.get(k) for k in names}
        saved_torch = kad_copilot.kad_torch
        kad_copilot.kad_torch = Stub
        try:
            for k in names:
                os.environ.pop(k, None)
            kad_copilot.KadCopilot({'_KC_BACKEND': 'torch'})
            os.environ.update(KAGG_KAD_BACKEND='torch', KAGG_KAD_DEVICE='cuda', KAGG_KAD_CHECKPOINT='/m/it1s2.pt',
                              KAGG_KAD_TEMPERATURE='0.7', KAGG_KAD_SEED='1002')
            kad_copilot.KadCopilot({})
        finally:
            kad_copilot.kad_torch = saved_torch
            for k, v in saved_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        self.assertEqual((made[0]['temperature'], made[0]['seed']), (0.0, None))    # the graph's own settings
        self.assertEqual(made[1], dict(checkpoint='/m/it1s2.pt', device='cuda', temperature=0.7, seed=1002))

    def test_levers_off_change_nothing(self):
        copilot, pairs = self.run_turns({'_KC_SELL': False, '_KC_HANDS': False, '_KC_FROM_STEP': 8, '_KC_EVERY': 1})
        self.assertGreater(copilot.report.get('calls', 0), 0)
        for _, before, after in pairs:
            self.assertEqual(after, before)

    def test_failure_is_closed_unless_required(self):
        broken = kad_copilot.KadCopilot({'_KC_FROM_STEP': 8, '_KC_EVERY': 1})
        broken.advisor.advise = lambda: (_ for _ in ()).throw(RuntimeError('model gone'))
        turns = game(12)
        for o in turns:
            a = action()
            self.assertEqual(broken.apply(o, copy.deepcopy(a)), a)
        self.assertEqual(broken.report.get('failures'), 1)       # off for the rest of the game after one failure
        required = kad_copilot.KadCopilot({'_KC_FROM_STEP': 8, '_KC_EVERY': 1}, required=True)
        required.advisor.advise = lambda: (_ for _ in ()).throw(RuntimeError('model gone'))
        with self.assertRaises(RuntimeError):
            for o in turns:
                required.apply(o, action())

    def test_check_refuses_unknown_items_and_jobs(self):
        with self.assertRaises(ValueError):
            kad_copilot.check({'_KC_SELL_ITEMS': ['GOLD']})
        with self.assertRaises(ValueError):
            kad_copilot.check({'_KC_JOBS': ['WEED']})
        with self.assertRaises(ValueError):
            kad_copilot.check({'_KC_EVERY': 0})


class EditAndRequirementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.seed = json.loads(Path(SEED_GRAPH).read_text())

    def kad_graph(self, **params):
        return graph_edits.apply_edit(copy.deepcopy(self.seed), {'channels': {'kad_copilot': True},
                                                                 'parameters': params or {'_KC_EVERY': 4}})

    def test_edit_inserts_the_stage_with_its_pins(self):
        graph = self.kad_graph(_KC_EVERY=3)
        ids = [n['id'] for n in graph['turn']['nodes']]
        self.assertEqual(ids[ids.index('kad_copilot') + 1], 'sanitize')
        node = next(n for n in graph['turn']['nodes'] if n['id'] == 'kad_copilot')
        self.assertEqual(node['parameters'], {'_KC_EVERY': 3})
        pins = graph['provenance']['runtime_bundle_hashes']
        for name in graph_edits.KAD_FILES:
            self.assertIn(name, pins)
        self.assertTrue(graph_edits.channels(graph)['kad_copilot'])

    def test_requirement(self):
        evo.check_required_kad(self.kad_graph(_KC_EVERY=4, _KC_TO_STEP=671))
        with self.assertRaisesRegex(ValueError, 'must stay enabled'):
            evo.check_required_kad(self.seed)
        evo.check_required_kad(self.kad_graph(_KC_SELL=False, _KC_HANDS=False, _KC_TO_STEP=671))   # advice only
        with self.assertRaisesRegex(ValueError, 'every 12 turns'):
            evo.check_required_kad(self.kad_graph(_KC_EVERY=24, _KC_TO_STEP=671))
        with self.assertRaisesRegex(ValueError, '4 days'):
            evo.check_required_kad(self.kad_graph(_KC_FROM_STEP=100, _KC_TO_STEP=150))
        with self.assertRaisesRegex(ValueError, 'liquidation'):
            evo.check_required_kad(self.kad_graph(_KC_TO_STEP=700))
        with self.assertRaisesRegex(ValueError, 'sell lever'):
            evo.check_required_kad(self.kad_graph(_KC_SELL_P=0.99, _KC_TO_STEP=671))
        off = self.kad_graph(_KC_EVERY=4, _KC_TO_STEP=671)
        next(n for n in off['turn']['nodes'] if n['id'] == 'kad_copilot')['enabled'] = False
        with self.assertRaisesRegex(ValueError, 'must stay enabled'):
            evo.check_required_kad(off)


if __name__ == '__main__':
    unittest.main()
