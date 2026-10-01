"""The land_plot stage: land purchase, hiring after the engine, plot work, sales, the edit, and a real game."""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import graph_edits
from hazel_runtime import land_plot

HERE = Path(__file__).resolve().parent
SEED_GRAPH = HERE / 'evolution_results' / 'ladder_2026-09-26' / 'seed_graph.json'


def tiles(unlocked=('NW', 'NE', 'SW')):
    quadrant = {(False, False): 'NW', (True, False): 'NE', (False, True): 'SW', (True, True): 'SE'}
    return [[None if quadrant[(x >= 5, y >= 5)] in unlocked else 'LOCKED' for x in range(10)] for y in range(10)]


def obs(step, unlocked=('NW', 'NE', 'SW'), money=9000.0, hands=10, shed=None, seeds=None, invs=None, grid=None,
        hires_today=10):
    return {'step': step, 'player': 0,
            'farms': [{'money': money, 'tiles': grid or tiles(unlocked), 'farmer': [4, 4],
                       'hands': [[4, 4]] * hands, 'unlocked_quadrants': list(unlocked), 'hires_today': hires_today},
                      {'money': 0.0, 'tiles': tiles(), 'farmer': [4, 4], 'hands': [], 'unlocked_quadrants': ['NW'],
                       'hires_today': 0}],
            'market': {'prices': {'WHEAT': 25, 'EGG': 50, 'FERTILIZER': 100, 'TOMATO': 60}, 'inventory': {}},
            'town': {'unlocked_shops': []},
            'private': {'shed': shed or {}, 'seeds': seeds or {}, 'inventories': invs or [{}] * (hands + 1)}}


def action(hands=10, market=None):
    return {'farmer': ['PASS'], 'hands': [['PASS']] * hands, 'market': market if market is not None else [['SELL', 'MILK', 2], []]}


class LandTest(unittest.TestCase):
    def test_settings_are_checked(self):
        with self.assertRaisesRegex(ValueError, '_LP_USE'):
            land_plot.check({'_LP_USE': 'COW'})
        with self.assertRaisesRegex(ValueError, '_LP_TILES'):
            land_plot.check({'_LP_TILES': 25})
        self.assertEqual(land_plot.check({})['_LP_DAY'], land_plot.PARAMETERS['_LP_DAY'])
        self.assertEqual(len(land_plot.PLOT), 24)
        self.assertNotIn((5, 5), land_plot.PLOT)
        self.assertTrue(all(x >= 5 and y >= 5 for x, y in land_plot.PLOT))

    def test_nothing_before_the_day(self):
        plot = land_plot.LandPlot({'_LP_DAY': 12})
        a = action()
        self.assertIs(plot.apply(obs(11 * 24 + 5), a), a)

    def test_buys_se_then_sw_first_when_missing_and_only_with_money(self):
        plot = land_plot.LandPlot({'_LP_DAY': 12, '_LP_MIN_MONEY': 2000})
        out = plot.apply(obs(12 * 24), action())
        self.assertEqual(out['market'], [['SELL', 'MILK', 2], [], ['BUY_LAND']])   # appended: the engine's slots stay
        plot = land_plot.LandPlot({'_LP_DAY': 9, '_LP_MIN_MONEY': 2000})
        out = plot.apply(obs(9 * 24, unlocked=('NW', 'NE')), action())
        self.assertEqual(sum(o == ['BUY_LAND'] for o in out['market']), 2)          # SW ($2k) then SE ($4k)
        plot = land_plot.LandPlot({'_LP_DAY': 9, '_LP_MIN_MONEY': 2000})
        out = plot.apply(obs(9 * 24, unlocked=('NW', 'NE'), money=7000.0), action())
        self.assertNotIn(['BUY_LAND'], out['market'])                               # 7000 - 6000 < 2000
        plot = land_plot.LandPlot({'_LP_DAY': 12})
        out = plot.apply(obs(12 * 24), action(market=[['BUY_LAND'], []]))
        self.assertEqual(out['market'], [['BUY_LAND'], []])                         # the engine buys: wait

    def test_no_land_against_a_copy_of_our_engine(self):
        grid = tiles()
        for x in range(10):
            grid[0][x] = {'kind': 'PLANT', 'crop': 'WHEAT'}
        o = obs(12 * 24, grid=grid)
        o['farms'][1].update(unlocked_quadrants=['NW', 'NE', 'SW'], tiles=copy.deepcopy(grid))
        plot = land_plot.LandPlot({'_LP_DAY': 12})
        self.assertNotIn(['BUY_LAND'], plot.apply(o, action())['market'])          # a mirror: keep the race
        self.assertEqual(land_plot.similarity(o['farms'][0], o['farms'][1]), 1.0)
        plot = land_plot.LandPlot({'_LP_DAY': 12, '_LP_MIRROR': 1.01})
        self.assertIn(['BUY_LAND'], plot.apply(o, action())['market'])
        o['farms'][1]['tiles'][0][:5] = [{'kind': 'PLANT', 'crop': 'MELON'}] * 5  # half the tiles differ
        plot = land_plot.LandPlot({'_LP_DAY': 12})
        self.assertIn(['BUY_LAND'], plot.apply(o, action())['market'])

    def test_the_engine_owning_se_turns_the_plot_off(self):
        plot = land_plot.LandPlot({'_LP_DAY': 12})
        plot.apply(obs(12 * 24, unlocked=('NW', 'NE', 'SW', 'SE')), action())
        self.assertEqual(plot.states[0]['phase'], 'off')

    def farm(self, params, hour=5, **kw):
        """A plot that bought SE at hour 0 of day 12 and is asked to act at `hour`."""
        plot = land_plot.LandPlot(params)
        plot.apply(obs(12 * 24), action())
        return plot, obs(12 * 24 + hour, unlocked=('NW', 'NE', 'SW', 'SE'), **kw)

    def test_hires_after_the_engine_and_the_tape(self):
        plot, o = self.farm({'_LP_DAY': 12, '_LP_WORKERS': 2, '_LP_HIRE_HOUR': 4}, hour=3)
        out = plot.apply(o, action())
        self.assertNotIn(['HIRE'], out['market'])                                   # before _LP_HIRE_HOUR
        o = dict(o, step=12 * 24 + 4)
        out = plot.apply(o, action(), tape_rest=[{'market': [['HIRE']]}])
        self.assertNotIn(['HIRE'], out['market'])                                   # the tape hires later today
        out = plot.apply(dict(o, step=12 * 24 + 5), action(market=[['HIRE']]))
        self.assertEqual(out['market'].count(['HIRE']), 1)                          # engine hires this turn
        out = plot.apply(dict(o, step=12 * 24 + 6), action(), tape_rest=[{'market': []}])
        self.assertEqual(out['market'].count(['HIRE']), 2)
        self.assertEqual(plot.states[0]['pending'], {'first': 11, 'count': 2})

    def work(self, params, grid, invs=None, shed=None, seeds=None, hour=6):
        """Hire at `hour`, then return the plot and the next observation with the two new hands at (5, 5)."""
        plot, o = self.farm(params, hour=hour)
        plot.apply(o, action(), tape_rest=[])
        nxt = obs(12 * 24 + hour + 1, unlocked=('NW', 'NE', 'SW', 'SE'), hands=11, grid=grid,
                  invs=invs or [{}] * 12, shed=shed or {}, seeds=seeds or {})
        nxt['farms'][0]['hands'] = [[4, 4]] * 10 + [[5, 5]]
        return plot, nxt

    def test_geese_build_buy_feed_and_harvest(self):
        grid = tiles(('NW', 'NE', 'SW', 'SE'))
        plot, o = self.work({'_LP_DAY': 12, '_LP_USE': 'GOOSE', '_LP_TILES': 2}, grid)
        out = plot.apply(o, action(hands=11))
        self.assertEqual(out['hands'][10], ['EAST'])                                # to (6, 5), the first coop
        self.assertIn(['BUY_ANIMAL', 'GOOSE', 2], out['market'])
        # standing on the coop with a goose in hand: place it; with wheat: feed a hungry goose first
        grid[5][6] = {'kind': 'COOP', 'animal': None}
        o['farms'][0]['hands'][10] = [6, 5]
        o['step'] += 1
        o['private']['inventories'] = [{}] * 11 + [{'GOOSE': 1}]
        self.assertEqual(plot.apply(o, action(hands=11))['hands'][10], ['PLACE', 'GOOSE', 1])
        grid[5][6] = {'kind': 'COOP', 'animal': 'GOOSE', 'fed_today': False, 'cared_today': False, 'yield_units': 2}
        o['step'] += 1
        o['private']['inventories'] = [{}] * 11 + [{'WHEAT': 1}]
        self.assertEqual(plot.apply(o, action(hands=11))['hands'][10], ['FEED'])
        grid[5][6] = dict(grid[5][6], fed_today=True, cared_today=True)
        grid[5][7] = {'kind': 'COOP', 'animal': 'GOOSE', 'fed_today': True, 'cared_today': True}
        o['step'] += 1
        self.assertEqual(plot.apply(o, action(hands=11))['hands'][10], ['HARVEST'])

    def test_a_goose_placed_today_needs_no_feed_trip(self):
        grid = tiles(('NW', 'NE', 'SW', 'SE'))
        grid[5][6] = {'kind': 'COOP', 'animal': 'GOOSE', 'fed_today': False, 'cared_today': False, 'placed_day': 12}
        plot, o = self.work({'_LP_DAY': 12, '_LP_USE': 'GOOSE', '_LP_TILES': 1}, grid, shed={'WHEAT': 5})
        o['farms'][0]['hands'][10] = [6, 5]
        self.assertEqual(plot.apply(o, action(hands=11))['hands'][10], ['CARE'])

    def test_hungry_geese_send_the_hand_to_the_shed_for_wheat(self):
        grid = tiles(('NW', 'NE', 'SW', 'SE'))
        grid[5][6] = {'kind': 'COOP', 'animal': 'GOOSE', 'fed_today': False, 'cared_today': True, 'placed_day': 10}
        grid[5][7] = {'kind': 'COOP', 'animal': 'GOOSE', 'fed_today': False, 'cared_today': True, 'placed_day': 10}
        plot, o = self.work({'_LP_DAY': 12, '_LP_USE': 'GOOSE', '_LP_TILES': 2}, grid, shed={'WHEAT': 5})
        self.assertEqual(plot.apply(o, action(hands=11))['hands'][10], ['PICKUP', 'WHEAT', 2])   # at (5, 5)
        o['farms'][0]['hands'][10] = [7, 6]
        o['step'] += 1
        self.assertEqual(plot.apply(o, action(hands=11))['hands'][10], ['WEST'])     # back to the shed

    def test_wheat_plants_waters_and_harvests_at_peak(self):
        grid = tiles(('NW', 'NE', 'SW', 'SE'))
        plot, o = self.work({'_LP_DAY': 12, '_LP_USE': 'WHEAT', '_LP_TILES': 2}, grid, seeds={'WHEAT': 3})
        plot.states[0]['seeds'] = 2
        o['farms'][0]['hands'][10] = [6, 5]
        self.assertEqual(plot.apply(o, action(hands=11))['hands'][10], ['PLANT', 'WHEAT'])
        grid[5][6] = {'kind': 'PLANT', 'crop': 'WHEAT', 'planted_day': 12, 'watered_today': False, 'yield_units': 0}
        o['step'] += 1
        self.assertEqual(plot.apply(o, action(hands=11))['hands'][10], ['WATER'])
        self.assertEqual(plot.states[0]['seeds'], 1)
        grid[5][6] = {'kind': 'PLANT', 'crop': 'WHEAT', 'planted_day': 8, 'watered_today': False, 'yield_units': 4}
        o['step'] += 1
        self.assertEqual(plot.apply(o, action(hands=11))['hands'][10], ['HARVEST'])  # age 4: peak, no watering

    def test_seeds_the_engine_plants_this_turn_come_first(self):
        grid = tiles(('NW', 'NE', 'SW', 'SE'))
        plot, o = self.work({'_LP_DAY': 12, '_LP_USE': 'WHEAT', '_LP_TILES': 2}, grid, seeds={'WHEAT': 1})
        plot.states[0]['seeds'] = 2
        o['farms'][0]['hands'][10] = [6, 5]
        a = action(hands=11)
        a['farmer'] = ['PLANT', 'WHEAT']
        self.assertEqual(plot.apply(o, a)['hands'][10], ['PASS'])

    def test_sells_the_stock_and_keeps_the_herd_feed(self):
        plot, o = self.farm({'_LP_DAY': 12, '_LP_USE': 'WHEAT', '_LP_SELL_RATIO': 0.7}, hour=2)
        plot.states[0]['stock'] = {'WHEAT': 12}
        o['private']['shed'] = {'WHEAT': 15}
        o['farms'][0]['tiles'][0][0] = {'kind': 'PASTURE', 'animal': 'COW'}
        out = plot.apply(o, action())
        self.assertIn(['SELL', 'WHEAT', 12], out['market'])                        # 15 - 2 feed for the cow
        plot.states[0]['stock'] = {'WHEAT': 8}                                      # below HOLD_CAP
        o['step'] += 1
        o['market']['prices']['WHEAT'] = 15                                          # below 0.7 x 25
        self.assertNotIn('SELL', [m[0] for m in plot.apply(o, action())['market'] if m and m[0] == 'SELL' and m[1] == 'WHEAT'])

    def test_produce_moves_to_the_stock_overnight(self):
        grid = tiles(('NW', 'NE', 'SW', 'SE'))
        plot, o = self.work({'_LP_DAY': 12, '_LP_USE': 'GOOSE', '_LP_TILES': 2}, grid)
        o['private']['inventories'] = [{}] * 11 + [{'EGG': 3, 'FERTILIZER': 1}]
        plot.apply(o, action(hands=11))
        nxt = obs(13 * 24, unlocked=('NW', 'NE', 'SW', 'SE'), hands=0, grid=grid, shed={'EGG': 3, 'FERTILIZER': 1})
        out = plot.apply(nxt, action(hands=0))
        self.assertEqual(plot.states[0]['workers'], {})
        self.assertIn(['SELL', 'EGG', 3], out['market'])
        self.assertIn(['SELL', 'FERTILIZER', 1], out['market'])


class EditTest(unittest.TestCase):
    def setUp(self):
        self.graph = json.loads(SEED_GRAPH.read_text())

    def test_the_edit_inserts_the_stage_before_sanitize(self):
        out = graph_edits.apply_edit(self.graph, {'channels': {'land_plot': True},
                                                  'parameters': {'_LP_USE': 'WHEAT', '_LP_TILES': 10}})
        ids = [n['id'] for n in out['turn']['nodes']]
        self.assertEqual(ids[ids.index('land_plot') + 1], 'sanitize')
        node = next(n for n in out['turn']['nodes'] if n['id'] == 'land_plot')
        self.assertEqual(node['parameters'], {'_LP_USE': 'WHEAT', '_LP_TILES': 10})
        self.assertIn('land_plot.py', out['provenance']['runtime_bundle_hashes'])
        both = graph_edits.apply_edit(out, {'channels': {'tactic': True},
                                            'tactic': 'def tactic(obs, action, memory, info):\n    return None'})
        ids = [n['id'] for n in both['turn']['nodes']]
        self.assertEqual(ids[ids.index('land_plot') + 1:ids.index('land_plot') + 3], ['tactic', 'sanitize'])
        self.assertIn('LAND PLOT (channel land_plot=on', graph_edits.describe_controls(out))

    def test_bad_values_and_missing_stage_are_refused(self):
        with self.assertRaisesRegex(ValueError, 'land_plot refused'):
            graph_edits.apply_edit(self.graph, {'channels': {'land_plot': True}, 'parameters': {'_LP_USE': 'PIG'}})
        with self.assertRaisesRegex(ValueError, 'no effect'):
            graph_edits.apply_edit(self.graph, {'parameters': {'_LP_TILES': 10}})

    def test_mixing_carries_the_plot(self):
        donor = graph_edits.apply_edit(self.graph, {'channels': {'land_plot': True}, 'parameters': {'_LP_DAY': 13}})
        edit = graph_edits.migration_edit(self.graph, donor, self.graph)
        self.assertEqual(edit['channels'], {'land_plot': True})
        mixed = graph_edits.apply_edit(self.graph, edit)
        self.assertEqual(graph_edits.settings(mixed), graph_edits.settings(donor))


GAME = r'''
import json, os, sys
os.environ['KAGG_GRAPH_PATH'] = sys.argv[1]
sys.path.insert(0, os.getcwd())
import agent_graph
engine = agent_graph.get_engine()
from kaggle_environments import make
env = make('kaggriculture', configuration={'seed': 20260925, 'episodeSteps': int(sys.argv[2])}, debug=False)
env.run([agent_graph.agent, lambda obs, config: {}])
farm = env.steps[-1][0]['observation']['farms'][0]
se = [farm['tiles'][y][x] for y in range(5, 10) for x in range(5, 10)]
print(json.dumps({'statuses': [s['status'] for s in env.steps[-1]], 'fallbacks': engine.fallback_count,
                  'error': engine.last_error, 'report': engine.plot.report, 'quadrants': farm['unlocked_quadrants'],
                  'geese': sum(1 for t in se if isinstance(t, dict) and t.get('animal') == 'GOOSE')}))
'''


class GameTest(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('KAGG_SLOW_TESTS'), 'set KAGG_SLOW_TESTS=1 for the real game (about a minute)')
    def test_a_goose_plot_in_a_real_game(self):
        graph = graph_edits.apply_edit(json.loads(SEED_GRAPH.read_text()), {
            'channels': {'land_plot': True},
            'parameters': {'_LP_USE': 'GOOSE', '_LP_TILES': 4, '_LP_DAY': 12, '_LP_MIN_MONEY': 0}})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'graph.json'
            path.write_text(json.dumps(graph_edits.repinned(graph)))
            proc = subprocess.run([sys.executable, '-c', GAME, str(path), str(18 * 24)], cwd=HERE,
                                  capture_output=True, text=True, timeout=900,
                                  env=dict(os.environ, KAGG_ORACLE_BACKEND='numpy', CUDA_VISIBLE_DEVICES=''))
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        report = json.loads(proc.stdout.strip().splitlines()[-1])
        self.assertEqual(report['statuses'], ['DONE', 'DONE'])
        self.assertEqual(report['fallbacks'], 0, report['error'])
        self.assertIn('SE', report['quadrants'], report)
        self.assertGreaterEqual(report['geese'], 3, report)
        self.assertGreater(report['report'].get('feeds', 0), 0, report)


if __name__ == '__main__':
    unittest.main()
