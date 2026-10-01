"""Tests of KAD-HP-1 (policy.py, policy_train.py, play.PolicyPlayer): the labels of a synthetic game
whose answers are known, the inputs, the heads and the loss, a smoke training run and its resume (as
train_job.py's self-test runs them), the player on an observation, and, when a real replay archive
is at REPLAY_ZIP, the labels' invariants on a real game."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data import (COMMAND_SLOTS, FEATURE_DIM, FEATURE_NAMES, ITEMS, OPERATIONS, encode_action, episode_arrays,
                  field_sizes)
from policy import (JOBS, MARKET_SLOTS, NOW, NOW_TABLE, ORDERS, TILES, UNITS, JOB_TABLE, Policy, PolicyConfig,
                    PolicyWindows, commands, inputs_at, policy_labels, policy_loss)
import play

REPLAY_ZIP = Path.home() / 'kaggriculture/replays/kaggriculture-episodes-2026-08-28.zip'
UNIT = {name: i for i, name in enumerate(FEATURE_NAMES)}


def place(row, unit, x, y):
    row[UNIT[f'own.unit.{unit}.exists']] = 1.0
    row[UNIT[f'own.unit.{unit}.x']] = x / 9
    row[UNIT[f'own.unit.{unit}.y']] = y / 9


def game(T=60):
    """X, A of a synthetic seat: the farmer walks east along row 0 and plants and waters at the far
    end; a hand works turns 20-39; a few market orders."""
    X = np.zeros((T, FEATURE_DIM), dtype=np.float32)
    A = np.zeros((T, COMMAND_SLOTS * 3), dtype=np.int64)
    x = 0
    for t in range(T):
        place(X[t], 0, x, 0)
        action = {'hands': [], 'market': []}
        if x < 9:
            action['farmer'] = ['EAST']
            x += 1
        else:
            action['farmer'] = ['PLANT', 'WHEAT'] if t % 3 == 0 else ['WATER'] if t % 3 == 1 else ['PASS']
        if 20 <= t < 40:
            place(X[t], 1, 5, 5)
            action['hands'] = [['HARVEST'] if t % 2 else ['PASS']]
        if t == 0:
            action['market'] = [['BUY_SEED', 'WHEAT', 5]]
        if t == 19:
            action['market'] = [[], ['HIRE']]
        if t == 30:
            action['market'] = [['SELL', 'WHEAT', 3]]
        A[t] = encode_action(action)
    return X, A


def dataset(root, files=4, T=60):
    """A tiny WindowDataset directory of synthetic seats, with the records' ratings and results."""
    records = []
    for i in range(files):
        X, A = game(T)
        split = 'val' if i == files - 1 else 'train'
        path = f'{split}/episode_{i}_seat0.npz'
        (root / split).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(root / path, X=X, A=A, episode_id=np.asarray(str(i)), seat=np.int64(0))
        records.append(dict(path=path, episode_id=str(i), seat=0, split=split, length=T, source='day-2026-09-25.zip',
                            team=f'team{i % 2}', team_rating=2900.0 - 10 * i, win=float(i % 2), margin=0.02 * i,
                            expected_margin=0.01, day='2026-09-25'))
    (root / 'manifest.json').write_text(json.dumps({'feature_dim': FEATURE_DIM, 'field_sizes': field_sizes,
                                                    'files': records}))
    return root


def observation(step=5, hands=1):
    tiles = [[None for _ in range(10)] for _ in range(10)]
    tiles[4][4] = {'kind': 'PLANT', 'crop': 'WHEAT', 'watered_today': True, 'consecutive_unwatered': 0,
                   'yield_units': 1, 'max_lifespan_step': 120, 'planted_day': 0, 'fertilized_until_day': -1}
    farm = {'farmer': [4, 4], 'hands': [[5, 4]] * hands, 'hires_today': hands, 'money': 2500.0, 'tiles': tiles,
            'unlocked_quadrants': ['NW']}
    products = ('WHEAT', 'CARROT', 'TOMATO', 'STRAWBERRY', 'MELON', 'EGG', 'MILK', 'WOOL', 'FERTILIZER')
    return {'farms': [farm, dict(farm, hands=[])], 'market': {'inventory': {p: 100 for p in products},
                                                              'prices': {p: 50 for p in products}},
            'town': {'unlocked_shops': ['BAKERY']}, 'day': 0, 'hour': step % 24, 'step': step, 'player': 0,
            'private': {'inventories': [{'WHEAT': 2}] + [{}] * hands, 'seeds': {'WHEAT': 5}, 'shed': {'WHEAT': 150}}}


class TestClasses(unittest.TestCase):
    def test_tables(self):
        self.assertEqual(len(JOBS), 40)
        self.assertEqual(len(ORDERS), 22)
        op, item = {s: i for i, s in enumerate(OPERATIONS)}, {s: i for i, s in enumerate(ITEMS)}
        self.assertEqual(JOBS[JOB_TABLE[op['PLANT'], item['WHEAT']]], ('PLANT', 'WHEAT'))
        self.assertEqual(JOB_TABLE[op['EAST'], 0], -1)
        self.assertEqual(NOW[NOW_TABLE[op['NONE']]], 'PASS')
        self.assertEqual(NOW[NOW_TABLE[op['EAST']]], 'EAST')
        self.assertEqual(NOW[NOW_TABLE[op['WATER']]], 'DO')


class TestLabels(unittest.TestCase):
    def test_walk_then_plant(self):
        X, A = game()
        L = policy_labels(X, A)
        units = L[:, :UNITS * 5].reshape(len(X), UNITS, 5)
        plant = JOBS.index(('PLANT', 'WHEAT'))
        # turns 0-8 walk east; at turn 9 the farmer is at (9, 0) = tile 9: t % 3 == 0 plants at 9
        for t in range(9):
            self.assertEqual(tuple(units[t, 0, [0, 2, 3, 4]]), (plant, 9, NOW.index('EAST'), 1), t)
        self.assertEqual(tuple(units[9, 0, [0, 2, 3]]), (plant, 9, NOW.index('DO')))
        self.assertEqual(units[10, 0, 0], JOBS.index(('WATER', None)))
        self.assertEqual(units[11, 0, 3], NOW.index('PASS'))
        # the hand: present 20-39, harvests at (5, 5) on odd turns; absent elsewhere
        self.assertEqual(tuple(units[20, 1, [0, 2, 3, 4]]), (JOBS.index(('HARVEST', None)), 55, NOW.index('PASS'), 1))
        self.assertEqual(units[39, 1, 3], NOW.index('DO'))
        self.assertEqual(units[45, 1, 4], 0)
        # the last turns have no job ahead once the farmer stops at a pass? (turn 59: 59 % 3 == 2: PASS)
        self.assertEqual(tuple(units[59, 0, [0, 2]]), (0, 9))
        market = L[:, UNITS * 5:UNITS * 5 + MARKET_SLOTS * 2].reshape(len(X), MARKET_SLOTS, 2)
        self.assertEqual(ORDERS[market[0, 0, 0]], ('BUY_SEED', 'WHEAT'))
        self.assertEqual(market[0, 0, 1], 5)
        self.assertEqual((ORDERS[market[19, 0, 0]], ORDERS[market[19, 1, 0]]), (('NONE', None), ('HIRE', None)))
        self.assertEqual((ORDERS[market[30, 0, 0]], market[30, 0, 1]), (('SELL', 'WHEAT'), 3))
        np.testing.assert_array_equal(L[:, -COMMAND_SLOTS * 3:], A)

    def test_a_unit_that_jumps_is_another_unit(self):
        X, A = game(4)
        for t in range(4):
            place(X[t], 1, 1 if t < 2 else 6, 1)   # the hand in slot 1 is replaced at turn 2
        A[3] = encode_action({'farmer': ['PASS'], 'hands': [['WATER']], 'market': []})
        units = policy_labels(X, A)[:, :UNITS * 5].reshape(4, UNITS, 5)
        self.assertEqual(units[1, 1, 0], 0)                            # no job ahead for the first hand
        self.assertEqual(tuple(units[2, 1, [0, 2]]), (JOBS.index(('WATER', None)), 16))

    def test_inputs(self):
        X, _ = game()
        first, later = inputs_at(X, 0), inputs_at(X, 30)
        self.assertEqual(first['globals'].shape, (9, len(first['globals'][0])))
        self.assertEqual(first['globals_mask'].tolist(), [True] + [False] * 8)
        self.assertTrue(later['globals_mask'].all())
        self.assertEqual(first['tiles'].shape, (TILES, 26))
        self.assertEqual(first['rows'].shape, (10, 260))
        self.assertEqual(later['units'].shape, (UNITS, 15))
        self.assertEqual(later['units_mask'][:2].tolist(), [True, True])
        self.assertEqual(later['unit_tile'][:2].tolist(), [9, 55])


def tiny(teams=2):
    torch.manual_seed(0)
    return Policy(PolicyConfig(width=32, layers=1, heads=2, market_layers=1, condition_dim=4, teams=teams))


class TestModel(unittest.TestCase):
    def test_loss_backward_act(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds = PolicyWindows(dataset(Path(tmp)), 'train', stride=5)
            batch = torch.utils.data.default_collate([ds[i] for i in range(6)])
        model = tiny()
        condition, team = torch.zeros(6, 8), torch.tensor([0, 1, 2, 0, 1, 2])
        loss, parts = policy_loss(model, model(batch, condition, team), batch)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(set(parts), {'job', 'amount', 'target', 'now', 'order', 'order_amount'})
        loss.backward()
        self.assertTrue(all(p.grad is not None for p in model.parameters()))
        model.eval()
        job, amount, target, now, orders, amounts = model.act(batch, condition, team)
        self.assertEqual((job.shape, target.shape, orders.shape), ((6, UNITS), (6, UNITS), (6, MARKET_SLOTS)))
        chosen = commands(job, amount, now, orders, amounts)
        self.assertEqual(chosen.shape, (6, COMMAND_SLOTS, 3))
        action = play.decode(chosen[0].reshape(-1).numpy(), 1, {})
        self.assertEqual(len(action['hands']), 1)

    def test_masked_loss_equals_the_kept_items_mean(self):
        from policy import _ce
        torch.manual_seed(0)
        logits, labels = torch.randn(4, 7, 5), torch.randint(0, 5, (4, 7))
        keep = torch.rand(4, 7) > 0.5
        reference = torch.nn.functional.cross_entropy(logits[keep], labels[keep])
        self.assertAlmostEqual(float(_ce(logits, labels, keep)), float(reference), places=5)
        self.assertEqual(float(_ce(logits, labels, torch.zeros_like(keep))), 0.0)

    def test_commands_of_known_choices(self):
        job = torch.zeros(1, UNITS, dtype=torch.long)
        job[0, 0] = JOBS.index(('PICKUP', 'WHEAT'))
        amount = torch.full((1, UNITS), 7)
        now = torch.full((1, UNITS), NOW.index('PASS'))
        now[0, 0] = NOW.index('DO')
        now[0, 1] = NOW.index('NORTH')
        orders = torch.zeros(1, MARKET_SLOTS, dtype=torch.long)
        orders[0, 1] = ORDERS.index(('SELL', 'MILK'))
        amounts = torch.full((1, MARKET_SLOTS), 4)
        action = play.decode(commands(job, amount, now, orders, amounts)[0].reshape(-1).numpy(), 2, {})
        self.assertEqual(action, {'farmer': ['PICKUP', 'WHEAT', 7], 'hands': [['NORTH'], ['PASS']],
                                  'market': [[], ['SELL', 'MILK', 4]]})


class TestTrainingRun(unittest.TestCase):
    def test_smoke_run_resume_and_player(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, out = dataset(Path(tmp) / 'data'), Path(tmp) / 'run'
            base = [sys.executable, str(HERE / 'policy_train.py'), '--data', str(data), '--out', str(out), '--smoke',
                    '--batch-size', '4', '--workers', '0', '--buffer-files', '2', '--condition', '--team-min-files', '1',
                    '--rating-halving', '50', '--margin-doubling', '0.1', '--recency-halving', '14']
            for leg in (['--eval-on-start', '--steps', '6', '--eval-every', '3'],
                        ['--resume', str(out / 'last.pt'), '--schedule-hours', '0.0001', '--eval-every', '100']):
                result = subprocess.run(base + leg, cwd=HERE, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr[-3000:])
            scores = json.loads((out / 'scores.json').read_text())
            self.assertTrue((out / 'complete').exists() and (out / 'best.pt').exists())
            self.assertGreaterEqual(scores['step'], 6)
            for key in ('val_loss', 'val_loss_weighted', 'first_active_command_accuracy',
                        'first_active_command_accuracy_weighted', 'farmer_command_accuracy', 'target_accuracy'):
                self.assertIn(key, scores)
            saved = torch.load(out / 'best.pt', map_location='cpu', weights_only=False)
            self.assertEqual(saved['conditions']['teams'], ['team0', 'team1'])
            self.assertIn(saved['conditions']['prompt_team'], ('team0', 'team1'))
            player = play.load_player(out / 'best.pt')
            self.assertIsInstance(player, play.PolicyPlayer)
            for step in range(3):
                action = player(observation(step))
                self.assertEqual(len(action['hands']), 1)
                self.assertIsInstance(action['market'], list)


class TestRealReplay(unittest.TestCase):
    def test_labels_on_a_real_game(self):
        if not REPLAY_ZIP.exists():
            self.skipTest('no real replay archive')
        with zipfile.ZipFile(REPLAY_ZIP) as z:
            for member in sorted(m for m in z.namelist() if m.endswith('.json')):
                try:
                    (X, A), _ = episode_arrays(json.loads(z.read(member)))
                    break
                except ValueError:   # a game the codec rejects (DataError)
                    continue
        L = policy_labels(X, A)
        units = L[:, :UNITS * 5].reshape(len(X), UNITS, 5)
        from policy import unit_tiles
        tiles = unit_tiles(X)
        command = A.reshape(len(X), COMMAND_SLOTS, 3)
        do = (units[..., 3] == NOW.index('DO')) & (units[..., 4] == 1)
        self.assertGreater(do.sum(), 100)
        t, u = np.nonzero(do)
        # a unit doing its job now: the job is this turn's command, at the tile it stands on
        self.assertTrue((units[t, u, 2] == tiles[t, u]).all())
        self.assertTrue((JOB_TABLE[command[t, u, 0], command[t, u, 1]] == units[t, u, 0]).all())
        jobs = {JOBS[j][0] for j in np.unique(units[..., 0][units[..., 4] == 1])}
        self.assertTrue({'PLANT', 'WATER', 'HARVEST'} <= jobs, jobs)
        # a unit walking to its job moves one tile closer (or around) most of the time
        walk = (units[:-1, :, 3] < 4) & (units[:-1, :, 0] > 0) & (units[:-1, :, 4] == 1) & (units[1:, :, 4] == 1)
        t, u = np.nonzero(walk)
        target = units[t, u, 2]
        before = np.abs(tiles[t, u] % 10 - target % 10) + np.abs(tiles[t, u] // 10 - target // 10)
        after = np.abs(tiles[t + 1, u] % 10 - target % 10) + np.abs(tiles[t + 1, u] // 10 - target // 10)
        self.assertGreater((after < before).mean(), 0.8)


if __name__ == '__main__':
    unittest.main()
