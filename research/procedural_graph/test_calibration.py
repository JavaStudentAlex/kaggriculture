"""Predictor calibration: the quantile-matching fit, its guards, the wrapper's report parsing and
the game-set payload (arena/calib_fit.py, arena/calibrate.py, arena/calib_payload.py)."""
from __future__ import annotations

import gzip
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from arena import calibrate

HERE = Path(__file__).resolve().parent

PRODUCTS = ['WHEAT', 'EGG', 'MILK']


def load_fit():
    """calib_fit.py is a script; import its functions without running main()."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('calib_fit', HERE / 'arena' / 'calib_fit.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CalibrationFitTests(unittest.TestCase):
    def synthetic(self, n=20000, scale=1.25):
        """old scores in [0, 1]; the new model's are `scale` times higher; WHEAT sells, EGG never,
        MILK sells but the new model is 3x inflated (to hit the clip)."""
        rng = np.random.default_rng(0)
        old = rng.uniform(0, 1, size=(n, 3)).astype(np.float32)
        new = old * np.array([scale, scale, 3.0], dtype=np.float32)
        sells = (rng.uniform(0, 1, size=(n, 3)) < old).astype(np.float32)
        sells[:, 1] = 0   # EGG: never sold
        d = {'truth4': sells, 'truth24': sells}
        for key in ('s4', 's24'):
            d[f'old_{key}'], d[f'new_{key}'] = old, new
        d['old_u24'], d['new_u24'] = old * 4, new * 4
        return d

    def test_factor_undoes_a_uniform_inflation(self):
        fit = load_fit()
        factors, _ = fit.fit(self.synthetic(scale=1.25), 'old', 'new', PRODUCTS)
        for metric in ('score_4', 'score_24', 'units_24'):
            self.assertAlmostEqual(factors[metric]['WHEAT'], 0.8, delta=0.02)

    def test_guards_no_sales_means_no_scaling_and_factors_are_clipped(self):
        fit = load_fit()
        factors, report = fit.fit(self.synthetic(), 'old', 'new', PRODUCTS)
        self.assertEqual(factors['score_4']['EGG'], 1.0)   # base rate 0: no evidence
        self.assertEqual(factors['score_4']['MILK'], fit.FACTOR_RANGE[0])   # 1/3 clipped to 0.5
        self.assertTrue(any('clipped' in line for line in report))

    def test_wrapper_reads_auc_rows_from_a_report(self):
        report = ('\n== score_4 (thresholds [0.3])\n'
                  '  WHEAT       base 27.9%  AUC old 0.682 new 0.642  factor 0.830  fire new*f/old: 1%/1%\n'
                  '  EGG         base  0.0%  AUC old nan new nan  factor 1.000  fire new*f/old: 0%/0%\n')
        rows = calibrate.auc_rows(report)
        self.assertEqual(rows[('score_4', 'WHEAT')], (27.9, '0.682', '0.642'))
        self.assertEqual(rows[('score_4', 'EGG')], (0.0, 'nan', 'nan'))


class GameSetPayloadTests(unittest.TestCase):
    def test_game_set_becomes_trace_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            gs = tmp / 'game_set'
            gs.mkdir()
            games = [{'trace': '00000_a@b_seed11_aseat0.json.gz', 'game_seed': 11, 'seats': [0], 'group': 'a@b'},
                     {'trace': '00001_a_seed12_aseat1.json.gz', 'game_seed': 12, 'seats': [0, 1], 'group': 'a'}]
            with zipfile.ZipFile(gs / 'traces.zip', 'w') as z:
                for g in games:
                    z.writestr(g['trace'], gzip.compress(b'[]'))
            sha = hashlib.sha256((gs / 'traces.zip').read_bytes()).hexdigest()
            (gs / 'games.json').write_text(json.dumps({'traces_sha256': sha, 'games': games}))
            model = HERE / 'hazel_runtime' / 'checkpoint'
            eid = 'test-calibration-game-set'
            try:
                subprocess.run([sys.executable, str(HERE / 'arena' / 'calib_payload.py'), '--game-set', str(gs),
                                '--model', f'old={model}', '--model', f'new={model}', '--stride', '1',
                                '--eval-id', eid], check=True, capture_output=True)
                payload = HERE / 'runs' / 'arena' / eid / 'payload'
                jobs = json.loads((payload / 'jobs.json').read_text())['jobs']
                self.assertEqual([(j['trace'], j['game_seed'], j['seats'], j['group']) for j in jobs],
                                 [(g['trace'], g['game_seed'], g['seats'], g['group']) for g in games])
                self.assertEqual(len({j['seed'] for j in jobs}), 2)   # unique job keys for colab_run
                with zipfile.ZipFile(payload / 'traces.zip') as z:
                    self.assertEqual(sorted(z.namelist()), sorted(g['trace'] for g in games))
            finally:
                import shutil
                shutil.rmtree(HERE / 'runs' / 'arena' / eid, ignore_errors=True)

    def test_committed_game_set_is_intact(self):
        gs = HERE / 'calibration' / 'game_set'
        meta = json.loads((gs / 'games.json').read_text())
        self.assertEqual(hashlib.sha256((gs / 'traces.zip').read_bytes()).hexdigest(), meta['traces_sha256'])
        with zipfile.ZipFile(gs / 'traces.zip') as z:
            self.assertEqual(sorted(z.namelist()), sorted(g['trace'] for g in meta['games']))
        self.assertEqual(len(meta['games']), 600)


if __name__ == '__main__':
    unittest.main()
