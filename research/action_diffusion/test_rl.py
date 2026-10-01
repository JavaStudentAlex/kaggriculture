"""Tests of the expert-iteration pieces: sampling and the seed check in play (policy._pick, fit_plants,
Policy.act, play.PolicyPlayer), the single market pass giving the two-pass choices, policy_train.py's
--keep-conditions, kad_arena.py's farm census, and kad_rl.py's selection (with the farm bonus), encoding,
training data, comparison, baseline, purge and bootstrap.
With a real replay archive (test_policy.REPLAY_ZIP) and the KAD-HP-1 checkpoint at REAL_MODEL, the
market pass is also checked on the real model and real turns."""
from __future__ import annotations

import dataclasses
import gzip
import hashlib
import json
import subprocess
import sys
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data import CROPS, FEATURE_DIM, episode_arrays, field_sizes
from policy import (JOBS, MARKET_SLOTS, NOW, PLANT_JOBS, UNITS, Policy, PolicyConfig, PolicyWindows, _pick,
                    fit_plants, inputs_at)
import kad_rl
import play
from test_policy import REPLAY_ZIP, dataset, observation, tiny
from train import condition_table

REAL_MODEL = Path.home() / 'kagg-colab/runs/kad_hp1_colab/resume/best.pt'


def two_pass_act(model, batch, condition=None, team=None):
    """Policy.act as it was before the single market pass: every head's most likely choice, the market
    decoder run twice per slot."""
    h, pad = model.encode(batch, condition, team)
    tiles, units = model.split(h)
    job = model.job(units).argmax(-1)
    z = units + model.job_embed(job)
    amount = model.amount(z).argmax(-1)
    target = torch.einsum('buw,btw->but', model.target_query(z), model.target_key(tiles)).argmax(-1)
    target = torch.where(job > 0, target, batch['unit_tile'].long())
    at = torch.gather(tiles, 1, target[..., None].expand(-1, -1, tiles.shape[-1]))
    now = model.now(z + model.target_embed(at)).argmax(-1)
    orders = torch.zeros(len(h), MARKET_SLOTS, dtype=torch.long)
    amounts = torch.zeros_like(orders)
    for k in range(MARKET_SLOTS):
        order_logits, _ = model.market_heads(h, pad, orders, amounts)
        orders[:, k] = order_logits[:, k].argmax(-1)
        _, amount_logits = model.market_heads(h, pad, orders, amounts)
        amounts[:, k] = amount_logits[:, k].argmax(-1)
    return job, amount, target, now, orders, amounts


def saved_tiny():
    model = tiny().eval()
    return dict(config=dataclasses.asdict(model.config), model=model.state_dict(), policy={},
                conditions=dict(prompt=dict(day='2026-09-25', rating_gap=0.0, win=1.0, excess_margin=0.03),
                                prompt_team='team1', teams=['team0', 'team1']))


class TestPick(unittest.TestCase):
    def test_greedy_and_draws(self):
        torch.manual_seed(0)
        logits = torch.randn(50, 7)
        self.assertTrue(torch.equal(_pick(logits), logits.argmax(-1)))
        first = _pick(logits, 1.0, torch.Generator().manual_seed(3))
        self.assertTrue(torch.equal(first, _pick(logits, 1.0, torch.Generator().manual_seed(3))))
        self.assertFalse(torch.equal(first, logits.argmax(-1)))

    def test_frequencies_follow_the_temperature(self):
        p = torch.tensor([0.7, 0.2, 0.1])
        logits = p.log().expand(40000, 3)
        for temperature, expected in ((1.0, p), (0.5, p ** 2 / (p ** 2).sum())):
            drawn = _pick(logits, temperature, torch.Generator().manual_seed(1))
            share = torch.bincount(drawn, minlength=3).float() / len(drawn)
            self.assertTrue(torch.allclose(share, expected, atol=0.01), (temperature, share, expected))


class TestFitPlants(unittest.TestCase):
    def setup(self, crops, tiles, now=None):
        """Units 0.. planting `crops` (None: no PLANT job) on `tiles`, all DO unless `now` says otherwise."""
        job = torch.zeros(1, UNITS, dtype=torch.long)
        now_ = torch.full((1, UNITS), NOW.index('PASS'))
        present = torch.zeros(1, UNITS, dtype=torch.bool)
        unit_tile = torch.zeros(1, UNITS, dtype=torch.long)
        for u, (crop, tile) in enumerate(zip(crops, tiles)):
            present[0, u] = True
            unit_tile[0, u] = tile
            job[0, u] = int(PLANT_JOBS[CROPS.index(crop)]) if crop else JOBS.index(('WATER', None))
            now_[0, u] = NOW.index((now or {}).get(u, 'DO'))
        return job, now_, unit_tile, present

    def test_within_seeds_nothing_changes(self):
        job, now, tiles, present = self.setup(['WHEAT', 'WHEAT', 'CARROT'], [0, 1, 2])
        seeds = torch.tensor([[2, 1, 0, 0, 0]])
        new_job, new_now = fit_plants(job, torch.zeros(1, UNITS, len(JOBS)), now, tiles, present, seeds)
        self.assertTrue(torch.equal(new_job, job) and torch.equal(new_now, now))

    def test_short_crops_switch_by_preference_then_pass(self):
        job, now, tiles, present = self.setup(['WHEAT', 'WHEAT', 'WHEAT', None, 'WHEAT'], [0, 1, 2, 3, 4],
                                              now={4: 'NORTH'})
        logits = torch.zeros(1, UNITS, len(JOBS))
        logits[0, 1, PLANT_JOBS[CROPS.index('TOMATO')]] = 5.0   # unit 1 likes tomatoes, which it has none of
        logits[0, 1, PLANT_JOBS[CROPS.index('MELON')]] = 3.0
        logits[0, 1, PLANT_JOBS[CROPS.index('CARROT')]] = 1.0
        seeds = torch.tensor([[1, 1, 0, 0, 1]])   # a wheat, a carrot and a melon seed
        new_job, new_now = fit_plants(job, logits, now, tiles, present, seeds)
        self.assertEqual(int(new_job[0, 0]), PLANT_JOBS[CROPS.index('WHEAT')])    # the farmer keeps the wheat
        self.assertEqual(int(new_job[0, 1]), PLANT_JOBS[CROPS.index('MELON')])    # its best crop with seeds
        self.assertEqual(int(new_job[0, 2]), PLANT_JOBS[CROPS.index('CARROT')])   # the last seed
        self.assertEqual(int(new_now[0, 2]), NOW.index('DO'))
        self.assertEqual(int(new_now[0, 3]), NOW.index('DO'))   # watering is untouched
        self.assertEqual(int(new_now[0, 4]), NOW.index('NORTH'))   # walking to plant later: untouched
        job, now, tiles, present = self.setup(['WHEAT', 'WHEAT'], [0, 1])
        new_job, new_now = fit_plants(job, logits, now, tiles, present, torch.tensor([[1, 0, 0, 0, 0]]))
        self.assertEqual(int(new_now[0, 1]), NOW.index('PASS'))   # no seed of any crop left

    def test_tiles_that_are_not_empty_pass(self):
        job, now, tiles, present = self.setup(['WHEAT', 'WHEAT', 'WHEAT'], [5, 7, 7])
        empty = torch.ones(1, 100, dtype=torch.bool)
        empty[0, 5] = False   # a plant already grows there
        new_job, new_now = fit_plants(job, torch.zeros(1, UNITS, len(JOBS)), now, tiles, present,
                                      torch.tensor([[9, 0, 0, 0, 0]]), empty)
        self.assertEqual([int(new_now[0, u]) for u in range(3)], [NOW.index('PASS'), NOW.index('DO'), NOW.index('PASS')])


class TestAct(unittest.TestCase):
    def test_single_market_pass_gives_the_two_pass_choices(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds = PolicyWindows(dataset(Path(tmp)), 'train', stride=3)
            batch = torch.utils.data.default_collate([ds[i] for i in range(12)])
        for seed in range(3):
            model = tiny()
            torch.manual_seed(seed)
            for p in model.parameters():   # random heads, so that the choices vary
                p.data.normal_(0, 0.3)
            model.eval()
            condition, team = torch.randn(12, 8), torch.randint(0, 3, (12,))
            with torch.no_grad():
                old = two_pass_act(model, batch, condition, team)
            new = model.act(batch, condition, team)
            for a, b in zip(old, new):
                self.assertTrue(torch.equal(a, b))

    def test_real_model_on_real_turns(self):
        if not (REAL_MODEL.exists() and REPLAY_ZIP.exists()):
            self.skipTest('no real checkpoint or replay here')
        saved = torch.load(REAL_MODEL, map_location='cpu', weights_only=False)
        model = Policy(PolicyConfig(**saved['config']))
        model.load_state_dict(saved['model'])
        model.eval()
        with zipfile.ZipFile(REPLAY_ZIP) as z:
            for member in sorted(m for m in z.namelist() if m.endswith('.json')):
                try:
                    (X, _), _ = episode_arrays(json.loads(z.read(member)))
                    break
                except ValueError:
                    continue
        turns = [0, 50, 150, 300, 450, 600, 700]
        batch = torch.utils.data.default_collate([inputs_at(X, t) for t in turns])
        team = torch.full((len(turns),), 1)
        with torch.no_grad():
            old = two_pass_act(model, batch, None, team)
        new = model.act(batch, None, team)
        for a, b in zip(old, new):
            self.assertTrue(torch.equal(a, b))


class TestPlayer(unittest.TestCase):
    def test_draws_replay_and_seeds_reach_the_model(self):
        saved = saved_tiny()
        runs = []
        for seed in (4, 4, 5):
            player = play.PolicyPlayer(saved, temperature=1.0, sample_seed=seed, fit_seeds=True)
            seen = []
            act = player.model.act

            def spy(*args, act=act, seen=seen):
                seen.append((args[5], args[6]))   # seeds, empty
                return act(*args)
            player.model.act = spy
            runs.append([json.dumps(player(observation(step, hands=2))) for step in range(4)])
        self.assertEqual(runs[0], runs[1])
        self.assertNotEqual(runs[0], runs[2])
        seeds, empty = seen[-1]
        self.assertEqual(seeds.tolist(), [[5, 0, 0, 0, 0]])
        self.assertFalse(bool(empty[0, 44]))   # tiles[4][4] holds wheat
        self.assertTrue(bool(empty[0, 45]))
        greedy = play.PolicyPlayer(saved)
        self.assertIsNone(greedy.generator)
        self.assertEqual(json.dumps(greedy(observation(0))), json.dumps(play.PolicyPlayer(saved)(observation(0))))


class TestKeepConditions(unittest.TestCase):
    def test_fine_tune_keeps_the_starting_prompt_and_teams(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            base = [sys.executable, str(HERE / 'policy_train.py'), '--smoke', '--batch-size', '4', '--workers', '0',
                    '--buffer-files', '2', '--condition', '--team-min-files', '1', '--steps', '2', '--eval-every', '2']
            first = subprocess.run(base + ['--data', str(dataset(tmp / 'a')), '--out', str(tmp / 'a_run')], cwd=HERE,
                                   capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr[-3000:])
            start = torch.load(tmp / 'a_run' / 'last.pt', map_location='cpu', weights_only=False)['conditions']
            other = dataset(tmp / 'b', files=5)   # other teams and a newer day
            manifest = json.loads((other / 'manifest.json').read_text())
            for i, f in enumerate(manifest['files']):
                f.update(team=f'other{i % 3}', day='2026-09-28', source='day-2026-09-28.zip')
            (other / 'manifest.json').write_text(json.dumps(manifest))
            second = subprocess.run(base + ['--data', str(other), '--out', str(tmp / 'b_run'), '--init-from',
                                            str(tmp / 'a_run' / 'last.pt'), '--keep-conditions'],
                                    cwd=HERE, capture_output=True, text=True)
            self.assertEqual(second.returncode, 0, second.stderr[-3000:])
            self.assertIn('"event": "init_from"', second.stdout)
            end = torch.load(tmp / 'b_run' / 'last.pt', map_location='cpu', weights_only=False)['conditions']
            self.assertEqual(end['teams'], start['teams'])
            self.assertEqual((end['prompt'], end['prompt_team']), (start['prompt'], start['prompt_team']))


def loop(root, **config):
    """A kad_rl.Loop on `root` without its constructor (no state file, no machine)."""
    self = kad_rl.Loop.__new__(kad_rl.Loop)
    self.root = Path(root)
    self.config = dict(kad_rl.DEFAULTS, **config)
    self.payload = self.root / 'payload'
    self.models = self.payload / 'models'
    self.state_path = self.root / 'state.json'
    self.state = dict(iteration=0, model='m0', eval_tag='m0', rejections=0, history=[])
    return self


def row(variant, opponent, seed, seat, own, opp, role='train', **extra):
    rewards = [None, None]
    rewards[seat], rewards[1 - seat] = own, opp
    return dict(tag=f'{variant}@{opponent}', variant=variant, opponent=opponent, seed=seed, a_seat=seat, role=role,
                group=f'{opponent}|{seed}|{seat}', rewards=rewards, statuses=['DONE', 'DONE'], errors=[], **extra)


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))


class TestLoop(unittest.TestCase):
    def test_select_keeps_sampled_games_that_beat_greedy(self):
        with tempfile.TemporaryDirectory() as tmp:
            rl = loop(tmp)
            d = Path(tmp) / 'it1'
            write(d / 'results.jsonl', [
                row('g', 'a', 1, 0, 50_000, 150_000), row('s1', 'a', 1, 0, 40_000, 150_000),
                row('s2', 'a', 1, 0, 61_000, 149_000),                     # s2 wins this group by +12k
                row('g', 'a', 2, 1, 55_000, 140_000), row('s1', 'a', 2, 1, 51_000, 140_000),
                row('s2', 'a', 2, 1, 54_000, 141_000),                     # greedy best: nothing kept
                row('g', 'b', 3, 0, 1, 2), row('s1', 'b', 3, 0, 9, 2),     # s2 not in yet: undecided
                dict(row('s2', 'b', 3, 0, 5, 2), errors=[{'phase': 'action'}])])
            encoded = []
            rl.encode = lambda it, directory, r, prompt, gain: encoded.append((r['variant'], r['seed'], gain)) or 'x'
            rl.select(1, d, ['g', 's1', 's2'], 'g', prompt={})
            decided = {e['group']: e for e in kad_rl.read_jsonl(d / 'selected.jsonl')}
            self.assertEqual(set(decided), {'a|1|0', 'a|2|1'})
            self.assertEqual((decided['a|1|0']['selected'], decided['a|1|0']['gain']), ('s2', 12_000))
            self.assertIsNone(decided['a|2|1']['selected'])
            self.assertEqual(encoded, [('s2', 1, 12_000)])
            rl.select(1, d, ['g', 's1', 's2'], 'g', prompt={})   # decided groups stay decided
            self.assertEqual(len(kad_rl.read_jsonl(d / 'selected.jsonl')), 2)

    def test_select_scores_farms_against_the_mean_of_the_samples(self):
        def farm(day10, day20):
            return {'0': {'10': dict(value=day10), '20': dict(value=day20)},
                    '1': {'10': dict(value=0), '20': dict(value=0)}}

        with tempfile.TemporaryDirectory() as tmp:
            rl = loop(tmp, farm_weight=2.0, farm_days=[10, 20])
            d = Path(tmp) / 'it1'
            write(d / 'results.jsonl', [
                # s2 has the best margin, s1 a farm worth 1,500 more: 2 x 1,500 beats 2,000 of margin
                row('s1', 'a', 1, 0, 90_000, 150_000, farm=farm(1_000, 3_000)),   # -60k + 2 x 2,000 = -56k
                row('s2', 'a', 1, 0, 92_000, 150_000, farm=farm(0, 1_000)),       # -58k + 2 x 500 = -57k
                row('s3', 'a', 1, 0, 80_000, 150_000, farm=farm(1_000, 1_000)),   # -70k + 2 x 1,000 = -68k
                # a game without a census: the group is scored by margins alone
                row('s1', 'a', 2, 0, 90_000, 150_000, farm=farm(5_000, 5_000)),
                row('s2', 'a', 2, 0, 92_000, 150_000),
                row('s3', 'a', 2, 0, 80_000, 150_000, farm=farm(0, 0))])
            encoded = []
            rl.encode = lambda it, directory, r, prompt, gain: encoded.append((r['variant'], r['seed'], gain)) or 'x'
            rl.select(1, d, ['s1', 's2', 's3'], None, prompt={})
            decided = {e['group']: e for e in kad_rl.read_jsonl(d / 'selected.jsonl')}
            first, second = decided['a|1|0'], decided['a|2|0']
            self.assertEqual((first['selected'], first['by_margin'], first['gain']), ('s1', 's2', 4_333))   # - mean
            self.assertEqual(first['farms'], {'s1': 2_000, 's2': 500, 's3': 1_000})
            self.assertEqual((second['selected'], second['by_margin'], second['gain']), ('s2', 's2', 4_667))
            self.assertIsNone(second['farms']['s2'])
            self.assertEqual(encoded, [('s1', 1, 4_333), ('s2', 2, 4_667)])

    def test_farm_value_and_stats(self):
        r = row('c1', 'a', 1, 1, 60_000, 150_000, role='eval',
                farm={'1': {'10': dict(value=1_000), '20': dict(value=3_000)},
                      '0': {'10': dict(value=8_000), '20': dict(value=12_000)}})
        self.assertEqual(kad_rl.farm_value(r, [10, 20]), 2_000)
        self.assertEqual(kad_rl.farm_value(r, [10, 20], seat=0), 10_000)
        self.assertIsNone(kad_rl.farm_value(r, [10, 25]))
        self.assertIsNone(kad_rl.farm_value(row('c1', 'a', 1, 1, 1, 2), [10]))
        s = kad_rl.stats([r, row('c1', 'a', 2, 0, 1, 2)], [10, 20])   # the second has no census
        self.assertEqual((s['mean_farm'], s['opponent_farm'], s['games']), (2_000, 10_000, 2))
        self.assertNotIn('mean_farm', kad_rl.stats([r]))

    def test_variants_sample_as_play_says(self):
        rl = loop('/nonexistent', play=dict(temperature=0.7, sample_seed=7, fit_seeds=False), temperatures=[0.7, 1.0])
        variants, greedy = rl.variants(3)
        self.assertIsNone(greedy)
        self.assertEqual(variants, {'it3s1': dict(temperature=0.7, sample_seed=3001, fit_seeds=False),
                                    'it3s2': dict(temperature=1.0, sample_seed=3002, fit_seeds=False)})
        rl.config['greedy_reference'] = True
        variants, greedy = rl.variants(3)
        self.assertEqual((greedy, variants[greedy]['temperature'], len(variants)), ('it3g', 0.0, 3))

    def test_baseline_plays_as_play_says_and_runs_the_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rl = loop(tmp, baseline_checks={'fit': {'fit_seeds': True}}, eval_opponents=['a'], eval_seeds=3)
            rl.state['eval_tag'] = None
            (root / 'payload' / 'agent').mkdir(parents=True)
            (root / 'payload' / 'agent' / 'main.py').write_text('')

            def wait(proc, directory, jobs):
                write(directory / 'results.jsonl', [
                    row(j['variant'], j['opponent'], j['seed'], j['a_seat'], 60_000 + 5_000 * (j['variant'] == 'm0fit'),
                        150_000, role='eval') for j in jobs])

            rl.arena = lambda directory, name, jobs, episodes: None
            rl.wait = wait
            rl.baseline()
            settings = {name: json.loads((root / 'payload' / 'bundles' / name / 'settings.json').read_text())
                        for name in ('m0', 'm0fit')}
            self.assertEqual(settings['m0'], dict(checkpoint='../../models/m0.pt', device='cuda', temperature=0.0,
                                                  sample_seed=7, fit_seeds=False))
            self.assertEqual(settings['m0fit'], dict(settings['m0'], fit_seeds=True))
            record = rl.state['history'][-1]
            self.assertEqual((record['eval']['games'], record['eval']['mean_margin']), (6, -90_000))
            self.assertEqual((record['checks']['fit']['mean_delta'], record['checks']['fit']['better']), (5_000, 6))
            self.assertEqual(rl.state['eval_tag'], 'm0')

    def test_encode_labels_the_seat_with_the_prompt(self):
        if not REPLAY_ZIP.exists():
            self.skipTest('no real replay archive')
        with zipfile.ZipFile(REPLAY_ZIP) as z, tempfile.TemporaryDirectory() as tmp:
            for member in sorted(m for m in z.namelist() if m.endswith('.json')):
                doc = json.loads(z.read(member))
                try:
                    episode_arrays(doc)
                    break
                except ValueError:
                    continue
            rl = loop(tmp)
            d = Path(tmp) / 'it1'
            (d / 'episodes').mkdir(parents=True)
            with gzip.open(d / 'episodes' / 'game.json.gz', 'wt') as fh:
                json.dump(doc, fh)
            prompt = dict(day='2026-09-25', excess_margin=0.036, team='Boey')
            name = rl.encode(1, d, row('s1', 'a', 7, 1, 60_000, 150_000, episode='game.json.gz'), prompt, 900)
            records = kad_rl.read_jsonl(Path(tmp) / 'own' / 'records.jsonl')
            self.assertEqual(records[0]['episode_id'], name)
            # honest labels: a lost game stays a loss; the gain over the group sets the margin condition
            self.assertEqual((records[0]['seat'], records[0]['team'], records[0]['win'], records[0]['margin']),
                             (1, 'Boey', 0.0, 0.09))
            (Path(tmp) / 'data').mkdir()
            (Path(tmp) / 'data' / 'manifest.json').write_text(json.dumps(dict(
                feature_dim=FEATURE_DIM, field_sizes=field_sizes,
                files=[dict(r, path='../' + r['path']) for r in records])))
            windows = PolicyWindows(Path(tmp) / 'data', 'train', stride=1)
            X, labels = windows.read(0)
            self.assertEqual(X.shape[0], records[0]['length'])
            self.assertTrue(np.array_equal(X, episode_arrays(doc)[1][0]))
            from model import encode_condition
            self.assertTrue(np.allclose(condition_table(windows)[0],
                                        encode_condition('2026-09-25', 0.0, 0.0, 0.09)))

    def test_training_data_mixes_kept_games_and_top_replays(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rl = loop(tmp, replay_days=['2026-09-24', '2026-09-25'], val_day='2026-09-25', replay_ratio=2.0,
                      replay_top_rank=2)
            for day in rl.config['replay_days']:
                files = []
                for i in range(8):
                    split = 'val' if i == 0 else 'train'
                    files.append(dict(path=f'{split}/e{i}_seat0.npz', episode_id=f'{day}-{i}', seat=0, split=split,
                                      length=10, win=float(i % 2), team_rank=1 + i % 4))
                files.append(dict(files[1], split='train', episode_id=f'2026-09-25-0'))   # a val game elsewhere
                write_manifest(root / 'days' / day, files)
            write(root / 'own' / 'records.jsonl', [dict(path=f'own/k{i}.npz', episode_id=f'k{i}', seat=0,
                                                        split='train', length=10) for i in range(2)])
            counts = rl.training_data(1)
            manifest = json.loads((root / 'data' / 'manifest.json').read_text())
            train = [f for f in manifest['files'] if f['split'] == 'train']
            val = [f for f in manifest['files'] if f['split'] == 'val']
            self.assertEqual(counts, dict(own=2, replay=4, replay_pool=4, val=1))
            self.assertEqual([f['path'] for f in train[:2]], ['../own/k0.npz', '../own/k1.npz'])
            self.assertTrue(all(f['win'] == 1 and f['team_rank'] <= 2 for f in train[2:]))
            self.assertTrue(all(f['path'].startswith('../days/') for f in train[2:] + val))
            self.assertNotIn('2026-09-25-0', {f['episode_id'] for f in train})
            self.assertEqual(val[0]['episode_id'], '2026-09-25-0')

    def test_compare_pairs_the_same_games(self):
        with tempfile.TemporaryDirectory() as tmp:
            rl = loop(tmp)
            write(Path(tmp) / 'it0' / 'results.jsonl', [row('m0', 'a', s, seat, 50_000, 150_000, role='eval')
                                                        for s in (1, 2) for seat in (0, 1)])
            write(Path(tmp) / 'it1' / 'results.jsonl', [row('c1', 'a', s, seat, 50_000 + 1000 * s, 150_000, role='eval')
                                                        for s in (1, 2, 3) for seat in (0, 1)])
            result = rl.compare('c1', 'm0')
            self.assertEqual((result['games'], result['mean_delta'], result['better'], result['worse']), (4, 1500, 4, 0))
            self.assertEqual(result['new']['losses'], 4)

    def test_purge_drops_undecided_games_without_episodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            rl = loop(tmp)
            d = Path(tmp) / 'it2'
            (d / 'episodes').mkdir(parents=True)
            (d / 'episodes' / 'kept.json.gz').write_text('x')
            write(d / 'results.jsonl', [row('g', 'a', 1, 0, 1, 2, episode='gone.json.gz'),    # decided: stays
                                        row('g', 'a', 2, 0, 1, 2, episode='gone.json.gz'),    # undecided, gone
                                        row('s1', 'a', 2, 0, 1, 2, episode='kept.json.gz'),   # undecided, on disk
                                        row('c2', 'a', 9, 0, 1, 2, role='eval', episode=False)])
            write(d / 'selected.jsonl', [dict(group='a|1|0', selected=None)])
            rl.purge(d)
            left = [(r['variant'], r['seed']) for r in kad_rl.read_jsonl(d / 'results.jsonl')]
            self.assertEqual(left, [('g', 1), ('s1', 2), ('c2', 9)])

    def test_pack_builds_the_bootstrap(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / 'best.pt'
            model.write_bytes(b'weights')
            out = subprocess.run([sys.executable, str(HERE / 'kad_rl.py'), 'pack', '--model', str(model), '--out', tmp,
                                  '--set', 'iterations=2', '--set', 'eval_opponents=["abo_v57"]',
                                  '--set', 'train_opponents=["abo_v43"]'], capture_output=True, text=True)
            self.assertEqual(out.returncode, 0, out.stderr[-2000:])
            with tarfile.open(Path(tmp) / 'kadrl.tar.gz') as tar:
                names = set(tar.getnames())
                config = json.loads(tar.extractfile('./config.json').read())
                agent_graph = tar.extractfile('./payload/agent/agent_graph.py').read()
                policy_graph = json.loads(tar.extractfile('./payload/agent/policy_graph.json').read())
                self.assertEqual(hashlib.sha256(agent_graph).hexdigest(),
                                 policy_graph['provenance']['entrypoint_sha256'])
            for name in ('./code/kad_rl.py', './code/policy_train.py', './payload/arena.py', './payload/bundle_agent.py',
                         './payload/kad_arena.py', './payload/agent/main.py',
                         './payload/agent/hazel_runtime/kad_copilot.py', './payload/agent/hazel_runtime/kad/kad_torch.py',
                         './payload/bundles/abo_v57/main.py', './payload/bundles/abo_v43/main.py',
                         './payload/models/m0.pt', './requirements.txt'):
                self.assertIn(name, names)
            self.assertEqual((config['iterations'], config['eval_opponents']), (2, ['abo_v57']))
            # the player is Juniper Knoll's graph with the copilot acting on the GPU (both levers on)
            node = next(n for n in policy_graph['turn']['nodes'] if n['id'] == 'kad_copilot')
            self.assertEqual({k: node['parameters'][k] for k in ('_KC_SELL', '_KC_HANDS', '_KC_BACKEND')},
                             {'_KC_SELL': True, '_KC_HANDS': True, '_KC_BACKEND': 'torch'})


def import_kad_arena():
    sys.path.insert(0, str(HERE.parent / 'procedural_graph' / 'arena'))   # kad_arena imports arena
    try:
        import kad_arena
    except ImportError as exc:   # no kaggle_environments here
        raise unittest.SkipTest(f'kad_arena does not import: {exc}')
    return kad_arena


class TestFarmCensus(unittest.TestCase):
    def test_census_counts_land_animals_and_crops(self):
        kad_arena = import_kad_arena()
        tiles = [[None] * 10 for _ in range(10)]
        tiles[0][0] = {'kind': 'PASTURE', 'animal': 'COW'}
        tiles[0][1] = {'kind': 'COOP', 'animal': 'GOOSE'}
        tiles[0][2] = {'kind': 'PASTURE'}                    # an empty structure costs nothing
        tiles[1][0] = {'kind': 'PLANT', 'crop': 'WHEAT'}
        tiles[1][1] = {'kind': 'PLANT', 'crop': 'STRAWBERRY'}
        tiles[1][2] = {'kind': 'WEED'}
        tiles[9][9] = 'LOCKED'
        ours = dict(money=1234.6, hands=[[1, 1]], farmer=[4, 4], unlocked_quadrants=['NW', 'NE', 'SW'], tiles=tiles)
        theirs = dict(money=3000.0, hands=[], farmer=[4, 4], unlocked_quadrants=['NW'],
                      tiles=[[None] * 10 for _ in range(10)])
        steps = [[{'observation': {'farms': [ours, theirs]}}, {'observation': {}}] for _ in range(481)]
        census = kad_arena.farm_census(steps)
        self.assertEqual(sorted(census['0'], key=int), ['5', '10', '15', '20'])   # day 25 is past the end
        day = census['0']['10']
        self.assertEqual(day['value'], 1_000 + 2_000 + 400 + 300 + 10 + 100)
        self.assertEqual((day['quadrants'], day['hands'], day['money']), (3, 1, 1235))
        self.assertEqual(day['animals'], {'GOOSE': 1, 'COW': 1, 'SHEEP': 0})
        self.assertEqual((day['crops']['WHEAT'], day['crops']['STRAWBERRY'], day['crops']['MELON']), (1, 1, 0))
        self.assertEqual((census['1']['20']['value'], census['1']['20']['quadrants']), (0, 1))

    def test_census_of_a_real_replay(self):
        kad_arena = import_kad_arena()
        if not REPLAY_ZIP.exists():
            self.skipTest('no real replay archive')
        with zipfile.ZipFile(REPLAY_ZIP) as z:
            member = sorted(m for m in z.namelist() if m.endswith('.json'))[0]
            doc = json.loads(z.read(member))
        census = kad_arena.farm_census(doc['steps'])
        self.assertEqual(set(census), {'0', '1'})
        for seat in census.values():
            self.assertEqual(sorted(seat, key=int), ['5', '10', '15', '20', '25'])
            for day in seat.values():
                self.assertTrue(1 <= day['quadrants'] <= 4 and day['value'] >= 0 and day['money'] >= 0)
        # a top game's farms grow: land or animals by day 20 on at least one side
        self.assertTrue(any(s['20']['quadrants'] > 1 or sum(s['20']['animals'].values()) for s in census.values()))


def write_manifest(directory, files):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'manifest.json').write_text(json.dumps(dict(
        version=1, alignment='next', feature_dim=FEATURE_DIM, field_sizes=field_sizes, codec={}, split={},
        files=files)))


if __name__ == '__main__':
    unittest.main()
