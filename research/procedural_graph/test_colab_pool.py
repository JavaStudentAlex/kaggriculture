"""arena/colab_pool.py with a fake Colab CLI: batches, bundle reuse, VM loss, retries, idle stop,
and graph_gauntlet.ColabPoolExecutor talking to it."""
from __future__ import annotations

import json
import re
import tempfile
import threading
import unittest
from pathlib import Path

from arena import colab_pool
import graph_gauntlet


def job(i, a='g_cand', b='opp'):
    return {'tag': f'{a}@{b}', 'a': f'bundles/{a}', 'b': f'bundles/{b}', 'seed': 100 + i, 'a_seat': i % 2}


def run_dir(root, bundles=('g_cand', 'opp')):
    root.mkdir(parents=True, exist_ok=True)
    for f in ('arena.py', 'bundle_agent.py'):
        if not (root / f).exists():
            (root / f).write_text(f'# {f}\n')
    for name in bundles:
        (root / 'bundles' / name).mkdir(parents=True, exist_ok=True)
        (root / 'bundles' / name / 'main.py').write_text(f'# {name}\n')
    for sub in ('jobs', 'games', 'logs'):
        (root / sub).mkdir(exist_ok=True)
    return root


class FakeCLI:
    """Plays every started chunk after `polls_until_done` polls; Colab deletes `doomed` VMs."""

    def __init__(self, command):
        self.command = command

    @classmethod
    def reset(cls):
        cls.running, cls.deleted, cls.chunks, cls.polls, cls.uploads = set(), set(), {}, {}, []
        cls.doomed, cls.error_once, cls.errored, cls.played, cls.polls_until_done = {}, set(), set(), [], 2
        cls.made, cls.bundle_uploads, cls.harness_uploads = [], [], []

    def new(self, session, high_mem, gpu=None):
        FakeCLI.made.append(session)
        FakeCLI.running.add((self.command, session))
        return True, 'Session READY'

    def upload(self, session, local, remote):
        FakeCLI.uploads.append((session, remote, Path(local).name))
        if remote.endswith('_jobs.json'):
            chunk = remote.rsplit('/', 1)[1][:-len('_jobs.json')]
            FakeCLI.chunks[chunk] = json.loads(Path(local).read_text())['jobs']
        if '_files_' in remote:
            import tarfile
            with tarfile.open(local) as tar:
                names = tar.getnames()
            FakeCLI.bundle_uploads.append((session, sorted({n.split('/')[1] for n in names
                                                            if n.startswith('bundles/') and n.count('/') >= 1})))
            FakeCLI.harness_uploads.append((session, sorted(n for n in names if '/' not in n and n.endswith('.py'))))
        return True, ''

    def exec_file(self, session, path, timeout):
        text = Path(path).read_text()
        lost = f"[colab] Session '{session}' appears to be lost (404/401). Cleaning up."
        if session in FakeCLI.deleted:
            return lost
        if 'COLAB_POOL_READY' in text:
            return 'COLAB_POOL_READY 8 Python 3.12.13'
        if 'COLAB_POOL_FILES_OK' in text:
            return 'COLAB_POOL_FILES_OK'
        if 'COLAB_POOL_STARTED' in text:
            chunk = re.search(r"root, chunk, workers = '[^']*', '([^']*)'", text).group(1)
            FakeCLI.polls[chunk] = 0
            return 'COLAB_POOL_STARTED 42'
        chunk = re.search(r"root, chunk = '[^']*', (None|'[^']*')", text).group(1).strip("'")
        if session in FakeCLI.doomed:
            FakeCLI.doomed[session] -= 1
            if FakeCLI.doomed[session] < 0:
                FakeCLI.deleted.add(session)
                FakeCLI.running = {r for r in FakeCLI.running if r[1] != session}
                return lost
        if chunk == 'None':
            return 'COLAB_POOL_STATUS ' + json.dumps({'results': 0, 'done': False, 'alive': False, 'tail': ''})
        FakeCLI.polls[chunk] += 1
        done = FakeCLI.polls[chunk] >= FakeCLI.polls_until_done
        n = len(FakeCLI.chunks[chunk]) if done else 0
        return 'COLAB_POOL_STATUS ' + json.dumps({'results': n, 'done': done, 'alive': not done, 'tail': ''})

    def download(self, session, remote, local):
        if session in FakeCLI.deleted:
            return False, 'lost'
        chunk = remote.split('/batches/')[1].split('/')[0]
        if remote.endswith('arena.log'):
            Path(local).write_text(f'arena log of {chunk}\n')
            return True, ''
        if FakeCLI.polls.get(chunk, 0) < FakeCLI.polls_until_done:
            Path(local).write_text('')
            return True, ''
        lines = []
        for j in FakeCLI.chunks[chunk]:
            FakeCLI.played.append((session, j['seed'], j['a_seat']))
            rewards = [1000.0, 900.0] if j['a_seat'] == 0 else [900.0, 1000.0]
            if j['seed'] in FakeCLI.error_once and j['seed'] not in FakeCLI.errored:
                FakeCLI.errored.add(j['seed'])
                lines.append(json.dumps(dict(j, statuses=['ERROR', 'DONE'], rewards=None, errors=['boom'])))
            else:
                lines.append(json.dumps(dict(j, statuses=['DONE', 'DONE'], rewards=rewards, errors=[])))
        Path(local).write_text(''.join(line + '\n' for line in lines))
        return True, ''

    def stop(self, session):
        FakeCLI.running = {r for r in FakeCLI.running if r[1] != session}
        return 0, ''

    def sessions(self):
        mine = sorted(s for c, s in FakeCLI.running if c == self.command)
        return '\n'.join(f'[{s}] ep-{s} | Hardware: CPU' for s in mine) or 'No active sessions found on server.'

    def refresh(self, session, endpoint):
        return 'deleted' if session in FakeCLI.deleted else 'ok'


class ColabPoolTests(unittest.TestCase):
    def setUp(self):
        FakeCLI.reset()
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.logs = []

    def tearDown(self):
        self.tmp.cleanup()

    def pool(self, vms=('colab2:hm', 'colab2:hm'), **kw):
        return colab_pool.ColabPool(self.root / 'pool', 'evo', list(vms), poll=0, cli_factory=FakeCLI,
                                    log=self.logs.append, sleep=lambda s: None, **kw)

    def request(self, name, jobs, root=None):
        root = root or run_dir(self.root / 'run')
        (self.root / 'pool' / 'inbox').mkdir(parents=True, exist_ok=True)
        (self.root / 'pool' / 'inbox' / f'{name}.json').write_text(
            json.dumps({'name': name, 'root': str(root), 'jobs': jobs}))

    def outbox(self, name):
        done = json.loads((self.root / 'pool' / 'outbox' / f'{name}.done').read_text())
        rows = [json.loads(l) for l in (self.root / 'pool' / 'outbox' / f'{name}.jsonl').read_text().splitlines()]
        return done, rows

    def test_request_split_over_vms_and_everything_stopped(self):
        jobs = [job(i) for i in range(10)]
        self.request('r1', jobs)
        self.pool().serve(once=True)
        done, rows = self.outbox('r1')
        self.assertEqual((done['games'], done['finished'], done['missing']), (10, 10, 0))
        self.assertEqual({(r['seed'], r['a_seat']) for r in rows}, {(j['seed'], j['a_seat']) for j in jobs})
        self.assertEqual(len({s for s, _, _ in FakeCLI.played}), 2)    # both VMs played
        self.assertFalse(FakeCLI.running)                                # every VM stopped
        self.assertIn('COLAB_VMS_LEFT=0', self.logs)
        self.assertFalse(list((self.root / 'pool' / 'inbox').glob('*.json')))
        self.assertIn('arena log of', (self.root / 'pool' / 'outbox' / 'r1.log').read_text())

    def test_bundle_reuse_within_a_vm_life(self):
        root = run_dir(self.root / 'run', bundles=('g_cand', 'g_next', 'opp'))
        pool = self.pool(vms=('colab2:hm',))
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for name, cand in (('r1', 'g_cand'), ('r2', 'g_next')):
                self.request(name, [job(i, a=cand) for i in range(3)], root)
                path, req = pool.next_request()
                results, logs = pool.play(req, tmp)
                pool.finish(path, req, results, logs)
            pool.stop_all()
        self.assertEqual([names for _, names in FakeCLI.bundle_uploads], [['g_cand', 'opp'], ['g_next']])
        # the harness goes up with the first request only (its bytes did not change)
        self.assertEqual([names for _, names in FakeCLI.harness_uploads], [['arena.py', 'bundle_agent.py'], []])
        self.assertEqual(self.outbox('r2')[0]['finished'], 3)

    def test_deleted_vm_games_replayed_elsewhere(self):
        FakeCLI.doomed['evo-0-1'] = 0      # answers no poll: deleted right after it started its chunk
        jobs = [job(i) for i in range(8)]
        self.request('r1', jobs)
        self.pool().serve(once=True)
        done, rows = self.outbox('r1')
        self.assertEqual((done['finished'], done['missing']), (8, 0))
        self.assertNotIn('evo-0-1', {s for s, _, _ in FakeCLI.played})
        self.assertIn('evo-0-2', FakeCLI.made)                            # the slot got a replacement
        self.assertFalse(FakeCLI.running)

    def test_short_lived_vms_use_up_the_replacements(self):
        for n in (1, 2, 3):
            FakeCLI.doomed[f'evo-0-{n}'] = 0
        self.request('r1', [job(i) for i in range(8)])
        self.pool(replacements=2).serve(once=True)
        self.assertEqual(self.outbox('r1')[0]['finished'], 8)            # the other slot played them
        self.assertIn('evo-0-3', FakeCLI.made)
        self.assertNotIn('evo-0-4', FakeCLI.made)                        # slot 0 is out of replacements

    def test_long_lived_vms_do_not_use_up_the_replacements(self):
        # Colab ends VMs that ran for hours: such a slot is healthy and keeps getting VMs
        for n in (1, 2, 3):
            FakeCLI.doomed[f'evo-0-{n}'] = 0
        self.request('r1', [job(i) for i in range(8)])
        self.pool(replacements=2, long_life=0).serve(once=True)
        self.assertEqual(self.outbox('r1')[0]['finished'], 8)
        self.assertIn('evo-0-4', FakeCLI.made)
        self.assertFalse(FakeCLI.running)

    def test_errored_game_retried_once(self):
        FakeCLI.error_once = {101}
        self.request('r1', [job(i) for i in range(4)])
        self.pool(vms=('colab2:hm',)).serve(once=True)
        done, rows = self.outbox('r1')
        self.assertEqual((done['finished'], done['errored']), (4, 0))
        self.assertEqual(sum(1 for _, seed, _ in FakeCLI.played if seed == 101), 2)

    def test_idle_pool_stops_vms(self):
        pool = self.pool(vms=('colab2:hm',), idle_stop=0)
        self.request('r1', [job(0)])
        with tempfile.TemporaryDirectory() as tmp:
            path, req = pool.next_request()
            pool.finish(path, req, *pool.play(req, Path(tmp)))
        self.assertTrue(FakeCLI.running)
        (self.root / 'pool' / 'STOP').write_text('')
        calls = []
        pool.sleep = lambda s: calls.append(s)
        pool.idle_since = 0
        pool.serve()
        self.assertFalse(FakeCLI.running)

    def test_executor_round_trip(self):
        root = run_dir(self.root / 'run')
        jobs = [job(i) for i in range(6)]
        (root / 'jobs' / 'g_cand.json').write_text(json.dumps({'jobs': jobs}))
        # two games already finished in an earlier attempt: only four go to the pool
        with (root / 'games' / 'g_cand.jsonl').open('w') as fh:
            for j in jobs[:2]:
                fh.write(json.dumps(dict(j, statuses=['DONE', 'DONE'], rewards=[1.0, 0.0], errors=[])) + '\n')
        pool = self.pool(vms=('colab2:hm',))
        executor = graph_gauntlet.ColabPoolExecutor(self.root / 'pool', poll=0.01, timeout=30)
        t = threading.Thread(target=executor.run, args=(root, 'g_cand', ['g_cand', 'opp']))
        t.start()
        while not list((self.root / 'pool' / 'inbox').glob('*.json')):
            pass
        pool.serve(once=True)
        t.join()
        rows = [json.loads(l) for l in (root / 'games' / 'g_cand.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows), 6)
        self.assertEqual(len(FakeCLI.played), 4)
        self.assertIn('arena log of', (root / 'logs' / 'g_cand.log').read_text())


if __name__ == '__main__':
    unittest.main()
