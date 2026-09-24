"""arena/colab_run.py with a fake Colab CLI: sharding, resume, merging and VM cleanup."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest

from arena import colab_run


def payload(root, n):
    jobs = [{'tag': 'best@mohui', 'a': 'bundles/best', 'b': 'bundles/mohui', 'seed': 100 + i, 'a_seat': i % 2}
            for i in range(n)]
    root.mkdir(parents=True)
    (root / 'jobs.json').write_text(json.dumps({'jobs': jobs}))
    (root / 'arena.py').write_text('# arena\n')
    return jobs


def result(job, margin):
    rewards = [0.0, 0.0]
    rewards[job['a_seat']], rewards[1 - job['a_seat']] = 1000.0 + margin, 1000.0
    return json.dumps(dict(job, statuses=['DONE', 'DONE'], rewards=rewards, errors=[]))


class FakeCLI:
    """One account: records calls, plays every uploaded shard at once, or fails as told."""
    calls, refuse, broken, stuck, running = [], set(), set(), set(), set()

    def __init__(self, command):
        self.command = command

    def new(self, session, high_mem):
        FakeCLI.calls.append((self.command, 'new', session, high_mem))
        if self.command in FakeCLI.refuse:
            return False, 'Service Unavailable'
        FakeCLI.running.add((self.command, session))
        return True, 'Session READY'

    def upload(self, session, local, remote):
        FakeCLI.calls.append((self.command, 'upload', session, remote))
        if remote.endswith('_shard.json'):
            FakeCLI.shards = getattr(FakeCLI, 'shards', {})
            FakeCLI.shards[session] = json.loads(Path(local).read_text())['jobs']
        return True, ''

    def exec_file(self, session, path, timeout):
        text = Path(path).read_text()
        if 'COLAB_ARENA_STARTED' in text:
            return 'COLAB_ARENA_STARTED 42 8 Python 3.12.13'
        if self.command in FakeCLI.broken:
            raise RuntimeError('websocket closed')
        return 'COLAB_ARENA_STATUS ' + json.dumps({'results': len(FakeCLI.shards[session]), 'done': True,
                                                   'alive': False, 'tail': 'ARENA_DONE'})

    def download(self, session, remote, local):
        FakeCLI.calls.append((self.command, 'download', session, remote))
        assert Path(local).is_absolute(), local   # the real CLI runs from the home directory
        jobs = FakeCLI.shards.get(session, [])
        if self.command in FakeCLI.broken:
            jobs = jobs[:1]   # a VM that died after one game: partial results
        Path(local).write_text(''.join(result(j, 5.0) + '\n' for j in jobs))
        return True, ''

    def stop(self, session):
        FakeCLI.calls.append((self.command, 'stop', session))
        if (self.command, session) in FakeCLI.stuck:
            FakeCLI.stuck.discard((self.command, session))   # the first stop does not take
        else:
            FakeCLI.running.discard((self.command, session))
        return 0, ''

    def sessions(self):
        mine = sorted(s for c, s in FakeCLI.running if c == self.command)
        return '\n'.join(f'[{s}] m-hm | Hardware: CPU' for s in mine) or 'No active sessions found on server.'


class ColabRunTests(unittest.TestCase):
    def setUp(self):
        FakeCLI.calls, FakeCLI.refuse, FakeCLI.broken, FakeCLI.shards = [], set(), set(), {}
        FakeCLI.stuck, FakeCLI.running = set(), set()

    def runner(self, tmp, vms):
        # relative paths, as typed on the command line
        return colab_run.ColabRun(Path(os.path.relpath(tmp / 'payload')), Path(os.path.relpath(tmp / 'results.jsonl')),
                                  'r', vms, poll=0.0, cli_factory=FakeCLI, log=lambda m: None)

    def test_shards_follow_capacity_and_every_session_is_stopped(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            payload(tmp / 'payload', 20)
            missing = self.runner(tmp, ['colab2:hm', 'colab4:std']).run()
            self.assertEqual(missing, 0)
            self.assertEqual({s: len(j) for s, j in FakeCLI.shards.items()}, {'r-0': 16, 'r-1': 4})
            self.assertIn(('colab2', 'new', 'r-0', True), FakeCLI.calls)
            self.assertIn(('colab4', 'new', 'r-1', False), FakeCLI.calls)
            stops = sorted(c[2] for c in FakeCLI.calls if c[1] == 'stop')
            self.assertEqual(stops, ['r-0', 'r-1'])
            self.assertEqual(len((tmp / 'results.jsonl').read_text().splitlines()), 20)

    def test_resume_plays_only_missing_games_and_merges_without_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            jobs = payload(tmp / 'payload', 6)
            (tmp / 'results.jsonl').write_text(''.join(result(j, 1.0) + '\n' for j in jobs[:4]))
            self.assertEqual(self.runner(tmp, ['colab2:hm']).run(), 0)
            self.assertEqual(len(FakeCLI.shards['r-0']), 2)
            lines = (tmp / 'results.jsonl').read_text().splitlines()
            self.assertEqual(len(lines), 6)
            self.assertEqual(len({(json.loads(l)['seed'], json.loads(l)['a_seat']) for l in lines}), 6)

    def test_a_failing_vm_is_still_stopped_and_its_partial_results_are_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            payload(tmp / 'payload', 8)
            FakeCLI.broken = {'colab3'}
            missing = self.runner(tmp, ['colab2:std', 'colab3:std']).run()
            self.assertEqual(missing, 3)          # colab3 played 1 of its 4 games
            self.assertEqual(sorted(c[2] for c in FakeCLI.calls if c[1] == 'stop'), ['r-0', 'r-1'])
            self.assertEqual(len((tmp / 'results.jsonl').read_text().splitlines()), 5)

    def test_a_vm_that_cannot_be_created_leaves_its_games_for_a_rerun(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            payload(tmp / 'payload', 4)
            FakeCLI.refuse = {'colab5'}
            missing = self.runner(tmp, ['colab1:std', 'colab5:std']).run()
            self.assertEqual(missing, 2)
            self.assertEqual([c[2] for c in FakeCLI.calls if c[1] == 'stop'], ['r-0'])
            FakeCLI.refuse = set()
            self.assertEqual(self.runner(tmp, ['colab1:std']).run(), 0)

    def test_attach_finishes_sessions_a_killed_runner_left_behind(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            jobs = payload(tmp / 'payload', 6)
            FakeCLI.shards = {'r-0': jobs[:3], 'r-1': jobs[3:]}   # uploaded by the runner that died
            missing = self.runner(tmp, ['colab2:hm', 'colab2:hm']).attach()
            self.assertEqual(missing, 0)
            self.assertFalse([c for c in FakeCLI.calls if c[1] in ('new', 'upload')])
            self.assertEqual(sorted(c[2] for c in FakeCLI.calls if c[1] == 'stop'), ['r-0', 'r-1'])
            self.assertEqual(len((tmp / 'results.jsonl').read_text().splitlines()), 6)

    def test_after_the_results_no_session_is_left(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            payload(tmp / 'payload', 4)
            FakeCLI.stuck = {('colab2', 'r-1')}           # its first stop does not take
            logs = []
            runner = self.runner(tmp, ['colab2:hm', 'colab2:hm'])
            runner.log = logs.append
            self.assertEqual(runner.run(), 0)
            self.assertIn('COLAB_VMS_LEFT=0', logs)
            self.assertEqual([c[2] for c in FakeCLI.calls if c[1] == 'stop'].count('r-1'), 2)
            self.assertEqual(FakeCLI('colab2').sessions(), 'No active sessions found on server.')

    def test_remote_scripts_render(self):
        setup = colab_run.SETUP.format(root=colab_run.REMOTE, workers=8)
        poll = colab_run.POLL.format(root=colab_run.REMOTE)
        compile(setup, 'setup', 'exec')
        compile(poll, 'poll', 'exec')
        self.assertIn("'--workers', '8'", setup)
        self.assertEqual(colab_run.parse_status('x\nCOLAB_ARENA_STATUS {"results": 3, "done": false}\n'),
                         {'results': 3, 'done': False})


if __name__ == '__main__':
    unittest.main()
