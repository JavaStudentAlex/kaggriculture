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
    doomed, deleted, polls, polls_until_done = set(), set(), {}, 1

    def __init__(self, command):
        self.command = command

    def new(self, session, high_mem, gpu=None):
        FakeCLI.calls.append((self.command, 'new', session, high_mem))
        if self.command in FakeCLI.refuse:
            if self.command in getattr(FakeCLI, 'halfmade', set()):
                FakeCLI.running.add((self.command, session))   # reported as failed, but the VM exists
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
            assert "'--trace-dir', 'traces'" in text
            if self.command in getattr(FakeCLI, 'broken_setup', set()):
                return 'COLAB_ARENA_ERROR requirements'
            return 'COLAB_ARENA_STARTED 42 8 Python 3.12.13'
        if 'COLAB_ARENA_PACKED' in text:
            return f"COLAB_ARENA_PACKED {len(FakeCLI.shards.get(session, []))}"
        if self.command in FakeCLI.broken:
            raise RuntimeError('websocket closed')
        lost = f"[colab] Session '{session}' appears to be lost (404/401). Cleaning up."
        if session in FakeCLI.deleted:
            return lost
        n = FakeCLI.polls[session] = FakeCLI.polls.get(session, 0) + 1
        if session in FakeCLI.doomed:   # plays one game, then Colab deletes it
            if n >= 2:
                FakeCLI.deleted.add(session)
                FakeCLI.running.discard((self.command, session))
                return lost
            return 'COLAB_ARENA_STATUS ' + json.dumps({'results': 1, 'done': False, 'alive': True, 'tail': ''})
        done = n >= FakeCLI.polls_until_done
        return 'COLAB_ARENA_STATUS ' + json.dumps({'results': len(FakeCLI.shards[session]), 'done': done,
                                                   'alive': not done, 'tail': 'ARENA_DONE' if done else ''})

    def download(self, session, remote, local):
        FakeCLI.calls.append((self.command, 'download', session, remote))
        assert Path(local).is_absolute(), local   # the real CLI runs from the home directory
        if session in FakeCLI.deleted:
            return False, 'lost'
        jobs = FakeCLI.shards.get(session, [])
        if session in FakeCLI.doomed:
            jobs = jobs[:1]
        if remote.endswith('traces.tgz'):
            import io, tarfile
            with tarfile.open(local, 'w:gz') as tar:
                for j in jobs:
                    data = b'{}'
                    info = tarfile.TarInfo(f"{j['tag']}_seed{j['seed']}_aseat{j['a_seat']}.json.gz")
                    info.size = len(data)
                    tar.addfile(info, io.BytesIO(data))
            return True, ''
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
        return '\n'.join(f'[{s}] ep-{s} | Hardware: CPU' for s in mine) or 'No active sessions found on server.'

    def refresh(self, session, endpoint):
        FakeCLI.calls.append((self.command, 'refresh', session))
        return 'deleted' if session in FakeCLI.deleted else 'ok'


class ColabRunTests(unittest.TestCase):
    def setUp(self):
        FakeCLI.calls, FakeCLI.refuse, FakeCLI.broken, FakeCLI.shards = [], set(), set(), {}
        FakeCLI.stuck, FakeCLI.running, FakeCLI.broken_setup, FakeCLI.halfmade = set(), set(), set(), set()
        FakeCLI.doomed, FakeCLI.deleted, FakeCLI.polls, FakeCLI.polls_until_done = set(), set(), {}, 1

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
            self.assertEqual(len(list((tmp / 'traces').glob('*.json.gz'))), 20)   # one trace per game
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

    def test_a_refused_vm_gives_its_games_to_the_vms_that_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            payload(tmp / 'payload', 4)
            FakeCLI.refuse = {'colab5'}
            missing = self.runner(tmp, ['colab1:std', 'colab5:std']).run()
            self.assertEqual(missing, 0)
            self.assertEqual(len(FakeCLI.shards['r-0']), 4)
            # r-1 gets a precautionary stop right after its failed create, r-0 after its games
            self.assertEqual(sorted(c[2] for c in FakeCLI.calls if c[1] == 'stop'), ['r-0', 'r-1'])

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

    def test_start_only_leaves_vms_playing_until_attach_pulls_and_removes_them(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            payload(tmp / 'payload', 6)
            FakeCLI.refuse = set()
            runner = colab_run.ColabRun(Path(os.path.relpath(tmp / 'payload')), Path(os.path.relpath(tmp / 'results.jsonl')),
                                        'r', ['colab2:hm', 'colab2:hm'], poll=0.0, cli_factory=FakeCLI,
                                        log=lambda m: None, detach=True)
            self.assertEqual(runner.run(), 0)
            self.assertFalse([c for c in FakeCLI.calls if c[1] in ('stop', 'download')])
            self.assertEqual(sorted(s for c, s in FakeCLI.running), ['r-0', 'r-1'])
            self.assertFalse((tmp / 'results.jsonl').exists())
            # later: attach pulls results and traces, then removes every VM
            self.assertEqual(self.runner(tmp, ['colab2:hm', 'colab2:hm']).attach(), 0)
            self.assertEqual(len((tmp / 'results.jsonl').read_text().splitlines()), 6)
            self.assertEqual(len(list((tmp / 'traces').glob('*.json.gz'))), 6)
            self.assertEqual(FakeCLI.running, set())

    def test_start_only_still_removes_a_vm_whose_setup_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            payload(tmp / 'payload', 2)
            FakeCLI.broken_setup = {'colab3'}
            runner = colab_run.ColabRun(tmp / 'payload', tmp / 'results.jsonl', 'r', ['colab3:std'], poll=0.0,
                                        cli_factory=FakeCLI, log=lambda m: None, detach=True)
            runner.run()
            self.assertEqual([c[2] for c in FakeCLI.calls if c[1] == 'stop'], ['r-0'])
            self.assertEqual(FakeCLI.running, set())

    def test_a_vm_reported_as_failed_but_created_is_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            payload(tmp / 'payload', 4)
            FakeCLI.refuse = FakeCLI.halfmade = {'colab2'}
            runner = colab_run.ColabRun(tmp / 'payload', tmp / 'results.jsonl', 'r', ['colab2:hm', 'colab1:std'],
                                        poll=0.0, cli_factory=FakeCLI, log=lambda m: None)
            self.assertEqual(runner.run(), 0)
            self.assertEqual(len(FakeCLI.shards['r-0']), 4)   # the High-RAM VM that "failed" plays
            self.assertEqual(FakeCLI.running, set())

    def test_a_vm_colab_deletes_is_replaced_and_only_its_unplayed_games_are_replayed(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            payload(tmp / 'payload', 8)
            FakeCLI.doomed = {'r-1'}   # plays one game, which the 10-minute pull keeps, then is deleted
            logs = []
            runner = colab_run.ColabRun(tmp / 'payload', tmp / 'results.jsonl', 'r', ['colab1:std', 'colab3:std'],
                                        poll=0.0, cli_factory=FakeCLI, log=logs.append, pull_every=0.0)
            self.assertEqual(runner.run(), 0)
            self.assertIn(('colab3', 'new', 'r-1-1', False), FakeCLI.calls)
            self.assertEqual(len(FakeCLI.shards['r-1-1']), 3)
            self.assertEqual(len((tmp / 'results.jsonl').read_text().splitlines()), 8)
            self.assertEqual(len(list((tmp / 'traces').glob('*.json.gz'))), 8)
            self.assertEqual(FakeCLI.running, set())
            self.assertIn('COLAB_VMS_LEFT=0', logs)
            self.assertIn('[r-1-1] colab3 link: https://colab.research.google.com/notebooks/empty.ipynb'
                          '?dbu=%2Ftun%2Fm%2Fep-r-1-1#datalabBackendUrl=https://colab.research.google.com/tun/m/ep-r-1-1',
                          logs)

    def test_tokens_are_refreshed_while_a_vm_plays(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            payload(tmp / 'payload', 2)
            FakeCLI.polls_until_done = 3
            runner = colab_run.ColabRun(tmp / 'payload', tmp / 'results.jsonl', 'r', ['colab1:std'], poll=0.0,
                                        cli_factory=FakeCLI, log=lambda m: None, refresh_every=0.0)
            self.assertEqual(runner.run(), 0)
            self.assertGreaterEqual(FakeCLI.calls.count(('colab1', 'refresh', 'r-0')), 3)

    def test_jobs_file_plays_only_the_listed_games(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            jobs = payload(tmp / 'payload', 12)
            lost = [j for i, j in enumerate(jobs) if i % 3]   # the shards of VMs that died
            (tmp / 'lost.json').write_text(json.dumps({'jobs': lost}))
            runner = colab_run.ColabRun(tmp / 'payload', tmp / 'results.jsonl', 'r2', ['colab1:std', 'colab3:std'],
                                        poll=0.0, cli_factory=FakeCLI, log=lambda m: None, jobs=tmp / 'lost.json')
            self.assertEqual(runner.run(), 0)
            played = sorted(j['seed'] for shard in FakeCLI.shards.values() for j in shard)
            self.assertEqual(played, sorted(j['seed'] for j in lost))
            self.assertEqual(len((tmp / 'results.jsonl').read_text().splitlines()), 8)

    def test_a_vm_without_a_local_record_is_reported_as_left(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            payload(tmp / 'payload', 2)
            FakeCLI.running = {('colab1', 'r-10')}   # another session whose name starts with r-1
            logs = []
            runner = self.runner(tmp, ['colab1:std', 'colab1:std'])
            runner.log = logs.append
            unnamed = '[?] m-s-lost | Hardware: CPU'
            listing = FakeCLI.sessions
            FakeCLI.sessions = lambda self: listing(self) + '\n' + unnamed
            try:
                runner.run()
            finally:
                FakeCLI.sessions = listing
            self.assertNotIn(('colab1', 'stop', 'r-10'), FakeCLI.calls)   # exact names only
            self.assertIn("COLAB_VMS_LEFT=1 ['colab1:m-s-lost (no local record)'] -- stop them by hand", logs)

    def test_remote_scripts_render(self):
        setup = colab_run.SETUP.format(root=colab_run.REMOTE, workers=8, trace_args="['--trace-dir', 'traces']")
        poll = colab_run.POLL.format(root=colab_run.REMOTE)
        pack = colab_run.PACK.format(root=colab_run.REMOTE)
        for name, code in (('setup', setup), ('poll', poll), ('pack', pack)):
            compile(code, name, 'exec')
        self.assertIn("'results.jsonl'] + ['--trace-dir', 'traces']", setup)
        self.assertIn("'--workers', '8'", setup)
        self.assertEqual(colab_run.parse_status('x\nCOLAB_ARENA_STATUS {"results": 3, "done": false}\n'),
                         {'results': 3, 'done': False})


    def test_large_upload_goes_in_parts_and_is_joined(self):
        """A file above UPLOAD_CHUNK is uploaded in parts; the VM joins them and checks size and hash."""
        class LocalCLI(colab_run.ColabCLI):
            def __init__(self):
                self.uploads = []

            def _run(self, args, timeout):
                if args[0] == 'upload':
                    self.uploads.append(args[-1])
                    Path(args[-1]).write_bytes(Path(args[-2]).read_bytes())
                    return 0, 'Uploaded'
                raise AssertionError(args)

            def exec_file(self, session, path, timeout):
                import subprocess
                import sys
                return subprocess.run([sys.executable, str(path)], capture_output=True, text=True).stdout

        chunk = colab_run.UPLOAD_CHUNK
        colab_run.UPLOAD_CHUNK = 1000
        try:
            with tempfile.TemporaryDirectory() as tmp:
                src, remote = Path(tmp) / 'big.bin', str(Path(tmp) / 'remote.bin')
                src.write_bytes(os.urandom(2500))
                cli = LocalCLI()
                ok, out = cli.upload('s', src, remote)
                self.assertTrue(ok, out)
                self.assertEqual(len(cli.uploads), 3)
                self.assertEqual(Path(remote).read_bytes(), src.read_bytes())
                self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()), ['big.bin', 'remote.bin'])
        finally:
            colab_run.UPLOAD_CHUNK = chunk

    def test_cli_output_is_redacted(self):
        text = 'failed: eyJhbGciOiJSUzI1NiJ9.eyJhdWQiOiJncHUifQ.c2lnbmF0dXJl (Caused by SSLError)'
        self.assertEqual(colab_run.redact(text), 'failed: <token> (Caused by SSLError)')


if __name__ == '__main__':
    unittest.main()
