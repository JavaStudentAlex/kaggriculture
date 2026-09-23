"""Persistent, process-isolated saved-agent RPC. Standard library only.

Wrappers use BundleAgent(bundle, entrypoint='main.py'). Each instance lazily
starts its own clean interpreter; no candidate/oracle modules are imported into
the caller. JSON-compatible observations/configurations/actions cross a JSONL
pipe. Exceptions, exits and deadlines are errors, never fallback actions.
"""
from __future__ import annotations

import atexit
import importlib.abc
import importlib.machinery
import importlib.util
import json
import math
import os
from pathlib import Path
import selectors
import subprocess
import sys
import threading
import time
import traceback
import weakref


class BundleAgentError(RuntimeError):
    """Remote import/action failure, child exit or invalid RPC response."""


class BundleAgentTimeout(TimeoutError):
    """The child failed to complete import or an action before its deadline."""


_LIVE = weakref.WeakSet()
_THREAD_VARS = (
    'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
    'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS',
    'OMP_THREAD_LIMIT', 'NUMBA_NUM_THREADS',
)


def _environment(bundle):
    env = {k: v for k, v in os.environ.items() if not k.startswith('KAGG_')}
    dependency = next((p for p in (bundle / 'mohui_v66',
                                  bundle / 'dependencies' / 'mohui_v66', bundle)
                       if (p / 'candidate_v66_meta_closed_loop.py').is_file()), bundle)
    env.update({
        'KAGG_MOHUI_DIR': str(dependency),
        'KAGG_ORACLE_SRC': str(bundle),
        'KAGG_TTM_DIR': str(bundle / 'checkpoint'),
        'KAGG_OPP_MODEL_SRC': str(bundle / 'opponent_model'),
        'KAGG_TASK_DIR': str(bundle),
        'KAGG_ORACLE_BACKEND': 'numpy',
        'KAGG_ORACLE_DEVICE': 'cpu',
        'CUDA_VISIBLE_DEVICES': '',
    })
    env.update({key: '1' for key in _THREAD_VARS})
    return env


class BundleAgent:
    """One persistent child per instance (and per owning parent PID).

    startup_timeout includes imports; timeout bounds each complete request,
    including pipe writes. close() is idempotent. Failed/closed instances do not
    silently restart or lose episode state. start() exposes import diagnostics
    for local preflight without running a game. Children inherit stderr only.
    """

    def __init__(self, bundle, entrypoint='main.py', *, timeout=60.0,
                 startup_timeout=120.0):
        self.bundle = Path(bundle).resolve()
        self.entrypoint = entrypoint
        entry = (self.bundle / entrypoint).resolve()
        if not entry.is_relative_to(self.bundle) or not entry.is_file():
            raise ValueError(f'entrypoint must be a file inside bundle: {entry}')
        self.timeout = float(timeout)
        self.startup_timeout = float(startup_timeout)
        if not all(math.isfinite(t) and t > 0 for t in (self.timeout, self.startup_timeout)):
            raise ValueError('timeouts must be positive finite seconds')
        self._owner = os.getpid()
        self._proc = None
        self._buffer = bytearray()
        self._lock = threading.RLock()
        self._closed = False
        self._seq = 0
        self.metadata = None
        _LIVE.add(self)

    @property
    def pid(self):
        return self._proc.pid if self._proc else None

    def _check_owner(self):
        if self._owner != os.getpid():
            # A multiprocessing fork must never share RPC streams or kill the
            # original parent's child. Close only this process's duplicate FDs.
            if self._proc:
                self._proc.stdin.close()
                self._proc.stdout.close()
            self._proc = None
            self._owner = os.getpid()
            self._buffer = bytearray()
            self._lock = threading.RLock()
            self._closed = False
            self._seq = 0
            self.metadata = None

    def _wait(self, stream, event, deadline):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise BundleAgentTimeout(f'{self.bundle.name}: RPC deadline exceeded')
        with selectors.DefaultSelector() as selector:
            selector.register(stream, event)
            if not selector.select(remaining):
                raise BundleAgentTimeout(f'{self.bundle.name}: RPC deadline exceeded')

    def _read(self, deadline):
        while b'\n' not in self._buffer:
            self._wait(self._proc.stdout, selectors.EVENT_READ, deadline)
            try:
                chunk = os.read(self._proc.stdout.fileno(), 65536)
            except BlockingIOError:
                continue
            if not chunk:
                code = self._proc.poll()
                raise BundleAgentError(f'{self.bundle.name}: child exited/closed RPC (exit={code})')
            self._buffer.extend(chunk)
        line, _, rest = self._buffer.partition(b'\n')
        self._buffer = bytearray(rest)
        try:
            reply = json.loads(line)
        except (ValueError, UnicodeError) as exc:
            raise BundleAgentError(f'{self.bundle.name}: invalid JSONL reply') from exc
        if not isinstance(reply, dict):
            raise BundleAgentError(f'{self.bundle.name}: RPC reply is not an object')
        if not reply.get('ok'):
            raise BundleAgentError(
                f"{self.bundle.name}: remote {reply.get('phase', 'unknown')} "
                f"{reply.get('error_type', 'Error')}: {reply.get('error', '')}\n"
                f"{reply.get('traceback', '')}")
        return reply

    def _write(self, data, deadline):
        view = memoryview(data)
        while view:
            self._wait(self._proc.stdin, selectors.EVENT_WRITE, deadline)
            try:
                n = os.write(self._proc.stdin.fileno(), view)
            except BlockingIOError:
                continue
            except BrokenPipeError as exc:
                raise BundleAgentError(f'{self.bundle.name}: child closed request pipe') from exc
            view = view[n:]

    def start(self):
        self._check_owner()
        with self._lock:
            if self._closed:
                raise BundleAgentError(f'{self.bundle.name}: agent is closed/failed')
            if self._proc is not None:
                return self.metadata
            deadline = time.monotonic() + self.startup_timeout
            try:
                self._proc = subprocess.Popen(
                    [sys.executable, '-I', '-u', '-B', str(Path(__file__).resolve()),
                     '--child', str(self.bundle), self.entrypoint],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=None, bufsize=0, cwd=self.bundle,
                    env=_environment(self.bundle), close_fds=True,
                )
                os.set_blocking(self._proc.stdin.fileno(), False)
                os.set_blocking(self._proc.stdout.fileno(), False)
                reply = self._read(deadline)
                if reply.get('phase') != 'ready':
                    raise BundleAgentError(f'{self.bundle.name}: missing ready handshake')
                self.metadata = reply['metadata']
                return self.metadata
            except BaseException:
                self.close()
                raise

    def __call__(self, observation, configuration=None):
        self._check_owner()
        with self._lock:
            self._seq += 1
            # Strict JSON before launch: no default=str, pickle or lossy fallback.
            request = json.dumps({'id': self._seq, 'observation': observation,
                                  'configuration': configuration},
                                 allow_nan=False, separators=(',', ':')).encode() + b'\n'
            self.start()
            try:
                deadline = time.monotonic() + self.timeout
                self._write(request, deadline)
                reply = self._read(deadline)
                if reply.get('id') != self._seq:
                    raise BundleAgentError(f'{self.bundle.name}: mismatched response ID')
                return reply['action']
            except BaseException:
                self.close()
                raise

    def close(self):
        if self._owner != os.getpid():
            self._check_owner()
        with self._lock:
            self._closed = True
            proc, self._proc = self._proc, None
            if proc is None:
                return
            try:
                proc.stdin.close()  # EOF exits an idle child normally.
                try:
                    proc.wait(timeout=0.25)
                except subprocess.TimeoutExpired:
                    proc.terminate()
                    try:
                        proc.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
            finally:
                proc.stdout.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def _cleanup():
    for client in list(_LIVE):
        client.close()


atexit.register(_cleanup)


class _BundleImports(importlib.abc.MetaPathFinder):
    """Prefer frozen top-level modules even if legacy main inserts an absolute path.

    Normal Python/installed dependency imports still work. This is namespace
    isolation for trusted agents, not a security sandbox.
    """

    def __init__(self, roots):
        self.roots = roots

    def find_spec(self, fullname, path=None, target=None):
        if path is not None or '.' in fullname:
            return None
        return importlib.machinery.PathFinder.find_spec(fullname, self.roots)


def _metadata(bundle, entry):
    modules = {}
    for name, module in list(sys.modules.items()):
        file = getattr(module, '__file__', None)
        if file and Path(file).resolve().is_relative_to(bundle):
            modules[name] = str(Path(file).resolve().relative_to(bundle))
    champ = sys.modules.get('champion')
    model = getattr(champ, '_MODEL', None)
    return {
        'pid': os.getpid(), 'entrypoint': str(entry.relative_to(bundle)),
        'bundle_modules': modules,
        'environment': {k: v for k, v in os.environ.items()
                        if k.startswith('KAGG_') or k in _THREAD_VARS},
        'oracle_stats': getattr(champ, 'ORACLE_STATS', None),
        'oracle_backend': getattr(model, 'backend', None),
        'oracle_model_dir': getattr(model, 'model_dir', None),
    }


def _child(bundle_arg, entrypoint):
    # Reserve a private protocol FD, then redirect fd 1 itself. print(), native
    # library writes, os.write(1, ...) and agent subprocesses all go to stderr.
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), 'w', buffering=1, encoding='utf-8')
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr

    def send(message):
        protocol.write(json.dumps(message, allow_nan=False, separators=(',', ':')) + '\n')
        protocol.flush()

    def error(phase, exc, request_id=None):
        send({'ok': False, 'phase': phase, 'id': request_id,
              'error_type': type(exc).__name__, 'error': str(exc),
              'traceback': traceback.format_exc()})

    try:
        bundle = Path(bundle_arg).resolve()
        entry = (bundle / entrypoint).resolve()
        if not entry.is_relative_to(bundle):
            raise ValueError('entrypoint escapes bundle')
        # Also sanitize here so direct --child invocation is deterministic.
        env = _environment(bundle)
        os.environ.clear()
        os.environ.update(env)
        roots = list(dict.fromkeys(map(str, (entry.parent, bundle,
                     Path(env['KAGG_MOHUI_DIR']), bundle / 'opponent_model'))))
        sys.path[:0] = roots
        sys.meta_path.insert(0, _BundleImports(roots))
        spec = importlib.util.spec_from_file_location('_saved_bundle_main', entry)
        if spec is None or spec.loader is None:
            raise ImportError(f'cannot load {entry}')
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        agent = getattr(module, 'agent', None)
        if not callable(agent):
            raise TypeError(f'{entry} has no callable agent')
        send({'ok': True, 'phase': 'ready', 'metadata': _metadata(bundle, entry)})
    except BaseException as exc:
        error('import', exc)
        return 1

    for line in sys.stdin:
        request = None
        try:
            request = json.loads(line)
            action = agent(request['observation'], request['configuration'])
            send({'ok': True, 'id': request['id'], 'action': action})
        except BaseException as exc:
            error('action', exc, request.get('id') if isinstance(request, dict) else None)
            return 1
    return 0


if __name__ == '__main__':
    if len(sys.argv) != 4 or sys.argv[1] != '--child':
        raise SystemExit('internal usage: pool_upgrade_bundle_agent.py --child BUNDLE ENTRYPOINT')
    raise SystemExit(_child(sys.argv[2], sys.argv[3]))
