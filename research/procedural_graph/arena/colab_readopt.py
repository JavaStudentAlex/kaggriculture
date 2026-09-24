#!/usr/bin/env python3
"""Re-register running Colab VMs whose local CLI records were pruned, and restart their keep-alive.

The Colab CLI drops a session's local record, and kills its keep-alive daemon, when one
`list_assignments` call misses it. On 2026-09-24 that happened to three VMs that were still
playing, right after the PC woke up. Such a VM shows up in `colab<N> sessions` as
`[?] <endpoint>`. Exec, download, stop and keep-alive by name then all fail, and without
pings Colab deletes the VM. This script puts the record back from the server's listing
(endpoint, runtime proxy URL and token) and starts a new keep-alive daemon, as `colab new` does:

    python3 arena/colab_readopt.py <N> <endpoint>=<session name> [<endpoint>=<session name> ...]

<N> is the account number of the `colab<N>` wrapper. The endpoints come from `colab<N>
sessions`; the names are the run's `<run-name>-<index>` (also in
~/.config/colab-cli/history/<name>.jsonl, event session_created). Runs under the CLI's own
Python. It prints no tokens.
"""
import os
import sys

TOOL_PYTHON = os.path.expanduser('~/.local/share/uv/tools/google-colab-cli/bin/python')


def main():
    if len(sys.argv) < 3 or not sys.argv[1].isdigit():
        raise SystemExit(__doc__)
    account, pairs = sys.argv[1], sys.argv[2:]
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.expanduser(f'~/.config/colab-cli/acc{account}_adc.json')
    try:
        import colab_cli  # noqa: F401
    except ImportError:   # re-run under the CLI's interpreter, which has colab_cli
        os.execv(TOOL_PYTHON, [TOOL_PYTHON, os.path.abspath(__file__), *sys.argv[1:]])
    from colab_cli.auth import AuthProvider
    from colab_cli.commands.session import spawn_keep_alive
    from colab_cli.common import state
    from colab_cli.state import SessionState

    state.config_path = os.path.expanduser(f'~/.config/colab-cli/sessions_acc{account}.json')
    state.auth_provider = AuthProvider('adc')
    names = dict(pair.split('=', 1) for pair in pairs)
    found = set()
    for a in state.client.list_assignments():
        name = names.get(a.endpoint)
        if not name:
            continue
        found.add(a.endpoint)
        if state.store.get(name):
            print(f'{name}: already registered, left as it is')
            continue
        s = SessionState(name=name, token=a.runtime_proxy_info.token, url=a.runtime_proxy_info.url,
                         endpoint=a.endpoint, accelerator=a.accelerator.value, variant=a.variant.name,
                         machine_shape=a.machine_shape.name)
        state.store.add(s)   # before the daemon, which exits if it finds no record
        try:
            state.client.keep_alive_assignment(a.endpoint)
            ping = 'ok'
        except Exception as exc:
            ping = f'failed ({type(exc).__name__})'
        s.keep_alive_pid = spawn_keep_alive(a.endpoint, name, auth_provider=state.auth_provider,
                                            config_path=state.config_path)
        state.store.add(s)
        state.history.log_event(name, 'session_readopted', {'endpoint': a.endpoint, 'pid': s.keep_alive_pid})
        print(f'{name}: re-registered, ping {ping}, keep-alive pid {s.keep_alive_pid}')
    for endpoint, name in names.items():
        if endpoint not in found:
            print(f'{name}: {endpoint} is not on the server any more (deleted)')


if __name__ == '__main__':
    main()
