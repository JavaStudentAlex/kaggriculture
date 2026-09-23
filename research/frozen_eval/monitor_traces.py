"""Monitor the 4 trace-mining shard kernels on Kaggle."""
from pathlib import Path
import json, subprocess, sys, time

RUN_DIR = Path(__file__).resolve().parent / 'runs' / 'iter27_trace_mining'
SHARDS = 4
EVALUATION_ID = 'iter27-trace-mining-20260923-400x2-v1'


def kaggle_status(ref):
    p = subprocess.run(['kaggle', 'kernels', 'status', ref],
                       capture_output=True, text=True, timeout=30)
    return (p.stdout + p.stderr).strip()


def download_output(ref, dest):
    dest.mkdir(parents=True, exist_ok=True)
    p = subprocess.run(['kaggle', 'kernels', 'output', ref, '-p', str(dest)],
                       capture_output=True, text=True, timeout=300)
    return p.returncode, (p.stdout + p.stderr).strip()


def main():
    once = '--once' in sys.argv
    interval = 180

    while True:
        states = []
        for i in range(SHARDS):
            ref = f'sunshinethroughfog/kagg-iter27-trace-{i}'
            status = kaggle_status(ref)
            entry = {'shard': i, 'status': status}

            if 'COMPLETE' in status or 'ERROR' in status:
                dest = RUN_DIR / 'remote_results' / str(i)
                rc, msg = download_output(ref, dest)
                entry['downloaded'] = rc == 0

                summary = dest / 'results' / 'summary.json'
                if summary.exists():
                    s = json.loads(summary.read_text())
                    entry['overall'] = s.get('overall', {})
                    entry['eval_status'] = s.get('status')

                traces_dir = dest / 'results' / 'traces'
                if traces_dir.exists():
                    traces = list(traces_dir.glob('*.json.gz'))
                    entry['traces'] = len(traces)
                    entry['trace_mb'] = round(sum(t.stat().st_size for t in traces) / 1e6, 1)

                # Rule: immediately delete completed/failed Kaggle kernel after artifacts are collected
                del_res = subprocess.run(['kaggle', 'kernels', 'delete', '-y', ref], capture_output=True, text=True, timeout=30)
                entry['deleted'] = del_res.returncode == 0

            states.append(entry)

        (RUN_DIR / 'monitor_status.json').write_text(json.dumps(states, indent=2))
        print(json.dumps(states, indent=2), flush=True)

        if once:
            return

        all_done = all('COMPLETE' in s['status'] or 'ERROR' in s['status'] for s in states)
        if all_done:
            print('ALL SHARDS TERMINAL', flush=True)
            return

        time.sleep(interval)


if __name__ == '__main__':
    main()
