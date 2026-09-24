#!/usr/bin/env python3
"""Build an arena payload: graph variants, opponent bundles, harness and job list.

    python research/procedural_graph/arena/payload.py --eval-id ID \
        --pairs cadence3:hazel,edges:hazel,hazel:mohui13 --seeds 40

A graph variant is a full copy of the current graph bundle (agent_graph.py as main.py,
policy_graph.json, hazel_runtime/) with the variant's graph edits from variants.json;
provenance hashes are re-pinned so the graph's integrity checks still pass. Opponents:
the submitted packages (hazel, copper, orchard: MANIFEST archive files only), the
bundled Mohui v66 backbone (mohui), mohui13 (backbone + the 13/9 opening of the top
Mohui-based ladder opponents) and willow (Willow Ford, the closest local agent to the
Harvest Current submission). One game per seed: the engine is seat-symmetric and the
agents deterministic (a swapped-seat replay gives identical cash), so the candidate's
seat alternates with the seed index instead of playing both seats.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
PG = HERE.parent
REPO = PG.parents[1]
SUBMISSIONS = REPO / 'shinka/champions/submissions'
PACKAGES = {'hazel': ('hazel_weir', 'MANIFEST.json'), 'copper': ('copper_weir', 'MANIFEST.json'),
            'orchard': ('orchard_tide', 'SUBMISSION_MANIFEST.json')}
NODE_FEATURES = ('parameters', 'enabled', 'order')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def variant_graph(spec, library):
    graph = json.loads((PG / 'policy_graph.json').read_text())
    if spec.get('as_committed'):
        return graph
    graph['surgical'] = copy.deepcopy(spec['surgical'])
    edits = {}
    for name in spec.get('use', []):
        for node, attrs in library[name].items():
            target = edits.setdefault(node, {})
            for key, value in attrs.items():
                if key == 'parameters':
                    target.setdefault('parameters', {}).update(value)
                else:
                    target[key] = value
    for node, attrs in spec.get('nodes', {}).items():
        edits.setdefault(node, {}).update(attrs)
    ids = {n['id'] for chain in ('turn', 'market') for n in graph[chain]['nodes']}
    if set(edits) - ids:
        raise SystemExit(f'unknown graph nodes in variant: {sorted(set(edits) - ids)}')
    for chain in ('turn', 'market'):
        for node in graph[chain]['nodes']:
            for key in NODE_FEATURES:
                node.pop(key, None)
            node.update(copy.deepcopy(edits.get(node['id'], {})))
    return graph


def write_graph_bundle(dst, graph, name, checkpoint='committed'):
    """A runnable copy of the graph agent: main.py (agent_graph.py), hazel_runtime/ and
    the given graph, named `name` and re-pinned to the copied runtime files."""
    graph = copy.deepcopy(graph)
    shutil.copytree(PG / 'hazel_runtime', dst / 'hazel_runtime',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.bak'))
    shutil.copy2(PG / 'agent_graph.py', dst / 'main.py')
    runtime = dst / 'hazel_runtime'
    if checkpoint == 'hazel':
        hazel = SUBMISSIONS / 'hazel_weir' / 'checkpoint'
        for f in ('config.json', 'labels.json', 'model.safetensors', 'scaler.npz'):
            shutil.copy2(hazel / f, runtime / 'checkpoint' / f)
        (runtime / 'checkpoint' / 'calibration.json').unlink(missing_ok=True)
        graph['source']['oracle_model_sha256'] = sha(runtime / 'checkpoint/model.safetensors')
        graph['model_checkpoint'] = {'name': "ttm_c256_h96_ft_2026-09-13 (Hazel Weir's checkpoint)",
                                     'model_sha256': graph['source']['oracle_model_sha256']}
    elif checkpoint != 'committed':
        # a model directory (e.g. models/ttm_c256_h96_ft_2026-09-23): same architecture, plain weights
        src = Path(checkpoint)
        for f in ('config.json', 'labels.json', 'model.safetensors', 'scaler.npz'):
            shutil.copy2(src / f, runtime / 'checkpoint' / f)
        (runtime / 'checkpoint' / 'calibration.json').unlink(missing_ok=True)
        graph['source']['oracle_model_sha256'] = sha(runtime / 'checkpoint/model.safetensors')
        graph['model_checkpoint'] = {'name': src.resolve().name, 'source': str(src),
                                     **{f'{f.split(".")[0]}_sha256': sha(runtime / 'checkpoint' / f)
                                        for f in ('model.safetensors', 'scaler.npz', 'config.json', 'labels.json')}}
    pins = graph['provenance']['runtime_bundle_hashes']
    for relative in list(pins):
        pins[relative] = sha(runtime / relative)
    graph['provenance']['entrypoint_sha256'] = sha(dst / 'main.py')
    graph['name'] = name
    (dst / 'policy_graph.json').write_text(json.dumps(graph, indent=2) + '\n')
    return dst


def build_variant(out, name, spec, library):
    write_graph_bundle(out / 'bundles' / name, variant_graph(spec, library), f'arena variant {name}',
                       spec.get('checkpoint', 'committed'))


def build_opponent(out, name):
    dst = out / 'bundles' / name
    if name in PACKAGES:
        folder, manifest = PACKAGES[name]
        src = SUBMISSIONS / folder
        for f in json.loads((src / manifest).read_text())['archive_files']:
            (dst / f).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src / f, dst / f)
    elif name in ('mohui', 'mohui13'):
        shutil.copytree(SUBMISSIONS / 'hazel_weir' / 'mohui_v66', dst,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        if name == 'mohui13':
            shutil.copy2(HERE / 'mohui13_main.py', dst / 'main.py')
    elif name == 'willow':
        # Willow Ford (roster, 2026-09-08): Mohui v66 + Keiz opening + town-shop harvesting.
        # Closest local agent to the Harvest Current submission (56085502, ladder 1669.8),
        # whose package is not downloadable: it reproduces Harvest Current's first 388-416
        # moves in three of its ladder replays (feeding the recorded observations).
        shutil.copytree(REPO / 'shinka/champions/dependencies/mohui_v66', dst / 'mohui_v66',
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        shutil.copy2(REPO / 'shinka/champions/roster/champ_20260908_150703_avg101195.py', dst / 'main.py')
    else:
        raise SystemExit(f'unknown opponent {name}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval-id', required=True)
    ap.add_argument('--pairs', required=True, help='a:b,... a = variant, graph or opponent, b = opponent')
    ap.add_argument('--graph', action='append', default=[], metavar='NAME=FILE',
                    help='play a saved graph file as-is under NAME (e.g. an evolution best_graph.json)')
    ap.add_argument('--checkpoint', action='append', default=[], metavar='NAME=DIR',
                    help='play the --graph NAME with the predictor in DIR instead of the committed one')
    ap.add_argument('--seeds', type=int, default=40)
    ap.add_argument('--seed-salt', type=int, default=20260924)
    ap.add_argument('--mirror-seeds', type=int, default=4, help='cap for the mirror sanity variant')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    out = Path(args.out or PG / 'runs' / 'arena' / args.eval_id / 'payload')
    library = json.loads((HERE / 'variants.json').read_text())
    pairs = [tuple(p.split(':')) for p in args.pairs.split(',') if p]
    graphs = dict(g.split('=', 1) for g in args.graph)
    checkpoints = dict(c.split('=', 1) for c in args.checkpoint)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    built = set()
    for a, b in pairs:
        for name in (a, b):
            if name in built:
                continue
            if name in graphs:
                write_graph_bundle(out / 'bundles' / name, json.loads(Path(graphs[name]).read_text()),
                                   f'arena graph {name}', checkpoints.get(name, 'committed'))
            elif name in library['variants']:
                build_variant(out, name, library['variants'][name], library['edits'])
            else:
                build_opponent(out, name)
            built.add(name)
    shutil.copy2(HERE / 'arena.py', out / 'arena.py')
    shutil.copy2(REPO / 'shinka/evolution/pool_upgrade_bundle_agent.py', out / 'bundle_agent.py')
    seeds = random.Random(args.seed_salt).sample(range(10_000_000, 2_000_000_000), args.seeds)
    jobs = []
    for i, seed in enumerate(seeds):
        for a, b in pairs:
            if a == 'mirror' and i >= args.mirror_seeds:
                continue
            jobs.append({'tag': a if b == 'hazel' else f'{a}@{b}', 'a': f'bundles/{a}', 'b': f'bundles/{b}',
                         'seed': seed, 'a_seat': i % 2})
    manifest = {'evaluation_id': args.eval_id, 'pairs': pairs, 'seeds': seeds, 'jobs': jobs,
                'variants': {n: library['variants'][n] for n in sorted(built) if n in library['variants']},
                'graphs': {n: {'file': graphs[n], 'sha256': sha(Path(graphs[n])), 'checkpoint': checkpoints.get(n, 'committed')}
                           for n in sorted(built) if n in graphs}}
    (out / 'jobs.json').write_text(json.dumps(manifest, indent=1) + '\n')
    files = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*')) if p.is_file()}
    (out / 'files.json').write_text(json.dumps(files, indent=0) + '\n')
    print(json.dumps({'payload': str(out), 'jobs': len(jobs), 'bundles': sorted(built), 'files': len(files),
                      'bytes': sum(p.stat().st_size for p in out.rglob('*') if p.is_file())}, indent=1))


if __name__ == '__main__':
    main()
