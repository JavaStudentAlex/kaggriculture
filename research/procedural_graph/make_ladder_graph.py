#!/usr/bin/env python3
"""A policy graph whose production engine is a public ladder agent (hazel_runtime/engines.py).

    python research/procedural_graph/make_ladder_graph.py --engine abo_v57 --out <graph.json>

The engine's files are copied from shinka/champions/ladder/<engine>/ (agent/ and SOURCE.json)
into hazel_runtime/engines/<engine>/ (an existing copy must be identical). The graph is the
--base graph (default: the feed15 graph) with:
- `engine` (and no engine_parameters) on the backbone turn node;
- the farmer, hands and market channels disabled, so the backbone's action is passed through
  and the graph plays exactly as the public agent does: evolution starts from that mirror and
  turns our layers (the oracle-driven market stages, the rescues, the extensions) back on only
  where they win;
- the optional oracle_guard turn stage (the predictor's front-run sells) present but disabled;
- no experimental switches and the surgical overrides off;
- engines.py and the engine's files pinned in provenance.runtime_bundle_hashes.
The oracle still observes every turn (the market stages need its forecast when they are on).
"""
from __future__ import annotations

import argparse
import copy
import filecmp
import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
LADDER = REPO / 'shinka' / 'champions' / 'ladder'
ENGINES = HERE / 'hazel_runtime' / 'engines'
BASE = HERE / 'evolution_results' / 'feed_fix_2026-09-24' / 'feed15_graph.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def install_engine(name):
    """hazel_runtime/engines/<name>/ from the ladder bundle; returns its files relative to hazel_runtime."""
    src, dst = LADDER / name, ENGINES / name
    meta = json.loads((src / 'SOURCE.json').read_text())
    if dst.exists():
        same = filecmp.cmp(src / 'SOURCE.json', dst / 'SOURCE.json', shallow=False) and all(
            (dst / 'agent' / rel).is_file() and sha(dst / 'agent' / rel) == digest for rel, digest in meta['files'].items())
        if not same:
            raise SystemExit(f'{dst} differs from {src}: remove it first if the ladder bundle was updated')
    else:
        (dst / 'agent').mkdir(parents=True)
        shutil.copy2(src / 'SOURCE.json', dst / 'SOURCE.json')
        for rel in meta['files']:
            (dst / 'agent' / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src / 'agent' / rel, dst / 'agent' / rel)
    return (['engines.py', 'oracle_guard.py', f'engines/{name}/SOURCE.json']
            + [f'engines/{name}/agent/{rel}' for rel in meta['files']], meta)


def add_oracle_guard(graph):
    """The optional oracle_guard turn stage right after `market` (disabled), in the execution
    chain and as a concept node, so graph edits can switch the predictor's front-run on."""
    turn = graph['turn']
    if any(n['id'] == 'oracle_guard' for n in turn['nodes']):
        return
    at = next(i for i, n in enumerate(turn['nodes']) if n['id'] == 'market') + 1
    turn['nodes'].insert(at, {'id': 'oracle_guard', 'binding': 'oracle_guard', 'enabled': False,
                              'summary': 'Predictor front-run sells added to the engine market orders'})
    after = turn['nodes'][at + 1]['id']
    edges = [e for e in turn['edges'] if not (e['source'] == 'market' and e['target'] == after)]
    k = next(i for i, e in enumerate(turn['edges']) if e['source'] == 'market')
    edges[k:k] = [{'source': 'market', 'target': 'oracle_guard', 'relation': 'NEXT'},
                  {'source': 'oracle_guard', 'target': after, 'relation': 'NEXT'}]
    turn['edges'] = edges
    graph['nodes'].append({
        'id': 'oracle_guard', 'name': 'oracle guard',
        'description': ('When the opponent-supply predictor (TinyTimeMixer oracle) gives score_4 >= _OG_SCORE for a '
                        'product we hold, sell spare shed stock into the engine\'s first empty market slot before '
                        'the opponent\'s units push the price down; never below _OG_PRICE_RATIO x base price.'),
        'bindings': ['oracle_guard'], 'binding_semantics': 'Executable stage'})
    graph['edges'].append({'source': 'oracle_observe', 'target': 'oracle_guard', 'relation': 'INFORMS', 'scope': 'turn'})


def ladder_graph(base, engine, meta, files):
    graph = copy.deepcopy(base)
    add_oracle_guard(graph)
    for node in graph['turn']['nodes']:
        if node['id'] == 'backbone':
            node['engine'] = engine
            node.pop('engine_parameters', None)
        if node['id'] in ('farmer', 'hands', 'market'):
            node['enabled'] = False
    graph['experimental'] = dict(graph.get('experimental') or {}, switches={})
    graph['surgical'] = {'enabled': False}
    graph['version'] = '6.0.0'
    graph['name'] = f'Ladder graph on {engine}'
    graph['description'] = (f"Production engine: {meta['title']} ({meta['url']}, Apache-2.0), played as-is with "
                            'the farmer, hands and market channels passed through; our layers are switched back '
                            'on by graph edits.')
    graph.setdefault('model_checkpoint', {})
    graph['engine'] = {'name': engine, 'notebook': meta['notebook'], 'notebook_id': meta.get('notebook_id'),
                       'entry': meta['entry'], 'files': meta['files']}
    pins = graph['provenance']['runtime_bundle_hashes']
    for rel in files:
        pins[rel] = None
    for rel in pins:   # every pin re-read from the current runtime (graph_runtime.py gained engines)
        pins[rel] = sha(HERE / 'hazel_runtime' / rel)
    graph['provenance']['entrypoint_sha256'] = sha(HERE / 'agent_graph.py')
    return graph


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--engine', required=True, help='a bundle under shinka/champions/ladder')
    ap.add_argument('--base', type=Path, default=BASE)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    files, meta = install_engine(args.engine)
    graph = ladder_graph(json.loads(args.base.read_text()), args.engine, meta, files)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(graph, indent=2) + '\n')
    print(json.dumps({'graph': str(args.out), 'engine': args.engine, 'pinned_engine_files': len(files)}))


if __name__ == '__main__':
    main()
