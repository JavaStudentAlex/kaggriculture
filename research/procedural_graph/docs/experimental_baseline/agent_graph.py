"""Current procedural graph: complete submitted Hazel behavior plus mapped concepts.

One isolated interpreter per agent is required (as in the frozen evaluator).
Legacy explicit graph paths still use the preserved pre-merge implementation.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
_ENGINE = None
_LEGACY = None


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def get_engine():
    global _ENGINE, _LEGACY
    if _ENGINE is not None:
        return _ENGINE
    graph_path = Path(os.environ.get('KAGG_GRAPH_PATH', CURRENT_DIR / 'policy_graph.json'))
    graph = json.loads(graph_path.read_text())
    if graph.get('runtime') != 'hazel_merged_v1':
        # Explicit historical graph evaluation stays historical, never silently upgraded.
        if not os.environ.get('KAGG_GRAPH_PATH'):
            raise ValueError('Current graph must declare hazel_merged_v1 runtime')
        sys.path.insert(0, str(CURRENT_DIR))
        evo = CURRENT_DIR.parents[1] / 'shinka/evolution'
        sys.path.insert(0, str(evo))
        _LEGACY = _load('_preserved_legacy_graph', CURRENT_DIR / 'hazel_import/premerge/agent_graph.py')
        _ENGINE = _LEGACY.get_engine()
        return _ENGINE
    bundle = CURRENT_DIR / 'hazel_runtime'
    for relative, expected in graph['provenance']['runtime_bundle_hashes'].items():
        path = (bundle / relative).resolve()
        if not path.is_relative_to(bundle.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Runtime source fingerprint mismatch: {relative}')
    for name in ['champion', 'kagg_oracle', 'kagg_ttm_numpy', 'features', 'mechanics', 'candidate_v66_meta_closed_loop']:
        loaded = sys.modules.get(name)
        file = getattr(loaded, '__file__', None)
        if file and not Path(file).resolve().is_relative_to(bundle.resolve()):
            raise RuntimeError(f'Foreign {name} module already loaded: use process-isolated agents')
    os.environ.update(KAGG_MOHUI_DIR=str(bundle / 'mohui_v66'), KAGG_ORACLE_SRC=str(bundle),
                      KAGG_TTM_DIR=str(bundle / 'checkpoint'), KAGG_OPP_MODEL_SRC=str(bundle / 'opponent_model'),
                      KAGG_ORACLE_BACKEND='numpy', KAGG_ORACLE_DEVICE='cpu', CUDA_VISIBLE_DEVICES='')
    sys.path.insert(0, str(bundle))
    champion = _load('champion', bundle / 'champion.py')
    runtime = _load('_merged_hazel_runtime', bundle / 'graph_runtime.py')
    _ENGINE = runtime.HazelGraph(champion, graph_path)
    # Require every conceptual node and edge to resolve, not just the execution chains.
    allowed = set(_ENGINE.turn_ids + _ENGINE.market_ids)
    nodes = graph['nodes']
    ids = [n['id'] for n in nodes]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate merged concept node')
    if any(not n.get('bindings') or not set(n['bindings']) <= allowed for n in nodes):
        raise ValueError('Unbound merged concept')
    if any(e['source'] not in ids or e['target'] not in ids for e in graph['edges']):
        raise ValueError('Dangling merged edge')
    if {b for n in nodes for b in n['bindings']} != allowed:
        raise ValueError('Missing submitted action stage')
    return _ENGINE


def agent(obs, configuration=None):
    engine = get_engine()
    if _LEGACY is not None:
        return _LEGACY.agent(obs, configuration)
    return engine.agent(obs, configuration)
