"""Merge existing graph concepts with complete source-bound submitted execution.

Only local current graph/entrypoint integration; no remote restart or submission.
"""
from pathlib import Path
import hashlib
import json
import shutil
from graph_runtime import MARKET_STAGES, TURN_STAGES

HERE = Path(__file__).resolve().parent
PG = HERE.parent
REPO = PG.parents[1]

MAPPING = {
    'state_audit': ['state'],
    'observation_recovery': ['backbone', 'sanitize'],
    'terminal_liquidation': ['investment_freeze', 'early_liquidation', 'final_liquidation'],
    'wage_defense': ['wage_liquidity'],
    'shed_headroom': ['predrop_headroom'],
    'town_shop_preempt': ['town_and_fertilizer'],
    'oracle_frontrun': ['oracle_frontrun'],
    'capital_compounding': ['backbone', 'investment_freeze'],
    'closed_loop_hydration_pipeline': ['backbone', 'farmer', 'hands'],
    'keiz_opening_scalp': ['opening_scalp'],
    'tiered_shed_pressure_valve': ['shed_pressure'],
    'fertilizer_monetization': ['town_and_fertilizer'],
    'early_endgame_liquidation': ['early_liquidation'],
    'final_turn_liquidation': ['final_liquidation'],
    'action_scheduler': ['routine_dispatch', 'sanitize'],
    'execution_feedback': ['oracle_record'],
}
RESOLUTIONS = {
    'state_audit': 'Submitted derived state is authoritative. No watering-can or well fields are invented.',
    'observation_recovery': 'Use submitted exception fallback and sanitization; runtime cannot request extra observations.',
    'terminal_liquidation': 'Use submitted stepped liquidation, not a blanket step-696 dump.',
    'wage_defense': 'Use submitted narrow product-buy reserve filter at hour>=20 or steps220..245 after day1. Do not add the conflicting always-on cash floor.',
    'shed_headroom': 'Use carried inventory from private.inventories and protected pre-drop clearance from hour20, capacity100 margin2.',
    'town_shop_preempt': 'Use submitted demand/cadence/price/batch logic; do not impose unimplemented static 12-16-unit reserves.',
    'oracle_frontrun': 'Use full submitted product-aware forecasts and reserves, not only WOOL/MILK.',
    'capital_compounding': 'Retain all submitted production/hiring/land decisions. Old prose day13 freeze and nine-worker cap were not implemented and conflict with submitted behavior.',
    'closed_loop_hydration_pipeline': 'Retain actual production routing and farmer/hand maintenance. Old watering-can refill and well rules are unsupported intent, not new game mechanics.',
    'keiz_opening_scalp': 'Correct old prose: buy35 WHEAT at step0, sell30 at step1.',
    'tiered_shed_pressure_valve': 'Use exact submitted pressure thresholds and boolean predicates, including activation at80.',
    'fertilizer_monetization': 'Preserve town-cadence-gated fertilizer sales: hold>=3, price>=45, retain1, batch<=5.',
    'early_endgame_liquidation': 'Include forecast activation from640, ordinary activation680, and phased sales from693.',
    'final_turn_liquidation': 'Preserve exact final liquidation and price>=1 eligibility; do not force invalid zero-price orders.',
    'action_scheduler': 'Ten-order cap applies to market channel, not farmer/hand actions. Preserve submitted order ordering and legal duplicate batches.',
    'execution_feedback': 'Record final submitted action; successful execution is observed on subsequent state, not assumed.',
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    active = PG / 'policy_graph.json'
    old = json.loads(active.read_text())
    if old.get('runtime') == 'hazel_merged_v1':
        raise SystemExit('Already merged; refusing to overwrite')
    backup = HERE / 'premerge'
    backup.mkdir(exist_ok=False)
    for name in ['policy_graph.json', 'agent_graph.py', 'graph_engine.py']:
        shutil.copy2(PG / name, backup / name)
    executable = json.loads((HERE / 'policy_graph.json').read_text())
    source = REPO / 'shinka/champions/submissions/hazel_weir'
    sm = json.loads((source / 'MANIFEST.json').read_text())
    assert digest(source / 'champion.py') == sm['champion']['sha256']
    runtime = PG / 'hazel_runtime'
    runtime.mkdir(exist_ok=False)
    for relative in sm['archive_files']:
        target = runtime / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, target)
    shutil.copy2(HERE / 'graph_runtime.py', runtime / 'graph_runtime.py')
    (runtime / 'SOURCE_MANIFEST.json').write_text(json.dumps(sm, indent=2) + '\n')
    nodes = []
    for node in old['nodes']:
        key = node['id']
        bindings = MAPPING[key]  # reject unmapped intent, do not silently ignore
        nodes.append({**node, 'previous_description': node.get('description'),
                      'description': RESOLUTIONS[key], 'bindings': bindings,
                      'binding_semantics': 'Alias to shared executable stages; never execute twice'})
    existing = {n['id'] for n in nodes}
    for key, summary in [(k, s) for k, s in TURN_STAGES] + [(k, s) for k, _, s in MARKET_STAGES]:
        if key not in existing:
            nodes.append({'id': key, 'name': key, 'description': summary,
                          'bindings': [key], 'binding_semantics': 'Executable stage'})
            existing.add(key)
    # Concept aliases map onto the same stages instead of generating duplicate actions.
    alias_edges = [{'source': n['id'], 'target': b, 'relation': 'IMPLEMENTS', 'scope': 'concept'}
                   for n in nodes for b in n['bindings'] if n['id'] != b]
    edges = []
    for section in ['turn', 'market']:
        for e in executable[section]['edges']:
            edges.append({**e, 'scope': section, 'relation': 'EXECUTES_BEFORE'})
    merged = {
        'version': '4.0.0', 'runtime': 'hazel_merged_v1',
        'name': 'current_procedural_graph_merged_with_full_hazel_submission',
        'description': 'Existing strategy concepts reconciled with all submitted action channels and complete inherited controller. Executable stages are source-bound; obsolete prose cannot suppress orders.',
        'execution_contract': {'entry_node': 'backbone', 'traversal': 'turn then nested market, in validated dependency order',
                               'action_limit': {'market': 10, 'farmer': 1, 'hands': 'one per hired hand'},
                               'conflicts': 'Submitted executable behavior wins over previously unimplemented or contradictory graph prose. See merge_resolutions.',
                               'mutation_contract': 'Bindings/source/runtime must be changed together and revalidated. Legacy prose-only graph mutators are incompatible.'},
        'source': executable['source'], 'turn': executable['turn'], 'market': executable['market'],
        'nodes': nodes, 'edges': edges + alias_edges,
        'merge_resolutions': RESOLUTIONS,
        'provenance': {'previous_graph_sha256': digest(backup / 'policy_graph.json'),
                       'previous_graph_backup': 'hazel_import/premerge/policy_graph.json',
                       'submission_id': sm['kaggle']['submission_id'],
                       'runtime_bundle_hashes': {str(p.relative_to(runtime)): digest(p) for p in sorted(runtime.rglob('*')) if p.is_file()}},
    }
    active.write_text(json.dumps(merged, indent=2) + '\n')
    print(json.dumps({'merged_graph': str(active), 'concept_nodes': len(nodes),
                      'turn_stages': len(merged['turn']['nodes']), 'market_stages': len(merged['market']['nodes']),
                      'preserved_old_concepts': len(old['nodes']), 'backup': str(backup), 'runtime': str(runtime)}, indent=2))


if __name__ == '__main__':
    main()
