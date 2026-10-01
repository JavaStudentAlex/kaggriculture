"""Single-change variants of the two submitted bundles (cliproxyapi, ~/kagg-evo/juniper_review/bundles): Juniper Knoll
with one of its four changes undone (j_*), Aspen Vale with one of Juniper's engine changes added (a_*).

    python make_variants.py
"""
import json
import shutil


def variant(src, name, edit):
    shutil.copytree(f'bundles/{src}', f'bundles/{name}', dirs_exist_ok=True)
    g = json.load(open(f'bundles/{name}/policy_graph.json'))
    edit(g['turn']['nodes'])
    json.dump(g, open(f'bundles/{name}/policy_graph.json', 'w'), indent=1)


def engine(nodes):
    assert nodes[1]['id'] == 'backbone' or 'engine_parameters' in nodes[1], nodes[1].get('id')
    return nodes[1]['engine_parameters']


def guard_off(nodes):
    assert nodes[7]['id'] == 'oracle_guard'
    nodes[7]['enabled'] = False


variant('juniper', 'j_noguard', guard_off)
variant('juniper', 'j_look3', lambda n: engine(n).pop('_S809_LOOK'))
variant('juniper', 'j_ca15', lambda n: engine(n).pop('_CA_MARGIN'))
variant('juniper', 'j_sr12', lambda n: engine(n).__setitem__('_SR_MARGIN', 12))
variant('aspen', 'a_look4', lambda n: engine(n).__setitem__('_S809_LOOK', 4))
variant('aspen', 'a_ca22', lambda n: engine(n).__setitem__('_CA_MARGIN', -22.0))
variant('aspen', 'a_sr14', lambda n: engine(n).__setitem__('_SR_MARGIN', 14))
