#!/usr/bin/env python3
"""Build the "yarn store second" variant of a graph bundle.

    python research/procedural_graph/arena/yarn_second_fix.py <bundle dir> <out dir>

Linden Brook's ladder audit (shinka/champions/evidence/linden_brook_loss_audit_20260925):
when the town's second shop (unlocked on day 6) is a YARN_STORE and the first is neither a
YARN_STORE nor a PET_CAFE, the Mohui v66 backbone has no route for it and stays on its default
plan (9 cows, 5 sheep, no more sheep bought); it lost all 9 such ladder games to opponents with
~10 sheep. The v66 meta layer already switches routes at step 144 (its bakery and farmers
branches); this adds one rule there: with no route active, no curve counter and no v65 attack
mode, take `bakery_yarn`. That route is the default route up to step 144 and the yarn plan from
step 144 on (2 cows and 9 sheep bought, twice the wool sold); the pet route is the same plan.

The bundle is copied, the one backbone file is patched by exact replacement (the script fails
unless the anchor occurs exactly once), the Apache-2.0 NOTICE gets a change line, and the
graph's runtime pins are recomputed so the bundle's integrity checks pass. `apply(dir)` does the
same in place; make_graph_submission.py --yarn-second-fix uses it on the staged submission.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

TARGET = 'hazel_runtime/mohui_v66/candidate_v66_meta_closed_loop.py'
NOTICE = 'hazel_runtime/mohui_v66/NOTICE'
HEADER = '# Modified by mohui666 in 2026; see NOTICE for the change summary.\n'
ANCHOR = '''            state["coke_armed"] = False

    route = state["route"]
'''
PATCH = '''            state["coke_armed"] = False
        # Yarn store second (day 6) after any first shop but YARN_STORE / PET_CAFE, with no
        # route active: the default route keeps 9 cows / 5 sheep. bakery_yarn is the default
        # route up to step 144 and the yarn plan from here (the pet route's plan too).
        if (second == "YARN_STORE" and shops[0] not in ("YARN_STORE", "PET_CAFE")
                and not state["route"] and not curve_active
                and not _V65._ATTACK_STATE[seat].get("active")):
            _activate(state, "yarn_second", "bakery_yarn", step)

    route = state["route"]
'''
NOTE = ('# Modified 2026-09-25 in the kaggriculture repository: yarn store as the second shop -> '
        'bakery_yarn route at step 144 (research/procedural_graph/arena/yarn_second_fix.py).\n')
NOTICE_LINE = ('\n2026-09-25, kaggriculture repository: candidate_v66_meta_closed_loop.py takes the\n'
               'bakery_yarn route at step 144 when the second town shop is a YARN_STORE and no\n'
               'other route is active (research/procedural_graph/arena/yarn_second_fix.py).\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def apply(dst):
    """Patch the graph bundle in `dst` in place and re-pin its graph; returns the patch record."""
    dst = Path(dst)
    target = dst / TARGET
    text = target.read_text()
    if text.count(ANCHOR) != 1 or not text.startswith(HEADER):
        raise SystemExit('anchor not found exactly once: the backbone file is not the expected v66')
    target.write_text(HEADER + NOTE + text[len(HEADER):].replace(ANCHOR, PATCH))
    (dst / NOTICE).write_text((dst / NOTICE).read_text() + NOTICE_LINE)
    graph_path = dst / 'policy_graph.json'
    graph = json.loads(graph_path.read_text())
    pins = graph['provenance']['runtime_bundle_hashes']
    for relative in pins:
        pins[relative] = sha(dst / 'hazel_runtime' / relative)
    graph['provenance']['entrypoint_sha256'] = sha(dst / 'main.py')
    graph['name'] = graph.get('name', '') + ' + yarn-second route'
    graph['backbone_patch'] = {'file': TARGET, 'script': 'research/procedural_graph/arena/yarn_second_fix.py',
                               'sha256': pins['mohui_v66/candidate_v66_meta_closed_loop.py']}
    graph_path.write_text(json.dumps(graph, indent=2) + '\n')
    return graph['backbone_patch']


def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    if dst.exists():
        raise SystemExit(f'{dst} exists')
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    patch = apply(dst)
    print(json.dumps({'bundle': str(dst), 'patched': patch['file'], 'sha256': patch['sha256']}))


if __name__ == '__main__':
    main()
