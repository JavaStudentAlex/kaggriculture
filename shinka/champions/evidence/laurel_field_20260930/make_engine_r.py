"""Make hazel_runtime/engines/tetsutani_demand_0927r: tetsutani_demand_0927 with its hard-coded cash reserves and the
input planner's return test turned into module constants (same values), so graph engine_parameters can tune them.

    python make_engine_r.py <engines dir>

At the defaults the engine plays exactly like tetsutani_demand_0927.
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

engines = Path(sys.argv[1])
src, dst = engines / 'tetsutani_demand_0927', engines / 'tetsutani_demand_0927r'
text = (src / 'agent' / 'main.py').read_text(encoding='utf-8')

EDITS = [
    ("    if farm['money'] < 12000 or obs['market']['prices']['TOMATO'] < CROP_MIN_PRICE:\n",
     "    if farm['money'] < _XR_TOMATO_MONEY or obs['market']['prices']['TOMATO'] < CROP_MIN_PRICE:\n"),
    ("    if farm['money']<budget+3000:\n        _V219_REPORT['budget_declines']+=1;return action\n",
     "    if farm['money']<budget+_XR_V219_RESERVE:\n        _V219_REPORT['budget_declines']+=1;return action\n"),
    ("    if farm['money']<budget+(3000 if initial else 1000):\n",
     "    if farm['money']<budget+(_XR_SHEEP_RESERVE_FIRST if initial else _XR_SHEEP_RESERVE):\n"),
    ("            if value<1.5*cost+50 or farm['money']<total_cost+cost+3000:break\n",
     "            if value<_XR_INPUT_ROI*cost+50 or farm['money']<total_cost+cost+_XR_INPUT_RESERVE:break\n"),
    ("    if obs['farms'][obs['player']]['money'] < 7000 + 3000 + spend:\n",
     "    if obs['farms'][obs['player']]['money'] < _XR_DAY11_RESERVE + spend:\n"),
    ("    if farm['money'] < 12000:\n        return False\n    if _cxtb_expected_revenue(obs) < _CXTB_MIN_REVENUE:\n",
     "    if farm['money'] < _XR_TOMATO_MONEY:\n        return False\n    if _cxtb_expected_revenue(obs) < _CXTB_MIN_REVENUE:\n"),
]
for old, new in EDITS:
    assert text.count(old) == 1, old
    text = text.replace(old, new)

HEADER = '''# Modified 2026-09-30 by team "Sunshine through fog" (Apache-2.0 section 4b): the literal cash reserves of six
# checks and the input planner's return test are read from these constants; their values are the original literals,
# so at these values the agent plays exactly as the notebook's.
_XR_TOMATO_MONEY = 12000        # money before the tomato plot qualifies (V219 and its CXTB replacement)
_XR_V219_RESERVE = 3000         # cash kept after the V219 crop-worker budget
_XR_SHEEP_RESERVE_FIRST = 3000  # cash kept after the first V233 sheep purchase
_XR_SHEEP_RESERVE = 1000        # cash kept after later V233 sheep purchases
_XR_INPUT_ROI = 1.5             # the input planner needs value >= this x cost + 50
_XR_INPUT_RESERVE = 3000        # cash the input planner keeps after its plans
_XR_DAY11_RESERVE = 10000       # money (plus the spend) the day-11 VE check requires
'''
lines = text.split('\n')
# after the leading comment/docstring block and imports: insert before the first def/class at column 0
at = next(i for i, l in enumerate(lines) if l.startswith(('def ', 'class ')))
lines[at:at] = HEADER.rstrip('\n').split('\n') + ['']
text = '\n'.join(lines)

if dst.exists():
    shutil.rmtree(dst)
shutil.copytree(src, dst)
(dst / 'agent' / 'main.py').write_text(text, encoding='utf-8')
meta = json.loads((src / 'SOURCE.json').read_text())
meta['name'] = 'tetsutani_demand_0927r'
meta['files'] = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted((dst / 'agent').iterdir())
                 if f.is_file()}
meta['note'] = ('tetsutani_demand_0927 with seven cash-reserve and return literals made module constants '
                '(_XR_*, same values), 2026-09-30; ' + meta.get('note', ''))
(dst / 'SOURCE.json').write_text(json.dumps(meta, indent=1, ensure_ascii=False) + '\n')
print('written', dst, meta['files'])
