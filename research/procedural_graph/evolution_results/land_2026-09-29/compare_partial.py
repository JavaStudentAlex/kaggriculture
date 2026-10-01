"""The seed's games replayed by the new runtime (the interrupted baseline request) against the cached margins."""
import json, sys
sys.path.insert(0, sys.argv[1])
from graph_gauntlet import margins
rows = [json.loads(l) for l in open(sys.argv[2]) if l.strip()]
found, errors = margins(rows, sys.argv[3])
old = json.load(open(sys.argv[4]))['margins']
same = sum(1 for k, v in found.items() if old.get(k) == v)
diff = [(k, old.get(k), v) for k, v in found.items() if old.get(k) != v]
print(f'{len(rows)} rows, {len(found)} margins, {len(errors)} errors: {same} equal to the cached margins, {len(diff)} differ')
for d in diff[:10]:
    print('  differs:', d)
print('PARTIAL_INERT_EXIT=' + ('0' if found and not diff else '1'))
