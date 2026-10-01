"""Re-key the seed's carried margins to the land run's opponent digests (the gauntlet's digests include absolute
paths, so identical bundle copies in a new run dir get new digests). Only opponents whose bundle files are
byte-identical (relative paths + contents) to the old run's keep their cached games."""
import hashlib, json, sys
from pathlib import Path
PG, RUN, OLD, PLAN, NEW_B, OLD_B = sys.argv[1:7]
sys.path.insert(0, PG)
from graph_gauntlet import ColabPoolExecutor, Gauntlet  # noqa: E402
from pool_upgrade_state import atomic_json  # noqa: E402
run, old = Path(RUN), Path(OLD)


def rel_digest(d):
    h = hashlib.sha256()
    for p in sorted(x for x in d.rglob('*') if x.is_file() and '__pycache__' not in x.parts and x.suffix != '.pyc'):
        h.update(str(p.relative_to(d)).encode())
        h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


g = Gauntlet(run, ColabPoolExecutor(Path.home() / 'kagg-evo' / 'pool'), 20, plan=json.loads(Path(PLAN).read_text()))
fp = g.prepare()
rec = json.loads((old / 'scores' / f'{OLD_B}.json').read_text())
same = {tag: g.opponent_digests[tag] for tag in rec['opponents']
        if tag in g.opponent_digests and (old / 'bundles' / tag).is_dir()
        and rel_digest(old / 'bundles' / tag) == rel_digest(run / 'bundles' / tag)}
atomic_json(run / 'scores' / f'{NEW_B}.json', {'fingerprint': fp, 'opponents': same, 'margins': rec['margins']})
usable = sum(1 for k in rec['margins'] if k.split('|')[0] in same)
print(f'fingerprint {fp[:12]}; {len(same)} of {len(rec["opponents"])} opponents identical; {usable} of '
      f'{len(rec["margins"])} margins usable -> scores/{NEW_B}.json')
