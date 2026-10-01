"""Replay opponents for every lost or tied game of an index (cliproxyapi): bundles that already exist in HAVE are
copied, the others are built from gzipped replays with make_replay_opponents.build.
    python build_lost5.py INDEX GZ_DIR OUT_DIR HAVE_DIR"""
import gzip, json, shutil, sys
from pathlib import Path
sys.path.insert(0, "/home/alex/kagg-evo/repo/shinka/champions/replay_opponents")
import make_replay_opponents as M
index, gz, out, have = json.load(open(sys.argv[1])), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
out.mkdir(parents=True, exist_ok=True)
built = copied = missing = 0
for g in index:
    if g["res"] not in ("L", "T"):
        continue
    dst = out / f"replay_{g['id']}"
    if dst.exists():
        continue
    if (have / dst.name).exists():
        shutil.copytree(have / dst.name, dst)
        copied += 1
        continue
    f = gz / f"episode-{g['id']}-replay.json.gz"
    if not f.exists():
        missing += 1
        continue
    M.build(g, json.loads(gzip.decompress(f.read_bytes())), out)
    built += 1
print(f"built {built}, copied {copied}, missing {missing}, total {len(list(out.glob('replay_*')))}")
