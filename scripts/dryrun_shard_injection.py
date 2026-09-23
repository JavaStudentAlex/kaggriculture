"""Dry-run: verify shard injection produces valid, correctly-populated code.

Does NOT push to Kaggle — only renders the template exactly as
prepare_and_push_shards() would, then validates the result.
"""
import ast
import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/alex/kaggriculture/research/procedural_graph")
from kaggle_island_evolution import _load_tunnel_credentials, _load_typesafe_key  # noqa: E402

SHARDS_DIR = Path("/home/alex/kaggriculture/kaggle_shards")
template_code = (SHARDS_DIR / "template" / "run_shard.py").read_text(encoding="utf-8")

typesafe_key = _load_typesafe_key()
tunnel_key, tunnel_host = _load_tunnel_credentials()
graph_json_str = json.dumps({"nodes": [], "edges": [], "_dryrun": True})
shard_id = 0

code = template_code.replace(
    'SHARD_ID = int(os.environ.get("SHARD_ID", "0"))', f"SHARD_ID = {shard_id}")
code = code.replace('TYPESAFE_KEY_PLACEHOLDER = ""',
                    f"TYPESAFE_KEY_PLACEHOLDER = {repr(typesafe_key)}")
code = code.replace('SSH_KEY_PLACEHOLDER = ""',
                    f"SSH_KEY_PLACEHOLDER = {repr(tunnel_key)}")
code = code.replace('SSH_HOST_PLACEHOLDER = ""',
                    f"SSH_HOST_PLACEHOLDER = {repr(tunnel_host)}")
code = code.replace(
    'cand_path = WORKING_DIR / "candidate_graph.json"',
    f'(WORKING_DIR / "candidate_graph.json").write_text({repr(graph_json_str)}, encoding="utf-8")\n'
    f'    cand_path = WORKING_DIR / "candidate_graph.json"')

checks = []

# 1. Rendered code must parse
try:
    tree = ast.parse(code)
    checks.append(("rendered shard parses as valid Python", True, ""))
except SyntaxError as e:
    checks.append(("rendered shard parses as valid Python", False, str(e)))
    tree = None

# 2. Placeholders must actually be populated (not left empty)
def literal_of(name):
    if tree is None:
        return None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == name:
                    try:
                        return ast.literal_eval(node.value)
                    except Exception:
                        return None
    return None

ts = literal_of("TYPESAFE_KEY_PLACEHOLDER")
sk = literal_of("SSH_KEY_PLACEHOLDER")
sh = literal_of("SSH_HOST_PLACEHOLDER")
sid = literal_of("SHARD_ID")

checks.append(("TYPESAFE key injected", bool(ts), f"{len(ts or '')} chars"))
checks.append(("SSH key injected", bool(sk) and "PRIVATE KEY" in (sk or ""), f"{len(sk or '')} chars"))
checks.append(("SSH host injected", bool(sh), str(sh)))
checks.append(("SHARD_ID is literal int", isinstance(sid, int), str(sid)))

# 3. No placeholder left unreplaced
leftovers = [n for n in ("TYPESAFE_KEY_PLACEHOLDER", "SSH_KEY_PLACEHOLDER", "SSH_HOST_PLACEHOLDER")
             if f'{n} = ""' in code]
checks.append(("no empty placeholders remain", not leftovers, str(leftovers)))

# 4. Inner SIFT wiring present
for fn in ("bootstrap_llm_tunnel", "shutdown_llm_tunnel", "run_inner_sift_evolution",
           "_score_graph_on_calibration", "_tunnel_llm_call"):
    present = tree is not None and any(
        isinstance(n, ast.FunctionDef) and n.name == fn for n in tree.body)
    checks.append((f"function {fn}() defined", present, ""))

# 5. Calibration and evaluation seeds must be disjoint
ns = {}
for node in (tree.body if tree else []):
    if isinstance(node, ast.Assign):
        for t in node.targets:
            if isinstance(t, ast.Name) and t.id in (
                    "SEEDS_SEAT0", "SEEDS_SEAT1", "CALIB_SEEDS_SEAT0",
                    "CALIB_SEEDS_SEAT1", "EVAL_SEEDS_SEAT0", "EVAL_SEEDS_SEAT1"):
                try:
                    ns[t.id] = eval(compile(ast.Expression(node.value), "<s>", "eval"), {}, dict(ns))
                except Exception:
                    pass

if "CALIB_SEEDS_SEAT0" in ns and "EVAL_SEEDS_SEAT0" in ns:
    o0 = set(ns["CALIB_SEEDS_SEAT0"]) & set(ns["EVAL_SEEDS_SEAT0"])
    o1 = set(ns["CALIB_SEEDS_SEAT1"]) & set(ns["EVAL_SEEDS_SEAT1"])
    checks.append(("calibration/eval seeds disjoint (seat0)", not o0, f"overlap={len(o0)}"))
    checks.append(("calibration/eval seeds disjoint (seat1)", not o1, f"overlap={len(o1)}"))
    checks.append(("eval block size",
                   len(ns["EVAL_SEEDS_SEAT0"]) == 35 and len(ns["EVAL_SEEDS_SEAT1"]) == 35,
                   f"{len(ns['EVAL_SEEDS_SEAT0'])}+{len(ns['EVAL_SEEDS_SEAT1'])} per champion"))
    checks.append(("calib block size",
                   len(ns["CALIB_SEEDS_SEAT0"]) == 15,
                   f"{len(ns['CALIB_SEEDS_SEAT0'])}+{len(ns['CALIB_SEEDS_SEAT1'])} per champion"))

print("=" * 68)
print("SHARD INJECTION DRY-RUN")
print("=" * 68)
failed = 0
for name, ok, detail in checks:
    mark = "PASS" if ok else "FAIL"
    if not ok:
        failed += 1
    print(f"  [{mark}] {name}" + (f"  ({detail})" if detail else ""))

print("=" * 68)
print(f"{len(checks) - failed}/{len(checks)} checks passed")
out = Path("/tmp/dryrun_shard_0.py")
out.write_text(code, encoding="utf-8")
print(f"Rendered shard written to {out} ({len(code)} bytes)")
sys.exit(1 if failed else 0)
