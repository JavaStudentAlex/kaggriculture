"""Execute the REAL tunnel bootstrap code from the rendered shard.

Extracts bootstrap_llm_tunnel / shutdown_llm_tunnel / _tunnel_llm_call from
/tmp/dryrun_shard_0.py (the exact code Kaggle would run) and exercises them
against the live host. Proves the injected credentials + tunnel logic work
end-to-end without needing a Kaggle kernel.
"""
import ast
import sys
import types
from pathlib import Path

src = Path("/tmp/dryrun_shard_0.py").read_text(encoding="utf-8")
tree = ast.parse(src)

WANT_FUNCS = {"bootstrap_llm_tunnel", "shutdown_llm_tunnel", "_tunnel_llm_call",
              "_extract_json_block"}
WANT_VARS = {"SSH_KEY_PLACEHOLDER", "SSH_HOST_PLACEHOLDER", "SSH_USER_PLACEHOLDER",
             "SSH_PORT_PLACEHOLDER", "LLM_PROXY_PORT", "LLM_PROXY_KEY",
             "_TUNNEL_PROC", "TUNNEL_ACTIVE", "INNER_SIFT_MODELS"}

kept = []
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in WANT_FUNCS:
        kept.append(node)
    elif isinstance(node, ast.Assign):
        for t in node.targets:
            if isinstance(t, ast.Name) and t.id in WANT_VARS:
                kept.append(node)
                break

mod_ast = ast.Module(body=kept, type_ignores=[])
ast.fix_missing_locations(mod_ast)

shard = types.ModuleType("shard_under_test")
shard.__dict__.update({
    "os": __import__("os"), "sys": sys, "json": __import__("json"),
    "time": __import__("time"), "subprocess": __import__("subprocess"),
    "Path": Path,
    "Dict": dict, "List": list, "Any": object, "Optional": object, "Tuple": tuple,
})
exec(compile(mod_ast, "<shard>", "exec"), shard.__dict__)

print("=" * 68)
print("LIVE TUNNEL TEST — running the shard's own bootstrap code")
print("=" * 68)
print(f"  host : {shard.SSH_USER_PLACEHOLDER}@{shard.SSH_HOST_PLACEHOLDER}:{shard.SSH_PORT_PLACEHOLDER}")
print(f"  proxy: 127.0.0.1:{shard.LLM_PROXY_PORT}")
print(f"  key  : {len(shard.SSH_KEY_PLACEHOLDER)} chars")
print("-" * 68)

failures = 0

ok = shard.bootstrap_llm_tunnel(timeout_sec=60)
print(f"  [{'PASS' if ok else 'FAIL'}] bootstrap_llm_tunnel() -> {ok}")
if not ok:
    failures += 1

if ok:
    # Real LLM call through the tunnel, using the shard's own helper
    for model in shard.INNER_SIFT_MODELS:
        try:
            out = shard._tunnel_llm_call(
                model,
                "You are a test harness. Reply with exactly the requested token.",
                "Reply with exactly: INNER_SIFT_OK",
                temperature=0.0, max_tokens=32,
            )
            good = "INNER_SIFT_OK" in out
            print(f"  [{'PASS' if good else 'FAIL'}] _tunnel_llm_call({model}) -> {out.strip()[:60]!r}")
            if not good:
                failures += 1
        except Exception as e:
            print(f"  [FAIL] _tunnel_llm_call({model}) raised: {str(e)[:140]}")
            failures += 1

    # JSON extraction helper used to parse mutation responses
    try:
        got = shard._extract_json_block('prose ```json\n{"rationale":"r","graph":{"n":1}}\n``` tail')
        good = got.get("graph", {}).get("n") == 1
        print(f"  [{'PASS' if good else 'FAIL'}] _extract_json_block() -> {got}")
        if not good:
            failures += 1
    except Exception as e:
        print(f"  [FAIL] _extract_json_block() raised: {str(e)[:120]}")
        failures += 1

shard.shutdown_llm_tunnel()
key_gone = not Path("/tmp/kagg_tunnel_key").exists()
print(f"  [{'PASS' if key_gone else 'FAIL'}] shutdown_llm_tunnel() removed key file")
if not key_gone:
    failures += 1

print("=" * 68)
print("ALL TUNNEL TESTS PASSED" if failures == 0 else f"{failures} TEST(S) FAILED")
sys.exit(1 if failures else 0)
