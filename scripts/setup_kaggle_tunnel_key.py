#!/usr/bin/env python3
"""Add the Kaggle tunnel key to authorized_keys with strict port-forward-only restrictions.

Restrictions applied:
  command="/bin/false"       -> no shell, no command execution
  no-pty                     -> no terminal allocation
  no-agent-forwarding        -> can't hijack local ssh-agent
  no-X11-forwarding          -> no X11
  permitopen="127.0.0.1:8317" -> can ONLY forward to the LLM proxy port, nothing else
"""
from pathlib import Path

AUTH_KEYS = Path.home() / ".ssh" / "authorized_keys"
PUB_KEY = Path.home() / ".ssh" / "colab_tunnel_ed25519.pub"

RESTRICTIONS = (
    'command="/bin/false",no-pty,no-agent-forwarding,'
    'no-X11-forwarding,permitopen="127.0.0.1:8317"'
)

pub = PUB_KEY.read_text().strip()
key_body = " ".join(pub.split()[:2])  # type + base64, drop comment

existing = AUTH_KEYS.read_text()

if key_body in existing:
    print("Key already present in authorized_keys — checking restrictions...")
    for line in existing.splitlines():
        if key_body in line:
            if line.startswith("command="):
                print("  ✓ Already restricted:", line[:80])
            else:
                print("  ✗ Present but UNRESTRICTED — needs fixing")
else:
    entry = f'{RESTRICTIONS} {pub}\n'
    new_content = existing.rstrip("\n") + "\n# Kaggle tunnel: port-forward to LLM proxy only\n" + entry
    AUTH_KEYS.write_text(new_content)
    AUTH_KEYS.chmod(0o600)
    print("✓ Added restricted Kaggle tunnel key to authorized_keys")
    print(f"  Restrictions: {RESTRICTIONS}")

print("\n--- Current authorized_keys entries ---")
for i, line in enumerate(AUTH_KEYS.read_text().splitlines(), 1):
    if line.strip() and not line.startswith("#"):
        # Show restrictions + key type + comment, hide the base64 body
        parts = line.split()
        if line.startswith("command="):
            print(f"{i}: [RESTRICTED] ...{parts[-1]}")
        else:
            print(f"{i}: [full-access] ...{parts[-1]}")
