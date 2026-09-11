---
name: kaggle-account-access
description: Check or securely configure Kaggle CLI and authenticated account access on a development host. Use for reachability, missing CLI or credentials, authorized credential transfer, and read-only submission-list verification.
---

# Kaggle account access

Read [the secure account-access workflow](references/remote-account-access.md) before acting.

- Treat a request to check access as read-only. Install software or transfer credentials only when authorized.
- Verify from the actual target host/container and intended login shell, not from the agent's own machine.
- A public website response proves networking, not authentication. List known private submissions to test account access without consuming quota.
- Prefer an isolated `uv tool install kaggle` for CLI-only setup. Project Python SDK dependencies belong in the intended project virtual environment.
- Keep authentication under `~/.kaggle`, outside this repository. Never embed tokens in skills, command arguments, scripts, output, or Git.
- When transfer is authorized, use encrypted transport and stdin; directory permissions must be 0700 and token permissions 0600. Refuse symlink destinations and unexpected existing credentials.
- Verify transfer integrity without publishing credentials or their digests, then test a read-only authenticated request.
- A successful listing does not authorize submission, credential rotation, or account changes.
