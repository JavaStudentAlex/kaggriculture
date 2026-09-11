# Secure Kaggle account access on a remote development host

## Scope and evidence

Treat “check access” as read-only discovery, not authorization to install software or transfer credentials. An explicit request to configure/transfer account access authorizes that setup, but not a competition submission or account changes.

Separate four checks:
1. HTTPS reachability: a public website HTTP 200 establishes networking only.
2. Tool availability: inspect both the execution environment and the user's login shell.
3. Credential discovery: report existence and permissions, never contents or environment values.
4. Authenticated authorization: execute a read-only account-specific request from the destination. A successful public request or presence of a token is not proof.

## Minimal CLI installation

For account access alone, avoid installing a complete research stack or modifying project dependencies:

```bash
uv tool install kaggle
kaggle --version
```

This isolates the CLI and SDK in a tool environment. It does **not** make `import kaggle` available in every Python interpreter. If application code needs the SDK, install it in the application's intended virtual environment separately. Verify the CLI is on the login-shell PATH; if not, inspect `uv tool dir --bin` and apply the environment's normal PATH setup rather than hardcoding another user's path.

## Credential transfer

Prefer the existing modern `~/.kaggle/access_token` when it is the working authentication source. Do not copy a legacy `kaggle.json` unnecessarily; it may contain unrelated configuration as well as credentials.

- Use an encrypted SSH transport and, for Kubernetes, execute the receiver inside the intended pod/container.
- Read the credential into memory and pass its bytes as subprocess stdin. The remote Python receiver consumes stdin while its code is supplied via `python -c`.
- Keep secret bytes out of command arguments, shell interpolation, output, chat, scripts, and logs. Disable debug tracing. Do not print captured failure payloads without sanitizing them.
- Create `~/.kaggle` with mode `0700` and `access_token` with mode `0600`, outside any Git repository. Use exclusive creation (`os.O_WRONLY | os.O_CREAT | os.O_EXCL`) and refuse unexpected existing or symlink destinations. If a token already exists, compare safely; never overwrite a different credential silently.
- Verify transfer integrity in memory, for example by capturing the receiver's SHA-256 and comparing it with the source without publishing either digest. Also verify ownership and permissions.
- Avoid temporary credential copies entirely. If staging is unavoidable, restrict permissions from creation and securely clean it up immediately after verification.

## Read-only end-to-end verification

Run inside the actual destination, using an entered competition:

```bash
kaggle competitions submissions <competition-slug> --csv
```

Successful retrieval of the user's known submissions demonstrates account access and the relevant competition authorization. Check a known submission identifier where available; do not infer identity from a public leaderboard. This test consumes no submission quota.

Finally verify:
- `kaggle --version` and command resolution work in a fresh login shell;
- no credential/config file was introduced into the project working tree;
- no submission or account mutation occurred.

Report the credential location and permissions, installed CLI version, and successful read-only verification. Do not imply Python application integration or permission to submit was tested merely because listing worked.
