# Frozen iteration-27 evaluation

This evaluates the preserved iteration-27 graph without mutation or induction. The live evolution pool and master policy are not edited.

## Design

- 13 current pool entries + saved Hazel Weir and Copper Weir submission packages = 15 entries.
- 100 fresh seed values in seat 0 and 100 disjoint fresh seed values in seat 1 per entry.
- Same seat-specific seed blocks across opponents; seeds are reproducibly sampled outside the old evaluation range.
- 3,000 games total; 2,800 competitive games plus 200 self-control games.
- Copper's pool policy and submitted package are retained as separately labeled deployment variants, not claimed to be distinct strategies.
- 5 CPU shards; each gets a balanced contiguous 20-index portion of both seed lists, covering every opponent: 600 games/shard.
- Report per-opponent and per-seat wins/draws/losses, invalids, mean reward and timing. Never count crashed opponents as wins.
- Both seats run in separate persistent subprocesses, reset every match. This changes the evaluator isolation, not candidate code. Historical same-process scores are not directly comparable.
- Every match must end with both seats DONE, finite rewards and 720 states.
- No LLM proposals, evolution, API inference, SSH tunnels, credential upload or competition submissions. TinyTimeMixer is frozen local CPU/NumPy inference.
- Private RPC copy denies outbound socket connections and treats any detected attempt as an agent error.

## Provenance

Candidate graph canonical SHA256: `5abf926b81653557c9023ca60eb9f1656eb2e359c4641e70b36f7c8f3417ee49`.

Submission identities confirmed by authenticated Kaggle submission listing:
- Hazel Weir: 56246758, 2026-09-15.
- Copper Weir: 56239161, 2026-09-14.

Original submission archives are not available locally. Extracted saved package contents exist; policy, oracle, NumPy implementation and checkpoint weight hashes match the submission manifests. Do not claim archive-byte equivalence.

`build_payload.py` creates isolated bundles and a per-file hash manifest. `package_payload.py` prepares a private dataset upload. No secrets or general-purpose repository archives are included.

## Resource evidence

Official Kaggle notebook documentation retrieved 2026-09-23 lists 4 CPU cores, 30 GB RAM, 12-hour CPU session maximum, 20 GB persisted working output. Remote preflight also measured 4 CPUs / affinity 4 and MemTotal 32,870,492 kB. Five simultaneous CPU batch jobs is the previously observed account ceiling, not a guaranteed entitlement.

Kaggle: https://www.kaggle.com/docs/notebooks#technical-specifications
Colab: https://research.google.com/colaboratory/faq.html

Colab free managed runtimes without a positive compute balance prohibit distributed workers; no quota workarounds or multiple accounts. This batch uses Kaggle CPU kernels instead.

## Remote bootstrap pitfall

Kaggle may automatically extract uploaded `.tar.gz` dataset files. Resolve either the archive or a manifest with the correct evaluation_id under `/kaggle/input`, then copy/extract to writable scratch and verify all file hashes. An absent tar path does not mean the dataset failed to mount.

## Safety gates

Full local and remote smoke games against both submission packages and the self-control precede bulk dispatch. A queued kernel is not proof of successful games. Fetch remote result artifacts, verify evaluation_id, full episode completion and statuses. If smoke fails, do not dispatch thousands of matches.

Run identities, queue state and results are recorded under `runs/iter27_20260923/`; no claim of bulk launch or completed evaluation should be made based on this design document alone.
