# Active advanced evolution configuration

The supported dedicated-server entrypoint is `bash run_advanced_evolution.sh`.
It runs `highcpu_island_evolution.py`; older single-candidate or notebook launchers
are not replacements for this architecture.

## Preserved architecture

- Eight checkpointed islands, seven-model UCB1 mutation pool.
- Three mutation candidates per iteration with self-healing retries.
- Exact canonical graph duplicate rejection across island archives.
- Qwen3-Embedding **8B** through Ollama, followed by the semantic novelty judge
  for high-similarity collisions.
- SIFT pairwise ranking via `gpt-6-astra`; only the winner goes to simulation.
- Growing champion pool and periodic meta-supervisor.
- Minimum **20 seed matches per champion on seat 0 and 20 on seat 1**. With
  13 champions this schedules 520 matches. The CLI and evaluation entrypoint
  reject values below 20.

## Embeddings and checkpoint safety

- Exact Ollama model: `qwen3-embedding:8b`, 4096-dimensional output.
- Endpoint: `http://127.0.0.1:11434/api/embed`, overridden by `OLLAMA_EMBED_URL`
  only when another compatible Ollama endpoint is required.
- Full canonical JSON is submitted, with `truncate:false`; oversized input is
  an error, not a silent prefix-only comparison. No MiniLM fallback.
- Zero/non-finite vectors and incompatible dimensions fail closed.
- Stored source graphs and vectors have matching indices, and a backend and
  serialization identity. A legacy checkpoint rebuilds from source graphs;
  source-less old vectors cannot be converted and are not reused.
- `qwen_embedding_cache/` is model-versioned internally and graph-hash keyed.
- `migrate_qwen_checkpoint.py` is the one-time stopped-run migration utility.
  It preserves iterations, champion graphs, scores, histories and rejections.
  It cannot recover old SIFT non-winners that were never saved as graphs.
- New viable candidates are persisted as graph/vector pairs, including SIFT
  non-winners, preventing the old unpaired-vector problem from recurring.
- Incumbent evaluations are rebaselined when seeds/pool/evaluator fingerprints
  change. Old 10-per-seat scores are not comparable to new 20-per-seat scores.

## Limits

Cosine similarity is a heuristic, not proof of functional equivalence or novelty.
The retained **0.985** semantic-review trigger is provisional and has **not** been
calibrated on labeled Qwen graph pairs. Exact duplicate rejection is deterministic;
semantic uniqueness is not guaranteed. Invalid semantic-judge output does not
certify a candidate as novel.

## Verification

`python -m unittest -v test_qwen_novelty` covers request completeness, invalid
vectors, malformed judge responses, migration, paired archive restore, and cache
reuse. Live Ollama embedding and semantic-judge probes are separate from these
mocked safety tests. Do not describe scheduler mocks as completed simulations.
