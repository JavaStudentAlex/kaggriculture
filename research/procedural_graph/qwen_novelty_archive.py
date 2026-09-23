"""Versioned graph/vector pairs for the advanced island runner.

Legacy checkpoints retained vectors without their source graphs. Rebuild from
recoverable graphs, never reinterpret MiniLM vectors as Qwen vectors.
"""
from __future__ import annotations

import copy
import json
import logging
from pathlib import Path

from pool_upgrade_state import atomic_json, graph_fingerprint
from shinka_graph_novelty import (
    EMBEDDING_IDENTITY, get_graph_embedding, validate_embedding,
)


def register_graph(island, graph, embedding):
    validate_embedding(embedding)
    island.novelty_graphs.append(copy.deepcopy(graph))
    island.embeddings_archive.append(list(embedding))
    island.embedding_identity = EMBEDDING_IDENTITY


def restore_archives(islands, run_dir, next_iteration):
    """Re-embed recoverable legacy graphs; preserve all non-novelty island state.

    Incremental model-versioned cache allows interrupted migration to resume.
    Already-versioned graph/vector pairs must align or startup fails closed.
    """
    run_dir = Path(run_dir)
    cache_dir = run_dir / "qwen_embedding_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    report = []
    for island in islands:
        if getattr(island, "embedding_identity", None) == EMBEDDING_IDENTITY:
            graphs = getattr(island, "novelty_graphs", None)
            if not isinstance(graphs, list) or len(graphs) != len(island.embeddings_archive):
                raise ValueError(f"Unpaired novelty archive: {island.name}")
            for embedding in island.embeddings_archive:
                validate_embedding(embedding)
            if not graphs:
                raise ValueError(f"Empty novelty archive: {island.name}")
            continue

        old_count = len(island.embeddings_archive)
        graphs = [island.champion_graph]
        graphs.extend(getattr(island, "novelty_graphs", []))
        graphs.extend(r["graph"] for r in island.history if isinstance(r.get("graph"), dict))
        for path in sorted(run_dir.glob(f"cand_iter_*_{island.name}.json")):
            iteration = int(path.name.split("_")[2])
            if iteration < next_iteration:
                graphs.append(json.loads(path.read_text()))
        unique = {graph_fingerprint(graph): graph for graph in graphs}
        embeddings = []
        for digest, graph in unique.items():
            cache_file = cache_dir / f"{digest}.json"
            cached = json.loads(cache_file.read_text()) if cache_file.exists() else {}
            if (cached.get("identity") == EMBEDDING_IDENTITY
                    and cached.get("graph_fingerprint") == digest):
                emb = validate_embedding(cached["embedding"])
            else:
                logging.info("Embedding recovered graph %s / %s with Qwen", island.name, digest[:12])
                emb = get_graph_embedding(graph)
                atomic_json(cache_file, {"identity": EMBEDDING_IDENTITY,
                                        "graph_fingerprint": digest, "embedding": emb})
            embeddings.append(emb)
        island.novelty_graphs = copy.deepcopy(list(unique.values()))
        island.embeddings_archive = embeddings
        island.embedding_identity = EMBEDDING_IDENTITY
        item = {"island": island.name, "legacy_vectors": old_count,
                "recovered_graphs": len(unique),
                "note": "Legacy unpaired vectors replaced from available source graphs; unsaved candidates cannot be recovered"}
        report.append(item)
        logging.info("Qwen novelty archive rebuilt: %s", item)
    return report
