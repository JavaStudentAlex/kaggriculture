"""One-time migration with full source-state preservation checks; runner must be stopped."""
import copy
import json
import logging
from pathlib import Path
from types import SimpleNamespace

from pool_upgrade_state import atomic_json, load_checkpoint, save_checkpoint
from qwen_novelty_archive import restore_archives
from shinka_graph_novelty import EMBEDDING_IDENTITY


def main():
    logging.basicConfig(level=logging.INFO)
    run = Path(__file__).resolve().parent / "highcpu_evo_run"
    path = run / "pool_upgrade_checkpoint.json"
    before = json.loads(path.read_text())
    state, islands = load_checkpoint(path, SimpleNamespace)
    report = restore_archives(islands, run, state["next_iteration"])
    for old, new in zip(before["islands"], islands):
        for key, value in old.items():
            if key not in ("embeddings_archive", "embedding_identity", "novelty_graphs"):
                assert getattr(new, key) == value, key
    provenance = copy.deepcopy(state["provenance"])
    if report:
        provenance["qwen_archive_migration"] = report
    save_checkpoint(path, islands, state["next_iteration"], state["master_graph"],
                    state["baseline_score"], state["evaluation_fingerprint"], provenance)
    verified = json.loads(path.read_text())
    assert verified["next_iteration"] == before["next_iteration"]
    for i in verified["islands"]:
        assert i["embedding_identity"] == EMBEDDING_IDENTITY
        assert len(i["embeddings_archive"]) == len(i["novelty_graphs"])
        assert all(len(v) == 4096 for v in i["embeddings_archive"])
    summary = {"next_iteration": verified["next_iteration"], "islands": len(islands),
               "graph_vector_pairs": sum(len(i.novelty_graphs) for i in islands),
               "embedding_identity": EMBEDDING_IDENTITY, "migration": report}
    atomic_json(run / "qwen_migration_report.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
