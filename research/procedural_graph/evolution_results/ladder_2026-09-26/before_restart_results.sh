#!/bin/bash
# BEFORE_START of the restart for the results-based promotion rule (2026-09-28): the loop code staged in
# ~/kagg-evo/dev (graph_gauntlet.compare counts results, the first stage keeps candidates whose results
# improve, donors and the best island are ranked by results first, the prompt and knowledge say so)
# replaces the loop's copy. None of it is in the gauntlet fingerprint, so every cached game stays valid.
set -euo pipefail
PG=/home/alex/kagg-evo/repo/research/procedural_graph
rsync -a -c --exclude __pycache__ --exclude evolution_results/ladder_2026-09-26/plan.json \
    /home/alex/kagg-evo/dev/research/procedural_graph/ $PG/
echo "code synced"
cd $PG
/home/alex/kagg-evo/venv/bin/python -c "import graph_gauntlet, highcpu_island_evolution; print('imports ok')"
echo "BEFORE_START done"
