#!/bin/bash
# Rerun only the stage fidelity of Juniper Knoll (tmux kagg-fid-juniper, log ~/kagg-evo/juniper_knoll/fidelity2.log),
# every game in its own clean process (stage_fidelity_any.py, second version).
set -uo pipefail
D=/home/alex/kagg-evo/juniper_knoll
PY=/home/alex/kagg-evo/venv/bin/python
LAND=/home/alex/kagg-evo/land-20260929
$PY $D/stage_fidelity_any.py /home/alex/kagg-evo/repo/shinka/champions/submissions/JuniperKnoll.tar.gz \
    --with_scores $LAND/run/scores/g_d46eaa6322231c85.json \
    --without_scores /home/alex/kagg-evo/runs/ladder1/scores/g_16bb4af65a704f00.json \
    --run $LAND/run --plan $LAND/repo/research/procedural_graph/evolution_results/land_2026-09-29/plan.json \
    --opponent tetsutani_demand_0927 --opponent tetsutani_demand --opponent haodou_ledger_0928 \
    --opponent leoprovorov_forecast --opponent replay_114792276 --n 2 --workers 3
echo STAGE_FIDELITY_EXIT=$?
