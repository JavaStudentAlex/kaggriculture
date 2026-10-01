#!/bin/bash
# Build, validate and fidelity-check Aspen Vale with the oracle on (working name "Maple Crest"), run on cliproxyapi:
# Aspen Vale's graph (the public engine's 09-27 version, _SR_MARGIN 12, the rival emulator) plus the oracle guard with
# the settings the old engine's islands evolved (MILK, WOOL, STRAWBERRY; score 0.5, batch 4, keep 2, price ratio 0.75).
#   build_oracle_on.sh [GRAPH] [CHECKPOINT DIR] [CALIBRATION JSON]
# Defaults: aspen_guard_graph.json and the reference predictor in hazel_runtime/checkpoint (ttm_c256_h96_ft_2026-09-13,
# the model the guard's settings were evolved with), no calibration. The validation step prints each game's per-step
# timing: the predictor (numpy) and the emulator both run every turn, which no submission has done before.
set -uo pipefail
PG=/home/alex/kagg-evo/repo/research/procedural_graph
PY=/home/alex/kagg-evo/venv/bin/python
GRAPH=${1:-/home/alex/kagg-evo/aspen_vale/aspen_guard_graph.json}
CKPT=${2:-$PG/hazel_runtime/checkpoint}
CAL=${3:-}
NOTE="Aspen Vale (ladder1 Island-Next-Counter champion after iteration 68, arena bundle g_16bb4af65a704f00: the public engine's 09-27 version + _SR_MARGIN 12 + the rival emulator) with the oracle guard on (settings evolved on the old engine's islands: MILK/WOOL/STRAWBERRY, _OG_SCORE 0.5, _OG_BATCH 4, _OG_KEEP 2, _OG_PRICE_RATIO 0.75); graph $(basename $GRAPH), predictor $(basename $(dirname $CKPT/x))${CAL:+ + calibration}. Evidence: shinka/champions/evidence/aspen_vale_20260928."
cd $PG
$PY make_graph_submission.py --graph $GRAPH --checkpoint $CKPT ${CAL:+--calibration $CAL} \
    --name "Maple Crest" --note "$NOTE" --validate --fidelity 101 --python $PY --force
echo SUBMIT_BUILD_EXIT=$?
