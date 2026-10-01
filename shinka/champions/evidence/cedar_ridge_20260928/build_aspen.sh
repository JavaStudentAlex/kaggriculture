#!/bin/bash
# Build, validate and fidelity-check the candidate after Cedar Ridge (working name "Aspen Vale"): the ladder1
# Island-Next-Counter champion after iteration 68 = the public engine's 09-27 version + engine _SR_MARGIN 12 + the
# rival_emulator stage, no predictor. Then the emulator path in the package: gauntlet games where the emulator changed
# the margin must give the arena's cached margin to the dollar (stage_fidelity.py). Run on cliproxyapi.
set -uo pipefail
PG=/home/alex/kagg-evo/repo/research/procedural_graph
PY=/home/alex/kagg-evo/venv/bin/python
NOTE="ladder1 Island-Next-Counter champion after iteration 68 (arena bundle g_16bb4af65a704f00) = the public engine's 09-27 version (tetsutani_demand_0927) + engine _SR_MARGIN 12 + the rival_emulator stage (both tetsutani versions, _EM_LOCK 12, _EM_RACE on); no predictor. Candidate for Cedar Ridge's losses (shinka/champions/evidence/cedar_ridge_20260928)."
cd $PG
$PY make_graph_submission.py --graph evolution_results/ladder_2026-09-26/next_counter_emulator_graph.json \
    --name "Aspen Vale" --note "$NOTE" --validate --fidelity 101 --python $PY --force
echo SUBMIT_BUILD_EXIT=$?
$PY evolution_results/ladder_2026-09-26/stage_fidelity.py /home/alex/kagg-evo/repo/shinka/champions/submissions/AspenVale.tar.gz \
    --with g_16bb4af65a704f00 --without g_f1a649e61fcad8b0 --opponent tetsutani_demand_0927 --opponent tetsutani_demand --n 2
echo STAGE_FIDELITY_EXIT=$?
