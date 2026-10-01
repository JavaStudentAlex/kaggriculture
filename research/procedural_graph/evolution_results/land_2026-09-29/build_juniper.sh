#!/bin/bash
# Build, validate and fidelity-check "Juniper Knoll" on cliproxyapi (tmux kagg-build-juniper, log
# ~/kagg-evo/juniper_knoll/build.log): the land run's seed = the predictor-required run's seed = Aspen Vale +
# _S809_LOOK 4 + _CA_MARGIN -22 + _SR_MARGIN 14 + the oracle guard for every rival family, packaged with
# require_oracle off (make_juniper_graph.py). Built from ladder1's code copy (~/kagg-evo/repo), whose runtime equals
# the predictor run's except for the require_oracle check (the one that produced the seed's cached margins) and which
# built Aspen Vale and Maple Crest. Then the package must reproduce the seed's cached margins on gauntlet games where
# the seed and Aspen Vale differ (stage_fidelity_any.py).
set -uo pipefail
D=/home/alex/kagg-evo/juniper_knoll
PG=/home/alex/kagg-evo/repo/research/procedural_graph
PY=/home/alex/kagg-evo/venv/bin/python
LAND=/home/alex/kagg-evo/land-20260929
NOTE="Land run seed = predictor-required run seed (arena bundles g_2e6e1664c98cc2a9, land runtime g_d46eaa6322231c85): ladder1 Island-Next-Counter champion after iteration 92 (g_99508e7525447d95 = the public engine's 09-27 version + _SR_MARGIN 12 + the rival emulator + _S809_LOOK 4 + _CA_MARGIN -22) + _SR_MARGIN 14 + the oracle guard for every rival family (MILK/WOOL/STRAWBERRY, score 0.5, strong 0.6, batch 4, keep 2, price ratio 0.75, steps 256-696); predictor ttm_c256_h96_ft_2026-09-13 on numpy, no calibration; require_oracle off in the package. Against Aspen Vale on 727 cached development games: +\$54 a game, 400 better / 191 worse, results +18/-5, wins 581 -> 595 (research/procedural_graph/evolution_results/land_2026-09-29/README.md)."
cd $PG
$PY $D/make_juniper_graph.py $D/juniper_knoll_graph.json || { echo SUBMIT_BUILD_EXIT=1; exit 1; }
$PY make_graph_submission.py --graph $D/juniper_knoll_graph.json --checkpoint $PG/hazel_runtime/checkpoint \
    --name "Juniper Knoll" --note "$NOTE" --validate --fidelity 101 --python $PY --force
echo SUBMIT_BUILD_EXIT=$?
$PY $D/stage_fidelity_any.py /home/alex/kagg-evo/repo/shinka/champions/submissions/JuniperKnoll.tar.gz \
    --with_scores $LAND/run/scores/g_d46eaa6322231c85.json \
    --without_scores /home/alex/kagg-evo/runs/ladder1/scores/g_16bb4af65a704f00.json \
    --run $LAND/run --plan $LAND/repo/research/procedural_graph/evolution_results/land_2026-09-29/plan.json \
    --opponent tetsutani_demand_0927 --opponent tetsutani_demand --opponent haodou_ledger_0928 \
    --opponent leoprovorov_forecast --opponent replay_114792276 --n 2 --workers 5
echo STAGE_FIDELITY_EXIT=$?
