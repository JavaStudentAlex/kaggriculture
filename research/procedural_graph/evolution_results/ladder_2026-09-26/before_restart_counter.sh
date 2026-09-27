#!/bin/bash
# BEFORE_START of the restart at iteration 61 of run ladder1 (2026-09-27, after the mixing that follows
# iteration 60): Island-Oracle, whose champion was the same graph as Island-Market's and whose guard tweaks
# were all rejected since iteration 34, becomes Island-Next-Counter: a third island on the public engine's
# 09-27 version, starting from Island-Next-Production's champion, for rival counters on that engine.
set -euo pipefail
RUN=/home/alex/kagg-evo/runs/ladder1
PG=/home/alex/kagg-evo/repo/research/procedural_graph
cp $RUN/checkpoint.json $RUN/checkpoint_before_next_counter.json
/home/alex/kagg-evo/venv/bin/python $PG/evolution_results/ladder_2026-09-26/convert_island.py $RUN \
    --replace Island-Oracle --name Island-Next-Counter --from-island Island-Next-Production \
    --stage "channel: rival_counter" --stage "engine: the 09-27 engine's sale timing" \
    --focus "Rival counters on the public engine's 09-27 version (tetsutani_demand_0927) as the backbone: switch \
the rival_counter channel on and give counters per rival class, using the 09-27 engine's switchable constants. \
On this engine the old public engine (tetsutani_demand) should be class wheat92_seller and copies of the 09-27 \
engine class mirror. The 09-27 line loses 30 of its 128 games against the old public engine, where the old line \
with its mirror counter wins 122-6, and ties 14 of its 24 games against the plain 09-27 engine. Sale-timing \
constants that can be counters: the race layers (_RACE_*, e.g. _RACE_HORIZON_MIRROR and _RACE_HORIZON_ESCALATED, \
24), the night shed guard (_SR_*), order-slot priority (_OR2_*), ready-stock lead sells (_S738_*, look 4), \
_S758_ITEMS and the rival-sales predictor (_V92_P_*, _V92_Q_*, horizon 48)." \
    --apply
echo "BEFORE_START done"
