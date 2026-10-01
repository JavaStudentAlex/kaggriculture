#!/bin/bash
# BEFORE_START of the restart that brings the rival_emulator and tactic stages into run ladder1 (2026-09-28):
#   0. the runtime check (inert_test_em.sh) must have passed: graphs without the stages, rebuilt with the new
#      runtime, replayed their games exactly; otherwise nothing changes and the loop stays stopped;
#   1. the code staged in ~/kagg-evo/dev (the stages, their edits, validation and bundle support, the knowledge,
#      the ideas and the queue with the first two emulator experiments) replaces the loop's copy;
#   2. the cached margins carry over to the new runtime fingerprint (carry_fingerprint.py);
#   3. Island-Market keeps its champion and history as Island-Tactics, the island that writes tactics.
set -euo pipefail
R=/home/alex/kagg-evo/repo
PG=$R/research/procedural_graph
RUN=/home/alex/kagg-evo/runs/ladder1
PY=/home/alex/kagg-evo/venv/bin/python
OLD_FP=4b6b57a1c6390e6f87d1968bb6bd14b668910f6120c3b4e4c41794e4aefefc55
grep -q '^INERT_EM_EXIT=0' /home/alex/kagg-evo/inert_test_em.log
rsync -a -c --exclude __pycache__ --exclude evolution_results/ladder_2026-09-26/plan.json \
    /home/alex/kagg-evo/dev/research/procedural_graph/ $PG/
echo "code synced"
cd $PG
$PY -c "import graph_gauntlet, highcpu_island_evolution, graph_edits; from hazel_runtime import rival_emulator, tactic; print('imports ok')"
$PY evolution_results/ladder_2026-09-26/carry_fingerprint.py $RUN evolution_results/ladder_2026-09-26/plan.json \
    --old $OLD_FP --apply
FOCUS=$(cat <<'TXT'
Code tactics on the old public engine (tetsutani_demand), from Island-Market's champion: the tactic channel (CONTROLS: TACTIC) runs a Python function you write on our action every turn. Use it for decisions no constant can express: conditions on the rival's public farm (its crops, animals and money show what it will sell next), on its orders of this very turn (info['rival_action'], known with the rival_emulator channel on while the rival runs a public engine), on our shed and the prices. Targets: our sales ahead of the rival's sales of the same product, the last day's sale race (final cash decides; stock left in the shed at the end is worth nothing), and rivals with other economies (classes other_opening, other and nsell_opener), where no emulator helps. One small, narrow tactic per edit; to improve a promoted tactic, send its whole new source. The market's sale-timing constants (V9_RACE*, _RACE_*, _EV_*, _DP_*, _MP_*, _ADV_*, _OR2_*) stay editable here.
TXT
)
if grep -q '"Island-Tactics"' $RUN/checkpoint.json; then
    echo "Island-Tactics exists"
else
    cp $RUN/checkpoint.json $RUN/checkpoint_before_tactics.json
    $PY evolution_results/ladder_2026-09-26/convert_island.py $RUN --replace Island-Market --name Island-Tactics \
        --refocus --focus "$FOCUS" --stage "channel: tactic" --stage "channel: rival_emulator" \
        --stage "engine: market" --apply
fi
echo "BEFORE_START done"
