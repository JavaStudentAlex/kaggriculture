#!/bin/bash
# One batch of Birch Hollow's recorded games on cliproxyapi (the rival/run_batch.sh steps, plus replay opponents):
# the rival's per-step public state, a replay opponent per game (into the repo's replay_opponents, where payload.py
# finds them by name), with MATCH=1 every public bundle matched against the rival's moves; then the batch's replays
# are deleted unless KEEP=1 (the disk is small; the lost games' replays stay for ladder_seed_plan.py).
B=$1
BH=/home/alex/kagg-evo/bh
R=$BH/replays_$B
PY=/home/alex/kagg-evo/venv/bin/python
M=/home/alex/kagg-evo/repo/shinka/champions/ladder/match_ladder_games.py
cd /home/alex/kagg-evo/rival
nice -n 19 $PY rival_features.py $BH/index_$B.json $R features > $BH/features_$B.log 2>&1
nice -n 19 $PY /home/alex/kagg-evo/repo/shinka/champions/replay_opponents/make_replay_opponents.py $BH/index_$B.json $R \
    --results W,L,T > $BH/replay_opponents_$B.log 2>&1
if [ "${MATCH:-0}" = 1 ]; then
    mkdir -p $BH/match/$B
    { ls -d /home/alex/kagg-evo/repo/shinka/champions/ladder/*/ /home/alex/kagg-evo/alder_ford/cands/*/; \
      ls -d /home/alex/kagg-evo/alder_ford2/cands2/{dvorkin_v31,haodou_ledger,leo_ice_fire,lynn_idle}/; } | grep -v __pycache__ | \
      xargs -P 3 -I{} bash -c 'b=$(basename {}); timeout 3600 nice -n 19 '$PY' '$M' {} '$BH'/index_'$B'.json '$R' > '$BH'/match/'$B'/$b.jsonl 2> '$BH'/match/'$B'/$b.err'
fi
[ "${KEEP:-0}" = 1 ] || rm -f $R/*.json
echo "BATCH_DONE $B $(date -u +%T) replay opponents: $(grep -c . $BH/replay_opponents_$B.log) log lines, match lines: $(cat $BH/match/$B/*.jsonl 2>/dev/null | wc -l)" >> $BH/batches.log
