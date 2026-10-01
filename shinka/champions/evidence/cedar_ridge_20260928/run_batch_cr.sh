#!/bin/bash
# One batch of Cedar Ridge's recorded games on cliproxyapi (Birch Hollow's run_batch_bh.sh with this submission's
# paths and the notebooks recovered on 09-28): the rival's per-step public state, a replay opponent per game (into
# the repo's replay_opponents, where payload.py finds them by name), with MATCH=1 every public bundle matched against
# the rival's moves (the four recoveries that crash the matcher are left out); then the batch's replays are deleted
# unless KEEP=1 (the disk is small; the lost games' replays stay for ladder_seed_plan.py).
B=$1
CR=/home/alex/kagg-evo/cedar_ridge
R=$CR/replays_$B
PY=/home/alex/kagg-evo/venv/bin/python
M=/home/alex/kagg-evo/repo/shinka/champions/ladder/match_ladder_games.py
cd /home/alex/kagg-evo/rival
nice -n 19 $PY rival_features.py $CR/index_$B.json $R $CR/features > $CR/features_$B.log 2>&1
nice -n 19 $PY /home/alex/kagg-evo/repo/shinka/champions/replay_opponents/make_replay_opponents.py $CR/index_$B.json $R \
    --results W,L,T > $CR/replay_opponents_$B.log 2>&1
if [ "${MATCH:-0}" = 1 ]; then
    mkdir -p $CR/match/$B
    { ls -d /home/alex/kagg-evo/repo/shinka/champions/ladder/*/ /home/alex/kagg-evo/alder_ford/cands/*/ $CR/cands3/*/; \
      ls -d /home/alex/kagg-evo/alder_ford2/cands2/{haodou_ledger,lynn_idle,guru_master_v4,leoprovorov_forecast_0927}/; } | \
      grep -v -e __pycache__ -e /hak_2887/ -e /yhay_router0909/ | \
      xargs -P 3 -I{} bash -c 'b=$(basename {}); timeout 3600 nice -n 19 '$PY' '$M' {} '$CR'/index_'$B'.json '$R' > '$CR'/match/'$B'/$b.jsonl 2> '$CR'/match/'$B'/$b.err'
fi
[ "${KEEP:-0}" = 1 ] || rm -f $R/*.json
echo "BATCH_DONE $B $(date -u +%T) replay opponents: $(grep -c . $CR/replay_opponents_$B.log) log lines, match lines: $(cat $CR/match/$B/*.jsonl 2>/dev/null | wc -l)" >> $CR/batches.log
