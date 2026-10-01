#!/bin/bash
# One batch of Aspen Vale's recorded games on cliproxyapi (run_batch_cr.sh of ../cedar_ridge_20260928 with this
# submission's paths, plus the notebooks recovered after 09-28 08:04 UTC in cands4/): the rival's per-step public
# state, a replay opponent per game (into the repo's replay_opponents, where payload.py finds them by name), with
# MATCH=1 every public bundle matched against the rival's moves (the four recoveries that crash the matcher are left
# out); then the batch's replays are deleted unless KEEP=1 (the disk is small; the lost games' replays stay for
# ladder_seed_plan.py).
B=$1
AV=/home/alex/kagg-evo/aspen_vale
R=$AV/replays_$B
PY=/home/alex/kagg-evo/venv/bin/python
M=/home/alex/kagg-evo/repo/shinka/champions/ladder/match_ladder_games.py
cd /home/alex/kagg-evo/rival
nice -n 19 $PY rival_features.py $AV/index_$B.json $R $AV/features > $AV/features_$B.log 2>&1
nice -n 19 $PY /home/alex/kagg-evo/repo/shinka/champions/replay_opponents/make_replay_opponents.py $AV/index_$B.json $R \
    --results W,L,T > $AV/replay_opponents_$B.log 2>&1
if [ "${MATCH:-0}" = 1 ]; then
    mkdir -p $AV/match/$B
    { ls -d /home/alex/kagg-evo/repo/shinka/champions/ladder/*/ /home/alex/kagg-evo/alder_ford/cands/*/ \
            /home/alex/kagg-evo/cedar_ridge/cands3/*/ $AV/cands4/*/; \
      ls -d /home/alex/kagg-evo/alder_ford2/cands2/{haodou_ledger,lynn_idle,guru_master_v4,leoprovorov_forecast_0927}/; } | \
      grep -v -e __pycache__ -e /hak_2887/ -e /yhay_router0909/ | \
      xargs -P 3 -I{} bash -c 'b=$(basename {}); timeout 3600 nice -n 19 '$PY' '$M' {} '$AV'/index_'$B'.json '$R' > '$AV'/match/'$B'/$b.jsonl 2> '$AV'/match/'$B'/$b.err'
fi
[ "${KEEP:-0}" = 1 ] || rm -f $R/*.json
echo "BATCH_DONE $B $(date -u +%T) replay opponents: $(grep -c . $AV/replay_opponents_$B.log) log lines, match lines: $(cat $AV/match/$B/*.jsonl 2>/dev/null | wc -l)" >> $AV/batches.log
