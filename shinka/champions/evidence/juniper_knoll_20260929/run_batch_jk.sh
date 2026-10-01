#!/bin/bash
# One batch of Juniper Knoll's recorded games on cliproxyapi (run_batch_av.sh of ../aspen_vale_20260928 for this review):
# the batch's gzipped replays are unpacked into replays_<batch>, then the rival's per-step public state
# (rival_features.py) and, with MATCH=1, every public bundle matched against the rival's recorded moves (P at a time;
# the recoveries that crash the matcher are left out); the unpacked replays are deleted at the end.
B=$1
AV=/home/alex/kagg-evo/juniper_review
R=$AV/replays_$B
PY=/home/alex/kagg-evo/venv/bin/python
M=/home/alex/kagg-evo/repo/shinka/champions/ladder/match_ladder_games.py
mkdir -p $R $AV/features
for e in $(python3 -c "import json; print(' '.join(str(g['id']) for g in json.load(open('$AV/index_$B.json'))))"); do
    [ -f $R/episode-$e-replay.json ] || gunzip -c $AV/replays/episode-$e-replay.json.gz > $R/episode-$e-replay.json
done
cd /home/alex/kagg-evo/rival
nice -n 19 $PY rival_features.py $AV/index_$B.json $R $AV/features > $AV/features_$B.log 2>&1
echo "$(date -u +%T) features $B exit $?" >> $AV/batches.log
if [ "${MATCH:-0}" = 1 ]; then
    mkdir -p $AV/match/$B
    { ls -d /home/alex/kagg-evo/repo/shinka/champions/ladder/*/ /home/alex/kagg-evo/alder_ford/cands/*/ \
            /home/alex/kagg-evo/cedar_ridge/cands3/*/ /home/alex/kagg-evo/aspen_vale/cands4/*/; \
      ls -d /home/alex/kagg-evo/alder_ford2/cands2/{haodou_ledger,lynn_idle,guru_master_v4,leoprovorov_forecast_0927}/; } | \
      grep -v -e __pycache__ -e /hak_2887/ -e /yhay_router0909/ | \
      xargs -P ${P:-2} -I{} bash -c 'b=$(basename {}); timeout 3600 nice -n 19 '$PY' '$M' {} '$AV'/index_'$B'.json '$R' > '$AV'/match/'$B'/$b.jsonl 2> '$AV'/match/'$B'/$b.err'
fi
rm -f $R/*.json; rmdir $R 2>/dev/null
echo "BATCH_DONE $B $(date -u +%T) match lines: $(cat $AV/match/$B/*.jsonl 2>/dev/null | wc -l)" >> $AV/batches.log
