#!/bin/bash
# Aspen Vale's 44 ladder games since the 09-28 20:16 UTC review (25 losses avL2, 19 wins avW2) through the same
# pipeline as ../aspen_vale_20260928: exact money flows first (fast), then the rival's public state, replay
# opponents and a match against every public bundle, then the class/best-match summary. Replays stay until the
# end (KEEP=1) and are deleted after (copies on the PC in replays/ours/aspen_vale).
set -u
AV=/home/alex/kagg-evo/aspen_vale
PY=/home/alex/kagg-evo/venv/bin/python
cd $AV
for b in avL2 avW2; do
    # the upload from the PC may still be running: wait for every replay of the batch, and no rsync on it
    want=$(python3 -c "import json; print(len(json.load(open('index_$b.json'))))")
    until [ "$(ls replays_$b | grep -c 'json$')" -ge "$want" ] && ! pgrep -f "rsync.*replays_$b" > /dev/null; do sleep 10; done
    nice -n 10 $PY flows_av.py index_$b.json replays_$b flows_$b.json > flows_$b.log 2>&1
    echo "$(date -u +%T) flows $b exit $?"
done
for b in avL2 avW2; do
    MATCH=1 KEEP=1 bash run_batch_av.sh $b
    echo "$(date -u +%T) batch $b done"
    $PY summarize_av.py $b > summarize_$b.log 2>&1
    echo "$(date -u +%T) summarize $b exit $?"
done
rm -f replays_avL2/*.json replays_avW2/*.json
rmdir replays_avL2 replays_avW2 2>/dev/null
df -h / | tail -1
echo "AV2_EXIT=0"
