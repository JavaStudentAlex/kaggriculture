#!/bin/bash
# Counterfactual replays of Juniper Knoll's ladder games (cliproxyapi, tmux kagg-jk-cf): every (bundle, episode) job of
# JOBS ("bundle episode" lines) in its own isolated process (python -I, empty HOME, one BLAS thread, as the package
# validation), P at a time at nice 19; results appended to cf/<bundle>.jsonl, errors to cf/<bundle>.err.
cd /home/alex/kagg-evo/juniper_review
JOBS=${1:-jobs.txt}
P=${P:-4}
mkdir -p cf /tmp/jk_home
run_one() {
    b=$1; e=$2
    grep -q "\"episode\": $e," cf/$b.jsonl 2>/dev/null && return 0
    out=$(HOME=/tmp/jk_home OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout 1800 nice -n 19 \
          /home/alex/kagg-evo/venv/bin/python -I counterfactual.py bundles/$b index_all.json replays $e 2>> cf/$b.err)
    [ -n "$out" ] && echo "$out" >> cf/$b.jsonl || echo "$(date -u +%T) $b $e failed" >> cf/$b.err
}
export -f run_one
xargs -P $P -L 1 bash -c 'run_one $0 $1' < $JOBS
echo "CF_DONE $(date -u +%T) $JOBS: $(cat cf/*.jsonl | wc -l) results"
