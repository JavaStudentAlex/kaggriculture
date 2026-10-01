#!/bin/bash
# Stop the ladder1 evolution for good without ending the shared Colab pool (a copy of ~/kagg-evo/stop_ladder1.sh).
# The tmux wrapper runs `touch $POOL/STOP` when the loop exits, which ends the pool. Freezing the wrapper with SIGSTOP
# does not work: tmux 3.4 sends SIGCONT to a stopped pane process at once (on 2026-09-29 07:45 the frozen wrapper ran
# its touch when the loop was killed; the file was removed 27 s later, before the pool checked it between requests).
# So the wrapper is killed first (for a final stop the SIGHUP this sends the loop does not matter), then the loop.
# The iteration in flight is abandoned; the checkpoint written before its "ITERATION n" line stays, so a resume
# replays that iteration. The pool keeps playing a request already in flight; it cannot be cancelled.
set -u
POOL=/home/alex/kagg-evo/pool
LOG=/home/alex/kagg-evo/runs/ladder1.log
WRAPPER=$(tmux list-panes -t kagg-evo-ladder1 -F "#{pane_pid}" 2>/dev/null)
LOOP=$(pgrep -P "$WRAPPER" 2>/dev/null)
[ -n "$WRAPPER" ] && [ -n "$LOOP" ] || { echo "no running loop found"; exit 1; }
STOP_BEFORE=$([ -e "$POOL/STOP" ] && echo yes || echo no)
kill -KILL "$WRAPPER"
kill -KILL "$LOOP" 2>/dev/null
for _ in $(seq 15); do tmux has-session -t kagg-evo-ladder1 2>/dev/null || break; sleep 1; done
tmux kill-session -t kagg-evo-ladder1 2>/dev/null
echo "=== STOPPED $(date -u +%FT%T) by stop_ladder1.sh (loop $LOOP killed during: $(grep -E '] ITERATION [0-9]' "$LOG" | tail -1 | cut -c1-80)) ===" >> "$LOG"
if [ "$STOP_BEFORE" = no ] && [ -e "$POOL/STOP" ]; then
    rm -f "$POOL/STOP"; echo "removed a STOP file the wrapper created"
fi
[ -e "$POOL/STOP" ] && echo "WARNING: the pool's STOP file exists" || echo "pool STOP file absent (pool keeps running)"
tmux ls
echo "LADDER1_STOP_EXIT=0"
