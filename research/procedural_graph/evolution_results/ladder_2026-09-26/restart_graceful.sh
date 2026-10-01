#!/bin/bash
# Restart the ladder1 evolution after the gauntlet in flight, without waiting for an ITERATION line (a copy of
# ~/kagg-evo/restart_graceful.sh on cliproxyapi). restart_ladder1.sh kills at the next "ITERATION n" line; during a
# mixing round, which plays one gauntlet per island, that line can be hours away. Here the loop gets SIGTERM: it
# finishes the gauntlet in flight (an iteration, or one island's mixing offer), saves, and exits; a stopped mixing
# resumes with the islands not yet offered.
#
#   BEFORE_START='<command>' restart_graceful.sh [ITERATIONS]
#
# The tmux wrapper must not run its `touch $POOL/STOP` when the loop exits (that ends the Colab pool), and killing it
# first would close the pane and send SIGHUP to the loop (same process group). So the wrapper is frozen (SIGSTOP),
# the loop stopped with SIGTERM, and the wrapper killed only after the loop has exited (it is then a zombie of the
# frozen wrapper, so exit is detected from /proc, not with kill -0).
set -u
LOG=/home/alex/kagg-evo/runs/ladder1.log
POOL=/home/alex/kagg-evo/pool
ITERATIONS=${1:-200}
WRAPPER=$(tmux list-panes -t kagg-evo-ladder1 -F "#{pane_pid}" 2>/dev/null)
LOOP=$(pgrep -P "$WRAPPER" 2>/dev/null)
[ -n "$WRAPPER" ] && [ -n "$LOOP" ] || { echo "no running loop found"; exit 1; }
kill -STOP "$WRAPPER"
kill -TERM "$LOOP"
echo "$(date -u +%T) wrapper $WRAPPER frozen, SIGTERM to loop $LOOP; waiting for it to finish the gauntlet in flight"
alive() { [ -e "/proc/$1" ] && ! grep -q '^State:[[:space:]]*Z' "/proc/$1/status" 2>/dev/null; }
while alive "$LOOP"; do sleep 10; done
echo "$(date -u +%T) loop exited: $(grep -E 'Evolution stopped|\[MIX\]|] ITERATION [0-9]' "$LOG" | tail -2 | tr '\n' ' ')"
kill -KILL "$WRAPPER" 2>/dev/null
for _ in $(seq 30); do tmux has-session -t kagg-evo-ladder1 2>/dev/null || break; sleep 2; done
tmux kill-session -t kagg-evo-ladder1 2>/dev/null
[ -e "$POOL/STOP" ] && { echo "the pool's STOP file exists: not restarting"; exit 1; }
if [ -n "${BEFORE_START:-}" ]; then
    echo "$(date -u +%T) running: $BEFORE_START"
    bash -c "$BEFORE_START" || { echo "BEFORE_START failed: the loop was not restarted"; exit 1; }
fi
echo "=== RESTART $(date -u +%FT%T) --iterations $ITERATIONS ===" >> "$LOG"
tmux new -d -s kagg-evo-ladder1 "cd /home/alex/kagg-evo/repo/research/procedural_graph && /home/alex/kagg-evo/venv/bin/python highcpu_island_evolution.py --executor colab-pool --pool_dir $POOL --plan evolution_results/ladder_2026-09-26/plan.json --islands ladder --knowledge evolution_knowledge_ladder.md --ideas evolution_ideas_ladder.md --seed_graph evolution_results/ladder_2026-09-26/seed_graph.json --run_dir /home/alex/kagg-evo/runs/ladder1 --iterations $ITERATIONS --seeds_per_opponent 20 --supervisor_interval 6 --mix_interval 12 --queue evolution_queue_ladder.json --stage_fraction 0.3 --prefetch >> $LOG 2>&1; echo EVO_EXIT=\$? >> $LOG; touch $POOL/STOP"
sleep 30
tmux ls | grep kagg-evo-ladder1 && grep -E "RESTART|bandit models|MIXING|] ITERATION [0-9]" "$LOG" | tail -3
