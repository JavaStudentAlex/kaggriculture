#!/bin/bash
# Restart the ladder1 evolution at the next iteration boundary (new code, or a new --iterations).
#
#   restart_ladder1.sh [ITERATIONS] [--pool]
#
# The checkpoint is saved before "ITERATION n" is logged, so the loop is killed right after that
# line and resumes from it. The tmux wrapper is killed FIRST: after the loop exits it would touch the
# pool's STOP file, which must happen only when a run really ends. The new wrapper does that.
# With --pool the Colab pool is restarted as well (new pool code), while it is idle: the loop was
# killed during its proposals, before it sent a gauntlet. The old pool stops its VMs (it must report
# COLAB_VMS_LEFT=0, or nothing is restarted) and the new one makes new VMs on the loop's first request.
# BEFORE_START (environment, optional) is a shell command run after the loop is killed and before it
# starts again (e.g. upgrade_scores.py); if it fails, the loop is not restarted.
set -u
LOG=/home/alex/kagg-evo/runs/ladder1.log
POOL=/home/alex/kagg-evo/pool
ITERATIONS=200
RESTART_POOL=0
for arg in "$@"; do
    case "$arg" in
        --pool) RESTART_POOL=1 ;;
        *) ITERATIONS=$arg ;;
    esac
done
WRAPPER=$(tmux list-panes -t kagg-evo-ladder1 -F "#{pane_pid}" 2>/dev/null)
LOOP=$(pgrep -P "$WRAPPER" 2>/dev/null)
[ -n "$WRAPPER" ] && [ -n "$LOOP" ] || { echo "no running loop found"; exit 1; }
N=$(grep -c '] ITERATION [0-9]' "$LOG")
echo "$(date -u +%T) wrapper $WRAPPER loop $LOOP: waiting for the next iteration boundary (seen $N)"
until [ "$(grep -c '] ITERATION [0-9]' "$LOG")" -gt "$N" ]; do
    kill -0 "$LOOP" 2>/dev/null || { echo "loop $LOOP exited on its own"; break; }
    sleep 5
done
kill -KILL "$WRAPPER" 2>/dev/null
kill -KILL "$LOOP" 2>/dev/null
echo "$(date -u +%T) killed wrapper $WRAPPER and loop $LOOP at: $(grep '] ITERATION [0-9]' "$LOG" | tail -1)"
for _ in $(seq 60); do tmux has-session -t kagg-evo-ladder1 2>/dev/null || break; sleep 2; done
tmux kill-session -t kagg-evo-ladder1 2>/dev/null

if [ "$RESTART_POOL" = 1 ]; then
    OLD_RUN=$(ps -eo args | grep '[c]olab_pool.py' | sed -n 's/.*--run-name \([^ ]*\).*/\1/p' | head -1)
    EXITS=$(grep -c '^COLAB_POOL_EXIT=' "$POOL/pool.log")
    touch "$POOL/STOP"
    echo "$(date -u +%T) STOP sent to the pool; waiting for it to stop its VMs"
    until [ "$(grep -c '^COLAB_POOL_EXIT=' "$POOL/pool.log")" -gt "$EXITS" ]; do sleep 5; done
    LEFT=$(grep -o 'COLAB_VMS_LEFT=[0-9]*' "$POOL/pool.log" | tail -1)
    echo "$(date -u +%T) pool ended: $LEFT"
    if [ "$LEFT" != "COLAB_VMS_LEFT=0" ]; then
        echo "VMs are left: stop them by hand (AGENTS.md 3.1); nothing restarted"
        exit 1
    fi
    # the stopped sessions' CLI history files (AGENTS.md 3.1)
    [ -n "$OLD_RUN" ] && find /home/alex/kagg-colab/home/.config/colab-cli/history -name "${OLD_RUN}-[0-9]*-[0-9]*.jsonl" -delete 2>/dev/null
    rm -f "$POOL/STOP"
    POOL_RUN="evo$(date -u +%d%H%M)"
    echo "=== POOL RESTART $(date -u +%FT%T) run name $POOL_RUN ===" >> "$POOL/pool.log"
    tmux new -d -s kagg-colab-evo "cd /home/alex/kagg-colab && export HOME=/home/alex/kagg-colab/home PATH=/home/alex/kagg-colab/home/.local/bin:\$PATH && python3 arena/colab_pool.py --pool-dir $POOL --run-name $POOL_RUN --vm colab2:hm --vm colab2:hm --vm colab2:hm --vm colab2:hm --vm colab2:hm --idle-stop 1800 >> $POOL/pool.log 2>&1; echo COLAB_POOL_EXIT=\$? >> $POOL/pool.log"
    sleep 5
    tmux has-session -t kagg-colab-evo 2>/dev/null || { echo "the new pool did not start"; exit 1; }
    echo "$(date -u +%T) new pool $POOL_RUN started"
fi

if [ -n "${BEFORE_START:-}" ]; then
    echo "$(date -u +%T) running: $BEFORE_START"
    bash -c "$BEFORE_START" || { echo "BEFORE_START failed: the loop was not restarted"; exit 1; }
fi

echo "=== RESTART $(date -u +%FT%T) --iterations $ITERATIONS ===" >> "$LOG"
tmux new -d -s kagg-evo-ladder1 "cd /home/alex/kagg-evo/repo/research/procedural_graph && /home/alex/kagg-evo/venv/bin/python highcpu_island_evolution.py --executor colab-pool --pool_dir $POOL --plan evolution_results/ladder_2026-09-26/plan.json --islands ladder --knowledge evolution_knowledge_ladder.md --ideas evolution_ideas_ladder.md --seed_graph evolution_results/ladder_2026-09-26/seed_graph.json --run_dir /home/alex/kagg-evo/runs/ladder1 --iterations $ITERATIONS --seeds_per_opponent 20 --supervisor_interval 6 --mix_interval 12 --queue evolution_queue_ladder.json >> $LOG 2>&1; echo EVO_EXIT=\$? >> $LOG; touch $POOL/STOP"
sleep 30
tmux ls | grep kagg-evo-ladder1 && grep -E "RESTART|bandit models|MIXING|] ITERATION [0-9]" "$LOG" | tail -3
