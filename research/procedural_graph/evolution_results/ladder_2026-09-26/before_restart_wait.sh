#!/bin/bash
# BEFORE_START wrapper (2026-09-28): waits up to 2 h for the runtime check (inert_test_em.sh); brings the rival_emulator
# and tactic stages in if it passed (before_restart_emulator.sh), and otherwise lets the loop restart on its old code.
L=/home/alex/kagg-evo/inert_test_em.log
for _ in $(seq 720); do grep -q '^INERT_EM_EXIT=' $L && break; sleep 10; done
if grep -q '^INERT_EM_EXIT=0' $L; then
    bash /home/alex/kagg-evo/before_restart_emulator.sh
else
    echo "the runtime check did not pass ($(grep '^INERT_EM_EXIT=' $L | tail -1)): the loop restarts on its old code"
fi
