#!/bin/bash
# The runtime with the rival_emulator and tactic stages (staged in ~/kagg-evo/dev) must replay the games of graphs without the
# stages, or the cached margins cannot carry over (carry_fingerprint.py):
#   1. the Island-Market champion of 09-27 (no counter), renamed only, against the public engine's 09-27 version on
#      the 20 arena validation seeds: equal to vs_engine0927's `market` games;
#   2. the Island-Opening champion (the mirror counter, the Cedar Ridge submission), renamed only, on Birch Hollow's 63
#      recorded games: equal to backtest_bh's `opening` games.
# Renaming gives new bundles, built with the dev runtime. Prints INERT_EM_EXIT=0 only if both comparisons pass.
set -uo pipefail
PG=/home/alex/kagg-evo/dev/research/procedural_graph
RUN=/home/alex/kagg-evo/runs/ladder1
PY=/home/alex/kagg-evo/venv/bin/python
T=/home/alex/kagg-evo/tmp
cd $PG
$PY -c "
import json
for src, dst in (('evolution_results/ladder_2026-09-27/market_runtime_check.json', '$T/market_rt_0928.json'),
                 ('evolution_results/ladder_2026-09-26/opening_mirror_counter_graph.json', '$T/opening_rt_0928.json')):
    g = json.load(open(src))
    g['name'] = str(g.get('name')) + ' (runtime check 09-28)'
    json.dump(g, open(dst, 'w'), indent=1)
"
$PY ladder_validate.py --run_dir $RUN --pool_dir /home/alex/kagg-evo/pool --name inert_emt_market \
    --graph market_rt=$T/market_rt_0928.json --opponents tetsutani_demand_0927 --seeds 20 || exit 1
$PY ladder_validate.py --run_dir $RUN --pool_dir /home/alex/kagg-evo/pool --name inert_emt_opening \
    --graph opening_rt=$T/opening_rt_0928.json --seeds 0 --replays /home/alex/kagg-evo/bh/replay_bundles || exit 1
$PY evolution_results/ladder_2026-09-26/same_games.py $RUN vs_engine0927:market inert_emt_market:market_rt --expect 20 \
    && $PY evolution_results/ladder_2026-09-26/same_games.py $RUN backtest_bh:opening inert_emt_opening:opening_rt --expect 63
echo INERT_EM_EXIT=$?
