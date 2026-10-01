#!/bin/bash
# Candidates for the submission after Cedar Ridge (2026-09-28), on Cedar Ridge's ladder games (run on cliproxyapi):
# Cedar Ridge itself, the Island-Opening champion (Cedar Ridge + the rival emulator) and the Island-Next-Counter
# champion (the public engine's 09-27 version + the rival emulator).
#   1. backtest: every graph against the recorded moves of the 85 games against rivals rated 1,900+ or unlisted;
#   2. the 57 lost seeds from both seats against the pool bundle that plays like the rival (fresh games: a rival
#      bundle reacts to us as the real one did, where recorded moves go stale once we play differently).
# Harvest Ledger's 09-28 agent (haodou_ledger_0928), which played 3 of the losses exactly, joined the ladder pool first.
set -uo pipefail
CR=/home/alex/kagg-evo/cedar_ridge
R=/home/alex/kagg-evo/repo
PG=$R/research/procedural_graph
PY=/home/alex/kagg-evo/venv/bin/python
until grep -q "BATCH_DONE crW" $CR/batches.log 2>/dev/null; do sleep 30; done
mkdir -p $CR/replay_bundles
for id in $($PY -c "import json; print(' '.join(str(g['id']) for f in ('index_crL.json', 'index_crW.json') for g in json.load(open('$CR/' + f))))"); do
    [ -d $R/shinka/champions/replay_opponents/replay_$id ] && ln -sfn $R/shinka/champions/replay_opponents/replay_$id $CR/replay_bundles/replay_$id
done
echo "replay bundles: $(ls $CR/replay_bundles | wc -l)"
cat $CR/match/crL/*.jsonl > $CR/match_crL_all.jsonl
cd $PG
$PY ladder_seed_plan.py --losses $CR/index_crL.json=$CR/replays_crL --evidence $CR/match_crL_all.jsonl \
    --fallback tetsutani_demand --random-seeds 0 --out $CR/cr_plan.json > $CR/cr_plan.log 2>&1
$PY -c "
import json
seen, out = set(), []
for e in json.load(open('$CR/cr_plan.json')):
    if e.get('set') == 'lost' and not str(e['tag']).startswith('replay_') and e['seed'] not in seen:
        seen.add(e['seed'])
        out.append(e)
json.dump(out, open('$CR/cr_losses.json', 'w'), indent=1)
print(len(out), 'lost seeds')
"
G="--graph cedar=evolution_results/ladder_2026-09-26/opening_mirror_counter_graph.json --graph opening=island:Island-Opening --graph next_counter=island:Island-Next-Counter"
# both at once, so both requests are in the pool's inbox before the loop's next one
{ $PY ladder_validate.py --run_dir /home/alex/kagg-evo/runs/ladder1 --pool_dir /home/alex/kagg-evo/pool --name backtest_cr $G \
    --seeds 0 --replays $CR/replay_bundles > $CR/backtest_cr.log 2>&1; echo BACKTEST_CR_EXIT=$? >> $CR/backtest_cr.log; } &
{ $PY ladder_validate.py --run_dir /home/alex/kagg-evo/runs/ladder1 --pool_dir /home/alex/kagg-evo/pool --name lost_cr $G \
    --seeds 0 --lost $CR/cr_losses.json > $CR/lost_cr.log 2>&1; echo LOST_CR_EXIT=$? >> $CR/lost_cr.log; } &
wait
echo VALIDATE_CR_DONE
