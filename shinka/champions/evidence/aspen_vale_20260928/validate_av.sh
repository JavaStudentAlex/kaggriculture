#!/bin/bash
# Candidates for the submission after Aspen Vale (2026-09-28), on Aspen Vale's ladder games (run on cliproxyapi;
# validate_cr.sh of ../cedar_ridge_20260928 with this submission's games):
#   aspen       Aspen Vale itself (the 09-27 public engine, _SR_MARGIN 12, the rival emulator);
#   next_market Island-Next-Market's champion (Aspen Vale + _S809_LOOK 4, _SR_MARGIN 14);
#   aspen_ca22  Aspen Vale with leo_pi's carrot margin (_CA_MARGIN -15 -> -22);
#   crops       Island-Crops' champion (the old public engine with the guard, the counters and the emulator);
#   tactics     Island-Tactics' champion (the same line with counters against both public engine versions).
# 1. backtest: every graph against the recorded moves of the 67 games against rivals rated 1,900+ or unlisted;
# 2. the 31 lost seeds from both seats against the pool bundle that plays like the rival (fresh games: a rival
#    bundle reacts to us as the real one did, where recorded moves go stale once we play differently).
# leo_pi, which played one of the losses exactly, joined the ladder pool first.
set -uo pipefail
AV=/home/alex/kagg-evo/aspen_vale
R=/home/alex/kagg-evo/repo
PG=$R/research/procedural_graph
PY=/home/alex/kagg-evo/venv/bin/python
mkdir -p $AV/replay_bundles
for id in $($PY -c "import json; print(' '.join(str(g['id']) for f in ('index_avL.json', 'index_avW.json') for g in json.load(open('$AV/' + f))))"); do
    [ -d $R/shinka/champions/replay_opponents/replay_$id ] && ln -sfn $R/shinka/champions/replay_opponents/replay_$id $AV/replay_bundles/replay_$id
done
echo "replay bundles: $(ls $AV/replay_bundles | wc -l)"
cat $AV/match/avL/*.jsonl > $AV/match_avL_all.jsonl
cd $PG
$PY ladder_seed_plan.py --losses $AV/index_avL.json=$AV/replays_avL --evidence $AV/match_avL_all.jsonl \
    --fallback tetsutani_demand --random-seeds 0 --out $AV/av_plan.json > $AV/av_plan.log 2>&1
$PY -c "
import collections, json
seen, out = set(), []
for e in json.load(open('$AV/av_plan.json')):
    if e.get('set') == 'lost' and not str(e['tag']).startswith('replay_') and e['seed'] not in seen:
        seen.add(e['seed'])
        out.append(e)
json.dump(out, open('$AV/av_losses.json', 'w'), indent=1)
print(len(out), 'lost seeds; stand-ins', dict(collections.Counter(e['tag'] for e in out)))
"
G="--graph aspen=evolution_results/ladder_2026-09-26/next_counter_emulator_graph.json --graph next_market=island:Island-Next-Market --graph aspen_ca22=$AV/aspen_ca22_graph.json --graph crops=island:Island-Crops --graph tactics=island:Island-Tactics"
# both at once, so both requests are in the pool's inbox before the loop's next one
{ $PY ladder_validate.py --run_dir /home/alex/kagg-evo/runs/ladder1 --pool_dir /home/alex/kagg-evo/pool --name backtest_av $G \
    --seeds 0 --replays $AV/replay_bundles > $AV/backtest_av.log 2>&1; echo BACKTEST_AV_EXIT=$? >> $AV/backtest_av.log; } &
{ $PY ladder_validate.py --run_dir /home/alex/kagg-evo/runs/ladder1 --pool_dir /home/alex/kagg-evo/pool --name lost_av $G \
    --seeds 0 --lost $AV/av_losses.json > $AV/lost_av.log 2>&1; echo LOST_AV_EXIT=$? >> $AV/lost_av.log; } &
wait
echo VALIDATE_AV_DONE
