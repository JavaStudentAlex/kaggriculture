# Alder Ford's games 101-149, the public engine's new version, and rival classes (2026-09-27)

Alder Ford (Kaggle 56582140) had 149 ladder games at 09:29 UTC on 09-27: 78W-55L-16T. The 49 games after
the first 100 (22:49 UTC 09-26 to 09:29 UTC 09-27) went 20W-24L-5T, against stronger rivals than before
(4-12 against rivals rated 2,100-2,199). All 49 rivals are rated 1,900 or more, or unlisted.

## Files

- `fetch_new_games.py`, `index.json`: all 149 games (read-only API; merged with `../alder_ford_20260927/index.json`
  because Kaggle lists only a submission's newest ~200 episodes). `summary.py` prints the record by block and
  rival rating. The 49 new replays are in `replays/ours/alder_ford/` (not in git) and each is a replay opponent
  in `../../replay_opponents/` (131 in all).
- `stats.json`, `gaps.txt`: the 29 new lost or tied games re-simulated money-exact, and where the money went
  (`../alder_ford_20260927/gaps.py`).
- `families.py`, `families.json`, `families.txt`: the 49 games by the rival's step-0 wheat orders.
- `ladder_match.jsonl`, `best_match.txt`: every public bundle (the pool and 22 candidates) matched against the
  29 lost or tied games; `best_match_new_agents.txt`: the agents of notebooks re-run on 09-26/27
  (`recover_new.py`, `recover_b85.py`: static recovery, no notebook code runs).
- `classes.json`, `dump_classes.py`: the rival_counter class of all 132 recorded games (below).

## The 29 new lost or tied games

| rival style | games | what happened |
|---|---|---|
| copies of the public engine (old version) | 18 (5 ties, 13 losses) | 7 play it for all 719 moves (the 5 ties, hank0123 -$28, Sarthak Sharma -$1,037), 11 follow it for 145-712 moves. Same herd as ours; sale races of $2-$640 and one strawberry loss |
| the public engine's 09-27 version | 3 | Dean0016 (2,490) and GeHaohua play it for all 719 moves, CrazyML for 252: -$477 to -$1,708, mostly milk |
| strong agents no public notebook matches (2,100-2,330) | 7 | same herd; better wheat, milk and strawberry sales: -$100 to -$5,731 |
| Michael Timbs (13-wheat opener) | 1 | 200 tomatoes sold at about $300 against our 80: -$31,557; even until day 27 |

## The public engine's new version (tetsutani_demand_0927)

tetsutani's "Demand-Preserving Turn Sale Timing" notebook was re-run at 02:50 UTC on 09-27 with a new agent
(`VARIANT = "step1009_step1008_fortyfirst_final_fixedsell_closure"`, main.py sha256 55be5d5f..., archive now
`ARCHIVE_B85`). "Kaggriculture (TOP 2) Master Engine V4" is a byte-identical copy; lynn's "Idle Seller" and
haodou's "Harvest Ledger" play the same moves in these games. It opens buy 8 / sell 3, drops the old engine's
ready-stock and hour-window lead sells (`_ADV_*`, `_EV_*`, `_DP_*`, `_MP_*`) and adds a library-based predictor
of the rival's sales (`_V92_P_*`, `_V92_Q_*`), more race layers, numbered sale layers (`_S720_*`-`_S1009_*`),
a hybrid opening and rival-keyed routes. In the pool since 09-27 as `shinka/champions/ladder/tetsutani_demand_0927`.

Against it in the arena (20 validation seeds, `~/kagg-evo/vs_engine0927.sh`) every graph we had lost 2-18:
Alder Ford -$712 a game, Birch Hollow (`_ADV_LOOK` 4) -$696, + lead sells 12 -$663, the Island-Market champion
-$686. No sale-timing setting of the old engine answers it.

## Backtest 3: the 49 newer games

`~/kagg-evo/backtest3.sh` (the recorded moves of the 49 rivals; Alder Ford reproduces 20-24-5):

| graph | record | change a game | better / worse | L->W | T->W | W->L |
|---|---|---|---|---|---|---|
| `_ADV_LOOK` 4 + lead sells 12 | 34-15-0 | +$385 | 41 / 8 (p 2e-6) | 11 | 4 | 1 |
| Birch Hollow (`_ADV_LOOK` 4) | 28-21-0 | +$116 | 36 / 13 (p 0.001) | 8 | 2 | 2 |
| Island-Market champion | 28-21-0 | +$103 | 35 / 14 | 8 | 3 | 3 |

## Rival classes: does the rival play exactly as we do?

Every observation shows the rival's whole farm (money, hands, land, tiles with crops, pastures and coops);
only inventories, seeds and the shed are private. Rivals on the same route tapes have our farm for most of
the game whatever their family, so the farm alone does not tell families apart (a nearest-neighbour model on
the farm trajectory stayed undecided in most games). Money does, relative to ours on the same seed:

- copies of our engine keep our money past step 92 (their own changes show at steps 160-661, or never);
- "buy N / sell N-5" openers (Forecast, most 2965, the 09-27 engine) sold 3 spare wheat at step 0; at step 92
  our engine sells its 3 and they do not, so their money first falls behind ours at step 92 by the price of 3
  wheat (-$87 to -$90), with the market's wheat stock +3;
- other openings differ from step 1 to 89.

`hazel_runtime/rival_model.MirrorTracker` classifies by step 93 (mirror, nsell_opener, wheat92_seller,
other_opening, other). On all 132 recorded games (`classes.json`):

| opening family | mirror | nsell_opener | other_opening |
|---|---|---|---|
| public engine (buy 5 + seed) | 53 | 1 | 0 |
| Forecast (buy 8 / sell 3) | 0 | 23 | 0 |
| 2965 (buy 20 / sell 15) | 3 | 13 | 4 |
| 13-wheat | 0 | 0 | 3 |
| other | 2 | 14 | 15 |

What `_ADV_LOOK` 4 + lead sells 12 gains per class against the recorded moves (`class_value.py`):

| class | backtest 2 (82 games) | backtest 3 (49 games) |
|---|---|---|
| mirror | +$710 (25 better / 2 worse), 7-9-11 -> 21-6-0 | +$535 (30 / 1), 12-14-5 -> 27-4-0 |
| nsell_opener | +$221 (31 / 11), 27-15-0 -> 28-14-0 | +$141 (5 / 4), 3-6-0 -> 2-7-0 |
| other_opening | +$110 (8 / 5), 6-7-0 -> 7-6-0 | +$112 (6 / 3), 5-4-0 -> 5-4-0 |

Almost all the results it changes are against mirrors, where it was also safe against the reacting public
engine in the arena (20-0, +$345 over `_ADV_LOOK` 4); against reacting Forecast, Shepherd, pilkwang and
V55/V57 agents it cost $60-122 a game. So it is applied per class: the `rival_counter` stage with
`counters: {"mirror": {...}}`. Exactness test (`~/kagg-evo/counter_test.sh`, 60 arena games): Alder Ford with
that counter reproduces the static lead-sell graph against the public engine and Alder Ford itself against
the Forecast agent and V43, game for game.
