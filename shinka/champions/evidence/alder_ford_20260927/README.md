# Alder Ford's losses after 100 ladder games (2026-09-26, 23:00 UTC)

Alder Ford (Kaggle submission 56582140) had played 100 public ladder games by 22:29 UTC: 58W-31L-11T,
rated 2,085.9 (2,077.9 at 19:00). The first 91 games are analysed in `../alder_ford_20260926/`; its
nine newest games (21:08-22:29 UTC) went 3W-3L-3T against rivals rated 2,050-2,200.

## Files

- `index.json`: all 100 games (`fetch_games.py` of `../linden_brook_loss_audit_20260925/`, read-only API).
  The replays of the 82 games against rivals rated 1,900 or more, or unlisted, are in
  `replays/ours/alder_ford/` (not in git), and each is a replay opponent in `../../replay_opponents/`.
- `stats.json`: `../linden_brook_loss_audit_20260925/resim_games.py` on all 42 lost and tied games; every one
  re-simulates money-exact.
- `gaps.py` / `gaps.txt`: per game, where the money went (ours minus the rival's): sales by product net of
  product purchases, animals, seeds, hires, land; the gap by day; both herds; the rival's step-0 orders.
- `families.py` / `families.json`: the 82 recorded games by the rival's step-0 wheat orders.
- `bt_report.py`: a backtest report per graph (record, change a game, flips, per family); `v1cmp.py`: validate1 per
  opponent for several graphs. Both read the run on cliproxyapi.
- `ladder_match.jsonl`, `best_match.py`: `match_ladder_games.py` for the 18 lost or tied games not matched
  before, against 33 public bundles (the ladder pool and the candidates of `../alder_ford_20260926/`).

## Three kinds of lost or tied games

| kind | games | what happened |
|---|---|---|
| ties | 11 | In 10 the rival is the public engine (tetsutani's "Demand-Preserving" agent) move for move: all 719 moves for the 9 newly matched ones and for BorisV. Alder Ford plays it move for move too, so both farms earn the same every day. The eleventh (shuinno) opens with buy 13 / sell 8 and then earns exactly what we do every day. |
| close losses (at most $600) | 15 | Near-copies: in all 15 the rival ended with exactly our herd (the same route tape), from four opening families (the public engine 4, the Forecast family and its variants 8, the 2965 family 3). The gap opens after day 20, on milk, strawberries, wool or wheat. |
| large losses (more than $600) | 16 | A different strategy: 2965-family wheat trading, more geese, carrots, more cows, early hiring. |

**The close losses are sale races.** The three rivals that start as the public engine and then leave it
win by selling contested products first or in larger lots:
- test_money (the engine for 468 moves) sells 2 wool where the engine sells 1, and won by $328;
- Bhaskar #2 (385 moves) puts the strawberry sale before the wheat sale in the same turn (both seats' orders
  clear slot by slot) and won by $1,712;
- robikscube (80 moves) keeps a wheat unit the engine sells, and won by $393.

Summed over the 15 close losses and 11 ties, the gap is on milk (-$3,909), wheat (-$1,404), fertilizer
(-$1,027) and wool (-$629). Against King-damon (-$407) we grew a late tomato crop: +$8,093 on tomatoes, and
-$7,304 on the land and hires for it.

**The large losses are other strategies** (summed over 16 games, ours minus theirs): carrots -$18,107, eggs
-$17,934, wheat -$13,702, wool -$7,304; tomatoes +$9,296, fertilizer +$6,742 and melons +$4,762 in our favour.
- syouya tobita (-$9,275), extra eclaire, ADRIANO ALMEIDA: 2965-family wheat trading (the 20-in / 15-out
  opening).
- Alexander Sokolov, shivam kanojia: buy 3 wheat and hire early; more geese (eggs -$4,483 and -$10,450).
- Nikita Makarov: 11 cows and 3 sheep against our 9 and 5 (milk -$8,457).
- Sadettin Şamil Verdil: the 13-wheat opener with carrots (-$16,393 on carrots, +$12,922 on wheat and
  fertilizer).

## By rival family (the 82 recorded games)

| rival's step-0 wheat orders | games | W-L-T | mean margin |
|---|---|---|---|
| buy 5 + a wheat seed (the public engine and its clones) | 24 | 6-8-10 | -$210 |
| buy 8 / sell 3 + a seed (the Forecast family) | 18 | 14-4-0 | +$903 |
| buy 20 / sell 15 (the 2965 family) | 17 | 11-6-0 | -$104 |
| buy N / sell N-5 + a seed, N = 6-18 (the Forecast shape with another N) | 13 | 3-9-1 | -$370 |
| other openings | 10 | 6-4-0 | |

The public engine's clones are the largest group, and 10 of the 11 ties are against them.

## What to change: sell ready stock one turn earlier (`_ADV_LOOK` 3 -> 4)

The game clears both seats' orders slot by slot and, within a slot, prices both players' units in
lockstep: identical order lists earn identical money (the ties), and whoever sells a product in an
earlier turn or slot gets the better price. The engine's ready-stock layer sells milk, wool, eggs,
strawberries, melons, carrots and tomatoes up to `_ADV_LOOK` turns before its route tape does; the public
engine looks 3 turns ahead. Looking 4 turns ahead puts our sales one turn ahead of every rival running it.

**Against the recorded rivals** (backtest1: each of the first 72 recorded games replayed with the rival's
exact moves; `../alder_ford_20260926/README.md`), `_ADV_LOOK` 4 kept all 36 won games won, turned 6 of the 8
ties and 2 of the 28 losses (test_money, edwinis) into wins, and turned 1 tie into a loss (Mike Kim,
-$164): 44W-27L-1T instead of 36W-28L-8T, +$60 a game.

**On fresh arena games** (validate1: 20 arena validation seeds against each of the 14 pool agents, plus 10
held-out lost seeds from both seats, 300 games per graph), against the champion on the same games:

| games | champion W-L-T | with `_ADV_LOOK` 4 | change a game (better / worse) |
|---|---|---|---|
| tetsutani_demand (the public engine), 20 validation seeds | 12-0-8 | 19-1-0 | +$200 (18 / 2), p = 0.0004 |
| abo_v43, 20 validation seeds | 20-0-0 | 20-0-0 | +$268 (18 / 0) |
| the other 12 pool agents, 240 validation games | 206-34-0 | 212-28-0 | +$2 (114 / 99); -$42 to +$46 per agent, none significant |
| Rowan Glen's and Linden Brook's held-out lost seeds, both seats | 15-5-0 | 16-4-0 | -$8 (8 / 10) |
| all 300 | 253-39-8 | 267-33-0 | +$32 (158 / 111), p = 0.005 |

## Backtest 2: all 82 recorded games, simple sale-timing variants (2026-09-26 23:39 UTC)

`~/kagg-evo/backtest2.sh` on cliproxyapi (`ladder_validate.py --name backtest2 --seeds 0 --replays ...`; backtest1's
games reused): Alder Ford and seven variants against the recorded moves of all 82 games with rivals rated 1,900
or more, or unlisted (backtest1's 72 plus the 10 newest). Alder Ford reproduces its ladder record, 40W-31L-11T.
Each variant against it on the same 82 games (report script: `bt_report.py`):

| graph | record | change a game | better / worse | p | L->W | T->W | W->L | T->L |
|---|---|---|---|---|---|---|---|---|
| `_ADV_LOOK` 4 + lead sells 12 turns ahead (`_EV_H`, `_DP_H`, `_MP_H` 8 -> 12) | 56-26-0 | +$365 | 64 / 18 | 3e-7 | 7 | 11 | 2 | 0 |
| `_ADV_LOOK` 6 | 52-30-0 | +$70 | 42 / 37 | 0.65 | 5 | 7 | 0 | 4 |
| `_ADV_LOOK` 4 + guard without milk | 53-28-1 | +$63 | 56 / 19 | 2e-5 | 4 | 9 | 0 | 1 |
| `_ADV_LOOK` 4 | 51-30-1 | +$62 | 57 / 17 | 3e-6 | 2 | 9 | 0 | 1 |
| `_ADV_LOOK` 5 | 50-32-0 | +$62 | 39 / 38 | 1 | 5 | 5 | 0 | 6 |
| guard without milk | 41-30-11 | -$1 | 7 / 5 | 0.77 | 1 | 0 | 0 | 0 |
| the plain public engine (no guard, no `_CXD_BUDGET`) | 35-33-14 | -$33 | 11 / 26 | 0.02 | 0 | 0 | 2 | 3 |

- **Lead sells 12 turns ahead** is the big step. The engine's evening (hours 15-20), dawn (0-2) and midday
  (10-13) lead-sell layers move about three quarters of the carrot, tomato, strawberry, melon, egg, milk and wool
  sales the tape plans for the next `_H` turns into the current turn, in its first order slot, while the price is
  at or above its 12-turn average; the public engine looks 8 turns ahead. With 12
  and `_ADV_LOOK` 4, every one of the 11 ties becomes a win (+$162 to +$1,582) and so do 7 close losses
  (robikscube, Rohan L, test_money, Umataro Tenma, Yusuraume, ikevayansky, edwinis); two close wins against the
  Forecast family become close losses (aaaaLiu123 +$67 -> -$12, Nakul Maurya +$403 -> -$34). By family: the public
  engine +$758 a game, the 2965 family +$268, other openings +$242, the Forecast family +$117, the two 13-wheat
  games -$6.
- `_ADV_LOOK` 5 and 6 add nothing over 4: about as many games get worse as better (39 / 38 and 42 / 37), and they turn
  more ties into losses (6 and 4, against 1 for `_ADV_LOOK` 4).
- The guard's milk sales cost a little: without milk, +$1 on top of `_ADV_LOOK` 4.
- Our layers are worth $33 a game against these rivals: the plain engine would have tied 3 more games and lost 2
  more.

**Caveat.** A replay does not react. Selling 12 turns of planned sales early pushes the price down before the
rival's recorded sales, and a recorded rival sells into it anyway, where a live one might wait. The larger the
change, the more a replay can overstate it; fresh arena games against agents that react decide.

## Fresh-seed validation of the lead-sell variant (2026-09-27, 00:09 UTC)

`~/kagg-evo/validate_best.sh look4_lead12 <graph>` added `_ADV_LOOK` 4 + lead sells 12 turns ahead to validate1's 300
games (the same seeds as the table above; report `v1cmp.py opening adv_look4 look4_lead12`, run on cliproxyapi).
W-L-T per graph, then the lead-sell variant's change a game (better / worse games, exact sign test):

| games | Alder Ford | `_ADV_LOOK` 4 | + lead 12 | + lead 12 vs Alder Ford | + lead 12 vs `_ADV_LOOK` 4 |
|---|---|---|---|---|---|
| tetsutani_demand (the public engine) | 12-0-8 | 19-1-0 | 20-0-0 | +$545 (20 / 0, p 2e-6) | +$345 (19 / 1, p 4e-5) |
| haideptry_2965 | 14-6-0 | 16-4-0 | 16-4-0 | +$165 (14 / 6) | +$155 (14 / 6) |
| haideptry_2965_0926 | 16-4-0 | 17-3-0 | 18-2-0 | +$186 (14 / 6) | +$150 (16 / 4, p 0.01) |
| abo_v43 | 20-0-0 | 20-0-0 | 20-0-0 | +$237 (16 / 4, p 0.01) | -$30 (7 / 13) |
| mohui | 20-0-0 | 20-0-0 | 20-0-0 | +$81 (15 / 5, p 0.04) | +$103 (16 / 4, p 0.01) |
| tetsutani_shape_shop | 20-0-0 | 20-0-0 | 20-0-0 | +$96 (13 / 7) | +$64 (11 / 9) |
| abo_v57_open13 | 20-0-0 | 20-0-0 | 20-0-0 | +$27 (7 / 13) | -$30 (4 / 16, p 0.01) |
| robust_economy | 20-0-0 | 20-0-0 | 20-0-0 | -$20 (7 / 13) | +$5 (7 / 13) |
| haideptry_shepherd | 12-8-0 | 13-7-0 | 12-8-0 | -$64 (8 / 12) | -$110 (6 / 14) |
| abo_v57 | 17-3-0 | 16-4-0 | 17-3-0 | -$97 (7 / 13) | -$80 (7 / 13) |
| leoprovorov_forecast | 11-9-0 | 13-7-0 | 13-7-0 | -$98 (7 / 13) | -$78 (5 / 15, p 0.04) |
| abo_v55 | 19-1-0 | 20-0-0 | 20-0-0 | -$102 (8 / 12) | -$60 (8 / 12) |
| hanifnoerrofiq_pioneers | 17-3-0 | 17-3-0 | 15-5-0 | -$106 (9 / 11) | -$70 (8 / 12) |
| pilkwang_sep | 20-0-0 | 20-0-0 | 20-0-0 | -$119 (2 / 17, p 7e-4) | -$122 (1 / 19, p 4e-5) |
| all 280 validation games | 238-34-8 | 251-29-0 | 251-29-0 | +$52 (147 / 132, p 0.4) | +$17 (129 / 151, p 0.2) |
| held-out lost seeds, both seats (16 of them against haideptry_2965) | 15-5-0 | 16-4-0 | 18-2-0 | +$311 (16 / 4, p 0.01) | +$318 (16 / 4, p 0.01) |
| all 300 | 253-39-8 | 267-33-0 | 269-31-0 | +$69 (163 / 136, p 0.13) | +$37 (145 / 155, p 0.6) |

- Selling planned sales 12 turns early wins the sale races against the public engine and the 2965 family, the
  rivals of most of Alder Ford's ties and close losses. It gives away price against agents that sell later
  (pilkwang_sep 1 better / 19 worse, the Forecast family, Shepherd, V55/V57), mostly without changing the result.
- On fresh seeds its W-L equals `_ADV_LOOK` 4 alone (269-31 against 267-33) and its dollar gain is not
  significant. The replay backtest's +$303 a game over `_ADV_LOOK` 4 is larger than any live measurement.
- `_ADV_LOOK` 4 alone is significant on both the backtest and fresh seeds, and costs nothing anywhere. The loop
  promoted it on its own on Island-Market at iteration 39 (159W-109L, +$28 a game, p = 0.003).
- The lead-sell edit is queued for Island-Market, whose champion has `_ADV_LOOK` 4 (iteration 51), so the loop's
  302-game gauntlet (with the 24 replays) judges it.
