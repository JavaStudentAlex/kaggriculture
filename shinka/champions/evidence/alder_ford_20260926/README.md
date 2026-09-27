# Alder Ford's ladder losses: who beat it (2026-09-26)

Alder Ford (Kaggle submission 56582140, submitted 14:50 UTC) is the ladder1 evolution's Opening
champion: the tetsutani_demand engine, the predictor guard on milk, wool and strawberries, and
`_CXD_BUDGET` 1050. By 19:00 UTC it had played 70 ladder games, 46W-22L-2T, and was rated 2,077.9.

## Files

- `index.json`: every game (fetch order, our seat, both final cash amounts, the rival's team, submission
  and its rating when listed), from the read-only episodes API. Replays of the lost and tied games are in
  `replays/ours/alder_ford/` (not in git).
- `ladder_match.jsonl`: `match_ladder_games.py` for each of the 24 lost or tied games against 31
  bundles: the ladder pool (`shinka/champions/ladder/`) and 20 candidates recovered statically from
  newer public notebooks. Two more candidates failed to load. `haideptry_2965_0926` is now in the pool;
  `haideptry_shepherd_0926` is the Shepherd notebook's version of 09-26.
- `candidates.json`: the candidate notebooks, how each agent was recovered, and the sha256 of its files.
  The candidates themselves were not kept.

## What the matching found

- Played exactly (719 of 719 moves):
  - tetsutani_demand: Yusuraume (2055, won by $167) and BorisV (tie).
  - haideptry_2965_0926: ADRIANO ALMEIDA (2031, won by $1,703). This is the 2965 notebook's version of
    09-26 13:54 UTC; the pool's `haideptry_2965` is the version of 09-25.
- Long common prefixes:
  - tetsutani_demand: J.Moriuchi 372 moves, Shangshang Zhang 226, Igor V 150.
  - haideptry_2965_0926: Rohan L 241, edwinis 159.
  - leoprovorov_forecast: King-damon 216, AI After Hours 150.
- Several public notebooks play exactly like tetsutani's agent in these games: Farmer John and the Wheat
  Seller, Master Engine V3, and the Shepherd notebook's 09-26 version.
- The other 13 rivals (14 games) match no public agent for more than 91 moves. Their step-0 wheat orders:
  - buy 8 / sell 8: 3 games (two of them then buy 5);
  - buy 20 / sell 15 without a seed purchase: 2;
  - buy 8 / sell 3 (the Forecast family's opening): 2, one rival;
  - one each of buy 5 (tetsutani's opening), buy 6 / sell 1, 7 / 2, 10 / 10, 18 / 13, 13 (the 13-wheat
    opener) and buy 3 with five hires.

## Use

`research/procedural_graph/ladder_seed_plan.py --extend` added the 24 seeds to run ladder1's plan, from
both seats (48 games):

```sh
python research/procedural_graph/ladder_seed_plan.py \
  --extend research/procedural_graph/evolution_results/ladder_2026-09-26/plan.json --with-ties \
  --losses shinka/champions/evidence/alder_ford_20260926/index.json=replays/ours/alder_ford \
  --evidence shinka/champions/evidence/alder_ford_20260926/ladder_match.jsonl \
  --fallback tetsutani_demand --out research/procedural_graph/evolution_results/ladder_2026-09-26/plan.json
```

It was run on cliproxyapi, where the replays are. Each game got an opponent this way:

| rule | games | bundle |
|---|---|---|
| a pool agent matched at least 100 moves | 10 | tetsutani_demand 5, haideptry_2965_0926 3, leoprovorov_forecast 2 |
| no match, by opening | 6 | haideptry_2965_0926 2, leoprovorov_forecast 2, tetsutani_demand 1, abo_v57_open13 1 |
| no match, unknown opening | 8 | tetsutani_demand |

From iteration 31 each game is also played from our seat against a replay of the rival's recorded moves
(`shinka/champions/replay_opponents/`, `--replay-opponents`), which reproduces the ladder game exactly.

## Why the games were lost (2026-09-26 evening)

Two more files:
- `stats.json`: `../linden_brook_loss_audit_20260925/resim_games.py` run on the 24 games. It gives exact
  per-seat accounting: every sale and purchase, money by day, and farm snapshots. All 24 games
  re-simulate money-exact.
- `ourseat_plain.jsonl`: `match_ladder_games.py` with the seats swapped. The plain tetsutani_demand
  engine plays our seat against the rival's recorded moves, so the first difference is where our own
  layers first changed play.

Findings:
- **Our layers mostly did not act.** In 13 of the 24 games Alder Ford played the public engine move for
  move: the guard and `_CXD_BUDGET` 1050 never changed an action. Those 11 losses and 2 ties are the
  engine's own.
- **Where our layers acted, it was late:**
  - the guard's first sale came on days 16-24 (6 games);
  - in the other 5 games the first change was at step 600 (day 25), two of the engine's sells in the
    other order (fertilizer before milk).
  - In all 6 games where the guard sold, we ended with less milk, wool and egg income than the rival,
    usually with the same herd. Against Yusuraume, who runs the public engine exactly, the guard's milk
    sales cost the whole $167.
- **The arena reproduces the ladder when it has the rival's code.** Against the exact rivals the arena
  gives the ladder result to the dollar: Yusuraume -$167, ADRIANO ALMEIDA -$1,703, BorisV a tie. Against
  the stand-ins of the other 21 lost seeds, the champion wins 13, ties 5 and loses 3. The gauntlet
  therefore sees few of these losses.
- **The variants that beat the engine change little.** Their first deviations from it:
  - a cow instead of a goose (Rohan L, day 10) or instead of a sheep (Shangshang Zhang, day 9);
  - a few more units sold early: wool (Igor V, edwinis), eggs (J.Moriuchi), wheat (King-damon);
  - wheat kept instead of sold (あかつき).
- **The large losses are strategic:**
  - Nikita Makarov: 11 cows and 3 sheep against our 9 and 5, so 47 more milk units: -$7.5k in animal
    products.
  - Alexander Sokolov: the rival made about $15k net on wheat trading and sold 432 more fertilizer units.
    We paid $7k more for land and hands for a late tomato crop that earned $11.7k.
  - syouya tobita: our engine bought $134k of wheat and sold $140k, and still made $8k less net on wheat
    than the rival.

## Backtest against the recorded rivals (2026-09-26 22:15 UTC)

Alder Ford's 72 games against rivals rated 1,900 or more, or unlisted, each became a replay opponent
(`shinka/champions/replay_opponents/`). Eight graphs then played every replay from our seat
(`ladder_validate.py --name backtest1 --seeds 0 --replays ...`; report
`~/kagg-evo/runs/ladder1/validation/backtest1.json`). Alder Ford itself reproduced all 72 ladder margins to
the dollar: 36 won, 28 lost, 8 tied. Each other graph against it, on the same 72 games:

| graph | better / worse | mean change | p | record on the 72 |
|---|---|---|---|---|
| `_ADV_LOOK` 3 -> 4 | 49 / 15 | +$60 | 2e-5 | 44-27-1 |
| guard without MILK | 7 / 3 | +$11 | 0.34 | 37-27-8 |
| `_OG_SCORE` 0.6 | 8 / 8 | +$6 | 1 | 35-29-8 |
| guard stops on day 21 | 10 / 8 | -$2 | 0.8 | 34-28-10 |
| guard without `_CXD_BUDGET` 1050 (Oracle champion) | 3 / 11 | -$1 | 0.06 | 36-28-8 |
| plain public engine (seed) | 11 / 22 | -$16 | 0.08 | 31-30-11 |
| cows-only HERD2 | 1 / 12 | -$337 | 0.003 | 33-32-7 |

- `_ADV_LOOK` 4 (ready-stock sale advancing looks 4 turns ahead instead of 3) gains against every rival
  family: tetsutani's +$143 a game (16 better, 3 worse), the 2965 family +$64, the 13-wheat opener +$75,
  the Forecast family and the other openings +$7-8. Against the stand-in agents the gauntlet had rejected
  it (iteration 24: about -$95 a game against abo_v55, abo_v57 and leoprovorov_forecast). The stand-ins
  mislead here.
- Our layers help a little against the real rivals: the plain engine would have scored $16 a game less.
- Leaning to cows hurts: it recovered Nikita Makarov's game but cost thousands in others.

A replay does not react to our play, so these numbers hold for changes a rival would not answer. The
engine-like rivals at this level mostly play tapes.
