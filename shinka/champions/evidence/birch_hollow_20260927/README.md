# Birch Hollow's first 84 ladder games (2026-09-27)

Birch Hollow (Kaggle 56603928, submitted 08:50 UTC 09-27) is Alder Ford plus the engine's `_ADV_LOOK` 3 -> 4.
By 17:00 UTC it had played 84 games: 56W-28L-0T, rating 2,044 (Alder Ford 2,060 then). 63 of them were against
rivals rated 1,900 or more, or no longer listed: all 28 losses and 35 wins.

## Files

- `fetch_new_games.py`, `index.json`: the 84 games (read-only API). The replays of the 63 games against strong
  rivals are in `replays/ours/birch_hollow/` (not in git), and each is a replay opponent in
  `../../replay_opponents/` (194 in all).
- `index_bhL.json`: the 28 losses; `index_bhW1.json`, `index_bhW2.json`: the 35 wins, in two download batches.
- `run_batch_bh.sh`: one batch on cliproxyapi: the rival's per-step public state (`research/procedural_graph/rival/
  rival_features.py`), the replay opponents and, for the losses, `match_ladder_games.py` for every public bundle.
- `match_bhL_all.jsonl`, `best_match_bhL.txt`: 40 public bundles (the pool and the agents recovered from other
  notebooks) matched against the 28 losses. Four recoveries crash in the matcher and match nothing (dvorkin_v31,
  hak_2887, leo_ice_fire, yhay_router0909).
- `classes_bh.json`: the rival_counter class of each loss, in `index_bhL.json`'s order: [class, step of the first
  difference from our farm (null for a mirror), money gap then, result, margin, rival, rating].

## The 28 losses by rival class

- **mirror, 10**: they play the old public engine as we do (169-719 identical moves). These are the mirror counter's
  target.
- **nsell_opener, 11**: 4 play the public engine's 09-27 version (3 of them for all 719 moves, -$834 to -$1,367),
  3 are like Forecast, 2 like statma, 1 like 2965, 1 unknown.
- **other_opening, 7**: no public agent matches; three were lost by $7.8k-$17k.

From the ladder1 restart after iteration 55 (17:36 UTC) the 28 losses are in the evolution's plan: both seats against
the stand-in bundle and one game against each replay (plan 379 -> 463 pool games).

## Replay backtest (`ladder_validate.py --seeds 0 --replays`, report `runs/ladder1/validation/backtest_bh.json`)

Every graph plays the 63 recorded games against the rival's recorded moves, from Birch Hollow's seat.

| graph | bundle | W-L | mean margin | vs Birch Hollow |
|---|---|---|---|---|
| Birch Hollow | g_303b063b37c58a0c | 35-28 (its ladder record) | -$446 | |
| Island-Opening champion (+ `_OR2_SLOT_MARGIN` 0, mirror counter lead 12) | g_900a45ee4b3107a7 | 41-22 | -$149 | 41 better, 10 worse, 12 same, +$297 a game, p = 1.5e-5 |
| Island-Endgame champion | g_dcd22642378a4af0 | 41-22 | -$151 | 45 better, 11 worse, 7 same, +$295 a game, p = 5.4e-6 |

Both champions keep all 35 wins and win 6 of the 28 losses, all 6 against mirrors (6 of 10). They win no loss
against the other classes (0 of 11 nsell_openers, 0 of 7 other openings). The lost games, Birch Hollow's margin and the Opening champion's:

| episode | rival rating | class | Birch Hollow | Opening champion |
|---|---|---|---|---|
| 114106198 | 1,945 | other_opening | -7,828 | -7,825 |
| 114116533 | unlisted | other_opening | -16,986 | -14,716 |
| 114122483 | 1,923 | mirror | -1,489 | +145 **won** |
| 114125498 | unlisted | other_opening | -26 | -26 |
| 114128787 | 1,915 | nsell_opener | -50 | -54 |
| 114129863 | 1,908 | nsell_opener | -751 | -751 |
| 114131335 | 2,107 | mirror | -156 | -266 |
| 114140153 | 1,941 | other_opening | -242 | -230 |
| 114144563 | 1,941 | nsell_opener | -285 | -285 |
| 114149055 | 2,084 | mirror | -172 | -161 |
| 114151915 | 2,040 | mirror | -1,004 | -1,015 |
| 114153672 | 2,143 | nsell_opener | -1,367 | -1,367 |
| 114157680 | 1,938 | mirror | -15 | +1,450 **won** |
| 114158094 | unlisted | other_opening | -960 | -972 |
| 114159209 | 1,941 | nsell_opener | -1,004 | -988 |
| 114166657 | 2,101 | nsell_opener | -601 | -601 |
| 114171081 | 2,032 | nsell_opener | -1,542 | -1,542 |
| 114177166 | 2,076 | nsell_opener | -3,702 | -3,702 |
| 114178592 | 2,088 | mirror | -92 | +687 **won** |
| 114179935 | 2,077 | mirror | -38 | +960 **won** |
| 114194716 | 2,059 | nsell_opener | -90 | -204 |
| 114198804 | 2,063 | other_opening | -58 | -58 |
| 114210798 | 2,089 | mirror | -74 | +209 **won** |
| 114211660 | 2,331 | other_opening | -9,819 | -9,816 |
| 114228442 | 2,148 | nsell_opener | -834 | -832 |
| 114243132 | 2,015 | mirror | -478 | -48 |
| 114244384 | 1,954 | nsell_opener | -1,349 | -1,349 |
| 114253900 | 2,106 | mirror | -81 | +564 **won** |

The later champions on the same 28 lost replays, from the gauntlet's cache (iterations 56-58):
- lead 16 against mirrors (Island-Herd) wins 7 of the 10 mirror games, and lead 20 (Island-Crops) wins 9. Against the
  public engine itself, which answers our earlier sales, both do worse than lead 12 (-$44 and -$236 a game);
- the 09-27 engine's line (Island-Next-Production) wins 17 of the 28 (8 of 11 nsell_openers, 7 of 10 mirrors, 2 of 7
  other openings). A recorded rival fits the engine it played against, so these numbers flatter an engine that
  differs from Birch Hollow's. Against the stand-in bundles it wins about as many games as the Opening champion
  (320 vs 323 of 382).

The Opening champion was submitted as Cedar Ridge (Kaggle 56619997, 21:43 UTC 09-27; `../../submissions/cedar_ridge/`).
