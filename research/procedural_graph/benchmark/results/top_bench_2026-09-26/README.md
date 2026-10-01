# Top-player benchmark, 2026-09-26's top games (run 2026-09-27)

Five Kaggle CPU notebooks (`sunshinethroughfog/kaggriculture-top-bench-1..5`, built by `make_top_bench.py`, all
`TOP_BENCH_EXIT=0`) played the 80 highest-rated games of `kaggle/kaggriculture-episodes-2026-09-26` (average rating
3,030-3,083). In each game every graph took one top player's seat, from both seats, against the other player's
recorded moves on the game's seed: 160 games per graph, 640 in all. `k/results.jsonl` holds arena.py's lines,
`report.json` the summary of `top_bench_report.py`.

| graph | W-L against the recorded moves | our cash | ours minus the replaced top player's cash: mean (median) |
|---|---|---|---|
| Cedar Ridge (the old engine with the mirror counter, submitted 09-27) | 93-67 | $114,384 | +$8,660 (+$2,824) |
| tetsutani_demand (the old public engine) | 93-67 | $114,466 | +$8,742 (+$3,433) |
| Island-Next-Production's champion (09-27 engine) | 87-73 | $111,072 | +$5,349 (+$1,896) |
| tetsutani_demand_0927 (the public engine's 09-27 version) | 87-73 | $111,072 | +$5,349 (+$1,905) |

How to read it:
- The recorded moves do not react to us. With us in the other seat the replayed player earns a median $12,586
  (mean $22,733) less than in its real game, and 75 of 160 results differ from the real game's. So W-L and the
  mean margin flatter us; our cash against what the replaced player earned in the same seat is the fairer
  number, and its median is only +$2-3k.
- There is no sign of a large production gap: in the seat of a 3,000-rated player our engines earn about what
  that player earned. The ~900 rating points between us and them are not explained by what the farm produces,
  which points at the interaction (sale races, reactions to the rival): the rival emulator and the tactic stage.
- Cedar Ridge plays exactly like the old engine here: its mirror counter acts only against copies of our engine.
- The 09-27 engine, which beats the old one head-to-head in our arena, earns less against the top players:
  it sells fertilizer worth $16.3k a game at the quote against the old engine's $35.7k (other products are
  within $2k).
