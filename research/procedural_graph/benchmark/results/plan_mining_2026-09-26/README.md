# Plan mining: the top players' plans against our route tapes (run 2026-09-28)

`sunshinethroughfog/kaggriculture-plan-mining` (a Kaggle CPU notebook, `make_plan_mining.py`) ran `plan_mining.py` over
every game of `kaggle/kaggriculture-episodes-2026-09-22..26`: 2,990 games (average rating ~2,950+), both seats, 0
unreadable, ~15 min. `plans.jsonl.gz` holds its 5,980 rows (census at hour 0 of each census day in this first run).
`route_plans.py` counted the 41 route tapes of both engines (`route_plans.json`), and `plan_report.py` compared them
(`report_<engine>.md`, `candidates_<engine>.json`), with our own ladder games as "ours": Alder Ford's and Birch
Hollow's recorded losses and ties and Cedar Ridge's 57 losses (160 games, mostly lost, against rivals near 2,000).

- The top players do not play our tapes. Only 9% of their seats come within 0.3 of any of the 41 tapes (mean absolute
  difference of log1p purchases at days 6, 12, 20 and 29); the median distance is 0.50, where our own games sit at
  ~0.03 from our tapes. Choosing another tape per world cannot copy their economy.
- Their farms (mean per seat, top winners against our games):

  | at the start of | money | land | cows | geese | sheep | tomato tiles | carrot tiles | strawberry tiles |
  |---|---|---|---|---|---|---|---|---|
  | day 15 | $24.9k / $22.9k | 3.5 / 3.1 | 8.5 / 7.3 | 5.7 / 2.5 | 6.5 / 7.6 | 6.3 / 0 | 4.0 / 0.5 | 28.6 / 33.0 |
  | day 20 | $55.6k / $48.1k | 3.5 / 3.2 | 8.2 / 7.3 | 5.7 / 2.5 | 6.7 / 7.6 | 11.0 / 1.6 | 5.8 / 0.7 | 23.3 / 33.0 |
  | day 25 | $79.5k / $66.4k | 3.5 / 3.2 | 7.5 / 7.3 | 5.6 / 2.5 | 6.0 / 7.6 | 5.9 / 1.6 | 12.8 / 9.8 | 10.5 / 17.0 |

  More than twice our geese, tomatoes and carrots where we grow almost none, fewer strawberries, the third land
  quadrant earlier. The money rows compare top-vs-top games with our games against ~2,000 rivals, so they are only
  indicative.
- Four worlds where the top seats that played like tape 110 won clearly more (11-14 seats each, 67-82% won,
  +$2.3k to +$6.7k a game) than those near our tape (none): SMOOTHIE_SHOP|BAKERY, ICE_CREAM_SHOP|ICE_CREAM_SHOP and
  ICE_CREAM_SHOP|BAKERY (tape 110 instead of 105), FARMERS_MARKET|ICE_CREAM_SHOP (110 instead of 107). Queued on
  09-28 for Island-Herd as one `_V92_TABLE` edit; the samples are small and the gauntlet decides.
