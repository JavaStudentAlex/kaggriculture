# Orchard Tide (56193386) loss audit — 2026-09-14 06:45 UTC

All 97 scored public games of the first oracle-based submission (44 W – 53 L), replays
downloaded to `/results/kagg/our_replays/orchard_tide/` (SSD, ephemeral, 2.9 GB) with
`download_replays.py`; `analyse_games.py` produced `stats2.json` (per game: money by day,
income/spend by day from money deltas, per-seat supply by product from the market
accounting of `research/opponent_model/extract.py`, farm snapshots at hour 12 of each
day). `early.json` = the early-game survival columns quoted in `ideas.md` P1.0
(min cash days 3–10, animal deaths, animals and weeds on day 12, "mirror" = opponent's
first six BUY_ANIMAL orders identical to ours). `index.json` = game list with seat,
rewards, opponent and its leaderboard rating at audit time.
