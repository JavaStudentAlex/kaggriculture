# Hand patch of Copper Weir (2026-09-14)

`hazel_weir/` — **Hazel Weir** = Copper Weir + the fix for the two losing worlds found by the tournament trace
audit (`../../evidence/tournament_2026-09-14/copper_weir_loss_audit.md`). Not a Shinka program: three edits on
the pool file, everything else byte-identical (`summary.json` has both hashes and the change list).

| check | Hazel Weir | Copper Weir (same seeds, same opponents) |
|---|---|---|
| evaluator, 15-pool (`metrics.json`) | 87.7 % (526-74), $81,317 | 84.2 % (505-56-39T incl. its 39-tie mirror) |
| evaluator, vs the 14 others | **509-51 (90.9 %)** | 504-56 (90.0 %) — flips +9 / −4 |
| fresh tournament seeds, vs the 14 others (`challenger_fresh_seeds.txt`) | **516-44 (92.1 %)** | 483-77 (86.2 %) — flips +36 / −3 |
| seeds 12316 / 12720 (the losing worlds) | 6-0, no discards | 0-6, 11 units discarded |
| mirror Hazel Weir vs Copper Weir | 17-23, 16 losses under $25, +$1,940 cash | |

Higher average cash against every one of the 14 opponents on both seed sets. Passes the 75 % gate; crowning was
disabled for the check runs, so it is **not in `pool/`** — replacing Copper Weir (a near-clone) or adding it is a
curation decision.
