# Cedar Ridge's first 99 ladder games (2026-09-28)

Cedar Ridge (Kaggle 56619997, submitted 21:43 UTC 09-27) is the ladder1 Island-Opening champion after iteration 58:
Birch Hollow (the old public engine, `tetsutani_demand`, with `_ADV_LOOK` 4) plus the oracle guard and the mirror
counter (lead sells 12 against copies of our own engine). By 07:30 UTC 09-28 it had played 99 games: 42W-57L-0T,
rating 1,999.2 (Birch Hollow 1,989.5 and Alder Ford 2,002.7 then, both falling from 2,030-2,060 on 09-27).

| games | record | rivals' median rating |
|---|---|---|
| 1-25 (21:53-23:21 UTC) | 20-5 | 1,411 |
| 26-50 (23:22-00:38) | 10-15 | 2,134 |
| 51-75 (00:42-02:06) | 7-18 | 2,079 |
| 76-99 (02:10-07:30) | 5-19 | 2,044 |

Since reaching ~2,000 it went 22-52. All 57 losses were against rivals rated 1,900+ or no longer listed; 44 of them
by less than $1,500.

## Files

- `fetch_new_games.py`, `index.json`: the 99 games (read-only API); `index_crL.json` the 57 losses, `index_crW.json` the
  28 wins against strong rivals. The 85 replays are in `replays/ours/cedar_ridge/` (not in git), and each is a replay
  opponent in `../../replay_opponents/` (`replay_<episode>`).
- `recover_0928.py`: the agents of the 11 public notebooks updated since the 09-27 snapshot, recovered statically on
  cliproxyapi (no notebook code runs). New: Harvest Ledger's 09-28 agent (`haodou_ledger_0928`, now in the ladder
  pool), leoprovorov's "31415926…" notebook, haideptry's Shepherd of 09-27. The 09-28 versions of the 2965 Master
  Hybrid, Guru's master engine v4 and Kunal Desale's TTV1 ship Harvest Ledger's earlier agent; Guru's "v5" ships
  Dvorkin's v31 (which the matcher cannot run); two notebooks hold no agent.
- `run_batch_cr.sh`: on cliproxyapi, the rival's per-step public state (`research/procedural_graph/rival/
  rival_features.py`), the replay opponents and `match_ladder_games.py` for every public bundle (41).
- `summarize_cr.py`: the class (the rival counter's MirrorTracker) and best match of each loss.
- `validate_cr.sh`: the candidates for the next submission on these games (below).
- `before_restart_cr.sh`: the ladder1 restart that adds the 57 lost games to the evolution's plan.

## Who beat Cedar Ridge (57 losses)

By class: nsell_opener 35, mirror 12, other_opening 10. By the public agent that reproduced the rival's moves
longest:

| rival | losses | median margin |
|---|---|---|
| the public engine's 09-27 version, exactly (700+ of 719 moves) | 13 | -$795 |
| the public engine's 09-27 version, 100-700 moves | 10 | -$734 |
| the old public engine (mirror class), 100-700 moves | 9 | -$653 |
| Harvest Ledger's 09-28 agent, exactly | 3 | -$1,293 |
| haideptry's 2965 (3), shiiin9 order book, Shepherd, tt_metav4, 100+ moves | 6 | |
| no public agent | 16 | -$1,143 |

The 09-27 version of the public engine accounts for 23 of the 57 losses. In the arena our old-engine line loses to it
1-23 (-$989 a game), while the line on that engine with the rival emulator (Island-Next-Counter, iteration 68) beats it
23-1 and the old engine 114-14.

## The candidates on these games (`validate_cr.sh`, 09-28)

Three graphs, played through the ladder1 Colab pool by `ladder_validate.py` (reports `runs/ladder1/validation/
backtest_cr.json` and `lost_cr.json` on cliproxyapi):
- cedar: Cedar Ridge itself (`opening_mirror_counter_graph.json`, bundle g_900a45ee4b3107a7);
- opening: Island-Opening's champion then (the same line plus iteration 69's rival emulator, g_9b1302e12eece55d);
- next_counter: Island-Next-Counter's champion (the 09-27 engine, `_SR_MARGIN` 12, the rival emulator;
  g_16bb4af65a704f00), packaged as the candidate "Aspen Vale" (`build_aspen.sh`: validation, fidelity and stage
  fidelity pass).

**Backtest** (`backtest_cr`, 85 games): each graph in our seat against the rival's recorded moves of every replayed game.
Cedar Ridge replays its ladder results exactly. Recorded moves do not answer a different play, so this flatters
changes that make the rival react and understates the emulator (a recorded rival stops matching its engine).

| graph | the 57 lost games | the 28 won games | mean margin |
|---|---|---|---|
| cedar | 0-57 (-$1,166) | 28-0 (+$2,859) | +$160 |
| opening | 3-54 (-$987) | 28-0 (+$3,010) | +$329 |
| next_counter | 36-21 (+$1) | 25-3 (+$3,063) | +$1,010 |

Against cedar game by game: opening 49 better, 0 worse (+$170, p=4e-15); next_counter 71 better, 14 worse (+$850,
p=2e-10).

**Lost seeds** (`lost_cr`, 114 games a graph): fresh games on the 57 lost seeds, from both seats, against the public
agent that played most like the rival (`summarize_cr.py`; the old engine where none did). Only two stand-ins reproduce
the loss: Cedar Ridge loses all 46 games to them and wins 66 of the other 68.

| stand-in (seeds) | cedar | opening | next_counter |
|---|---|---|---|
| tetsutani_demand_0927 (12) | 0-24 (-$898) | 2-22 (-$482) | 24-0 (+$185) |
| haodou_ledger_0928 (11) | 0-22 (-$960) | 4-18 (-$264) | 14-8 (+$472) |
| tetsutani_demand (21) | 40-2 | 42-0 | 40-2 |
| haideptry_2965_0926 (5) | 10-0 | 10-0 | 4-6 |
| leoprovorov_forecast (5) | 10-0 | 10-0 | 8-2 |
| abo_v55, haideptry_shepherd, robust_economy (1 each) | 6-0 | 6-0 | 6-0 |
| all | 66-48 (+$396) | 74-40 (+$765) | 96-18 (+$735) |

Against cedar game by game: opening 96 better, 0 worse (+$368); next_counter 82 better, 32 worse (+$338).

**The evolution's gauntlet games** both have played (463: the pool, earlier lost seeds, Alder Ford's replays): cedar
348-115, next_counter 398-64-1. next_counter is ahead against haideptry_2965 (53-3 against 44-12), leoprovorov_forecast
(24-4 against 18-10), haideptry_shepherd (12-0 against 10-2) and the 09-27 engine (23-1 against 1-23), level against
haideptry_2965_0926 (10-6 each), and behind only against the old public engine (114-14 against 122-6, where Cedar Ridge's
mirror counter acts).

Verdict: next_counter (Aspen Vale) is the only candidate that beats the two rivals behind the reproducible losses, and
it is ahead on every view; its cost is a few more losses to the old engine and to the 2965 family on these seeds. The
57 losses joined the evolution's plan at the 10:29 UTC restart (`restart_cr.log`: plan 634 jobs, from iteration 78).
Submitted as Aspen Vale (Kaggle 56643792) at 14:31 UTC 09-28; Kaggle's validation passed at 14:35 (starting rating
600). Record: `../../submissions/aspen_vale/MANIFEST.json`.
