# Replay opponents: our lost ladder games, with the real rival's moves

A replay opponent plays one seat of a recorded ladder game: at every step it returns the action the
rival took there. Built by `make_replay_opponents.py` from a loss-audit index and its replays:

```sh
python shinka/champions/replay_opponents/make_replay_opponents.py \
  shinka/champions/evidence/alder_ford_20260926/index.json replays/ours/alder_ford   # lost and tied games
```

Each `replay_<episode>/` holds `main.py` (the player), `actions.json.gz` (the rival's 719 recorded actions)
and `SOURCE.json` (episode, seats, rival, seed, recorded rewards, the file's sha256). `arena/payload.py`
builds it as an opponent by its name.

## Why

Our stand-in agents are weaker than the real rivals. On Alder Ford's 21 lost or tied seeds without an exactly
matching public agent, the champion beat the stand-in 13 times and lost 3 times, while the real rivals had
beaten or tied it on every one. A gauntlet against the stand-ins cannot see those losses. Against a replay
opponent the champion replays the ladder game exactly. Checked on 2026-09-26 on all 24: in the gauntlet the
champion's margin against each replay equals the ladder margin to the dollar (e.g. -$6,771 against Nikita
Makarov's, -$1,148 against J.Moriuchi's).

## Limits

- The replay does not react. Against a changed agent it repeats the rival's moves; the engine refuses an
  order the rival can no longer fill or pay for. It is faithful while the change is small and late, like a
  sell-timing or herd edit. It is not a model of the rival's strategy.
- Only the seat we played exists. The gauntlet plays it from our Kaggle seat. The same seed from the other
  seat stays against the stand-in agent, so both seats are still played.
- Use them for Alder Ford's games, whose rivals faced the engine the candidates are built on. The rivals of
  Rowan Glen and Linden Brook faced our Mohui-based agent, which plays differently from the first step.

## In use

- 82 bundles: every game Alder Ford played by 22:29 UTC on 2026-09-26 against a rival rated 1,900 or more, or
  unlisted: 40 won, 31 lost, 11 tied (`--results W,L,T`).
- Run ladder1's plan (since 2026-09-26, iteration 31) plays each of Alder Ford's first 22 lost and 2 tied games
  against its replay: `ladder_seed_plan.py --replay-opponents shinka/champions/replay_opponents`, one job per
  game (`set: replay`).
- Ladder backtests play candidate graphs against all of them (`ladder_validate.py --seeds 0 --replays ...`):
  won games show what a change gives away, lost ones what it wins. Results:
  `../evidence/alder_ford_20260926/README.md` (72 games) and `../evidence/alder_ford_20260927/README.md` (82).
