# Juniper Knoll against Aspen Vale on the ladder (2026-09-29)

**Question** (user, ~20:20 UTC 09-29): Juniper Knoll (Kaggle 56679483) is Aspen Vale (56643792) with better parameters
and the predictor, yet it rates lower (1,866.8 against 2,066.1 at 20:22 UTC). Where did it fail, and why?

**Answer.** Juniper does not play worse than Aspen. On the same ladder games the two agents reach the same result
almost everywhere (section "The same games with the other agent"). Its four changes flip a few close games each way,
and none of them explains a rating gap. The gap is the ladder:
- Juniper started at 600 on the last day, is still climbing, and has 84 games.
- Aspen's rating is left over from a weaker field and is falling (8-27 in the hours Juniper played).
- Measured on the same hours, both perform at about 1,915-1,990.

By 21:46 UTC the gap had shrunk from 199 to 121 points: Juniper 1,896.7, Aspen 2,017.9.

Both agents lose the same way, as in Aspen's 09-29 review (`../aspen_vale_20260929/`):
- **Big losses**, to private agents that out-invest us by day 10.
- **Close races**, against copies of public engines that leave the public code partway.

## What Juniper changed

Juniper = Aspen's graph with four settings changed. The runtime code is byte-identical, and the submitted
`policy_graph.json` files differ only in:

| setting | Aspen | Juniper | what it does (engine `tetsutani_demand_0927`) |
|---|---|---|---|
| `_S809_LOOK` | 3 (engine default) | 4 | sells stock the route tape would sell in the next N turns now, when the quote is at least 50 |
| `_CA_MARGIN` | -15 (engine default) | -22 | plants carrots in place of wheat when carrots pay by more than this margin; acts from day 6 h16 |
| `_SR_MARGIN` | 12 | 14 | sells shed goods at hours 21-23 before the night drop overflows the shed, with this many units of headroom |
| oracle guard | off | on | the predictor (09-13 model, numpy) front-runs predicted rival sales of milk, wool and strawberries (score 0.5, batch 4, keep 2, steps 256-696) |

The case for it was Aspen's own development games: +$54 a game over 727 cached games, results +18/-5
(`research/procedural_graph/evolution_results/land_2026-09-29/README.md`). That is a net 13 results in 727 games,
1.8 %, or about 1.5 games in Juniper's first 84: far below what a ladder rating can show.

## The ratings

Kaggle's API gives no per-game rating, so the rivals' ratings here are their current ones (`index_jk.json`,
`index_av.json`, `fetch_new_games.py`).

| Juniper's games | time (UTC 09-29) | record | rivals' median rating |
|---|---|---|---|
| 1-20 | 15:07-16:15 | 17-2-1 | 1,334 |
| 21-40 | 16:19-17:23 | 13-7-0 | 1,736 |
| 41-60 | 17:27-18:23 | 13-7-0 | 1,829 |
| 61-80 | 18:27-19:55 | 14-6-0 | 1,831 |
| 81-84 | 20:03-20:19 | 2-2-0 | 1,898 |

| Aspen's games on 09-29 | time (UTC) | record | rivals' median rating |
|---|---|---|---|
| 1-20 | 00:02-06:34 | 11-9 | 2,005 |
| 21-40 | 08:23-13:43 | 7-13 | 2,128 |
| 41-60 | 13:59-15:59 | 8-12 | 2,150 |
| 61-80 | 16:11-19:51 | 3-17 | 2,110 |
| 81-84 | 19:59-20:23 | 2-2 | 2,072 |

The two submissions met no common rival. By the rivals' rating, over the same hours (since 15:07):

| rival rating | Juniper | Aspen |
|---|---|---|
| below 1,800 | 41-6-1 | - |
| 1,800-1,900 | 15-11 | - |
| 1,900-2,000 | 0-4 | 2-0 |
| 2,000-2,100 | 1-0 | 4-7 |
| 2,100-2,200 | 0-1 | 1-14 |
| 2,200 and up | - | 1-5 |
| no longer listed | 2-2 | 0-1 |

Performance rating (the Elo rating at which the expected score equals the actual one, rivals' current ratings):

| games | rated games | score | performance |
|---|---|---|---|
| Juniper, all 84 | 80 | 57.5 | 1,919 |
| Juniper, games 61-84 | 23 | 16.0 | 1,993 |
| Aspen, the same hours (since 15:07) | 34 | 8.0 | 1,914 |
| Aspen, 09-29 before 15:07 | 29 | 13.0 | 2,081 |
| Aspen, 09-28 | 45 | 27.0 | 2,072 |

No game of either agent ended in an error or timeout: all 149 recorded games (84 Juniper, 65 Aspen since 06:34) end
DONE/DONE. Juniper used at most 3.2 s of the 60 s overage bank after step 1 (median 0), Aspen at most 1.9 s
(`overage.py`). The predictor's cost on Kaggle is not a factor.

## The same games with the other agent

`counterfactual.py` replays a recorded game from its seed with another bundle in our seat. The rival's seat replays
its recorded moves. Until the bundle plays differently from what we sent, the game is the recorded one; from there on
the rival does not react. With the submitted bundle this is a fidelity check: **Juniper's and Aspen's own bundles
reproduce every replayed game move for move and to the dollar.** The bundles:
- `aspen`: Aspen Vale as submitted.
- `juniper`: Juniper Knoll as submitted.
- `j_noguard`, `j_look3`, `j_ca15`, `j_sr12`: Juniper with one change undone.
- `a_look4`, `a_ca22`, `a_sr14`: Aspen with one of Juniper's engine changes added.

They are all built by `make_variants.py` from the submitted tarballs (sha256 `4cc19f98…` and `139d095c…`).

RESULTS_TABLE

## Who beat Juniper

The rival's class is `rival_counter`'s MirrorTracker on the rival's public state, and its best match is the public
bundle that reproduced its recorded moves longest (`match_ladder_games.py` over 44 public bundles;
`summarize_jkL.log`, `best_match_jkL.txt`, `classes_jkL.json`). Juniper's 24 losses and 1 tie:

| rival | games | median margin |
|---|---|---|
| no public agent (10 with another opening than ours) | 11 | -$1,951 |
| our base engine `tetsutani_demand_0927`, left after 148-408 moves | 6 | -$1,647 |
| the old public engine `tetsutani_demand`, left after 256-292 moves | 3 | -$740 |
| `haideptry_2965_new`, `haodou_ledger_0928`, `statma_submit`, `tt_metav4_v13` (one each, left after 150-436 moves) | 4 | -$41 to -$4,710 |
| `tetsutani_demand_0927` exactly, all 719 moves (Stefano Blando) | 1 | -$4 |

- All 4 of the biggest losses (-$6,034 to -$16,229) are to private agents.
- The one exact public engine beat us by $4, a game Aspen wins against the same moves (+$20; `_CA_MARGIN`).

## Where the money went (exact)

`flows_jk.py` re-runs every recorded step through the environment's interpreter; the simulated money equals the
recorded money at every step of all 149 games. Ours minus the rival's, per game, by product (sales net of the same
product bought back) and by cost (positive = we spent less). `money_split.py`:

| category | Juniper, 25 lost or tied | Aspen, 45 lost (09-29) | Juniper, 59 won | Aspen, 20 won (09-29) |
|---|---|---|---|---|
| strawberry | -$1,684 | -$232 | +$983 | -$412 |
| wheat | -$906 | -$196 | +$279 | -$744 |
| tomato | -$869 | -$510 | +$161 | -$685 |
| egg | -$782 | -$729 | +$17 | -$931 |
| wool | -$489 | -$588 | +$1,263 | +$2,848 |
| fertilizer | -$84 | +$82 | +$263 | +$269 |
| milk | +$32 | +$119 | +$708 | +$602 |
| melon | +$477 | +$258 | +$810 | +$514 |
| carrot | +$754 | -$425 | +$1,630 | -$878 |
| land | +$160 | -$89 | -$68 | $0 |
| hires | +$251 | +$346 | +$378 | +$28 |
| seeds, animals | +$194 | +$213 | -$42 | +$132 |

Carrots are where Juniper differs: `_CA_MARGIN` -22 makes carrots its biggest earner against the rivals (+$1,630 a
win), where Aspen sells fewer carrots than its rivals. Its losses lean on strawberries because four of the five big
ones were strawberry-heavy private agents.

The big losses look the same for both agents (`stories_all.json`, `farms_av2.py stories_all.json 4000`). Our farm on
day 10 is the same in every game:
- two quadrants;
- 5 wheat, 20 strawberries and 12 melons;
- 6-9 cows, 4-7 sheep and 0-2 geese.

The rivals that beat us by thousands have a third quadrant, 15-30 wheat and often 6-8 geese by day 10, and tomatoes by
day 20. We lead in cash on day 10 in all of them and fall behind between days 15 and 25.

## What it means

MEANING

## Files

- `index_jk.json`: Juniper's 84 games; `index_av.json`: Aspen's 193 listed games; `index_all.json`: the 149 replayed
  (Juniper's 84, Aspen's 65 since 06:34 UTC 09-29).
- `fetch_new_games.py` (read-only API, run from the PC; `MIN_RATING=0` fetches every replay) and `ratings.py`, which
  showed that the episode listing has no ratings.
- Replays: `replays/ours/juniper_knoll/` and `replays/ours/aspen_vale/` on the PC (not in git); gzipped copies in
  `~/kagg-evo/juniper_review/replays` on cliproxyapi.
- On cliproxyapi, in `~/kagg-evo/juniper_review`:
  - `counterfactual.py`, `run_cf.sh`, `make_variants.py`; results in `cf/<bundle>.jsonl`, summarized by
    `cf_summary.py`, `cf_table.py` (`cf_table.txt`) and `first_moves.py`;
  - `flows_jk.py` (`flows_all.json`, `flows_av_today.json`), `story_jk.py` (`stories_all.json`), `money_split.py`,
    `farms_av2.py`;
  - `run_batch_jk.sh` and `summarize_jk.py` (rival features and the public-bundle match of the losses);
  - `overage.py` (`overage.json`).
- `match_seat.py`: the first difference only, without playing on (superseded by `counterfactual.py`).
