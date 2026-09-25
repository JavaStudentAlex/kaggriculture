# Linden Brook (56545702) loss audit — 2026-09-25

Linden Brook = the feed15 graph + predictor `ttm_c256_h96_ft_2026-09-24` + its ladder calibration
(`shinka/champions/submissions/linden_brook/MANIFEST.json`), submitted 2026-09-25 08:29 UTC.
By 20:35 UTC it had played **59 ladder games: 33 W – 26 L – 0 T** (seat 0: 16–12, seat 1: 17–14;
plus the validation self-play game), rating 1145.8. No errors or timeouts; every game ended DONE.

## How the numbers were made

1. `fetch_games.py 56545702 replays/ours/linden_brook index.json` (read-only Kaggle API): the
   submission's COMPLETED public episodes, our seat by the exact submission id, both rewards, the
   opponent's team and submission, and that submission's rating (`team-submissions`; `null` = no
   longer active). Replays: `replays/ours/linden_brook/episode-<id>-replay.json` (git-ignored,
   ~31 MB each, 1.8 GB). `index.json` here is the 20:40 UTC snapshot, with each game's seed.
2. `resim_games.py index.json replays/ours/linden_brook stats.json 3`: every replay is re-simulated
   through kaggle-environments 1.32.7 from its seed (`info.seed`; `configuration.seed` is null in
   Kaggle replays) and both seats' recorded actions (the action stored at `steps[t+1]` is the one
   taken from observation `t`). The engine's unit-commit, hire and land functions are wrapped, so
   every executed sale and purchase is logged with seat, step and price: per-product revenue is
   exact, not inferred. **All 59 games reproduced both seats' money on every one of the 719 steps.**
   The engine's randomness (weeds, the next town shop) comes from `Random((seed * 1_000_003) ^ day)`,
   so a game is fully determined by its seed and the two action tapes. (Needs about 11 s and
   300 MB per game; `stats.json` stores `me`/`opp` as the per-seat accounting.)
3. `town_shops.py index.json replays/ours/linden_brook shops.json`: the shops unlocked at the start
   of every day.
4. `aggregate.py stats.json shops.json > report.txt`: every table below.

## Findings

**1. The rating is its level.** By the current rating of the opponent's submission (20:40 UTC):
below 1000: 8 W – 0 L; 1000–1200: 22 W – 12 L; 1200–1400: 0 W – 4 L; 1400 and up: 0 W – 6 L;
inactive submissions: 3 W – 4 L. It never beat a submission rated ≥ 1200. Ten of the 26 losses
were to submissions made after ours (still settling).

**2. Losses are decided after day 15.** Revenue is level through day 14 (ours $43.3k vs theirs
$42.0k in losses), then the opponent out-earns us $83.5k to $62.2k over days 15–29. Mean final
cash in losses: $73.5k vs $88.6k. Hands and land are equal (3 quadrants each; 2 opponents bought a
4th), the opponent spends ~$5k more (animals, wheat), unsold stock at the end is ~0 on both sides,
and our wheat and fertilizer trading nets about the same in wins and losses (~$8k and ~$12–13k a
game).

**3. Our farm is a fixed script.** The Mohui v66 backbone under the Hazel runtime replays recorded
routes chosen by the order of the town's shops (`_v58_base_action` / `_v59_route` in
`../../submissions/linden_brook/hazel_runtime/mohui_v66/candidate_v60_curve_counter.py`):
- first shop YARN_STORE (unlocks on day 3) → the sheep route (6 cows / 11 sheep);
- first shop PET_CAFE and second YARN_STORE or PET_CAFE → the pet route (also sheep-heavy);
- a few step-72 "modes" that need an exact public signature (both players' money, wheat stock and
  the opponent's animals and crops): narrow recognisers of known opponents;
- everything else → the default route: 9 cows / 5 sheep, strawberries, wheat, melons.
In 59 games we held **0 geese**, grew **no tomatoes**, sold ~12 carrots a game and never bought the
4th quadrant. Day-14 animals (goose/cow/sheep): 0/9/5 in 43 games, 0/6/10–11 in 10, 0/8/5 in 6.

**4. Yarn store opening on day 6 → 0 W – 9 L.** With the yarn store second (day 6) after an
ice-cream, pizza, smoothie, brunch or farmers-market shop no route applies (after a bakery there is
one, but only behind an exact step-72 signature, which did not match in our one such game), and we
stay on 9 cows / 5 sheep. We lost all 9 such games (mean margin −$18.2k), to opponents with 10.0 sheep on
day 14 on average (median 9) against our 5; their wool revenue beat ours by $27.6k a game (median
$19.9k). Games on the sheep or pet route: 8 W – 2 L (yarn store on day 3: 7–1).

**5. 15 losses: products we never make.** (The other 2 losses: a yarn store on day 9, and a
$2.2k game.) Opponents with geese (2.9 on average) and
carrots/tomatoes earn from bakery / brunch / pet-café / farmers-market / pizza demand that we leave
to them: egg +$5.3k, carrot +$6.9k, tomato +$1.9k per game on average, while matching us on the
products we share.

**6. Our income depends on the town's draws.** We sell the same volumes in wins and losses
(≈ 262 strawberries and ≈ 250 milk a game on average). With low strawberry demand our strawberries fetch $66 a unit ($17.3k a
game, lost 16/29); with high demand $152 ($39.9k, lost 10/30). Milk: $36 vs $109 a unit ($8.9k vs
$27.9k). Stronger opponents shift production toward the shops the town unlocked; we don't.

**7. What the predictor can and cannot do.** These are production decisions made by the backbone's
routes; a better opponent-sale forecast (the oracle) changes when we sell, not what we grow. The
sell-timing gap is smaller: in losses the opponent averages $147 per wool unit to our $120 and $68
per milk unit to our $59.

## Losses (seed = `info.seed` of the replay)

| episode | seed | seat | opponent (rating now) | margin | town shops by day 12 | their sheep/geese d14 (ours) | their biggest revenue gains over us |
|---|---|---|---|---|---|---|---|
| 113222599 | 1025817768 | 0 | ayman elamin (1101.3) | -47,993 | bakery, **yarn**, pizza, **yarn** | 15/0 (5/0) | wool +87,735, tomato +7,354 |
| 113223849 | 131716916 | 0 | Shardul Gharat (1240.7) | -6,370 | bakery, smoothie, bakery, pizza | 4/2 (5/0) | wheat +8,707, egg +4,859 |
| 113226248 | 270102265 | 1 | rishavsaigal (1574.4) | -15,410 | smoothie, farmers mkt, pet café, farmers mkt | 6/3 (5/0) | carrot +8,848, egg +4,484 |
| 113226277 | 379442008 | 1 | Zhong Lyu (1383.4) | -36,899 | pizza, pizza, smoothie, smoothie | 5/3 (5/0) | tomato +28,875, milk +6,803 |
| 113228742 | 1863161608 | 0 | Phương Nguyễn Việt (1130.0) | -2,195 | pet café, bakery, pizza, bakery | 8/0 (5/0) | milk +1,043, wool +740 |
| 113229927 | 1825472073 | 1 | Kers Aoyagi (1136.0) | -11,270 | pizza, **yarn**, ice cream, ice cream | 9/0 (5/0) | wool +18,302, fertilizer +1,174 |
| 113231075 | 925888793 | 1 | K.nosaka (1024.9) | -7,098 | ice cream, **yarn**, pet café, brunch | 10/0 (5/0) | wool +29,336, strawberry +1,061 |
| 113238070 | 2101348097 | 0 | Anton Tikhonov (1560.9) | -16,829 | brunch, pet café, bakery, brunch | 6/5 (5/0) | wheat +48,703, egg +11,321 |
| 113238122 | 216987965 | 1 | zyichi (1090.9) | -6,064 | smoothie, **yarn**, bakery, smoothie | 8/0 (5/0) | wool +17,538, strawberry +694 |
| 113240486 | 2049321353 | 1 | wuy1hao (1202.4) | -16,716 | brunch, **yarn**, **yarn**, brunch | 8/0 (5/0) | wool +15,859, fertilizer +846 |
| 113245144 | 16002583 | 1 | AlexMoura2026 (inactive) | -35,089 | pet café, ice cream, pet café, farmers mkt | 6/3 (5/0) | carrot +32,059, egg +4,384 |
| 113246316 | 2043505370 | 0 | Busya PRIME (1071.3) | -14,036 | ice cream, **yarn**, brunch, farmers mkt | 9/0 (5/0) | wool +20,166, fertilizer +1,015 |
| 113252276 | 36705986 | 1 | Cuong Le (1138.7) | -6,275 | brunch, farmers mkt, pet café, pet café | 5/0 (5/0) | carrot +8,511, strawberry +1,406 |
| 113255835 | 1847291811 | 1 | Justin Mai 16 (1119.9) | -1,846 | farmers mkt, bakery, pet café, farmers mkt | 4/2 (5/0) | egg +3,785, wheat +494 |
| 113257010 | 1586964654 | 0 | williams (1129.6) | -7,997 | bakery, bakery, pizza, farmers mkt | 5/3 (5/0) | egg +5,403, carrot +3,297 |
| 113261712 | 1991302409 | 1 | Matin Urdu (inactive) | -16,430 | ice cream, **yarn**, pet café, smoothie | 11/0 (5/0) | wheat +40,967, wool +19,892 |
| 113263462 | 1010027077 | 0 | Ansh Agarwal (1831.8) | -18,388 | bakery, pizza, brunch, **yarn** | 6/3 (5/0) | egg +5,640, wool +4,152 |
| 113264061 | 1074457773 | 1 | Jesse Bullard (inactive) | -9,803 | pet café, bakery, brunch, brunch | 5/4 (5/0) | egg +7,978, carrot +5,574 |
| 113266414 | 526045270 | 0 | Xiang Li (1138.2) | -1,530 | pet café, pet café, smoothie, brunch | 4/3 (10/0) | egg +5,016, milk +965 |
| 113267586 | 162630429 | 0 | Speril (1177.6) | -16,426 | pet café, brunch, farmers mkt, bakery | 5/3 (5/0) | carrot +11,281, egg +4,865 |
| 113286389 | 1336616261 | 1 | Pranav Thota (1162.5) | -6,922 | ice cream, brunch, **yarn**, pizza | 11/0 (5/0) | wool +14,097, carrot +3,017 |
| 113326519 | 140659244 | 0 | PUN (inactive) | -8,690 | **yarn**, brunch, farmers mkt, pet café | 11/0 (11/0) | carrot +8,760, wool +874 |
| 113336002 | 1441443537 | 0 | z7777 (2371.7) | -17,033 | bakery, brunch, bakery, ice cream | 6/5 (5/0) | wheat +144,917, egg +10,171 |
| 113341838 | 258395137 | 1 | Dariush Afshar (1452.9) | -19,433 | pet café, brunch, bakery, pizza | 6/5 (5/0) | egg +8,380, carrot +6,622 |
| 113397704 | 1690491916 | 0 | volumatic (1234.5) | -18,339 | farmers mkt, **yarn**, ice cream, pet café | 9/0 (5/0) | wool +17,542, carrot +4,989 |
| 113421316 | 282724254 | 1 | edwinis (1411.1) | -25,602 | ice cream, **yarn**, farmers mkt, farmers mkt | 11/0 (5/0) | wool +21,712, tomato +10,980 |

## Follow-up: the yarn-second fix, played against Linden Brook (2026-09-25)

**Fix** (`research/procedural_graph/arena/yarn_second_fix.py`): one rule in the v66 routing
(`candidate_v66_meta_closed_loop.py`, step-144 block): if the second shop is a YARN_STORE, the
first is neither YARN_STORE nor PET_CAFE, and no route, curve counter or v65 attack mode is
active, take the `bakery_yarn` route. That route is the default route up to step 144 and the
yarn plan after it (the pet route's plan as well: 2 cows and 9 sheep bought from day 6).
Everything else is Linden Brook byte for byte (the arena `old` bundle equals the submitted
package in all 30 code, runtime and checkpoint files).

**Match**: fixed vs old on the 59 seeds of this audit, both seats, every game twice: 236 games on
5 Colab High-RAM VMs (run `yarnfix`, 2026-09-25 21:43–22:25 UTC; jobs in `yarnfix_jobs.json`,
report in `yarnfix_results.txt`, made by `research/procedural_graph/arena/yarnfix_report.py`).
- 236/236 DONE, no errors; every repeat identical to the dollar (the games are deterministic).
- The two agents play identically until step 144, so a seed's town up to day 6 is the same in
  both of its games. **8 of the 59 seeds opened the yarn store second: the fixed agent won all
  16 of those games (8 per seat), by $21,772 on average** (+$11.2k to +$31.8k). They are 8 of the
  9 Kaggle losses with a day-6 yarn store (the ninth, 925888793, opened it first in our games).
- Engine replays of those games (`arena/replay_trace.py`, all faithful): fixed 10–11 sheep and
  6 cows on day 13 vs old 5 sheep and 9 cows; wool 256–282 units ($52.4k a game on average) vs
  131 ($25.6k); milk $16.2k vs $22.0k; no animal starved or escaped on either side.
- The other 51 seeds are mirror games: identical cash per seat whichever agent sat there (48
  exact ties, 3 seeds a few dollars apart by seat), netting to zero.
- Caveat: the old agent stays on cows, so the fixed agent has the wool market to itself. Ladder
  opponents in these towns kept ~10 sheep; the gain there will be smaller than +$21.8k.

## Second pass: the market, the farm's waste, sell timing, one opponent family (2026-09-25)

`market_audit.py stats.json replays/ours/linden_brook market_audit.json 3` re-simulates the 59
games once more (again money-exact on every step of every game) with more of the engine wrapped:
every farmer and hand action and the ones that did nothing, the daily animal and plant refresh
(unfed days, escapes, care bonuses, production lost to caps, withered plants), crop decay, shed
overflow, refused market orders, every sale's price against the product's highest price in the 24
steps before it, each product's market stock on every step, and each seat's time bank.
`market_report.py market_audit.json stats.json shops.json > market_report.txt` makes the tables.

**8. The farm work is clean.** Losses, per game, ours | theirs: 19 | 48 unfed animal-days (of
~340), 0.2 | 1.8 animals escaped, 0 | 6 units lost to shed overflow, 1 | 2 withered plants, 0 | 1
crop units rotted. One small leak: ~11 FEED actions a game by a unit holding no wheat, which
leaves the animal unfed that day and forfeits ~5.5 care-bonus units (about $300 a game).

**9. Sell timing is on par with the opponents.** Price received over the product's highest price
in the previous 24 steps, losses, ours | theirs: wheat 0.96 | 0.97, strawberry 0.58 | 0.60, milk
0.55 | 0.57, melon 0.74 | 0.74, fertilizer 0.85 | 0.86; wool 0.69 | 0.76 is the one gap (the
yarn-store games). In wins ours is at least as high as theirs. The predictor steers timing, and
timing is not what loses these games.

**10. What we grow crashes when the town does not buy it.** By the engine's price curves
(`market_price`), a surplus of 59 wool, 62 strawberries, 76 milk or 158 melons puts the price at
$1, while wheat never falls below ~$17 and eggs still fetch $38 with 1,000 extra. The town takes 6
units a day per shop per product it buys (12 for the one-product shops) plus 1 a day at the town
centre; fertilizer has no buyer at all. We sell ~260 strawberries, ~250 milk and ~150 wool a game
whatever the town. In losses the milk market is oversupplied 61 % of the game and strawberries
30 %, and per game we sell 83 milk, 51 strawberries and 40 wool at $1 (wins: 47, 27, 48).

**11. Our fixed plan pays only when the town has the shops for it.** W–L by the shops buying milk
open on day 15: none 3–6, one 9–12, two or more 21–8. Our cows earn $25 (no milk shop), $47 (one)
or $139 (two or more) of milk per cow-day, against a feed of one wheat (~$40) a day. Counting the
shop instances buying milk or strawberries on day 15: 1–3 → 8 W – 14 L, 4 → 9–5, 5+ → 16–7.

**12. One opponent family takes most of our losses.** Twenty opponents had exactly 33 strawberry
and 23–24 wheat plants on day 14 (we have 38 and 22), and 14 of them earned exactly $14,389 from
melons: another version of the same public Mohui farm. **We went 5 W – 15 L against them (mean
−$11.0k) and 28–11 against everyone else (+$12.7k).** Nine of the ten opponents rated ≥ 1200 are in
this family, the six rated ≥ 1400 among them. Their herd follows the town: 0/6/11 or 0/8/9
geese/cows/sheep when the yarn store comes early (the case Rowan Glen fixes), 3–5 geese bought on
days 8–11 in 9 of the 20 games (none with a yarn store among the first two shops), and carrots and
tomatoes for pet cafés, farmers markets and pizza shops. Over the 20 games they out-earned us on
carrots (+$4.2k a game), wool (+$3.5k), eggs (+$2.9k) and tomatoes (+$2.0k); we out-earned them on
wheat (+$3.4k) and milk (+$0.6k). Geese pay in any town because eggs never crash: opponents with
geese made 1.43 eggs and $83 per goose-day, and we lost 13 of those 17 games. Our backbone has no
goose route (GOOSE appears only in its counting tables). The two strongest, z7777 (2372) and Anton
Tikhonov (1561), also traded wheat at volume: $160k and $64k of wheat sold.

**13. The late-game gap**, days 15–29 of the losses, theirs − ours per game: wool +$9.1k, carrot
+$4.6k, wheat +$3.2k, egg +$2.9k, tomato +$1.8k; milk −$1.2k.

**14. Time bank.** We use ~3.3 s of the 60 s overage in a game (lowest left: 55.4 s); no risk of a
timeout.

## Files

| file | what |
|---|---|
| `fetch_games.py` | lists the submission's games, rates the opponents, downloads the replays |
| `resim_games.py` | exact per-seat accounting by re-simulation (checks money on every step) |
| `town_shops.py` | the town's shop timeline per game |
| `aggregate.py` | losses against wins; writes `report.txt` |
| `index.json` | every game: time, seat, **seed**, result, cash, opponent, its rating |
| `stats.json`, `shops.json`, `report.txt` | the outputs (20:40 UTC snapshot, 59 games) |
| `yarnfix_jobs.json`, `yarnfix_results.txt` | the fixed-vs-old match on these seeds (jobs, result table) |
| `market_audit.py` | second pass: farm waste, market state, sell timing and time bank per seat, by re-simulation |
| `market_report.py` | its tables, losses against wins; writes `market_report.txt` |
| `market_audit.json`, `market_report.txt` | its outputs (the same 59 games) |

Rerun with the venv that has kaggle-environments 1.32.7 (and the Kaggle SDK for the fetch).
