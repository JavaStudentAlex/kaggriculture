# Ladder pool: the public agents that beat Linden Brook and Rowan Glen

The opponents that beat our submissions on the Kaggle ladder (2026-09-25/26) run public Kaggle
notebooks, or close variants of them. These bundles are those notebooks' agents, recovered
byte for byte, so the arena can play the ladder's opponents instead of our own old champions
(which our agents beat; only plain Mohui was still competitive).

| bundle | notebook | ladder evidence (match_ladder_games.py) |
|---|---|---|
| `tetsutani_demand` | tetsutani, "Demand-Preserving Turn Sale Timing" (cha22 lineage) | **exact**: AkiraIshikawa (rated 2155) and Shane Thivaharraja (2188), both beat Rowan Glen |
| `abo_v55` | Ahmed Berat Ozer, "Kaggriculture V55 — One-Turn Market Race Edge" | **exact**: Ansh Agarwal (1808-1832), beat Linden Brook |
| `abo_v57` | Ahmed Berat Ozer, "Kaggriculture V57 — Funding-Order Invariant" | **exact**: PUN, beat Linden Brook |
| `abo_v43` | Ahmed Berat Ozer, "Kaggriculture V43: Recovering Lost Harvests" | **exact**: Dariush Afshar (1453), beat Linden Brook |
| `haideptry_2965` | haideptry, "The 2965 Master Hybrid Engine" (V57 + order-book slot evaluator), the notebook's version of 09-25 | 150-196 steps: Matin Urdu, Hector Valverde, sneaky6767 (1920) |
| `haideptry_2965_0926` | the same notebook's version of 2026-09-26 13:54 UTC | **exact**: ADRIANO ALMEIDA (2031), beat Alder Ford; 159-241 steps: Rohan L, edwinis |
| `haideptry_shepherd` | haideptry, "The Shepherds Ledger Herd Safe Sovereign" (opening buy 8 / sell 3) | 150-169 steps: z7777 (2437), Rashid Khazeiynasab (2175) |
| `leoprovorov_forecast` | leoprovorov, "Four-Turn Forecast — Notebook Version 2" | 217 steps: Rashid Khazeiynasab (2175) |
| `hanifnoerrofiq_pioneers` | hanifnoerrofiq, "Pioneers of Kaggle Town - Candidate 2" (opening buy 6 / sell 5) | 211 steps: julien gaza (2091) |
| `pilkwang_sep` | pilkwang, "Kaggriculture: Structured Economic Policy" | 347-411 steps: Zhong Lyu (1383), edwinis (1411) |
| `robust_economy` | nihilisticneuralnet, "Kaggriculture: Population-Robust Economy" (Metav4 lineage) | 289 steps: AlexMoura2026 |
| `tetsutani_shape_shop` | tetsutani, "Shape the Shop Work the Pasture" (09-06): the ladder's 13-wheat opening (buy 13 at step 0, sell 9 at step 1) | **exact**: wuy1hao, random_numb, phi; 623 and 106 steps: YuRuiZe, williams (all beat Rowan Glen or Linden Brook; `../evidence/ladder_pool_20260926/open13_match.jsonl`) |
| `abo_v57_open13` | derived: V57 with the ladder's 13-wheat opening | the opening of 16 of Rowan Glen's first 28 losses; the stand-in before `tetsutani_shape_shop` was found, it matches none of those rivals past step 0 |

`tetsutani_demand` also plays exactly like Yusuraume (2055) and BorisV, who beat and tied Alder Ford, and
`leoprovorov_forecast` reproduces King-damon for 216 moves. Alder Ford's losses and 20 candidates from
newer notebooks: `../evidence/alder_ford_20260926/`.

"Exact" means the bundle, playing the opponent's seat of the recorded ladder game while our seat
replays our recorded actions, chooses the recorded action at every one of the 719 steps. The step
counts are the length of the identical prefix for the closest bundle. All matches over the 57 Rowan
Glen and 66 Linden Brook games: `../evidence/ladder_pool_20260926/ladder_match.jsonl`.

## Files

- `<bundle>/agent/`: the agent's files exactly as the notebook embeds them (Apache-2.0 texts and
  attributions are inside them); `<bundle>/SOURCE.json`: notebook, id, pull time, entry file and the
  sha256 of every file; `<bundle>/main.py`: `host_main.py`.
- `host_main.py`: loads the entry file the way kaggle_environments loads a submission (its code runs
  in a fresh namespace, the last callable it defines is the agent, called with (observation,
  configuration) cut to its argument count on structified inputs).
- `build_ladder_pool.py`: pulls each notebook read-only (`kaggle kernels pull -m`) and recovers the
  agent **statically**: the large literals of its code cells are decoded (base64/85, then
  zlib/gzip/lzma/bz2) until Python source or a tar archive appears. No notebook code runs. A changed
  notebook is reported; `--update` accepts it.
- `match_ladder_games.py`: the move-for-move test against a recorded ladder game. Market lists are
  compared slot by slot: these agents put empty `[]` entries in their order lists on purpose (both
  seats' orders clear index by index, so an empty entry delays the orders after it).

## Strength (Colab arena, 2026-09-26, 20 seeds per pair, seats alternating)

Round-robin 1: haideptry_2965 126-14 (+$5.9k a game), haideptry_shepherd 122-18, abo_v57 105-35,
abo_v55 82-58, robust_economy 64-76, abo_v43 41-99, pilkwang_sep 20-120, plain Mohui 0-140 (-$20.7k).
Round-robin 2: tetsutani_demand beats 2965 18-2 (+$1.1k), Shepherd 19-1, V57 19-1, Four-Turn Forecast
18-1, Pioneers 18-2 and the 13-opening V57 20-0; the 13-opening V57 loses to every V57-family agent
by $7-9k. Rowan Glen (our Mohui-based submission) lost 0-96 to V57, V55, Shepherd, 2965 and Robust
Economy (-$18k to -$20k a game, 28 lost ladder seeds x 2 seats + 40 random seeds), 0-20 to
Demand-Preserving, Four-Turn Forecast and Pioneers, 3-17 to the 13-opening V57, and beat Mohui 74-22.

The ladder-engine graphs (`research/procedural_graph/make_ladder_graph.py`) run one of these agents
as their production engine; `tetsutani_demand` is the engine of the 2026-09-26 evolution run.
