# KAD-HP-1 inside the graph runtime, without torch (2026-09-29)

A Kaggle agent gets no GPU, and torch cannot be relied on in the agent sandbox (Orchard Tide's first submission timed
out importing it). Agent bundles also carry only `hazel_runtime/`, not `research/action_diffusion`. So any stage that
uses KAD in the arena or on Kaggle must use this numpy port. A torch-based stage finds no model there and does nothing.

## Files

- `hazel_runtime/kad/kad_data.py`: `research/action_diffusion/data.py`, verbatim (sha256 f9767747caa5…). It is the
  observation encoder KAD was trained on.
- `hazel_runtime/kad/kad_numpy.py`: the Policy's forward pass in numpy (8-layer encoder, heads, greedy 10-slot market
  decoder) and `KadAdvisor`.
- `hazel_runtime/kad/weights.npz`: 41.5 MB, float16. It was exported by `export_kad.py` from
  `research/action_diffusion/runs/kaggle_hp1_2026-09-28/best.pt` (step 145k, the RL run's m0). The prompt the model
  plays with is folded in as `token_bias`: day 2026-09-25, rating gap 0, a win, excess margin 0.03644, team Boey.

## Check (cliproxyapi, `export_kad.py --replay kaggriculture-episodes-2026-09-18.zip --games 2 --turns 80`)

The check covers 320 positions, both seats of 2 top-player games.

| model | order probabilities vs torch | greedy orders + amounts equal | jobs equal |
|---|---|---|---|
| float32 port | max difference 1.1e-6 | 320/320 | 320/320 |
| float16 weights | max difference 0.0013 | 320/320 | 318/320 (near-ties) |

- `KadAdvisor.see` builds the same rows as `data.episode_arrays` in 320 of 320 positions.
- Result: `KAD_EXPORT_CHECK PASS`.
- Speed on one cliproxyapi core: 150 ms per `advise()`, and 249 ms for the first call, which loads the weights.
  Expect about 2.5x that on Kaggle, based on the predictor's measured ratio.

## API

```python
from kad.kad_numpy import KadAdvisor     # or hazel_runtime.kad.kad_numpy
adv = KadAdvisor()                       # weights load lazily on the first advise()
adv.see(obs)                             # EVERY turn (cheap): the model reads this turn and the 8 before it
a = adv.advise()                         # one forward pass, greedy like Policy.act at temperature 0
a['orders'], a['amounts']                # 10 slots of (op, item) and amount bins
a['probs']                               # [10, 22] order-class probabilities per slot (given the orders before it)
adv.probability(a, 'BUY_LAND')           # largest P of that order at any slot; ('SELL', 'MILK'), ('BUY_SEED', 'TOMATO'), ...
a['jobs']                                # [present units, 40] job-class probabilities (kad_numpy.JOBS)
adv.plant(a, 'TOMATO')                   # mean P over our units that planting TOMATO is their next job
```

The encoder raises on unknown shops or items (fail closed), so a stage should catch exceptions and turn itself off
for the game.

## Validation accuracies of this checkpoint (step 145k, 09-25 held-out)

These are per slot and per field. Most market slots are empty, so the high numbers are easy ones:

| field | accuracy |
|---|---|
| order | 88.9% |
| job | 85.2% |
| target | 91.8% |
| now | 94.4% |
| order amount | 64.6% |
| **whole market command right** | **40.3%** |
| first active command exactly right | 73.2% |

Alone, the model loses 278 of 280 games to the ladder bots (RL run 2 baseline, T=0.7).

## Engine facts a copilot stage must respect

- Ladder engines put empty `[]` entries in their order lists on purpose: they are one-slot delays, because both
  seats' orders clear index by index. Filling one changes the engine's own sale timing. The oracle guard does this
  only for a predicted rival dump.
- WHEAT is feed and seed stock. The guard never sells it.
- Seeds or products the engine never plants or uses are money lost.
- `WEED` is not a game command. The job operations are in `kad_numpy.JOB_OPS`.
- Route tapes assume where each unit is. A job done in place (WATER, HARVEST) does not move a unit, but a move
  desyncs the tape. HARVEST cuts a one-time crop before its peak.
