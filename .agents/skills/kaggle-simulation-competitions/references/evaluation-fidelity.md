# Evaluation Fidelity, Scenario Coverage, and Replay-Inspired Rules

## 1. Diagnose “Is local evaluation the same game?”

Separate three questions before recommending a better model:
1. **Mechanics fidelity:** same interpreter implementation, resolved configuration, episode horizon, and action semantics?
2. **Scenario coverage:** which seeds, event sequences, opponents, and seating combinations are actually tested?
3. **Execution fidelity:** direct Python callables versus submission sandbox, process lifecycle, dependencies, and CPU/time limits?

### Verification workflow
- Inspect the **deployed evaluator used by the run**, not just a local source copy. Deployment copies can lag behind or ahead of local files.
- Read its environment construction, configuration overrides, seed loops, opponent selection, and scoring aggregation.
- Check the installed package version and hash the actual environment module. Compare against the official version corresponding to the replay dates. Version labels alone do not exclude local modifications.
- Compare resolved configuration values, not only explicitly supplied overrides.
- Run full episodes through the same environment factory and inspect event histories and final states. Cheap no-op agents demonstrate whether shops/events exist; they do not establish competitive performance.
- For stronger fidelity evidence, execute recorded actions through the pinned engine and assert every recorded state transition. Match actions at replay index t with the observation at t−1. Verify the pinned engine's observation/action indexing directly.
- State the verification boundary: matching archived games does not guarantee compatibility with future engine updates or prove sandbox runtime equivalence.
- **Recorded-observation vs live policy input (`step` on P1):** Public replay JSON may omit `step` from seat 1 even though the live framework supplies it to the agent. When re-executing a policy from replay observations, explicitly set `obs['step'] = t` for either seat. Otherwise a backbone using `obs.get('step', 0)` repeatedly resets to turn zero; a wrapper's day/hour fallback does not repair the backbone. Verify the unchanged submitted policy reproduces recorded actions and state transitions before trusting any ablation. Use fresh processes per variant when dependencies contain mutable globals.
- **Diagnostic counterfactuals use fixed recorded opponent action tapes:** They measure total coupled outcomes against a non-reactive opponent, not live competitive win rates.
- **Atomic action validation coupling (e.g. `PLANT` in Kaggriculture):** When modifying market orders or seed reserves, be aware of cross-action validation rules. In `kaggriculture.py`, if the total unit `PLANT` actions for a crop in a turn exceed the actual available seeds in inventory, the engine drops *all* plant actions for that crop (`engine:920–933`). Market order filtering that reduces purchased seeds can thus trigger an unintended complete field-action blackout.
- **RNG-schedule coupling via field actions:** Changing an agent's early field tasks (e.g. via an aggressive `hire_first` reordering) alters tile vacancy at midnight. Because daily weed spawning draws random coordinates on empty tiles *before* town shops are drawn with replacement, a tile change can shift the random draw sequence and produce completely different town shops on subsequent days, even with an identical environment seed.
- **Four-way wrapper ablation:** Compare deployed wrapper, untouched backbone, opening-only wrapper, and overlay-only wrapper against the same recorded opponent actions, retaining an unchanged-policy reproduction control. Report both players' cash and margin, because changed supply changes the opponent's prices too. Track whether town draws remain the same. These diagnose interactions, not reactive-opponent win rates; a rule can repair one loss while breaking existing wins.

### Seed coverage trap
Repeated evaluations against more historical champions can still use only a handful of seeds. More opponent matches or generations do **not** necessarily imply broader event coverage. Reciprocal seats control seat bias; they do not replace new seeds.

A useful evaluation split is:
- **Fixed regression suite:** stable seeds and historical champions, retaining strict per-champion DRQ checks.
- **Development suite:** rotating/disjoint seed batches and diverse opponent families.
- **Held-out suite:** seeds and opponent instances not used for selection or threshold tuning.
- **Targeted stress suite:** concentrated shop demand, scarce commodities, oversupply, storage pressure, and late liquidation.

Do not promise monotonic performance outside the evaluated suite. Repeated optimization on fixed seeds risks selection overfitting even if the agent never observes the seed. Avoid replacing regression checks with rotating seeds; use both.

### Determinism precision
Same engine, configuration, initial state, seed, and action sequence should reproduce the game. “Deterministic” does not mean every match is identical, nor that agents know future random events. A shared random stream may be consumed differently as gameplay changes: do not assume a seed alone fixes shop draws independently of actions. Inspect RNG call order or test it before using seeds as event-sequence identifiers.

## 2. Translate exceptional replays into conditional rules

A replay is one realized interaction, not the competitor's policy or a counterfactual price forecast. Select ordinary controls from the same team, then check action similarity and build comparability; team names may aggregate multiple submissions.

Distinguish:
- **Volume effect:** more goods actually harvested/sold.
- **Realized-price effect:** comparable output sold at better prices.
- **Cost effect:** purchases, feed, labour, and expansion.
- **Context effect:** opponent supply and time-resolved town consumption.

Use executed fills, not requested order quantities. Final tile counts miss crops that produced and then expired. Gross buy-resell turnover is not profit.

### Safe rule structure
For each candidate rule define:
1. **Observable trigger:** existing shops, current inventory, visible opponent production, remaining horizon.
2. **Hypothesis:** expected shortage or exploitable timing advantage, with uncertainty.
3. **Action scope:** sales only, production allocation, or complete sub-plan.
4. **Resource budget:** cash, feed, land, worker time, and storage.
5. **Ownership:** which tiles/tasks/orders override the baseline and for how long.
6. **Exit/fallback:** invalidation conditions, maintenance responsibilities, and liquidation deadline.
7. **Test:** baseline versus baseline+rule on matched seeds, both seats, reactive opponents, and held-out scenarios.

A market-order adjustment can be relatively contained. Planting a different crop into a taped farm plan requires taking ownership of watering, harvest, routing, and inventory handling; inserting a single action then returning to an incompatible tape is unsafe. Prevent duplicate spending, contradictory unit tasks, and double-selling stock.

Never infer a universal rule such as “plant tomatoes on day 16” from one profitable game. Infer a conditional opportunity: enough observable demand, limited competing supply, sufficient growth lead time, and feasible harvest/sales capacity. Do not infer original policy intent or neural/search architecture solely from the action trace.

## 3. Prediction versus planning

Prices are endogenous: both players' sales/purchases and town consumption change market inventory, which determines price through known engine rules. Growing a crop changes potential supply, not market inventory directly.

Use the engine for known mechanics. Learn or estimate uncertain opponent sales, hidden stock, production changes, and responses. Evaluate candidate actions across plausible opponent/event scenarios rather than forecasting a single price path independent of our actions.

For trace-trained models:
- Filter or separate incompatible engine versions.
- Split by episode/time and preferably opponent/submission identity to reduce leakage.
- Use only live-observable information as decision inputs; replay-only opponent private stock may be a training label, not a live feature.
- Keep future shop draws out of inputs available before they unlock.
- Assess calibration and downstream interactive win rate, not price error alone.
- Replaying an opponent's fixed action tape is useful for diagnostics but does not reproduce that opponent's reaction to our changed strategy.
