# Hazel Weir: source-linked executable policy map

## 1. Scope, identity, and evidence boundaries

This is an audit of the **saved submission bundle**, not a replacement policy, evaluation result, or claim of efficacy. Its purpose is to specify what a behavior-preserving procedural graph must call, in what order, and with what state. No submission, training run, existing source, checkpoint, or skill was changed for this audit.

- Bundle: [`shinka/champions/submissions/hazel_weir`](../../../shinka/champions/submissions/hazel_weir/).
- Local submission identity: [`MANIFEST.json:2–5,57–67`](../../../shinka/champions/submissions/hazel_weir/MANIFEST.json#L57-L67) records Hazel Weir, submission **56246758**. “Latest” is supplied task context; this audit did not query Kaggle or independently establish current remote status.
- **Line-count discrepancy:** the supplied request says 922 lines. The saved `champion.py` actually has **928 `str.splitlines()` lines**, ending with `return final` at line 928. All lines were read, not just the first 922. The read-file tool's `total_lines` metadata reported 927, while its displayed final line and independent Python/AST inspection gave 928. References below use actual one-based Python source lines.
- Champion SHA-256: `9c8d1431dc2401eb7aa63091cb72d84f749273ed8462dfb75fbfe4f1618d568f`; independently computed and matches the local manifest.
- Oracle SHA-256: `94dd8fc5d600d06f2adcf99dd09669c007dd1d09c32a032b916c2115c950e24a`.
- NumPy network SHA-256: `e380ee0850854ea43770aee7067f0d9fe61e538bdf4d4eebe9277c9b0fceecda`.
- Bundled weights SHA-256: `3c94bfc5d5c8df7c586aff22a87ecb1dd4138a6baaa7fbf278fc57c80323f5ed`.
- Scaler SHA-256: `76e4867da2266dc108176fac3adc1953fa7a3a2787076786f13c73c9feea56ac`; config: `550419d6596d38a974be8f15a7b3c93465ce1dcf528adf83d87574ca2d14ea14`; labels: `c2a90375674ed20307f814ccb46733b442bcbffc1f5f94c2ccb3f31b11c98f75`.

Source comments containing historical win rates, purported probability calibration, model dates, performance claims, or “guarantees” are **not** evidence for those claims here. Conditions below describe code execution, including discrepancies with comments.

### Source abbreviations

All links are relative to this document. `C` = [`champion.py`](../../../shinka/champions/submissions/hazel_weir/champion.py), `O` = [`kagg_oracle.py`](../../../shinka/champions/submissions/hazel_weir/kagg_oracle.py), `B` = [`mohui_v66/candidate_v62_composite_bakery_yarn.py`](../../../shinka/champions/submissions/hazel_weir/mohui_v66/candidate_v62_composite_bakery_yarn.py), `V` = [`candidate_v66_meta_closed_loop.py`](../../../shinka/champions/submissions/hazel_weir/mohui_v66/candidate_v66_meta_closed_loop.py). Encoded modules have **virtual decoded line numbers**, explicitly distinguished from physical source lines in §5.

## 2. Graph boundary: exposed stages versus inherited atomic routines

**The bundle does not contain a procedural-graph implementation.** Node IDs below are an executable mapping for a proposed graph adapter. Calling a named node is not a claim that its internals are already graph-exposed.

### Graph-level control flow

```text
IMPORT bundle bootstrap → import backbone and initialize oracle

TURN(obs, configuration)
  G00 player index
  G01 ATOMIC_BACKBONE(obs, configuration)
      exception or non-dict → PASS base
  G02 observed hand count
  G03 ORACLE_OBSERVE(obs, configuration) → forecast or None
  try:
    G04 FARM_STATE(obs, seat, forecast) → shared mutable st
    G05 FARMER_RESCUE(base.farmer, st) → farmer; may set st._farmer_rescue
    G06 HAND_RESCUE(base.hands, st) → hands; consumes farmer rescue claim
    G07 MARKET_PIPELINE(base.market, st) → market
  except Exception:
    evolved = base                 # whole overlay block, not only failing channel
  G08 SANITIZE(evolved, base, observed_n_hands)
  G09 ORACLE_RECORD(final)
  return final
```

Provenance: [`C.agent:902–928`](../../../shinka/champions/submissions/hazel_weir/champion.py#L902-L928). Python evaluates the dictionary values farmer → hands → market in that order. All three receive **the same pre-action observation**, not a simulated post-farmer/post-hands state. Only the explicit rescue claim/wheat accounting passes between them. `farm_state` is not passed to Mohui.

| Boundary | What may be exposed without pretending to decompose inherited code | What remains atomic |
|---|---|---|
| G01 | Backbone call, input/output, selected-route/state telemetry if observed read-only | Every route controller, encoded route action, public signature router, residual controller, weed transaction, inherited seed/shed guard, terminal overlay |
| G03/G09 | Observe-before-decide and record-final-action events | History accounting, model cache, checkpoint inference, engine-dependent feature extraction |
| G04–G06 | Derived features, farmer rescue, hands rescue with exact sequencing | None hidden beyond documented functions |
| G07 | Ordered stages M00–M14 below, sharing the same order list, deferred state, helper closures, and local money | `_add_sell` must remain a shared semantics-preserving primitive if extracted |
| G08 | Final output filter and fallback branches | Do **not** replace it with the bundled adapter's stronger legality sanitizer |

Merely importing Mohui and exposing “choose backbone” is **not** a graph representation of all farmer/hand production decisions. Those decisions include thousands of table-selected atomic actions. Retaining the exact route data and controller closures is the behavior-preserving representation; a new planner that reproduces their intent is a different policy.

## 3. Bootstrap and dependency resolution

[`main.py:19–58`](../../../shinka/champions/submissions/hazel_weir/main.py#L19-L58): locate bundle via `__file__`, otherwise frame filename containing `champion.py`, otherwise reverse `sys.path`, otherwise cwd. Set defaults for `KAGG_MOHUI_DIR`, `KAGG_ORACLE_SRC`, `KAGG_TTM_DIR`, `KAGG_OPP_MODEL_SRC`; **existing environment overrides win**. Force `KAGG_ORACLE_BACKEND=numpy`, `CUDA_VISIBLE_DEVICES=''`; thread variables default to 1, not forcibly overwritten. Prepend bundle directory, import champion as a real module, and leave `kaggle_submission_agent` as the last defined callable. The submission does not enter `mohui_v66/main.py`.

[`C:33–86`](../../../shinka/champions/submissions/hazel_weir/champion.py#L33-L86): Mohui and oracle resolvers try environment override first, then repository/home fallbacks, accepting the first directory with the marker. Mohui import itself is outside the agent's runtime exception handler. Oracle import/model/tracker initialization is caught: errors increment `ORACLE_STATS`, store a 200-character exception, and permit forecast-free play. Eager load defaults on; `KAGG_ORACLE_EAGER=0` delays loading.

**Isolation implication:** preserve the bundled dependencies and weights, not similarly named current repository files. `setdefault`, ordinary imports named `champion`, `features`, `mechanics`, and process-global embedded module registration permit path/module-cache contamination. Hashing the champion alone is insufficient.

## 4. Inherited atomic backbone: executable selection stack

### 4.1 Call stack and patch ordering

The action call descends:

```text
v66 meta
 → v65 q2 Daniel
   → v65 q2 known-yarn repair
     → v65 cygn clone
       → v65 local combined
         → v62 composite
           → v60 composite
             → v59 mirror lead
               → final v58 agent
                 → advance all warmed route controllers + select action
```

It returns outward through each wrapper's overrides. Source: [v66:37–45,444–452](../../../shinka/champions/submissions/hazel_weir/mohui_v66/candidate_v66_meta_closed_loop.py#L37-L45), [Daniel:25–62](../../../shinka/champions/submissions/hazel_weir/mohui_v66/candidate_v65_curve_q2_daniel_backbone.py#L25-L62), [q2:25–120](../../../shinka/champions/submissions/hazel_weir/mohui_v66/candidate_v65_curve_q2_known_yarn_repair.py#L25-L120), [cygn:25–46](../../../shinka/champions/submissions/hazel_weir/mohui_v66/candidate_v65_curve_cygn_clone.py#L25-L46), [local:226–249](../../../shinka/champions/submissions/hazel_weir/mohui_v66/candidate_v65_local_combined.py#L226-L249), [B:4406–4407](../../../shinka/champions/submissions/hazel_weir/mohui_v66/candidate_v62_composite_bakery_yarn.py#L4406-L4407).

Import-time monkey patches matter: original v58 advancement → v62 bakery wrapper → v65 attack wrapper → v66 meta wrapper. Each wrapper first runs its captured predecessor, then may overwrite `actions['recovery']`. Recovery-mode/route hooks test outer v66 first, then attack, then v62, then original. V66 also wraps `_v58_base_action`. The final v58 `agent` definition replaces older `agent` definitions in the same module; reading the first `agent` alone would map the wrong policy.

### 4.2 Route construction, warm-up, default selection

[`B:3599–3632`](../../../shinka/champions/submissions/hazel_weir/mohui_v66/candidate_v62_composite_bakery_yarn.py#L3599-L3632): ten policy entries, in order `base_backbone, base_yarn, base_pet, recovery, smoothie, clone, known_yarn, ice_minimax, bakery_yarn, pizza_recovery`. At each early call construct **one** missing controller, replay its missed observation/configuration prefix, then call every constructed controller on the current observation. Clear warm history once all ten exist. Controllers not selected still mutate their histories. Additional independent clone/known-yarn/backbone controllers in cygn/q2/Daniel are also called every turn before their wrapper can return.

[`B._v56_visible_configuration:2279–2286`](../../../shinka/champions/submissions/hazel_weir/mohui_v66/candidate_v62_composite_bakery_yarn.py#L2279-L2286): strip keys whose lowercase name is `seed` or `randomseed` from dict configurations.

[`B._v58_base_action:3649–3663`](../../../shinka/champions/submissions/hazel_weir/mohui_v66/candidate_v62_composite_bakery_yarn.py#L3649-L3663): missing backbone → safe PASS. Before step 72 use backbone. First shop YARN → yarn. First PET and step ≥144 with second shop YARN or PET → pet. Otherwise backbone. This rule is below later patch overrides.

### 4.3 Original v58 branch predicates and first-match order

[`B._v58_recovery_mode:3696–3730`](../../../shinka/champions/submissions/hazel_weir/mohui_v66/candidate_v62_composite_bakery_yarn.py#L3696-L3730) is sampled at **step ==72 only**, if mode is unset. Let signature be `(first_shop, rounded_own_money, rounded_opp_money, rounded_market_WHEAT, opp_COW, opp_SHEEP, opp_WHEAT_plants, opp_MELON_plants)`. `_v58_number` is `int(round(float(value or 0)))`, conversion failure →0.

| Exact tuple prefix; all have animal/crop tail `(3,2,7,12)` | Latched mode |
|---|---|
| FARMERS_MARKET,141,193,9974 | farmers-recovery |
| BAKERY,145,195,9975 | recovery |
| SMOOTHIE_SHOP,145,193,9975 | recovery |
| BAKERY,142,142,9973 | recovery |
| ICE_CREAM_SHOP,142,191,9974 | recovery |
| ICE_CREAM_SHOP,145,195,9975 | ice-minimax |
| BAKERY,141,193,9974 | bakery-second-shop |
| PIZZA_SHOP,141,193,9974 | pizza-second-shop |
| Else/no shops | base |

[`B.agent:3785–3844`](../../../shinka/champions/submissions/hazel_weir/mohui_v66/candidate_v62_composite_bakery_yarn.py#L3785-L3844), after advancing controllers and taking the base proposal, returns in this order:

1. farmers-recovery → recovery, except step ≥144 and second SMOOTHIE → smoothie.
2. recovery → recovery.
3. ice-minimax → ice_minimax.
4. bakery-second-shop → bakery_yarn if step ≥144 and second YARN, otherwise base.
5. pizza-second-shop → pizza_recovery if step ≥144 and second PET, otherwise base.
6. At step96 with first YARN, latch known_yarn for `(own_money,opp_money,WHEAT_inventory,opp_COW,opp_SHEEP,opp_WHEAT,opp_MELON)` equal to `(214,214,9982,4,2,7,12)` or `(215,125,9979,4,2,7,12)`; thereafter return known_yarn.
7. First YARN from step72: compare complete public farms restricted to money, farmer, hands, unlocked quadrants, tiles; consecutive exact equality increments mirror streak, mismatch resets it. At step360 latch clone if streak ≥240; thereafter return clone.
8. Return base. Any exception in this function → hand-aligned safe PASS.

`_v59_route` mirrors these route choices for future-step lookups ([B:3874–3911](../../../shinka/champions/submissions/hazel_weir/mohui_v66/candidate_v62_composite_bakery_yarn.py#L3874-L3911)); it is also monkey-patched. Do not substitute only current selected action when an algorithm needs future raw route rows.

### 4.4 v62 and v65 public-state branches

- **v62 bakery:** [`B:4346–4396`](../../../shinka/champions/submissions/hazel_weir/mohui_v66/candidate_v62_composite_bakery_yarn.py#L4346-L4396). At step72, first BAKERY, own/opp money 151/39, WHEAT inventory9971, opponent assets exactly COW2/SHEEP2/WHEAT7/MELON12 → armed. Once armed, step≥144 and second YARN → active permanently. Armed overrides recovery with backbone before confirmation, known_yarn after; route hook follows that choice.
- **v65 attack:** [local:37–119](../../../shinka/champions/submissions/hazel_weir/mohui_v66/candidate_v65_local_combined.py#L37-L119). Step1 signature `(opp_money,opp_hands,WHEAT_inventory,WHEAT_price)==(11,5,9942,33)` arms. At exactly72, activate iff first shop is BAKERY/BRUNCH/FARMERS, own money149/hands0/assets COW3,SHEEP2,WHEAT7,MELON12; opponent money724/hands0/assets COW2,SHEEP3,WHEAT10,STRAWBERRY3; WHEAT inventory9975/price30. Active recovery and route become base_yarn.
- **Curve latch:** [local:124–249](../../../shinka/champions/submissions/hazel_weir/mohui_v66/candidate_v65_local_combined.py#L124-L249). Exactly step1, own money2864, rival1390, rival hands0, WHEAT inventory9941 activates. At that call return reordered base market: SELL WHEAT1, last matching COW/SHEEP buy orders, all HIREs, last matching WHEAT/MELON seed orders, cap10. Thereafter at step≥2 choose the 720-row Curve route directly, align hands, repair weed collisions, append BUY WHEAT1 at step2, apply inherited hour23 shed guard, and at718 replace market with observed-shed liquidation. If inactive return the already-computed base action.
- **Curve weed repair:** a pending per-actor FIFO takes priority; execute its head, append this turn's non-PASS scheduled action, drop empty queue. Without a queue, intended BUILD_PASTURE/BUILD_COOP/PLANT/PLACE on a weed becomes DIG with the intended op queued. Unlike the sparse planner's transaction, no fixed eight-turn expiry.
- **Cygn:** [cygn:34–46](../../../shinka/champions/submissions/hazel_weir/mohui_v66/candidate_v65_curve_cygn_clone.py#L34-L46). Curve active + step≥144 + first two shops exactly `(YARN_STORE,PIZZA_SHOP)` → independent clone controller action, seed trim, shed guard; else selected predecessor. No additional explicit Curve liquidation here.
- **q2 known yarn:** [q2:72–120](../../../shinka/champions/submissions/hazel_weir/mohui_v66/candidate_v65_curve_q2_known_yarn_repair.py#L72-L120). Curve step1 changes first existing SELL WHEAT quantity to2; step2 changes first existing BUY WHEAT to2; these are early returns, not insertions if missing. At72 with Curve active, full public signature must equal `_CURVE_Q2_REPAIR_STEP72`: FARMERS only; own205/opp220; both hands0 and COW3/MELON12/SHEEP2/WHEAT7; exact full market inventory/prices in the linked constant. Latch repair; choose independent known-yarn action, seed trim, shed guard, observed-shed liquidation at718.
- **Daniel:** [Daniel:29–62](../../../shinka/champions/submissions/hazel_weir/mohui_v66/candidate_v65_curve_q2_daniel_backbone.py#L29-L62). Same full step72 signature as q2 but shop tuple `(SMOOTHIE_SHOP,)`, Curve active → latch Daniel. Choose independent backbone controller, seed trim/shed guard, observed-shed liquidation at718. Otherwise retain predecessor.

**Shadowing:** champion opening M01 replaces step0 market with BUY WHEAT35 and prepends SELL WHEAT30 at step1, regardless of these underlying scalp quantities. It retains existing step1 SELLs, including a Curve SELL2 if that branch is reached. Later exact signatures are conditional code, not proof that Hazel's modified trajectory reaches them.

### 4.5 v66 full-signature routing

[`V._public_signature:79–95`](../../../shinka/champions/submissions/hazel_weir/mohui_v66/candidate_v66_meta_closed_loop.py#L79-L95) includes ordered complete shop tuple, rounded own money, own hand count, sorted complete own asset counts, corresponding opponent values, **all sorted market inventory and price entries**. It does not merely match a named opponent or first shop. Constants are literal full tuples; the table below identifies exact source ranges rather than weakening them to partial predicates.

[`V._advance:300–362`](../../../shinka/champions/submissions/hazel_weir/mohui_v66/candidate_v66_meta_closed_loop.py#L300-L362): after predecessor advancement, reset/update meta state. At step72 and **not Curve active**, use the following `if/elif` order:

| Exact constant/source | Meta transition |
|---|---|
| [`_TWOMOON_STEP72:131–145`](../../../shinka/champions/submissions/hazel_weir/mohui_v66/candidate_v66_meta_closed_loop.py#L131-L145) | historical_twomoon → yarn |
| [`_YAT_STEP72:147–161`](../../../shinka/champions/submissions/hazel_weir/mohui_v66/candidate_v66_meta_closed_loop.py#L147-L161) | historical_yat → recovery |
| [`_VICTOR_STEP72:99–113`](../../../shinka/champions/submissions/hazel_weir/mohui_v66/candidate_v66_meta_closed_loop.py#L99-L113) | arm victor, no immediate route |
| [`_BAKERY_STEP72:165–179`](../../../shinka/champions/submissions/hazel_weir/mohui_v66/candidate_v66_meta_closed_loop.py#L165-L179) | bakery → bakery_yarn |
| [`_FARMERS_STEP72:181–195`](../../../shinka/champions/submissions/hazel_weir/mohui_v66/candidate_v66_meta_closed_loop.py#L181-L195) | farmers → recovery |
| [`_NEIBYR_STEP72:197–211`](../../../shinka/champions/submissions/hazel_weir/mohui_v66/candidate_v66_meta_closed_loop.py#L197-L211) | neibyr → recovery |
| [`_C0NRAD_STEP72:213–227`](../../../shinka/champions/submissions/hazel_weir/mohui_v66/candidate_v66_meta_closed_loop.py#L213-L227) | c0nrad → ice_minimax |
| [`_JOHNBLAKE_STEP72:229–243`](../../../shinka/champions/submissions/hazel_weir/mohui_v66/candidate_v66_meta_closed_loop.py#L229-L243) | johnblake → known_yarn |
| [`_STEPHEN_STEP72:245–259`](../../../shinka/champions/submissions/hazel_weir/mohui_v66/candidate_v66_meta_closed_loop.py#L245-L259) | stephen → known_yarn |
| [`_COKE_STEP72:261–275`](../../../shinka/champions/submissions/hazel_weir/mohui_v66/candidate_v66_meta_closed_loop.py#L261-L275) | arm coke, no immediate route |

At exactly144: armed victor requires [`_VICTOR_STEP144:115–129`](../../../shinka/champions/submissions/hazel_weir/mohui_v66/candidate_v66_meta_closed_loop.py#L115-L129) to select clone, otherwise log rejection; clear arm. Bakery keeps bakery_yarn only if second shop YARN, else recovery. Farmers selects yarn if second YARN, else recovery. Armed coke requires [`_COKE_STEP144:277–291`](../../../shinka/champions/submissions/hazel_weir/mohui_v66/candidate_v66_meta_closed_loop.py#L277-L291) to select known_yarn, otherwise log rejection; clear arm. Meta route overrides action alias, recovery-mode hook, base-action hook, and future-route hook ([V:359–388](../../../shinka/champions/submissions/hazel_weir/mohui_v66/candidate_v66_meta_closed_loop.py#L359-L388)). Branch names are labels in source, not runtime identity inputs.

### 4.6 Inherited post-selection overlays

- **v59 mirror sale lead** ([B:3914–4025](../../../shinka/champions/submissions/hazel_weir/mohui_v66/candidate_v62_composite_bakery_yarn.py#L3914-L4025)): independently latch exact-public-farm clone at step≥96 after ≥90 consecutive equal observations. Operate only step≥120 and eligible shop family. First PIZZA excluded; PET requires second not FARMERS/PIZZA; SMOOTHIE second not ICE/PIZZA/SMOOTHIE; ICE second not BRUNCH/PET/SMOOTHIE; YARN second not ICE; remaining families allowed. First repay stored sell quantity from current market. If next step≥719 or divisible by72, stop. Otherwise find first next-turn raw-route SELL of MELON/STRAWBERRY/MILK/WOOL/EGG; quantity=min(8,scheduled,observed stock−already selling). Merge current sell or prepend if slot; record one-item debt and break. Full queue without merge breaks. Debt tracks requested preemption, not observed execution.
- **v60 opening route mutation** ([B:4040–4086](../../../shinka/champions/submissions/hazel_weir/mohui_v66/candidate_v62_composite_bakery_yarn.py#L4040-L4086)): for each of ten named routes, if exactly one BUY_PRODUCT WHEAT5 occurs at raw step1, move it to raw step0. Happens at import before policy construction. Champion later supersedes market opening.
- **Strawberry seed trim** ([B:4113–4142](../../../shinka/champions/submissions/hazel_weir/mohui_v66/candidate_v62_composite_bakery_yarn.py#L4113-L4142)): only when current market contains STRAWBERRY seed buys. Remaining demand=current action's strawberry PLANT count + all later selected raw-route PLANT counts. Cap each buy to remaining−private seeds, update projected seed holdings, omit zero buys. Current future-route hook may not describe an independent outer repair controller's proposal; preserve this distinction.
- **Hour23 shed guard** ([B:4171–4322](../../../shinka/champions/submissions/hazel_weir/mohui_v66/candidate_v62_composite_bakery_yarn.py#L4171-L4322)): only hour23. Project actor DROP/PICKUP/PLACE/HARVEST/COLLECT_FERTILIZER/FEED/FERTILIZE; DROP whose entire load exceeds room is changed to PASS. Project requested market sells/buys with capacity but without cash pricing. Excess=max(0,projected shed+projected bags−capacity). Sell finished goods in descending `(current price,item)` order; append WHEAT last, plus FERTILIZER before WHEAT on day≥29. Before day29, WHEAT availability ≤total shed+bag wheat−animal count. Merge sells or append below10; stop after excess cleared. This is distinct from champion's cheaper-first headroom stage.
- **v66 terminal actors** ([V:391–452](../../../shinka/champions/submissions/hazel_weir/mohui_v66/candidate_v66_meta_closed_loop.py#L391-L452)): steps713–716, actor with positive load of sellable products moves toward central shed access **only when Manhattan distance equals `718-step`**. At717/718 invoke decoded `monetizable_terminal_units`: loaded at access →DROP; at717 loaded distance1 →move; at717 empty at access standing on positive-yield tile →HARVEST. At718 replace market with decoded collision-ranked projected-shed liquidation. Champion market can subsequently overwrite this projected-deposit sale list.

## 5. Encoded inherited logic: decoded, not treated as an unexplained black box

### 5.1 Physical and virtual provenance

The audit recursively **AST-parsed and decoded** the Base85/zlib source and JSON payloads in memory, without importing the bundle, executing its module initialization, writing decoded files, or running a game.

Physical root: [`B._V51_BASE_SOURCE:15–1623`](../../../shinka/champions/submissions/hazel_weir/mohui_v66/candidate_v62_composite_bakery_yarn.py#L15-L1623). Decode `zlib.decompress(base64.b85decode(concatenated_literal)).decode('utf-8')` to virtual v50 source, 1,613 lines, SHA-256 `3b36352aa26f4fc6c58f11a543cda4734b29156516b6aae8cfbfd00eea512427`.

Inside that source:

- `_V50_BASE_SOURCE`, virtual lines10–1375, decodes virtual v49 source, 1,338 lines, SHA-256 `1f33ff8d7d2bcfb71813c0f02e78d84e1136688e11fac2abb0ab0ab7766c5310`.
- Virtual v49 `_V49_MODULES:35–524` is decoded JSON mapping module name → source string. Virtual v50 `_V50_MODULES:1408–1436` supplies `v50.hybrid` and `v50.suffix_router` similarly.
- Virtual line references below mean **line numbers within the decoded named source string**, not an invented physical file link. All resolve through the physical B payload above. The same prefix in `candidate_v60_curve_counter.py` through physical line4323 is text-identical to B; decoded shared payloads are identical (the later version-label difference is not a code-payload difference).

Active decoded modules and source provenance:

| Module | Main executed functions/virtual lines | Decoded SHA-256 |
|---|---|---|
| `v50.hybrid` | `build_route_value_hybrid`12–41, `safe_action`44–52 | `4b1d1ea9711d57775c36e12fc662b410b3979fc9575966a1d4c82762d8e638bf` |
| `scripts.v21_route_memory_search` | `replay_policy`4–8 | `a6fec2ac0aed7c763918921030fe038949ce11e380e14ee06e963c1e15de5c72` |
| `v23.planner` | `PlannerConfig`15–17, `build_sparse_planner`20–64 | `299d7289581f326380467f316e0594ba94e9350812bae9b8ed935a82c3e8ef07` |
| `scripts.v22_weed_repair` | `wrap_weed_repair`26–122 | `5cecabf289279305cc78c22bb7daed956e4eae2922b080b725326df678283ec5` |
| `v23.policy_library` | sell scoring38–51, `reorder_sell_slots`54–79 | `d49e36a3a51c190e571c05f2359c85bf1bb38f8af5fe12ba4397cb71419e9f7e` |
| `scripts.v22_market_impact` | price curve24–78, `impact_score`90–113 | `41a99cc81eaadcbb6c67885974c8133af3872f730c2b7389bf99602f52da3ed8` |
| `v23.state_encoder` | regime51–53, demand86–110, `encode_state`155–208 | `0f20488c2d154fa9d50bbd1db8b07433bd1e336b95535c11a7759f9eeb8314e0` |
| `v23.simulator` | `market_price`51–65, `execute_sell`68–78 | `c4cdeba82f3897e45c623b1a61acb1e180920196e24712b2ce0650538e2a3094` |
| `v24.market_maker` | `_project_private_after_actor`220–329, `_animal_count`188–193; expert constructed but disabled | `7c1a2f267ee4813bd7bcc5423bb9a49e80984666ff57987051092645b16ea885` |
| `v49.residual_controller` | config35–113, `ResidualValueController`309–816, builder819–834 | `f5a9c9f973e3c844cc0963954379bf1860fbfa4a18308f0e9a6f36d4f1d04d64` |
| `scripts.v19_terminal` and identical `v19_terminal` | projection68–128, distance177–187, terminal190–239, movement242–261, units264–308 | `5de2444e49b554e3957ba91f653604d0ec4a4e4bc3d1ceb4aca528ec74efd8a2` |

Other embedded modules were enumerated: `v44.gold_floor`755 lines, `v48.fast_route_router`173, `v50.suffix_router`74; older proposal routers/builders and alternative policy objects are initialized by nested source loading, but the final v58→v50 hybrid path does not call their old top-level agents. Likewise `plan_sell_quantities`, `build_dual_regime_planner`, `RouteLibrary.policy`, and unused terminal overlay modes are not additional live Hazel decisions. Do not promote all imported historical experiments to graph stages.

### 5.2 Atomic route proposals: exact data, not a generic farming policy

`v50.hybrid:22–31` deep-copies a route, requires exactly719 rows, wraps sparse planner, then residual controller with one `default` route and no route selector. `replay_policy:4–8` returns a deep copy of `actions[clamp(obs.step,0,len−1)]`, with PASS fallback for a falsey row. Thus farmer movement, planting, building, stocking, feeding, watering, care, harvesting, fertilizer collection/application, carrying, drops, hires, buys, and land expansion are largely **step-indexed full action data**, amended by the explicitly mapped observation-conditioned routines. They are not inferred anew from `farm_state`.

| Atomic route data in B | Physical encoded span | Decoded rows |
|---|---|---:|
| `_V56_BACKBONE_ROUTE = _V55_LIVE_ROUTE` | [1968–2122](../../../shinka/champions/submissions/hazel_weir/mohui_v66/candidate_v62_composite_bakery_yarn.py#L1968-L2122) | 719 |
| `_V56_YARN_ROUTE` | [2123–2273](../../../shinka/champions/submissions/hazel_weir/mohui_v66/candidate_v62_composite_bakery_yarn.py#L2123-L2273) | 719 |
| `_V57_PET_ROUTE` | [2338–2491](../../../shinka/champions/submissions/hazel_weir/mohui_v66/candidate_v62_composite_bakery_yarn.py#L2338-L2491) | 719 |
| `_V58_RECOVERY_ROUTE` | [2543–2687](../../../shinka/champions/submissions/hazel_weir/mohui_v66/candidate_v62_composite_bakery_yarn.py#L2543-L2687) | 719 |
| `_V58_SMOOTHIE_ROUTE` | [2688–2838](../../../shinka/champions/submissions/hazel_weir/mohui_v66/candidate_v62_composite_bakery_yarn.py#L2688-L2838) | 719 |
| `_V58_CLONE_ROUTE` | [2839–2997](../../../shinka/champions/submissions/hazel_weir/mohui_v66/candidate_v62_composite_bakery_yarn.py#L2839-L2997) | 719 |
| `_V58_KNOWN_YARN_ROUTE` | [2998–3144](../../../shinka/champions/submissions/hazel_weir/mohui_v66/candidate_v62_composite_bakery_yarn.py#L2998-L3144) | 719 |
| `_V58_ICE_MINIMAX_ROUTE` | [3145–3293](../../../shinka/champions/submissions/hazel_weir/mohui_v66/candidate_v62_composite_bakery_yarn.py#L3145-L3293) | 719 |
| `_V58_BAKERY_YARN_ROUTE` | [3294–3447](../../../shinka/champions/submissions/hazel_weir/mohui_v66/candidate_v62_composite_bakery_yarn.py#L3294-L3447) | 719 |
| `_V58_PIZZA_RECOVERY_ROUTE` | [3448–3598](../../../shinka/champions/submissions/hazel_weir/mohui_v66/candidate_v62_composite_bakery_yarn.py#L3448-L3598) | 719 |
| Curve direct route, other physical module | [curve_counter:4327–4329](../../../shinka/champions/submissions/hazel_weir/mohui_v66/candidate_v60_curve_counter.py#L4327-L4329) | 720 |

PET and BAKERY_YARN raw decoded JSON are identical (`042988978d40dd8048d0aa548e8300dfb3a814398a939291d299a3a5f738e89d` SHA-256 of compact JSON); their controller instances are still distinct stateful objects. Backbone compact-JSON hash before the v60 opening mutation is `0fce5e3f1c02196aeb6496b03f13e6489413b89e3bea91ad8abfcce49a23db23`; Curve is `c4e280a8bb68ac145d4b408d0078bd9eff99f1cc1ac5353c226ec80c65f43988`.

Decoded example, **before** v60 mutation and champion overlays: ten hybrid routes begin with PASS/empty market at0, farmer NORTH at1, BUY WHEAT5, BUY_SEED WHEAT7, BUY_SEED MELON12, five HIREs, BUY COW2, BUY SHEEP2. At2 farmer WEST and five hand moves/pickups; most relevant routes also SELL WHEAT1. Curve raw data instead begins BUY WHEAT53, then SELL WHEAT48 with reordered investments; **those raw Curve steps0/1 are not the local wrapper's active action path**. This is why reconstructing a route from its opening alone is unsafe.

Older `_V51_ROUTE`, `_V54_ROUTE`, and nested v50 GIN route also decode to719 rows but are not the selected final v58 route entries.

### 5.3 Sparse farming correction and sell-slot ranking

Decoded provenance from §5.1:

- `v23.planner:29–60`: wrap route replay with weed repair (replay_steps8, stop_on_match default false); choose demand alpha0.25 for regime `townCenterSellInterval >=24`, otherwise0; reorder existing sell slots. Default absent center interval here is12.
- `scripts.v22_weed_repair:46–117`: align current observed hand count. Per seat reset at step0, missing active-state initialization, or step decreasing (not equal). For active actor transaction: actor disappeared →expire; age1 retry intended op; ages2…9 replay that actor's previous raw-route action; otherwise expire. Then detect new BUILD_PASTURE or PLANT on WEED →DIG and store transaction. Does not repair BUILD_COOP or PLACE here. Other actors and market unchanged.
- `v23.policy_library:38–79`: only existing SELL positions are reordered, keeping non-SELL slot positions. Base score=`requested_qty * max(0,current_quote−price(inventory+requested_qty))`; with positive demand alpha multiply by `1+alpha*min(1,recovery_days/10)`, where recovery_days=`max(0,inventory+qty−10000)/max(.25,demand_per_day)`. Rank descending, tie by earlier original position. Uses requested, not executable, quantity.
- Decoded market price curves use per-product equilibrium10000, floor1, round integer price. `(base,scale,below_shape,target,above_shape,target)` = WHEAT `(25,400,sqrt,.8,log,.2)`, CARROT `(35,450,hinge,1,sqrt,.7)`, TOMATO `(60,200,hinge,.4,sqrt,.6)`, STRAWBERRY `(120,100,sqrt,.7,linear,1.6)`, MELON `(250,300,log,.2,sq,3.6)`, EGG `(50,332,hinge,.4,log,.2)`, MILK `(160,122,sqrt,.6,linear,1.6)`, WOOL `(200,105,log,.2,sq,3.2)`, FERTILIZER `(100,200,linear,.4,linear,.4)`. Amplitude normalizes shape at scale; hinge uses `u+8*max(0,u−1)^2`, `u=deviation/scale`. `execute_sell` adds each quote and increments inventory only above floor1.

### 5.4 Residual controller: separate from the TTM oracle

Effective config is decoded `ResidualConfig` defaults overridden by [`B._V54_CONFIG:1963`](../../../shinka/champions/submissions/hazel_weir/mohui_v66/candidate_v62_composite_bakery_yarn.py#L1963): active preemption `[72,700)`, max orders10, flow EMA alpha.35, inferred supply clamp0…80, near distance8, near streak12, front money deficit4500, supply threshold3, preempt horizon2, total batch8, minimum delta8, minimum price/base.45, exposure scale.08, preempt deficit4000, feed reserve1.5 days. Front and preempt items are seven products excluding WHEAT/FERTILIZER. Adaptive near-schedule prediction enabled: weight starts0, confidence positive alpha.35/negative alpha.55, no-event decay.99, at least2 evidence turns, activation confidence≥.30, ceiling1. **Deferral and market maker are disabled.** Terminal rule collision.

`ResidualValueController.apply`, virtual743–816, exact order:

1. Per-seat reset on empty state, step0, or decreasing step. Update inferred opponent flow only for adjacent observed steps: inventory delta minus controller's own preceding projected market net plus preceding-town draw. Clamp and EMA. Compare own/opponent positive supply overlap across preempt items: agreement=`2*overlap/max(1,own_total+opp_total)`; update adaptive confidence/event count, or decay if no flow event.
2. Compute `clone_distance`: hand-count difference +3×quadrant-count difference + L1 crop/animal/structure/weed counts (`terminal:153–187`). Within8 increments streak, else reset; ≥12 latches near permanently for episode.
3. Get sparse-planner proposal and safe-copy/align it; select controller's default raw route.
4. Repay preempted sells due now (`_repay:476–503`): subtract from matching sells; carry unpaid quantity to next turn if step<718.
5. `_release_pending:505–540`; `_defer:542–599` is disabled under effective config. If enabled, only steps240…679, <3 pending items, sell-only market, cash≥3500; defer up to12 eligible CARROT/STRAWBERRY/MELON/MILK/WOOL units when this tick's demand minus ceil(EMA supply) yields revenue gain≥20; release next tick or age≥2, value-priority within free slots. Release projection hardcodes capacity100.
6. `_preempt:601–741`: enabled +72≤step<700 +free slot + (near or deficit≥4000 or any EMA supply≥3). Project actor transfers, subtract current sells; compute wheat feed protection though wheat is not in preempt set. For each eligible product gather raw-route scheduled sales over next2 steps, subtract already owed quantities, cap by observed/projected availability and8. Require price/base≥.45. Forecast supply=`2*(EMA+.08*public_exposure)` plus near adaptive confidence×scheduled quantity only when evidence gates pass. Subtract known demand over horizon, compare per-unit-curve revenue now versus future; delta≥8 qualifies. Sort candidates descending `(delta,now,item,scheduled,allocations)`, prepend additional sells under total batch8 and order cap, record due quantities against future route steps.
7. Front existing eligible sells if near, money deficit≥4500, or eligible-product EMA peak≥3. Demand-adjusted reorder first; move eligible sells before all other orders. Otherwise reorder existing SELL slots with alpha.25. This fronting has no active_start/stop gate.
8. Market maker not applied when disabled. At718 replace market with collision-ranked terminal liquidation. Safe-copy, align, truncate10. Save own estimated executable market net, shops, step.

**Accounting caveat:** these controller histories record their own hypothetical selected proposals, including proposals from nonselected route controllers, before outer wrappers/champion modify them. Only the TTM tracker records the final Hazel action. `_executable_market_net:150–185` clips sells to projected stock and buys to room but does not model cash/price execution or floor1 non-supply. Thus the two “opponent flow” signals are neither identical nor interchangeable.

Shared `_project_private_after_actor` (decoded v24:220–329) projects DROP with discarded overflow, PICKUP, animal PLACE, shed PLACE, and FEED consuming **actor bag wheat**, only after locked-tile checks where appropriate. Does not credit harvested wheat in feed reserve. The disabled MarketMakerExpert remains instantiated but has no action effect under this config; its broad configuration/default values are not Hazel trade rules.

### 5.5 Decoded terminal primitives

Decoded `scripts.v19_terminal`:

- Shed access is four central cells; `projected_shed:68–128` adds legal current DROP/PLACE deposits with capacity100 and skips placing an animal into an empty matching structure. It does not subtract PICKUP.
- `terminal_market:190–239` sells every positive projected sellable holding, cap10, replace=True on live call paths. Collision rank=`(1+visible_opponent_exposure)*product_glut_weight*max(1,quote)*log1p(qty)`; glut weights STRAWBERRY2/MELON3.6/MILK2/WOOL3.2/EGG1.5/TOMATO1.3/others1; tie fixed product order. No champion-style observed-price≥1 exclusion.
- `_move_toward:242–261` chooses nearest access cell, ties by y then x; try horizontal movement before vertical, require in-bounds and not literal LOCKED; otherwise PASS. Not a global path planner.
- Unit terminal rules are described in §4.6. They count all bag contents as load at717/718; v66 early return-home overlay counts only sellable products.

## 6. Champion derived state and actor channels

### 6.1 G04 farm_state

[`C.farm_state:209–271`](../../../shinka/champions/submissions/hazel_weir/champion.py#L209-L271): use obs day/hour (default0), own farm money/positions/hands, private shed and bags, market prices and visible shops. Enumerate `tiles[row][column]`:

- None →empty; any string →locked counter.
- PLANT →plant list; thirsty iff consecutive_unwatered≥1 and not watered_today.
- PASTURE with truthy animal →animal list; hungry iff `(unfed≥1 OR (hour≥18 AND unfed≥0)) AND not fed_today`; PASTURE without animal →empty.
- WEED →weed list. **COOP is not included as an animal/empty/hungry case.**
- Shed used=sum numeric values; carried=sum integer-converted numeric bag values. Capacity is hardcoded100, room=max(0,100−used). Cash floor700 through day8,250 after, even though its market guard is disabled. Oracle is retained only if dict.

Reference data [`C:127–205`](../../../shinka/champions/submissions/hazel_weir/champion.py#L127-L205): base prices as §5.3. Champion shop membership SMOOTHIE strawberry/milk; ICE strawberry/milk/wheat; PIZZA tomato/milk/wheat; YARN wool; BAKERY egg/wheat; BRUNCH egg/wheat/strawberry; PET carrot; FARMERS carrot/tomato/strawberry/**melon**. Multiple shops are unioned for demand membership, not weighted by count. This FARMERS recipe differs from embedded v23's wheat/carrot/tomato/strawberry.

### 6.2 G05 farmer rescue

[`C.evolve_farmer_action:753–785`](../../../shinka/champions/submissions/hazel_weir/champion.py#L753-L785): copy base or PASS. Only base first verb exactly PASS, truthy tiles/position. Interpret position elements as `(r,c)` and index `tiles[r][c]`; bounds-check. In order for that cell: WEED →DIG; PLANT thirsty by≥1/unwatered →WATER; PASTURE animal hungry by same rule as state and **shed WHEAT>0** →FEED. Record `(r,c,verb)` in shared `st['_farmer_rescue']` on override. Local exceptions ignored; return copied base. No movement, harvest, fertilization, planting, investment, or nonidle rewrite. Not gated by `_ENABLE_HAND_RESCUE`.

### 6.3 G06 hand rescue

[`C.evolve_hand_actions:789–852`](../../../shinka/champions/submissions/hazel_weir/champion.py#L789-L852): convert each list/tuple base hand action to list, otherwise PASS; pad to current hand count (final truncation happens at sanitizer). With rescue enabled:

1. Sort thirsty by descending missed-water days then base product price; hungry similarly by missed-feed days then animal product price. These orderings can affect who consumes the limited shed-based feed budget.
2. Reserve farmer's claimed coordinate; wheat_avail=shed WHEAT minus1 if farmer FEED rescue.
3. Build coordinate→hand-index mapping for currently PASS (or empty-list) hand orders. Duplicate coordinates retain the **last** idle hand at that coordinate.
4. Visit weeds in scan order →DIG if an idle hand is on the unclaimed cell; mark claimed.
5. Visit hungry →FEED on unclaimed idle location while wheat_avail>0; decrement once per rescue.
6. Visit thirsty →WATER on unclaimed idle location.

No travel to rescue targets; no action-space search; no use of worker bag contents; no accounting for other nonidle hands already consuming feed or acting on the same cell. Claimed set covers only the farmer rescue and added hand rescues, not all inherited tasks. Positions are used directly as `(row,column)` map keys, whereas inherited routines use `(x,y)` with `tiles[y][x]`.

## 7. Champion market graph: exact ordered stages

### Shared M00 helper semantics

[`C.evolve_market_orders:284–349`](../../../shinka/champions/submissions/hazel_weir/champion.py#L284-L349): `step=obs.step` if not None, else `24*st.day+st.hour`; `mkt=list(base_orders or [])` (outer-list copy); local `money=st.money`. Per-turn protected-item set and per-seat persistent `_deferred_sells` map.

- `_already_selling` sums requested quantities of all list/tuple SELLs for item, regardless of executable holdings or order timing.
- `_add_sell(q<=0)` →False. `protect=True` puts item in protected set **before** success is known. Merge with the **first** matching SELL by replacing that entry and increasing quantity, without moving its slot; otherwise append when fewer than10 orders.
- At full queue, consider only unprotected SELLs; non-SELLs cannot be evicted. Value=`qty*observed price`, falling back to base price (unknown product50) when quote≤0. Choose first minimum-value sell. Replace only if candidate value **strictly greater than1.1×minimum**.
- On eviction, record `[max(evicted_qty,previous_deferred_qty),current_step]`, not sum; install replacement in the evicted slot. Failure leaves no new order. Helper effects (including protected membership and deferred records) persist through later stages of this call.

### M01–M14 ordered table

These are sequential transforms, **not mutually exclusive alternatives**, except explicit returns. Later merges/evictions can change earlier orders.

| ID | Condition and exact operation | Source |
|---|---|---|
| M01 opening | step0: return only BUY_PRODUCT WHEAT35. step1: prepend SELL WHEAT30, remove all inherited BUY_PRODUCT WHEAT orders, keep other orders including other WHEAT SELLs, return first10. No subsequent market stage runs. | [C351–360](../../../shinka/champions/submissions/hazel_weir/champion.py#L351-L360) |
| M02 deferred retry | Intended `step<=1: deferred.clear()` occurs **after M01 returns**, hence unreachable for ordinary0/1. Iterate snapshot of deferred keys; q=min(deferred qty,held−already requested). Delete if q≤0 or age>24 (not≥24). Try helper; on success pop item. Entries evicted during this stage join same dictionary but are not newly iterated unless their key was already in snapshot. | [C362–373](../../../shinka/champions/submissions/hazel_weir/champion.py#L362-L373) |
| M03 liquidity | `(hour>=20 OR 220<=step<=245) AND day>1`. For each BUY_PRODUCT in existing order, reserve=max(250,120×hands), except WHEAT when held wheat<n_animals →max(20,25×hands). Cost=current quote or base/default50 ×requested qty. Reject if money−cost<reserve; retained buys decrement local money sequentially. Sells do not credit money; other investments do not debit it. | [C375–390](../../../shinka/champions/submissions/hazel_weir/champion.py#L375-L390) |
| M04 end investment stop | day≥28 remove BUY_SEED/BUY_ANIMAL/BUY_LAND/HIRE **only orders with length≥2**. Add BUY_PRODUCT to canceled verbs if day≥29 or held WHEAT≥n_animals. Thus standard one-element HIRE and BUY_LAND survive this filter. | [C392–401](../../../shinka/champions/submissions/hazel_weir/champion.py#L392-L401) |
| M05 optional cash floor | Disabled. If enabled, sequentially reject length≥3 BUY_PRODUCT/BUY_SEED/BUY_LAND whose estimated cost leaves <700 on day≤8 or<250 after. Uses price/base quote, not actual seed/land costs; starts from money remaining after M03. | [C403–415](../../../shinka/champions/submissions/hazel_weir/champion.py#L403-L415) |
| M06 optional animal top-up | Disabled. If enabled: n_animals<17, queue<10, local money≥1500, day≤26; subtract already requested BUY_ANIMAL quantities, append GOOSE min(want,2), at least1. | [C417–426](../../../shinka/champions/submissions/hazel_weir/champion.py#L417-L426) |
| M07 reserve/pressure | Feed reserve R=0 if step≥696; else max(2,n_animals+6) if step≥672; else max(4,2×n_animals). At shed_used≥80, enabled pressure iterates positive holdings by descending `(qty,item)`. Floor ratio .05 at≥95, .15 at≥90, .35 otherwise; batch10 in all tiers. **Skip only if price<ratio×base AND price<1**. Thus any quote≥1 passes, even below ratio. Add min(held−already−R_for_wheat,10). | [C428–453](../../../shinka/champions/submissions/hazel_weir/champion.py#L428-L453) |
| M08 predrop headroom | hour≥20 and step<712; need=shed_used+carried−98. If positive, sort positive holdings by `(price,−qty)`, stable for exact ties. Skip price<1. Add min(held−already−R_for_wheat,need), protected=True; decrement need only on helper success. **Initial need does not subtract already planned sells.** | [C455–473](../../../shinka/champions/submissions/hazel_weir/champion.py#L455-L473) |
| M09 oracle short front-run | Enabled and nonempty score_4 dict; no explicit256 gate. Candidate products except MELON with score≥.30, descending score. R for wheat;1 reserve for FERTILIZER. Need held−reserve≥1, price≥.60×base, except nondemanded WOOL≥.70. Available=held−reserve−already. Batch=all available if score24≥.45 OR units24≥2 OR score4≥.60; else8 if score4≥.45; else6. Helper add positive min(available,batch). | [C475–514](../../../shinka/champions/submissions/hazel_weir/champion.py#L475-L514) |
| M10 shop cadence/rank/sales | 144≤step<712 and cadence true. Cadence=every4 steps OR shed≥60 and even step OR step≥672 and even step OR oracle bypass. Full thresholds and batch rules below. | [C516–618](../../../shinka/champions/submissions/hazel_weir/champion.py#L516-L618) |
| M11 fertilizer surplus | Inside same M10 cadence/window: held FERTILIZER≥3 and price≥45; add min(held−1−already,5)>0. | [C620–629](../../../shinka/champions/submissions/hazel_weir/champion.py#L620-L629) |
| M12 early liquidation | Active by default680≤step<712; also640≤step<680 if score24≥.30 for any WOOL/MILK/TOMATO/STRAWBERRY/EGG. Exact subphases below. | [C631–700](../../../shinka/champions/submissions/hazel_weir/champion.py#L631-L700) |
| M13 final market replacement | step≥712, construct separate liquidation queue from observed shed sorted descending qty×quote. Steps712…717: first sell up to10 per positive-price product, then second pass sell up to10 more from each remainder until10 slots; duplicates are intentional. Steps≥718: one sell of all held per item, max10. Both require price≥1. If queue nonempty return it, discarding all previous market work. If empty fall through. | [C702–748](../../../shinka/champions/submissions/hazel_weir/champion.py#L702-L748) |
| M14 finish | Return `mkt[:10]`. No final affordability, inventory, or legality simulation. | [C750](../../../shinka/champions/submissions/hazel_weir/champion.py#L750) |

### M10 detail: cadence bypass, price protection, batch sizing

[C521–618](../../../shinka/champions/submissions/hazel_weir/champion.py#L521-L618): bypass only with priority enabled and step≥256. Any demanded non-MELON score4 reaching threshold enables a **global** bypass: threshold .25 at shed≥70, .50 at shed<40, otherwise.40. No held-stock check for the triggering product. The global bypass also lowers every demanded item's price threshold.

Default product order: STRAWBERRY,MILK,WOOL,TOMATO,CARROT,EGG,WHEAT,MELON. With priority enabled and nonempty units24, stable-sort ascending `(units24,original_index)`, but MELON's supplied key is **forced0**. This changes insertion/preemption order of added sales; it does not globally reorder inherited sells.

Price floors:

- demanded: .85×base, or .70×base when global bypass;
- nondemanded MELON/TOMATO: .80×base; other nondemanded: .85×base;
- nondemanded MELON/WOOL during256≤step<640 with units24≥1.5 or score24≥.35: premium backlog=max(0,held−already); shield=max(.80,.95−.015×backlog); floor=max(ordinary floor,shield×base).

Availability=held−R for wheat, otherwise held. Need at least1 for MELON/TOMATO/EGG, otherwise2. This minimum check is on availability **before subtracting already selling**. Every emitted order still requires quote≥floor. Remaining=max(0,availability−already).

| Case | Batch cap before min(remaining,cap) |
|---|---|
| Priority enabled, step≥256, not MELON; demanded, score24≥.45 or units24≥2 | all remaining |
| Same; demanded, score24≥.30 or units24≥1.2 | 8 |
| Same; demanded, score24≥.15 or units24≥.6 | 6 |
| Same; demanded, otherwise | 3; if both forecast keys present and quote≥base, min(6,max(3,remaining//2)) |
| Same; nondemanded, score24≥.45 or units24≥2 | all remaining |
| Same; nondemanded, otherwise | 2; if both keys present, score24<.10 and units24<.6 →3; with quote≥base →min(4,max(3,remaining//2)) |
| Outside this branch | MELON2; other nondemanded3; other demanded4 |

**No-forecast distinction:** after256 the enabled-priority batch branch still executes with missing scores interpreted0. It then uses demanded3/nondemanded2; it does not revert to pre256 caps. `quiet_known` only permits expansions when both per-item keys exist.

### M12 detail: liquidation subphases

[C634–700](../../../shinka/champions/submissions/hazel_weir/champion.py#L634-L700), within active early_liq:

1. If score24 dict exists, all non-MELON products meeting score≥.30 before680 or≥.20 from680 are sorted descending score. Available=held−already−R_for_wheat−1_for_fertilizer. Quote floor.65×base before680/.52 after. Batch=all if score≥.45;8 if≥.35;6 otherwise.
2. At step≥680 independently add fixed high-value exits `(item,min_ratio,max_batch,min_available)` = `(MELON,.65,5,1)`, `(TOMATO,.75,4,1)`, `(FERTILIZER,.50,6,2)`, `(WOOL,.60,4,1)`. These do **not** preserve the fertilizer unit reserved in phase1.
3. At step≥693 sort shed by descending qty×quote, skip only if quote<.35×base **and** quote<1, add up to7 per item minus already selling. No feed reserve here: wheat may be sold during693…695 while R elsewhere is still positive. The code's “day29” comment starts at693, not the day29 boundary696.

### Disabled/unconsumed champion knobs

[C143–205](../../../shinka/champions/submissions/hazel_weir/champion.py#L143-L205): `_ENABLE_CASH_FLOOR=False`, `_ENABLE_ANIMAL_TOPUP=False`; rescue, pressure, priority, frontrun True. `_ENABLE_ORACLE_HOLD=False` and `_ORACLE_HOLD_SCORE=.50` have **no implemented hold branch** in this saved policy. `_WAGE_RESERVE_FLOOR=600` is not used; actual M03 reserves are computed separately. `_ANIMAL_YIELD` is unused; goose preference affects only disabled top-up. Do not expose unused constants as live decisions.

## 8. Oracle graph: observe, infer, forecast, record

### 8.1 Loading and caching

[O:70–130](../../../shinka/champions/submissions/hazel_weir/kagg_oracle.py#L70-L130) resolves model/features paths and imports features/mechanics. Bundled [`mechanics.py:11–19`](../../../shinka/champions/submissions/hazel_weir/opponent_model/mechanics.py#L11-L19) **unconditionally depends on installed `kaggle_environments` constants and market_price**. The later O fallback for land/hire/shed-access imports does not remove that dependency.

[O._NumpyModel:206–232](../../../shinka/champions/submissions/hazel_weir/kagg_oracle.py#L206-L232): load scaler, assert feature count, read alignment, construct NumPy model, assert150 channels and96 horizon, warm zero context. [O.get_model:275–286](../../../shinka/champions/submissions/hazel_weir/kagg_oracle.py#L275-L286) caches on `sys._kagg_oracle_registry[(model_dir,device_spec,backend_env)]`; alignment override is not in cache key. `_build_model` supports numpy/torch/auto, rejects unknown backend; auto falls back to NumPy only when torch/tsfm imports fail, not all later model errors. Torch branch picks device by pid round-robin with≥1GiB free, CPU fallback on transfer/warm failure. Submission bootstrap forces NumPy, so GPU policy is not the normal bundled entry path.

Current saved [`checkpoint/config.json`](../../../shinka/champions/submissions/hazel_weir/checkpoint/config.json) specifies context256, horizon96, channels150, patch length/stride32,8 patches, encoder192/decoder128,3 adaptive levels,2 layers each, std scaling, MSE, target channels141…149. [`labels.json:1`](../../../shinka/champions/submissions/hazel_weir/checkpoint/labels.json#L1) says next_action. Older docstrings mentioning512/64 patches or legacy alignment are not the saved checkpoint settings.

[`kagg_ttm_numpy.py:138–296`](../../../shinka/champions/submissions/hazel_weir/kagg_ttm_numpy.py#L138-L296): supported-config/weight-shape checks; per-context mean/variance in float64 with1e−5 floor, back to float32; patch and linear embedding; adaptive encoder factors4,2,1; layernorm/MLP GELU/gated patch and feature mixers; decoder channel mixing (gate before MLP), patch/feature mixing (gate after MLP); flatten head to96 steps; select targets and restore context location/scale. No online fitting or weight mutation. This is numerical prediction, not calibrated classification.

### 8.2 Observe-before-action state machine

[`O.OpponentTracker:346–366,528–608`](../../../shinka/champions/submissions/hazel_weir/kagg_oracle.py#L528-L608): one champion tracker object, not a dictionary keyed by seat. Model context default is checkpoint context; minimum real history from `KAGG_ORACLE_MIN_CONTEXT`, default context. Without model only, fallback context512. Reset clears history, arrays of `(720+96,150)` and `(720+96,9)`, turns, prediction, step, configuration.

On observation:

1. Seat from obs; step explicit or day×configured turnsPerDay+hour. Reset if step0, step≤previous step, or config missing. Resolve shop/center intervals, capacity, hire multiplier, board size.
2. If previous step exists, settle it against new market inventory; store signed supply, feed clipped nonnegative `log1p` supply into nine target channels, update opponent history. A missing turn leaves zero rows rather than reconstructing a skipped trajectory. `n_complete=step`, not an actual count of contiguous rows.
3. Compute **current**141 features using history only through previous turn; standardize `(x−mean)/std`; store row step. Keep `_Turn` with own initial shed, money, hires, quadrants, actor positions/bags, market inventory, shops. Prune saved turns older than step−3.
4. Set pred=None; if model exists and n_complete≥min_context, predict from rows `[step−context,step)`, left-zero-pad if early. Current-row features are not in the forecast block. Normal bundled first forecast is step256. Return forecast orNone.
5. After final sanitized action, `record_action` saves actual market orders and unit ops in the current `_Turn`.

### 8.3 Flow accounting and feature construction

[`O._our_executed/_shed_at_market/_settle:369–525`](../../../shinka/champions/submissions/hazel_weir/kagg_oracle.py#L369-L525): project actor-adjacent DROP/PICKUP/nonanimal PLACE in actor order, with capacity and DROP overflow removal. Then process own orders in list order, unit-by-unit, own cash/inventory/room constraints; HIRE Fibonacci cost, sequential land costs, fixed seed/animal costs, product curve buy price at inventory−1, sells add supply only if price>1. Earlier sells fund later own buys. No concurrent opponent-price interleaving is simulated. Projection does not model every actor operation (e.g. feed/fertilization/animal placement).

Total flow=`market_inventory_now−inventory_before+town_draw(previous_step,previous_shops)`. `next_action` label=total−estimated own flow. Legacy optional path attributes by requested own orders and inferred opponent flow from one step earlier, splitting total proportionally by absolute quantities when both active. Environment override controls alignment; absent/invalid labels fall back to legacy except unsupported alignment raises.

[`features.py:28–169`](../../../shinka/champions/submissions/hazel_weir/opponent_model/features.py#L28-L169): sorted feature names; time, both public money/hands, ratio, opponent hires/quadrants, own shed/room, public crop/animal/pending production/weed/water counts, shop multiplicities, each product inventory/deviation/price/base-ratio/own holdings/opponent pending/demand now and next day, cumulative positive opponent supply, time since positive supply, signed rolling4/24 supply sums. Own private bags are used for accounting, not opponent private data. History windows count recent stored observations; gaps are not expanded into explicit history deque entries. Bundled town draw counts duplicate shops, doubles single-product shops, adds town-center product draw on configured tick ([mechanics:58–76,115–123](../../../shinka/champions/submissions/hazel_weir/opponent_model/mechanics.py#L58-L76)).

### 8.4 Forecast semantics and failures

[`O.forecast:610–619`](../../../shinka/champions/submissions/hazel_weir/kagg_oracle.py#L610-L619): raw `(96,9)` log-space predictions; `units=expm1(max(pred,0))`; for horizons1/4/24/96, `units_k` is sum of clipped-unit forecasts and `score_k` is **maximum raw predicted log value**, not a probability. Policy uses score4, score24, units24 only. `score_k` can be negative; no probability calibration or NaN/Inf sanitization is supplied.

[`C._oracle_observe/_oracle_record:89–121`](../../../shinka/champions/submissions/hazel_weir/champion.py#L89-L121): eager-disabled lazy init only if trackerNone and accumulated errors==0; first load failure prevents retry under this gate. Observe errors returnNone and record short error; tracker may retain partial mutations. Successful non-None forecasts update count/mean latency. Record exceptions silently ignored. Lazy load does not update `ORACLE_DEVICE` from its initial `off` value, so that field alone is not proof of inactivity. Stats/model cache are process-lived, not reset per game.

## 9. Final sanitizer and fallbacks

[`C._clean_op/_sanitize:860–899`](../../../shinka/champions/submissions/hazel_weir/champion.py#L860-L899): accepts string verb →one-element list; list/tuple must be nonempty; verb converted to string and checked against allowed set; all arguments preserved verbatim.

Allowed unit verbs: NORTH,SOUTH,EAST,WEST,PASS,PICKUP,PLANT,WATER,HARVEST,FERTILIZE,BUILD_COOP,BUILD_PASTURE,DIG,PLACE,FEED,DROP,COLLECT_FERTILIZER,CARE. Market: BUY_SEED,BUY_PRODUCT,BUY_ANIMAL,SELL,HIRE,BUY_LAND.

- Non-dict action →return base unchanged.
- Farmer: cleaned evolved, else cleaned base farmer, else PASS.
- Hands: use evolved sequence; if entire channel not list/tuple use base hands or empty. Each malformed entry becomes PASS, **not corresponding base hand action**. Truncate/pad to pre-action observed hand count.
- Market: evolved sequence, otherwise base market or empty; remove disallowed/malformed operations and truncate10. Invalid entries are dropped, not replaced by base entries.
- No argument arity/type/item/quantity checks; no legality, cash, seeds, feed, capacity, locked-tile, or post-actor simulation. Thus the docstring's “schema-valid guarantee” is stronger than implementation.
- Sanitizer runs outside the overlay try block. Malformed base hands/market or exceptional objects can still raise. Player index/hand-count extraction also lies outside portions of protected execution.

[`mohui_v66/main.py:3–13`](../../../shinka/champions/submissions/hazel_weir/mohui_v66/main.py#L3-L13) wraps raw v66 in the **different** [`adapter.sanitize_action:711–785`](../../../shinka/champions/submissions/hazel_weir/mohui_v66/kaggriculture_agent/env/adapter.py#L711-L785), which canonicalizes, checks simultaneous seed overrequests, simulates legal farmer/hands using the engine, then clips market affordability/stock. **Hazel does not call that wrapper.** Substituting it is a behavior change, not a faithful sanitizer extraction.

## 10. State ownership, reset contract, and transaction boundaries

| State | Owner/reset semantics | Graph consequence |
|---|---|---|
| `_deferred_sells[seat]` | C183,312,362–373; created lazily, nominal clear at≤1 unreachable after opening returns; expired only age>24/stock unavailable/reissue | Preserve for exact parity; fixing game reset must be a separately tested change |
| `_TRACKER` | Single champion-level tracker; resets on repeated/decreasing step or0; not seat-keyed | One instance per game/seat; duplicate calls are destructive to history |
| Oracle registry/stats | Process-global model registry; champion stats persist | Cache key/environment/import state are part of reproducibility |
| `_V58_POLICIES`, warm history | Module-global; one new controller per early call; no outer per-game clearing, warm history cleared once constructed | A graph must not lazily construct only selected branch or skip unselected updates |
| v58/v59 state | Per seat; reset0 or step≤last; includes mode/mirror/known-yarn/clone/due | Calling twice at one step resets selection/lead state |
| v62 switch, v65 attack/Curve/q2/Daniel, v66 meta | Per seat; reset0 or step≤last; contain armed/active, pending FIFO, repair latch, events | Preserve each wrapper's separate state and order of updates |
| Decoded residual and weed controller | Per seat and per controller; reset0/empty/missing state or decreasing step, generally **not equal** | Repeated-step semantics differ from wrappers; do not impose one universal reset |
| `st._farmer_rescue` | One per-turn state dict, written before hands | Farmer and hands cannot run in parallel without changing coordination |
| Local market `money`, `protected_items`, list | One market call; helper closes over persistent deferred map | Stages cannot be independently reordered or replayed without preserving context |

[Wrapper reset sources: B3773–3797,4003–4011; local87–102,226–249; q2/Daniel/V as linked above.]

**Exceptions are not transactions.** Backbone policies update before champion overlay; if a later overlay fails, final action falls back to base but prior oracle observation, route histories, rescue local state, or deferred-map mutations are not rolled back. A graph implementation that rolls back all nodes, or individually catches channel exceptions, would not match this behavior.

## 11. Concrete risks and preservation checklist

These are source-level findings, not measured loss attribution or efficacy claims.

1. **Wrong entry/sanitizer:** importing `mohui_v66/main.py` adds strong legality simulation absent from Hazel. The direct raw-v66 call is required for parity.
2. **Hidden route/state omission:** ten warmed controllers plus independent repair controllers have evolving histories. Only exposing outer shop/market heuristics does not encode all production logic. Preserve exact compressed route data and embedded controller code as atomic dependencies.
3. **Opening return defeats deferred reset:** demonstrated by in-memory policy probe: seed deferred WOOL `[3,710]`; call new-game steps0/1 leaves it intact; step2 reissues SELL WOOL3 because negative age is not expired.
4. **Terminal observed/projected mismatch:** a probe with step718 observed WOOL2, carried5, and inherited projected SELL WOOL7 returned only SELL WOOL2. M13 replaces inherited terminal projected-deposit sales whenever its observed liquidation queue is nonempty. No claim about realized game outcome is made.
5. **Actor coordinate mismatch:** champion rescue indexes farmer position as row/column; inherited routines use x/y. On nonsymmetric positions this can target the transposed tile/hand match. Preserve actual code during extraction; repairing coordinates is a separate change.
6. **Rescue feed source mismatch and COOP omission:** champion gates FEED by shed wheat and ignores actor bag availability; inherited projection models FEED from actor bag. State ignores COOP animals, affecting hunger lists/n_animals/feed reserves. Game legality is not checked by final sanitizer.
7. **End-investment filter length bug:** M04 only removes operations with at least2 elements, so ordinary `[HIRE]` and `[BUY_LAND]` survive. One-element order forms are present in decoded route data.
8. **Price-floor conjunctions:** M07 and M12 phase3 accept any quote≥1, not the ratios named by comments. Changing AND to OR alters behavior.
9. **Protection is local, not universal:** feed reserve is applied only to added sells in specific stages, never imposed on all inherited/deferred sells; M12 phase3 lacks it even before696. Fertilizer reserve1 is removed by pressure/headroom/fixed early liquidation if those stages add sales. `_add_sell` merges without protecting quantities against prior overrequests.
10. **Headroom overcount:** need ignores already scheduled market sales and may sell additional quantities beyond computed end-of-day shortage; it also does not anticipate current-turn harvesting or every actor transfer. Protected items can consume eviction flexibility even when helper insertion fails.
11. **Cash approximation and queue economics:** M03 does not credit earlier sells or debit hires/land/seeds/animals; it uses single observed quotes. Reordering inherited non-SELLs changes their funding/stock semantics. Full-queue eviction only considers SELLs, so large investment queues can block additions.
12. **Internal flow versus final action:** inherited residual controllers account their own pre-Hazel proposals, including nonselected policy proposals; TTM accounts final action. They cannot share one “own executed flow” state without changing behavior. Oracle itself approximates concurrent-price execution and actor transfers.
13. **Recipe/config inconsistency:** champion FARMERS includes MELON, embedded v23 includes WHEAT, oracle imports installed engine recipe; inherited regime defaults center12 while oracle defaults24; champion day/step/capacity/order limits hardcode24/720/100/10 in places. Changing engine/config can desynchronize layers.
14. **Threshold sensitivity and non-probability scores:** score4/24 are maxima of log predictions, not probabilities. Model or NumPy numeric changes can move exact thresholds. Missing forecast after256 has different caps from pre256. No finite-value check is supplied.
15. **Process contamination:** ordinary import names, overwritten embedded `sys.modules`, env setdefault, cache surviving imports, and global non-seat-keyed tracker require isolation. Curve-source module loading re-executes the shared embedded package payload and registers module names again; bytes are identical here, but identities and captured globals must not be assumed interchangeable under a refactor.
16. **Timing endpoint mismatch:** ten inherited hybrid routes cover0…718 and clamp later requests to718; Curve has0…719; champion has distinct≥718 branch. This audit did not assert which terminal observation the current engine actually executes. Verify with the intended engine before changing endpoint logic.
17. **Weak sanitizer:** in-memory probe accepted farmer `['PLANT']` and market `['SELL','NOT_A_PRODUCT',-3]`; invalid hand verb became PASS instead of base WATER. These are actual local probe results, not fabricated engine responses.
18. **Comments/metadata drift:** 922 versus928 lines, checkpoint256 versus old512 descriptions, unused wage floor/hold flag, and differing sanitizer claims illustrate why source conditions take precedence over prose.

### Behavior-preserving extraction acceptance criteria (not performed here)

- Pin the listed source/checkpoint/scaler bytes and resolve all imports inside the bundle.
- Keep G01 atomic initially; expose telemetry without advancing a controller twice. Do not replace route data with generalized intent summaries.
- Preserve G04→G05→G06→G07 ordering, one overlay exception boundary, market insertion order/tie stability/early returns, helper sharing, and existing reset semantics.
- Feed oracle once before policy decisions and record only final sanitized action. No lookahead observation or intermediate-action recording.
- Compare original and candidate final actions **and persistent state transitions** on boundary steps0/1/2,72/96/120/144/256/360,640/672/680/693/696/700/711/712/713…719; pressure thresholds40/60/70/80/90/95; all score thresholds; missing forecasts; full queues; repeated/decreasing steps; no-hands/new-hands; failure paths.
- Any “fix” to bugs identified here belongs in a separately named intervention after parity, not silently in import reconstruction. No efficacy prediction follows from exposing the graph.

## 12. Audit verification actually performed

- Read full current champion, submission bootstrap, oracle, NumPy model, features/mechanics, all outward v65/v66 wrappers, B's executable selection/guard sections, Curve-specific sections, and the unused wrapper's sanitizer path.
- AST-indexed functions and physical line extents; recursively decoded nested source/module dictionaries and every top-level route payload; checked route lengths719 versus720, shared prefix equality, and representative decoded openings. Relevant active decoded functions were inspected with virtual line numbers.
- Computed actual SHA-256 values above and compared the champion/oracle/network/weights values with the saved manifest; no reliance on prior-session explanations.
- Ran only **in-memory isolated pure-policy probes** from selected champion AST constants/functions: deferred carryover, terminal observed-shed replacement, and sanitizer argument behavior. These did not import the bundle, load weights, create bytecode caches, access the network, or simulate a game.
- Scope limitation: this is source/documentation verification, not action-equivalence evaluation, a replay tournament, engine-mechanics proof, forecast-quality test, or measured performance assessment. Encoded fixed action tables remain source-linked atomic data rather than reproduced thousands of rows in this document.
