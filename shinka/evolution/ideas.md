# Kaggriculture Strategy Ideas & Mutation Directions

Research catalogue of verified engine mechanics, failure forensic audits from the live Kaggle ladder (Hazel Weir & Copper Weir submissions), and concrete evolutionary levers for Shinka mutations.

---

## 1. Verified Kaggle Ladder Failure Forensics (Hazel Weir & Copper Weir)

Audited across 133 ladder matches of Hazel Weir (`56246758`, 43.6% WR, rating 1314.7) and 131 matches of Copper Weir (`56239161`, 42.7% WR, rating 1306.1):

1. **Town Shop Neglect (The Macro Revenue Gap):**
   - High-rated ladder opponents ($100k–$150k+ cash) actively exploit town shops (`BRUNCH_SPOT`, `PET_CAFE`, `BAKERY`, `FARMERS_MARKET`).
   - Opponents produce and sell 100+ Carrots and 140+ Eggs, riding surging demand where Carrot prices rise from $35 to **$87+** and Eggs to **$65+**.
   - Hazel/Copper Weir remained locked in a Wheat/Melon monoculture with 0 Eggs, 0 Tomatoes, and <15 Carrots, forfeiting $15k–$30k of premium town revenues.
2. **Commodity Glut & The $1.00 Wool/Milk Trap:**
   - When both players build large flocks of Sheep and herds of Cows, the open market order book collapses.
   - By Day 20–22 (steps 480–540), **WOOL collapsed from $215 to $1.00**, and **MILK collapsed from $179 to $1.00**.
   - Holding animal produce until late turns forces dumping hundreds of units at $1.00 for negligible returns.
3. **Midnight Wage Starvation (Episode 109577172 Crash):**
   - At Step 239 (Day 9 Hour 23), bank cash dropped to $0.00.
   - At Midnight (Step 240), all hired-hand contracts expired and workers were fired for lack of wages.
   - Unwatered crops died, and the farm was paralyzed for 480 steps, finishing at **$0.00**.

---

## 2. Immediate Baseline Fixes in `initial.py`

1. **Strict Midnight Wage Reserve Guard:**
   - In hours 18–23, enforce a strict cash floor: `wage_reserve = max(600.0, n_hands * 120.0 + 100.0)`.
   - Filter any `BUY_PRODUCT`, `BUY_SEED`, `BUY_ANIMAL`, or `BUY_LAND` that draws cash below `wage_reserve`.
   - At hour 21+, if bank cash is below `wage_reserve`, trigger emergency sales of surplus non-feed inventory to restore the reserve before midnight.
2. **Pre-Step 540 Livestock Glut Protection:**
   - From step 480 onward, begin phased selling of Wool and Milk in small batches (up to 6 units) while prices remain $\ge \$15$, avoiding the $1.00 price cliff.
3. **Town Shop Crop Diversification:**
   - When `obs['town']['unlocked_shops']` contains `PET_CAFE`, `BAKERY`, or `BRUNCH_SPOT`, acquire seeds for `CARROT` and `TOMATO` and maintain stock for shop sales.
4. **Floor-Price Selling Dampener:**
   - Avoid dumping bulk product at prices $\le \$3$ unless shed capacity is critically full ($\ge 95\%$).

---

## 3. Evolutionary Search Directions for Shinka Mutations

### Direction A: Dynamic Town Shop Arbitrage
* **Mechanism:** Evolve an adaptive scoring function in `farm_state` that calculates the effective profit margin of each recipe in `obs['town']['unlocked_shops']` (e.g. Smoothie Shop vs Brunch Spot).
* **Action:** Re-weight planting in `evolve_farmer_action` and livestock purchasing in `evolve_market_orders` dynamically to match town shop demand.

### Direction B: Opponent Order-Flow Front-Running (Oracle Exploitation)
* **Mechanism:** Exploit `st["oracle"]` predictions (`score_4` and `score_24`).
* **Action:** When `score_4 >= 0.30` or `score_24 >= 0.40` on a commodity, the opponent is preparing to dump. Front-run them by executing a market sell 1–2 turns before them, securing the top of the price curve before their dump crashes the price.

### Direction C: Non-Linear Elasticity Selling
* **Mechanism:** Track market inventory changes in `obs['market']['inventory']` to estimate price elasticity.
* **Action:** Rather than dumping in large batches, pace orders into micro-batches when demand is inelastic, and hold stock when prices temporarily dip due to opponent sales.

### Direction D: Hired-Hand Dispatcher Optimization
* **Mechanism:** In `evolve_hand_actions`, refine the idle worker assignment priority.
* **Action:** Prioritize watering and harvesting high-value shop crops (Carrot, Strawberry, Tomato) and feeding high-yield animals over clearing low-priority weeds on empty tiles.
