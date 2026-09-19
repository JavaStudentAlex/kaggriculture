"""Strategic knowledge base and Kaggle ladder heuristics for Kaggriculture.

Distilled from forensic audits of live ladder replays (50 daily Kaggle datasets,
34,744 episodes), Hazel Weir match losses, and Shinka champion benchmarks.
Injected into Jev System One prompts to ground decisions in competitive reality.
"""
from __future__ import annotations

KAGGLE_STRATEGY_PLAYBOOK = {
    "core_mechanics": {
        "turns_per_day": 24,
        "total_days": 30,
        "total_steps": 720,
        "board_size": "10x10 grid",
        "shed_capacity": 100,
        "engine_order_cap": 10,  # Strict max market orders per turn
        "starting_cash": 3000.0,
    },
    "golden_rules": [
        {
            "id": "WAGE_DEFENSE_RULE",
            "name": "Midnight Payroll Defense",
            "principle": "At hour 24/midnight, payroll liability is deducted: (n_hands * 120 + 100).",
            "what_fails": "Entering hour 21 with cash < liability. Emergency dumping fertilizer earns <$30, hands are fired, crops wither in 2 turns, reward drops to $0.",
            "what_works": "Monitor net available cash continuously. By hour 18, if cash < payroll liability, halt all capital purchases and force top-priority liquidation."
        },
        {
            "id": "TOWN_SHOP_ARBITRAGE",
            "name": "Town Shop Premium Capture",
            "principle": "Town shops (Bakery, Pizza Shop, Brunch Spot) unlock every 3 days and purchase commodities at 2.0x-3.5x open-market rates.",
            "what_fails": "Dumping harvests on open market immediately at 0.70x-0.85x base price, leaving shed empty when shop contracts arrive.",
            "what_works": "Maintain a 12-16 unit inventory buffer of shop crops (Carrot, Tomato, Egg). Prioritize shop delivery orders in top order slots (slots 1-3)."
        },
        {
            "id": "ENGINE_ORDER_CAP",
            "name": "The 10-Order Ceiling",
            "principle": "The simulation strictly drops any orders submitted beyond 10 per tick.",
            "what_fails": "Hiring 11+ hands while queuing multiple buy and sell orders. Trade orders are dropped, causing liquidity stalls.",
            "what_works": "Cap the workforce at 9-10 hands across 22-26 plots (ratio of ~2.4 plots/hand). Preserves order bandwidth for market trades."
        },
        {
            "id": "EXPANSION_HORIZON",
            "name": "Day 13 Land Expansion Cutoff",
            "principle": "Recurring crops (Strawberry) take multiple days to mature and compound revenue.",
            "what_fails": "Buying land or slow seeds after Day 13-15. Cash is locked up in crops that never reach harvest maturity before Day 30.",
            "what_works": "Aggressively compound land and hands up to Day 12-13, then enforce a hard freeze on new plot purchases and focus 100% on harvest velocity."
        },
        {
            "id": "ORACLE_FRONTRUN",
            "name": "TTM Opponent Glut Preemption",
            "principle": "Opponent mass liquidations crash open-market prices to $1.00 floors.",
            "what_fails": "Holding luxury livestock commodities (Wool, Milk) while opponent floods the market.",
            "what_works": "When the TTM oracle indicates high glut probability (risk >= 0.50), preemptively liquidate goods 4-8 turns before the collapse."
        },
        {
            "id": "CLOSED_LOOP_WATERING",
            "name": "Uninterrupted Regrowth Clocks",
            "principle": "Strawberry plots require continuous moisture for tight 3-day recurring flushes.",
            "what_fails": "Workers harvesting ripe fruit and leaving plots dry because watering cans are empty.",
            "what_works": "Workers immediately refill cans at the well after dropping harvests at the shed, ensuring in-situ watering upon harvest."
        }
    ]
}


def build_jev_knowledge_summary() -> str:
    """Returns a compact textual summary of competitive strategies for Jev state."""
    lines = [
        "COMPETITIVE STRATEGY DIRECTIVES:",
        "1. Payroll: By hour 18, cash MUST cover midnight payroll (n_hands*120+100) or force emergency liquidation.",
        "2. Shops: Reserve 12-16 units for unlocked town shops (Pizza/Bakery) paying 2.0x-3.5x premium.",
        "3. Cap: Max 9-10 hands to prevent exceeding the 10-order engine action ceiling.",
        "4. Freeze: Stop buying land and Strawberry seeds at Day 13; prioritize harvest throughput.",
        "5. Oracle: Preemptively liquidate Wool/Milk before predicted opponent supply gluts.",
        "6. Labor: Guarantee workers have full watering cans to prevent unwatered strawberry delays."
    ]
    return " | ".join(lines)
