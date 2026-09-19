"""Test connectivity to TypeSafe AI System One / Jev API."""
import os
from pathlib import Path

env_path = Path("/home/alex/kaggriculture/.env")
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[len("export "):]
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip().strip("'\"")

from typesafe_sdk import TypeSafeClient, Choice, Noul, Score

print(f"TYPESAFE_API_KEY present: {bool(os.environ.get('TYPESAFE_API_KEY'))}")
client = TypeSafeClient()

print("Calling Jev System One...")
try:
    res = client.system_one(
        state={
            "farm_day": 5,
            "net_cash": 2500,
            "hands_count": 3,
            "unlocked_shops": ["PIZZA_SHOP"],
            "orders_in_shed": {"TOMATO": 10, "WHEAT": 25}
        },
        questions={
            "posture": Choice(
                instructions="What strategic posture should the farm adopt today?",
                criteria={
                    "shop_arbitrage": "Fulfill high margin pizza shop orders with tomatoes",
                    "capital_compound": "Aggressively expand farmland and hire more hands",
                    "wage_defense": "Preserve cash to pay hands"
                }
            ),
            "is_emergency": Noul(instructions="Is the farm in an immediate emergency liquidity crisis?"),
            "expansion_urgency": Score(
                instructions="How urgently should we expand land?",
                criteria=["low", "medium", "high"]
            )
        }
    )
    print("API SUCCESS!")
    print(f"Posture choice: {res.choices['posture'].choice}")
    print(f"Is emergency noul: {res.nouls['is_emergency'].noul}")
    print(f"Expansion score: {res.scores['expansion_urgency'].score}")
except Exception as e:
    print(f"API ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
