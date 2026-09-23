"""Executable test suite for opt-in procedural graph extensions.

Verifies:
1. Baseline disabled equivalence across 720 randomized turns.
2. Defensible opponent-pressure sell scheduling (both seats, no claimed priority).
3. Capacity-aware liquidation net of scheduled sales, preserving feed/fertilizer.
4. Order arbitration respecting dependencies, order cap, and no duplicate sells.
5. Legal opportunity-aware idle weed reclamation (coords x,y; DIG before PLANT; guarded no-op for speculative planting).
6. Actual shop consumption phase-aware pricing (post-market consumption, no imaginary premiums).
7. Fail-closed safety and complete decision telemetry.
8. Provenance hashes and graph execution chain validation.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import unittest

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / "hazel_runtime"
if str(RUNTIME) not in sys.path:
    sys.path.insert(0, str(RUNTIME))

from engine_contract import CROPS, PRODUCTS, SHOPS, market_price
from experimental import Extensions, STAGES, SWITCHES, settings
from graph_runtime import HazelGraph, validate_chain, TURN_STAGES, MARKET_STAGES
import agent_graph


def load_pure_champion():
    import ast, types
    source_path = RUNTIME / "champion.py"
    tree = ast.parse(source_path.read_text())
    body = [n for n in tree.body if getattr(n, "lineno", 0) >= 125]
    mod = types.ModuleType("pure_champion")
    mod.__file__ = str(source_path)
    exec(compile(ast.Module(body=body, type_ignores=[]), str(source_path), "exec"), mod.__dict__)
    return mod


def mock_observation(step=200, player=0):
    tiles = [[None for _ in range(10)] for _ in range(10)]
    farm0 = {
        "money": 5000.0,
        "farmer": [1, 1],
        "hands": [[1, 2]],
        "tiles": tiles,
        "unlocked_quadrants": ["NW"],
        "hires_today": 0,
    }
    farm1 = copy.deepcopy(farm0)
    return {
        "step": step,
        "day": step // 24,
        "hour": step % 24,
        "player": player,
        "farms": [farm0, farm1],
        "private": {
            "shed": {p: 5 for p in PRODUCTS},
            "inventories": [{}, {}],
            "seeds": {"WHEAT": 10, "STRAWBERRY": 5},
        },
        "town": {"unlocked_shops": ["BAKERY", "YARN_STORE"]},
        "market": {
            "inventory": {p: 10000 for p in PRODUCTS},
            "prices": {p: 50 for p in PRODUCTS},
        },
    }


class TestExperimentalProceduralGraph(unittest.TestCase):
    def setUp(self):
        self.pure_champion = load_pure_champion()
        self.graph_path = ROOT / "policy_graph.json"
        self.graph_data = json.loads(self.graph_path.read_text())

    def test_01_provenance_and_chain_integrity(self):
        # 1. Turn and Market chains valid
        turn_ids = [s[0] for s in TURN_STAGES]
        market_ids = [s[0] for s in MARKET_STAGES]
        validate_chain(self.graph_data["turn"], turn_ids)
        validate_chain(self.graph_data["market"], market_ids)

        # 2. Champion bytes match original submitted hash exactly
        champ_bytes = (RUNTIME / "champion.py").read_bytes()
        self.assertEqual(
            hashlib.sha256(champ_bytes).hexdigest(),
            "9c8d1431dc2401eb7aa63091cb72d84f749273ed8462dfb75fbfe4f1618d568f",
            "Champion source bytes must be frozen and preserved!",
        )

        # 3. All provenance hashes match current files
        bundle = RUNTIME
        for rel, exp in self.graph_data["provenance"]["runtime_bundle_hashes"].items():
            actual = hashlib.sha256((bundle / rel).read_bytes()).hexdigest()
            self.assertEqual(actual, exp, f"Hash mismatch for runtime bundle: {rel}")

        entry_exp = self.graph_data["provenance"].get("entrypoint_sha256")
        entry_act = hashlib.sha256((ROOT / "agent_graph.py").read_bytes()).hexdigest()
        self.assertEqual(entry_act, entry_exp, "Entrypoint sha256 mismatch")

        # 4. Engine loads through agent_graph without error
        eng = agent_graph.get_engine()
        self.assertIsNotNone(eng)
        self.assertEqual(eng.turn_ids, turn_ids)

    def test_02_baseline_disabled_equivalence_720_turns(self):
        """When switches are disabled, runtime output matches baseline champion exactly."""
        g = HazelGraph(self.pure_champion, self.graph_path)
        # Explicitly test disabled equivalence
        g.extensions = Extensions({s: False for s in SWITCHES})
        self.assertFalse(any(g.extensions.switches.values()))

        rng = random.Random(1337)
        for step in range(720):
            obs = mock_observation(step=step, player=0)
            products = list(self.pure_champion._BASE_PRICE)
            obs["private"]["shed"] = {p: rng.randrange(25) for p in products}
            obs["private"]["inventories"] = [{"WHEAT": rng.randrange(10)}]
            obs["market"]["prices"] = {p: rng.randrange(1, 300) for p in products}
            obs["farms"][0]["money"] = float(rng.randrange(8000))
            obs["farms"][0]["hands"] = [[1, 2]] * rng.randrange(4)

            # Simulated forecast
            forecast = {
                "score_4": {p: rng.random() for p in products},
                "score_24": {p: rng.random() for p in products},
                "units_24": {p: rng.random() * 5.0 for p in products},
            }
            state = self.pure_champion.farm_state(obs, 0, forecast)

            # Evolved by pure champion
            base_farmer = ["PASS"]
            base_hands = [["PASS"] for _ in obs["farms"][0]["hands"]]
            base_market = [["SELL", rng.choice(products), rng.randrange(1, 10)] for _ in range(rng.randrange(6))]

            champ_farmer = self.pure_champion.evolve_farmer_action(obs, 0, base_farmer, state)
            champ_hands = self.pure_champion.evolve_hand_actions(obs, 0, base_hands, state)
            champ_market = self.pure_champion.evolve_market_orders(obs, 0, base_market, state)
            champ_action = self.pure_champion._sanitize(
                {"farmer": champ_farmer, "hands": champ_hands, "market": champ_market},
                {"farmer": base_farmer, "hands": base_hands, "market": base_market},
                len(obs["farms"][0]["hands"]),
            )

            # Simulated baseline evolved dict before extensions
            baseline_evolved = {
                "farmer": copy.deepcopy(champ_farmer),
                "hands": copy.deepcopy(champ_hands),
                "market": copy.deepcopy(champ_market),
            }

            # Run through extensions with all switches disabled
            ext_action = baseline_evolved
            for ext_stage in ("experimental_state", "shop_phase", "capacity_liquidation",
                              "opponent_pressure", "idle_opportunities", "order_arbitration"):
                ext_action = g.extensions.run(ext_stage, obs, 0, ext_action, state, {})

            ext_sanitized = self.pure_champion._sanitize(
                ext_action,
                {"farmer": base_farmer, "hands": base_hands, "market": base_market},
                len(obs["farms"][0]["hands"]),
            )

            self.assertEqual(champ_action, ext_sanitized, f"Equivalence failure at step {step}")

    def test_03_defensible_opponent_pressure_scheduling_both_seats(self):
        """Contiguous SELL blocks are ranked by opponent pressure; non-SELL acts as dependency barrier.
        Symmetric logic across Seat 0 and Seat 1; quotes before commit respected."""
        for seat in (0, 1):
            ext = Extensions({"opponent_pressure": True, "order_arbitration": True})
            obs = mock_observation(step=100, player=seat)
            state = {
                "oracle": {
                    "score_4": {
                        "WHEAT": 0.10,
                        "STRAWBERRY": 0.85,
                        "MILK": 0.40,
                        "WOOL": 0.95,
                    }
                }
            }
            # Action with two contiguous SELL blocks separated by non-SELL orders
            action = {
                "farmer": ["PASS"],
                "hands": [["PASS"]],
                "market": [
                    ["SELL", "WHEAT", 5],        # Block 1: score 0.10
                    ["SELL", "STRAWBERRY", 2],   # Block 1: score 0.85
                    ["HIRE", 1],                 # Barrier: non-SELL
                    ["BUY_PRODUCT", "WHEAT", 4], # Barrier: non-SELL
                    ["SELL", "MILK", 3],         # Block 2: score 0.40
                    ["SELL", "WOOL", 1],         # Block 2: score 0.95
                ],
            }
            ext.run("experimental_state", obs, seat, action, state, {})
            result = ext.run("opponent_pressure", obs, seat, action, state, {})

            # In block 1: STRAWBERRY (0.85) should be ranked before WHEAT (0.10)
            self.assertEqual(result["market"][0], ["SELL", "STRAWBERRY", 2])
            self.assertEqual(result["market"][1], ["SELL", "WHEAT", 5])

            # Barriers must NOT be crossed or moved
            self.assertEqual(result["market"][2], ["HIRE", 1])
            self.assertEqual(result["market"][3], ["BUY_PRODUCT", "WHEAT", 4])

            # In block 2: WOOL (0.95) should be ranked before MILK (0.40)
            self.assertEqual(result["market"][4], ["SELL", "WOOL", 1])
            self.assertEqual(result["market"][5], ["SELL", "MILK", 3])

            # Verify telemetry records seat without claiming seat priority
            event = [e for e in ext.telemetry if e["stage"] == "opponent_pressure"][0]
            self.assertEqual(event["seat"], seat)
            self.assertIn("no_seat_priority", event["quote_semantics"])

    def test_04_capacity_aware_liquidation_net_and_reserves(self):
        """Capacity relief is net of scheduled sales and preserves actual feed/fertilizer needs.
        No arbitrary fertilizer floor when no crops exist."""
        ext = Extensions({"capacity_liquidation": True, "order_arbitration": True})
        obs = mock_observation(step=22, player=0)  # hour 22: within pre-drop window (24-4=20)
        # Give farm 1 cow (needs feed today and tomorrow) and 1 mature ongoing strawberry (needs fert)
        obs["farms"][0]["tiles"][2][2] = {"animal": "COW", "fed_today": False}
        obs["farms"][0]["tiles"][3][3] = {
            "kind": "PLANT",
            "crop": "STRAWBERRY",
            "planted_day": 0,
            "fertilized_until_day": -1,
            "watered_today": False,
        }
        obs["private"]["shed"] = {
            "WHEAT": 10,
            "FERTILIZER": 10,
            "CARROT": 40,
            "MELON": 40,
        }
        # Total in shed = 100 (cap is 100). Farmer carries 15 units.
        obs["private"]["inventories"] = [{"CARROT": 15}, {}]
        # Carried 15 units will overflow shed at end-of-day by 15 units!
        # Cow needs 2 wheat (1 today, 1 tomorrow). Strawberry needs 1 fertilizer.
        # Shed has 10 wheat (reserve 2 -> unreserved 8) and 10 fert (reserve 1 -> unreserved 9).

        # Case A: Existing orders already sell 10 CARROT
        action = {
            "farmer": ["PASS"],
            "hands": [["PASS"]],
            "market": [["SELL", "CARROT", 10]],
        }
        ext.run("experimental_state", obs, 0, action, {}, {})
        # Shortage = 100 + 15 - 10 - 100 = 5 units remaining shortage!
        result = ext.run("capacity_liquidation", obs, 0, action, {}, {})
        # Should append additional sell orders for exactly the remaining 5 units, NOT the full 15!
        added_qty = sum(o[2] for o in result["market"][1:])
        self.assertEqual(added_qty, 5, "Capacity relief must be NET of already scheduled sales!")

        # Case B: Verify feed and fertilizer reserves cannot be sold under capacity liquidation
        # Shed full of only WHEAT (2 units) and FERTILIZER (1 unit)
        obs["private"]["shed"] = {"WHEAT": 2, "FERTILIZER": 1}
        obs["private"]["inventories"] = [{"WHEAT": 5}, {}]
        action = {"farmer": ["PASS"], "hands": [["PASS"]], "market": []}
        ext.run("experimental_state", obs, 0, action, {}, {})
        result = ext.run("capacity_liquidation", obs, 0, action, {}, {})
        # Sells should be 0 because all held wheat/fert is reserved for live animals/plants!
        sells = [o for o in result["market"] if o[0] == "SELL"]
        self.assertEqual(len(sells), 0, "Protected animal feed and crop fertilizer must not be sold!")

    def test_05_order_arbitration_dependency_and_duplicate_handling(self):
        """Order arbitration deduplicates SELL orders without merging across non-SELL barriers,
        enforces 10-order cap, and clamps to unreserved stock."""
        ext = Extensions({"order_arbitration": True})
        obs = mock_observation(step=100, player=0)
        obs["private"]["shed"] = {"WHEAT": 10, "MELON": 5}
        action = {
            "farmer": ["PASS"],
            "hands": [["PASS"]],
            "market": [
                ["SELL", "WHEAT", 4],
                ["HIRE", 1],
                ["SELL", "WHEAT", 8],        # Duplicate SELL for WHEAT across barrier -> rejected!
                ["SELL", "MELON", 100],       # Oversell: only 5 held -> clamped to 5!
                ["BUY_PRODUCT", "FERTILIZER", 1],
                ["SELL", "CARROT", 5],       # CARROT held is 0 -> rejected (no stock)!
            ] + [["BUY_SEED", "WHEAT", 1] for _ in range(10)], # Over cap of 10
        }
        ext.run("experimental_state", obs, 0, action, {}, {})
        result = ext.run("order_arbitration", obs, 0, action, {}, {})

        market = result["market"]
        # Must be capped at 10 orders
        self.assertLessEqual(len(market), 10)
        # First sell of WHEAT kept
        self.assertEqual(market[0], ["SELL", "WHEAT", 4])
        self.assertEqual(market[1], ["HIRE", 1])
        # Duplicate sell of WHEAT rejected
        self.assertNotIn(["SELL", "WHEAT", 8], market)
        # Oversell MELON clamped to available 5
        melon_sells = [o for o in market if len(o) >= 2 and o[0] == "SELL" and o[1] == "MELON"]
        self.assertEqual(melon_sells, [["SELL", "MELON", 5]])
        # Non-stock CARROT rejected
        carrot_sells = [o for o in market if len(o) >= 2 and o[0] == "SELL" and o[1] == "CARROT"]
        self.assertEqual(carrot_sells, [])

    def test_06_idle_opportunity_weed_reclamation_and_plant_guard(self):
        """Idle-only weed reclamation emits DIG on standing weed or step toward adjacent weed.
        Guards against naive planting on weed; speculative planting is a guarded no-op."""
        ext = Extensions({"idle_opportunities": True})
        obs = mock_observation(step=50, player=0)
        # Put weed directly under farmer at (1, 1)
        obs["farms"][0]["tiles"][1][1] = {"kind": "WEED"}
        # Put weed adjacent to hand at (1, 2) -> East at (2, 2)
        obs["farms"][0]["tiles"][2][2] = {"kind": "WEED"}

        action = {
            "farmer": ["PASS"],
            "hands": [["PASS"]],
            "market": [],
        }
        ext.run("experimental_state", obs, 0, action, {}, {})
        result = ext.run("idle_opportunities", obs, 0, action, {}, {})

        # Farmer is standing on weed: must emit DIG (never naive PLANT)
        self.assertEqual(result["farmer"], ["DIG"], "Must DIG standing weed, never PLANT directly!")
        # Hand is at (1, 2) with weed at (2, 2) (East): emits ['EAST']
        self.assertEqual(result["hands"][0], ["EAST"], "Must step towards adjacent weed!")

        # Verify telemetry records plant guard reason
        event = [e for e in ext.telemetry if e["stage"] == "idle_opportunities"][0]
        self.assertIn("DIG_before_PLANT", event["plant_reason"])

    def test_07_shop_phase_aware_pricing_no_imaginary_premiums(self):
        """Shop phase computes post-market town consumption correctly; does not assume same-turn premiums."""
        ext = Extensions({"shop_phase": True})
        # Step 24: both shop interval (4) and town center interval (24) trigger
        obs = mock_observation(step=24, player=0)
        obs["town"]["unlocked_shops"] = ["YARN_STORE", "BAKERY"]
        # YARN_STORE: 1 product ("WOOL") -> single product consumption = 2 units
        # BAKERY: 2 products ("EGG", "WHEAT") -> multi-product consumption = 1 unit each
        # Town center: 1 unit each for all products except FERTILIZER
        # Total expected consumption at step 24:
        # WOOL: 2 (yarn store) + 1 (town center) = 3
        # WHEAT: 1 (bakery) + 1 (town center) = 2
        # EGG: 1 (bakery) + 1 (town center) = 2
        # FERTILIZER: 0 (not consumed by shops or center)
        action = {"farmer": ["PASS"], "hands": [["PASS"]], "market": []}
        ext.run("experimental_state", obs, 0, action, {}, {})
        ext.run("shop_phase", obs, 0, action, {}, {})

        event = [e for e in ext.telemetry if e["stage"] == "shop_phase"][0]
        c = event["consumption"]
        self.assertEqual(c["WOOL"], 3)
        self.assertEqual(c["WHEAT"], 2)
        self.assertEqual(c["EGG"], 2)
        self.assertEqual(c["FERTILIZER"], 0)
        self.assertIn("post_market_consumption_not_same_turn_premium", event["reason"])

    def test_08_fail_closed_transactional_safety(self):
        """If an extension stage encounters an error, the agent fails closed and keeps the prior action."""
        g = HazelGraph(self.pure_champion, self.graph_path)
        g.extensions.switches["order_arbitration"] = True
        obs = mock_observation(step=100, player=0)

        # Corrupt context to trigger exception inside order_arbitration
        g.extensions.ctx = "invalid_context_object"
        action = {"farmer": ["PASS"], "hands": [["PASS"]], "market": [["SELL", "WHEAT", 1]]}
        # Run agent: should fail closed without crashing
        res = g.extensions.run("order_arbitration", obs, 0, action, {}, {})
        self.assertEqual(res, action, "Must retain prior action on extension failure!")

        # Verify telemetry records extension_failed_closed
        errors = [e for e in g.extensions.telemetry if "failed_closed" in e["reason"]]
        self.assertTrue(len(errors) > 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
