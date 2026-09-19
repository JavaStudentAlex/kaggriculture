import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from graph_engine import ProceduralGraphEngine


class TestProceduralGraphEngine(unittest.TestCase):
    def setUp(self):
        self.engine = ProceduralGraphEngine()

    def test_graph_loading(self):
        self.assertGreaterEqual(len(self.engine.nodes), 6)
        self.assertGreaterEqual(len(self.engine.edges), 5)
        # Verify priority ordering
        priorities = [e["priority"] for e in self.engine.edges]
        self.assertEqual(priorities, sorted(priorities))

    def test_wage_defense_guard(self):
        # Scenario 1: Hour 19, Day 5, Cash $200, 3 hands ($460 liability) -> Guard MUST trigger
        obs = {"hour": 19, "day": 5, "step": 115, "farms": [{"hands": [{}, {}, {}]}], "player": 0}
        st = {"money": 200.0, "hands": [{}, {}, {}]}
        guards = self.engine.evaluate_guards(st, obs)
        self.assertTrue(guards["wage_defense"], "Wage defense should trigger on payroll deficit")

        # Scenario 2: Hour 19, Day 5, Cash $1000, 3 hands -> Guard should NOT trigger
        st["money"] = 1000.0
        guards = self.engine.evaluate_guards(st, obs)
        self.assertFalse(guards["wage_defense"], "Wage defense should not trigger when cash is healthy")

    def test_shed_headroom_guard(self):
        # Scenario: Shed has 80 units, hands carrying 10 units -> Total 90 >= 85 -> Must trigger
        obs = {
            "hour": 10, "day": 8, "step": 200,
            "private": {"shed": {"WHEAT": 50, "WOOL": 30}},
            "farms": [{"hands": [{"inventory": {"WHEAT": 5}}, {"inventory": {"WOOL": 5}}]}],
            "player": 0
        }
        st = {"money": 5000.0}
        guards = self.engine.evaluate_guards(st, obs)
        self.assertTrue(guards["shed_headroom"])

    def test_slot_arbitration_cap(self):
        # 4 nodes submit 3 orders each (total 12 orders). Engine must strictly cap at 10
        # and prioritize wage_defense > shed_headroom > town_shop > oracle_frontrun
        orders_by_node = {
            "wage_defense": [["SELL", "STRAWBERRY", 5], ["SELL", "FERTILIZER", 10]],
            "shed_headroom": [["SELL", "WOOL", 8], ["SELL", "MILK", 6]],
            "town_shop_preempt": [["BUY_SEED", "CARROT", 2], ["SELL", "CARROT", 4]],
            "oracle_frontrun": [["SELL", "WOOL", 10], ["SELL", "MILK", 10], ["SELL", "STRAWBERRY", 10]],
            "farm_execution": [["HIRE"], ["BUY_LAND"], ["BUY_SEED", "WHEAT", 5]]
        }
        final = self.engine.arbitrate_market_orders(orders_by_node, max_orders=10)
        self.assertEqual(len(final), 10)
        # Wage defense must be at the very front
        self.assertEqual(final[0], ["SELL", "STRAWBERRY", 5])
        self.assertEqual(final[1], ["SELL", "FERTILIZER", 10])
        # Shed headroom follows immediately
        self.assertEqual(final[2], ["SELL", "WOOL", 8])
        self.assertEqual(final[3], ["SELL", "MILK", 6])

    def test_sub_millisecond_latency(self):
        # Benchmark over 5,000 iterations to verify performance
        obs = {
            "hour": 20, "day": 12, "step": 300,
            "private": {"shed": {"WHEAT": 40, "WOOL": 20}},
            "farms": [{"hands": [{"inventory": {"WHEAT": 5}}]}],
            "town": {"unlocked_shops": ["BAKERY"]},
            "player": 0
        }
        st = {
            "money": 450.0,
            "hands": [{}, {}],
            "oracle_scores_4": {"WOOL": 0.35, "MILK": 0.10}
        }
        orders_by_node = {
            "wage_defense": [["SELL", "STRAWBERRY", 2]],
            "shed_headroom": [["SELL", "WHEAT", 10]],
            "oracle_frontrun": [["SELL", "WOOL", 5]]
        }

        t0 = time.perf_counter()
        N = 5000
        for _ in range(N):
            guards = self.engine.evaluate_guards(st, obs)
            final = self.engine.arbitrate_market_orders(orders_by_node)
        elapsed = time.perf_counter() - t0
        per_call_ms = (elapsed / N) * 1000.0
        print(f"\n[BENCHMARK] Average evaluation + arbitration latency: {per_call_ms:.4f} ms per turn")
        self.assertLess(per_call_ms, 0.5, "Graph evaluation must run well under 0.5 ms")


if __name__ == "__main__":
    unittest.main()
