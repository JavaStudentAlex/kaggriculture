#!/usr/bin/env python3
"""Cash at the day-1 hire and wheat left for step-0/1 wheat openings, computed with the engine.

    python opening_cash.py

Both seats buy wheat at step 0 and sell at step 1 (orders at index 0, cleared in lockstep),
then place the backbone route's step-1 orders (7 wheat seeds, 12 melon seeds, 5 hires,
2 cows, 2 sheep). Nothing trades between step 2 and the step-24 hire, so the cash after
step 1 is the cash at the hire: below $7 the route gets 3 hands instead of 4. The route
needs 5 wheat in the shed after step 1 to feed its cows on day 1.
"""
from kaggle_environments import make

ROUTE = ([["BUY_SEED", "WHEAT", 7], ["BUY_SEED", "MELON", 12]] + [["HIRE"]] * 5
         + [["BUY_ANIMAL", "COW", 2], ["BUY_ANIMAL", "SHEEP", 2]])
OPPONENTS = {"35/30 (Hazel, Willow)": (35, 30), "5/0 (plain Mohui)": (5, 0), "13/9 (Mohui13)": (13, 9)}
OURS = [(13, 9), (13, 8), (14, 9), (15, 10), (16, 11), (17, 12), (18, 13), (20, 15), (22, 17),
        (25, 20), (14, 10), (15, 11), (12, 7), (11, 6), (10, 5), (35, 30), (5, 0)]


def after_step_1(us, opp):
    env = make("kaggriculture", configuration={"seed": 1}, debug=False)
    env.reset()
    orders = [({"market": [["BUY_PRODUCT", "WHEAT", buy]] if buy else []},
               {"market": ([["SELL", "WHEAT", sell]] if sell else []) + ROUTE}) for buy, sell in (us, opp)]
    env.step([orders[0][0], orders[1][0]])
    env.step([orders[0][1], orders[1][1]])
    cash = [f["money"] for f in env.state[0].observation.farms]
    wheat = [env.state[i].observation.private["shed"]["WHEAT"] for i in (0, 1)]
    return cash, wheat


def main():
    print("ours   | " + " | ".join(f"{name:>30s}" for name in OPPONENTS))
    for pair in OURS:
        cells = []
        for opp in OPPONENTS.values():
            (mine, theirs), (wheat, _) = after_step_1(pair, opp)
            note = "short" if theirs < 7 else ""
            cells.append(f"us ${mine:3.0f} wheat {wheat} | them ${theirs:3.0f} {note:5s}")
        print(f"{pair[0]:2d}/{pair[1]:<3d} | " + " | ".join(f"{c:>30s}" for c in cells))


if __name__ == "__main__":
    main()
