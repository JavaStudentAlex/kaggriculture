import json, sys
sys.path.insert(0, "/home/jovyan/kaggriculture/research/opponent_model")
from mechanics import PRODUCTS
T = "/results/kagg/tournament_2026-09-14/traces"; ME = 1
tagA, tagB = sys.argv[1], sys.argv[2]; thr = float(sys.argv[3]) if len(sys.argv) > 3 else 50
A = json.load(open(f"{T}/{tagA}.json"))["steps"]; B = json.load(open(f"{T}/{tagB}.json"))["steps"]
m = lambda S, t: S[t][0]["observation"]["farms"][ME]["money"]
sells = lambda S, t: [o for o in ((S[t][ME].get("action") or {}).get("market") or []) if o[0] in ("SELL", "BUY_PRODUCT")]
nonsell = lambda S, t: [o[0] for o in ((S[t][ME].get("action") or {}).get("market") or []) if o[0] not in ("SELL", "BUY_PRODUCT")]
shed = lambda S, t: {k: v for k, v in ((S[t][ME]["observation"].get("private") or {}).get("shed") or {}).items() if k in PRODUCTS and v}
px = lambda S, t: S[t][0]["observation"]["market"]["prices"]
cum = 0
for t in range(313, 719):
    dA, dB = m(A, t + 1) - m(A, t), m(B, t + 1) - m(B, t)
    cum += dA - dB
    if abs(dA - dB) >= thr or sells(A, t + 1) != sells(B, t + 1):
        print(f"t={t:3d} d{t//24:<2}h{t%24:<2} dCW {dA:>+7,.0f} dOS {dB:>+7,.0f} diff {dA-dB:>+6,.0f} cum {cum:>+7,.0f}")
        if sells(A, t + 1) != sells(B, t + 1) or abs(dA - dB) >= thr:
            print(f"      CW: {sells(A, t+1)} +{len(nonsell(A, t+1))} other | shed {shed(A, t)}")
            print(f"      OS: {sells(B, t+1)} +{len(nonsell(B, t+1))} other | shed {shed(B, t)}")
            p = px(A, t); print(f"      prices: " + " ".join(f"{k[:4]}{v}" for k, v in p.items()))
