import json, collections, sys
T = "/results/kagg/tournament_2026-09-14/traces"; ME = 1
def led(tag):
    L = json.load(open(f"{T}/{tag}_ledger.json"))
    tot = [collections.Counter(), collections.Counter()]; units = [collections.Counter(), collections.Counter()]
    day = [collections.Counter(), collections.Counter()]; step = [collections.defaultdict(collections.Counter), collections.defaultdict(collections.Counter)]
    for st, pid, op, item, price in L:
        if pid < 0: continue
        sgn = 1 if op == "SELL" else -1
        key = item if op == "SELL" else f"buy:{item}" if op == "BUY_PRODUCT" else f"buy:{op[4:].lower()}:{item}"
        tot[pid][key] += sgn * price; units[pid][key] += 1; day[pid][st // 24] += sgn * price; step[pid][st][key] += sgn * price
    return tot, units, day, step
def money(tag):
    d = json.load(open(f"{T}/{tag}.json"))["steps"]; return [d[t][0]["observation"]["farms"][ME]["money"] for t in range(720)]
for seed in (12316, 12720):
    for opp in ("CiderRidge", "QuietBarley"):
        a, b = f"{seed}_CopperWeir_vs_{opp}", f"{seed}_OpenSluice_vs_{opp}"
        A, uA, dA, sA = led(a); B, uB, dB, sB = led(b); mA, mB = money(a), money(b)
        netA, netB = sum(A[ME].values()), sum(B[ME].values())
        print(f"\n=== seed {seed} vs {opp}: final CW {mA[-1]:,.0f} OS {mB[-1]:,.0f} ({mA[-1]-mB[-1]:+,.0f}); ledger net CW {netA:,.0f} OS {netB:,.0f} ({netA-netB:+,.0f}); hires etc. = {mA[-1]-3000-netA:,.0f} / {mB[-1]-3000-netB:,.0f}")
        print(f"{'item':22} {'units CW':>8} {'units OS':>8} {'$ CW':>9} {'$ OS':>9} {'$ diff':>7} {'$/u CW':>7} {'$/u OS':>7}")
        for k in sorted(set(A[ME]) | set(B[ME]), key=lambda k: -(abs(A[ME][k]) + abs(B[ME][k]))):
            if abs(A[ME][k] - B[ME][k]) >= 1 or True:
                print(f"{k:22} {uA[ME][k]:>8} {uB[ME][k]:>8} {A[ME][k]:>9,.0f} {B[ME][k]:>9,.0f} {A[ME][k]-B[ME][k]:>+7,.0f} {A[ME][k]/uA[ME][k] if uA[ME][k] else 0:>7.1f} {B[ME][k]/uB[ME][k] if uB[ME][k] else 0:>7.1f}")
        print("net diff by day (CW-OS): " + " ".join(f"d{d}:{dA[ME][d]-dB[ME][d]:+,.0f}" for d in range(30) if abs(dA[ME][d]-dB[ME][d]) >= 10))
        print(f"opponent ledger net: {sum(A[1-ME].values()):,.0f} (vs CW) / {sum(B[1-ME].values()):,.0f} (vs OS)")
        json.dump(dict(cw=dict(tot=A[ME], units=uA[ME], day=dA[ME]), os=dict(tot=B[ME], units=uB[ME], day=dB[ME])), open(f"{T}/{seed}_{opp}_ledger_summary.json", "w"))
