#!/usr/bin/env python3
"""Round-robin tie-break among the top run-2 programs on FRESH seeds.

Same protocol as evaluate.py (20 seeds with A on seat 0 + 20 disjoint seeds with
B on seat 0 per pairing) but on seed blocks the evolution never saw, so a program
that over-fitted the evaluator's fixed seeds gets no credit for it.
Run under the CUDA venv with the eval_once.sh environment.
"""
import os, sys, json, time, itertools, collections, math
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

EVO = "/home/jovyan/kaggriculture/shinka/evolution"
sys.path.insert(0, EVO)
import evaluate  # noqa: E402  (needs KAGG_* env set by the caller)

R = "/results/kagg/shinka_results"
POOL = "/home/jovyan/kaggriculture/shinka/champions/pool"
PROGS = {
    "r1g62_seed": f"{POOL}/champ_20260912_gen62_avg82350.py",
    "g33_crowned": f"{R}/gen_33/main.py",
    "g55": f"{R}/gen_55/main.py",
    "g60": f"{R}/gen_60/main.py",
    "g81": f"{R}/gen_81/main.py",
    "g82": f"{R}/gen_82/main.py",
    "g88": f"{R}/gen_88/main.py",
}
# fresh, disjoint from evaluate.py's 101..2020 / 70102..72021
SEAT0 = [5003 + 101 * i for i in range(1, 21)]
SEAT1 = [90007 + 101 * i for i in range(1, 21)]
assert not (set(SEAT0) | set(SEAT1)) & (set(evaluate.SEAT0_SEEDS) | set(evaluate.SEAT1_SEEDS))
OUT = sys.argv[1] if len(sys.argv) > 1 else "/results/kagg/eval_tiebreak_run2"
os.makedirs(OUT, exist_ok=True)


def main():
    names = list(PROGS)
    tasks, meta = [], []
    for a, b in itertools.combinations(names, 2):
        for s in SEAT0:
            tasks.append((PROGS[a], PROGS[b], s)); meta.append((a, b, 0))
        for s in SEAT1:
            tasks.append((PROGS[b], PROGS[a], s)); meta.append((a, b, 1))
    print(f"{len(tasks)} games, {len(names)} programs, {evaluate.WORKERS} workers", flush=True)
    t0 = time.time()
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=evaluate.WORKERS, mp_context=ctx) as pool:
        results = list(pool.map(evaluate.run_single_match_worker, tasks, chunksize=4))
    wall = time.time() - t0

    pair = collections.defaultdict(lambda: dict(w=0, l=0, t=0))
    agg = {n: dict(w=0, l=0, t=0, cash=0.0, g=0, w0=0, g0=0, w1=0, g1=0, crash=0) for n in names}
    games = []
    for (a, b, a_seat), (r0, r1, c0, c1) in zip(meta, results):
        ra, rb = (r0, r1) if a_seat == 0 else (r1, r0)
        ca, cb = (c0, c1) if a_seat == 0 else (c1, c0)
        games.append(dict(a=a, b=b, a_seat=a_seat, ra=ra, rb=rb, crash_a=ca, crash_b=cb))
        agg[a]["crash"] += ca; agg[b]["crash"] += cb
        for x, rx, ry, seat in ((a, ra, rb, a_seat), (b, rb, ra, 1 - a_seat)):
            agg[x]["g"] += 1; agg[x]["cash"] += rx
            agg[x]["g0" if seat == 0 else "g1"] += 1
            if rx > ry:
                agg[x]["w"] += 1; agg[x]["w0" if seat == 0 else "w1"] += 1
            elif rx < ry:
                agg[x]["l"] += 1
            else:
                agg[x]["t"] += 1
        k = (a, b)
        if ra > rb: pair[k]["w"] += 1
        elif ra < rb: pair[k]["l"] += 1
        else: pair[k]["t"] += 1

    # Bradley-Terry over the round-robin (ties = half a win each)
    W = collections.defaultdict(float); N = collections.defaultdict(float)
    for g in games:
        a, b = g["a"], g["b"]
        N[(a, b)] += 1; N[(b, a)] += 1
        if g["ra"] > g["rb"]: W[a] += 1
        elif g["rb"] > g["ra"]: W[b] += 1
        else: W[a] += 0.5; W[b] += 0.5
    p = {n: 1.0 for n in names}
    for _ in range(2000):
        new = {}
        for i in names:
            den = sum(N[(i, j)] / (p[i] + p[j]) for j in names if j != i)
            new[i] = (W[i] / den) if den > 0 and W[i] > 0 else 1e-9
        s = sum(new.values()) / len(new)
        p = {k: v / s for k, v in new.items()}
    ref = max(p.values())
    elo = {n: 400 * math.log10(max(p[n], 1e-12) / ref) for n in names}

    lines = [f"Tie-break round-robin on fresh seeds: {len(tasks)} games, {wall:.0f} s wall",
             f"seat-0 seeds {SEAT0[0]}..{SEAT0[-1]}, seat-1 seeds {SEAT1[0]}..{SEAT1[-1]} (20 + 20 per pairing)", "",
             f"{'program':12} {'W':>4} {'L':>4} {'T':>3} {'WR%':>6} {'seat0%':>7} {'seat1%':>7} {'avg cash':>9} {'BT-Elo':>7} {'crash':>5}"]
    for n in sorted(names, key=lambda x: -elo[x]):
        s = agg[n]
        lines.append(f"{n:12} {s['w']:>4} {s['l']:>4} {s['t']:>3} {100*s['w']/s['g']:>6.1f} "
                     f"{100*s['w0']/max(1,s['g0']):>7.1f} {100*s['w1']/max(1,s['g1']):>7.1f} "
                     f"{s['cash']/s['g']:>9,.0f} {elo[n]:>7.0f} {s['crash']:>5}")
    lines += ["", "head-to-head (row vs column, W-L-T from the row's side):", " " * 12 + "".join(f"{n:>13}" for n in names)]
    for a in names:
        row = f"{a:12}"
        for b in names:
            if a == b: row += f"{'-':>13}"; continue
            k = (a, b) if (a, b) in pair else (b, a)
            d = pair[k]
            w, l, t = (d["w"], d["l"], d["t"]) if k == (a, b) else (d["l"], d["w"], d["t"])
            row += f"{f'{w}-{l}-{t}':>13}"
        lines.append(row)
    text = "\n".join(lines)
    print(text, flush=True)
    json.dump(dict(programs=PROGS, seat0_seeds=SEAT0, seat1_seeds=SEAT1, wall_s=wall, aggregate=agg,
                   bt_elo=elo, pairs={f"{a}|{b}": v for (a, b), v in pair.items()}, games=games, text=text),
              open(os.path.join(OUT, "tiebreak.json"), "w"), indent=1)
    open(os.path.join(OUT, "tiebreak.txt"), "w").write(text + "\n")


if __name__ == "__main__":
    main()
