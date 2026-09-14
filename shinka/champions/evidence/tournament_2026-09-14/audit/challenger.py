#!/usr/bin/env python3
"""Paired check of a modified program against the tournament: play it against every pool
champion on the stage-1 seeds, in the seats the reference champion had, and compare game by
game with the reference's stage-1 record from tournament.json.

  python challenger.py --program <file> --reference "Copper Weir" --out DIR
"""
import os, sys, json, time, argparse, collections, multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import tournament as T

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--program", required=True); ap.add_argument("--reference", default="Copper Weir")
    ap.add_argument("--tournament", default=os.path.join(os.path.dirname(HERE), "tournament.json"))
    ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=108)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    champs = T.load_champions(); names = [c[0] for c in champs]
    order = {n: k for k, n in enumerate(names)}
    ref = args.reference
    tour = json.load(open(args.tournament))
    # reference's stage-1 games keyed by (opponent, seed) -> (ref cash, opp cash, ref seat)
    refg = {}
    for g in tour["games"]:
        if g["stage"] != 1 or ref not in (g["a"], g["b"]): continue
        if g["a"] == ref: refg[(g["b"], g["seed"])] = (g["ra"], g["rb"], g["a_seat"])
        else: refg[(g["a"], g["seed"])] = (g["rb"], g["ra"], 1 - g["a_seat"])
    tasks, meta = [], []
    for name, base, path, era in champs:
        if name == ref:
            # the reference never played itself: give it the seats of the pool-order convention
            seat_for = lambda seed: 1 if seed in T.STAGE1_SEAT0 else 0
        else:
            seat_for = lambda seed, n=name: refg[(n, seed)][2]
        for seed in T.STAGE1_SEAT0 + T.STAGE1_SEAT1:
            s = seat_for(seed)
            tasks.append((args.program, path, seed) if s == 0 else (path, args.program, seed)); meta.append((name, seed, s))
    print(f"{len(tasks)} games: {os.path.basename(args.program)} vs {len(champs)} champions, {args.workers} workers", flush=True)
    t0 = time.time(); res = [None] * len(tasks); done = 0
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
        futs = {pool.submit(T.evaluate.run_single_match_worker, t): i for i, t in enumerate(tasks)}
        for f in as_completed(futs):
            i = futs[f]; r0, r1, c0, c1 = f.result(); name, seed, s = meta[i]
            mine, theirs = (r0, r1) if s == 0 else (r1, r0); cm, ct = (c0, c1) if s == 0 else (c1, c0)
            res[i] = dict(opp=name, seed=seed, seat=s, mine=mine, theirs=theirs, crash_mine=cm, crash_opp=ct)
            done += 1
            if done % 100 == 0: print(f"  {done}/{len(tasks)} {time.time()-t0:,.0f}s", flush=True)
    wall = time.time() - t0
    # ---- report ---------------------------------------------------------------------------------
    by = collections.defaultdict(list)
    for r in res: by[r["opp"]].append(r)
    lines = [f"challenger {args.program} vs the pool on the stage-1 seeds ({len(tasks)} games, {wall:,.0f} s); reference {ref}",
             f"{'opponent':16} {'new W-L-T':>10} {'ref W-L-T':>10} {'flips +':>7} {'flips -':>7} {'new cash':>9} {'ref cash':>9} {'opp cash new/ref':>18}"]
    tw = tl = tt = rw = rl = rt = 0; tot_flip_p = tot_flip_m = 0; crashes = 0
    for name in names:
        rs = by[name]; w = sum(r["mine"] > r["theirs"] for r in rs); l = sum(r["mine"] < r["theirs"] for r in rs); t = len(rs) - w - l
        crashes += sum(r["crash_mine"] for r in rs)
        if name == ref:
            lines.append(f"{name:16} {f'{w}-{l}-{t}':>10} {'(self)':>10} {'':>7} {'':>7} {sum(r['mine'] for r in rs)/len(rs):>9,.0f} {sum(r['theirs'] for r in rs)/len(rs):>9,.0f}")
            continue
        rw_ = sum(refg[(name, r["seed"])][0] > refg[(name, r["seed"])][1] for r in rs); rl_ = sum(refg[(name, r["seed"])][0] < refg[(name, r["seed"])][1] for r in rs); rt_ = len(rs) - rw_ - rl_
        fp = sum((r["mine"] > r["theirs"]) and not (refg[(name, r["seed"])][0] > refg[(name, r["seed"])][1]) for r in rs)
        fm = sum((r["mine"] < r["theirs"]) and not (refg[(name, r["seed"])][0] < refg[(name, r["seed"])][1]) for r in rs)
        tw += w; tl += l; tt += t; rw += rw_; rl += rl_; rt += rt_; tot_flip_p += fp; tot_flip_m += fm
        lines.append(f"{name:16} {f'{w}-{l}-{t}':>10} {f'{rw_}-{rl_}-{rt_}':>10} {fp:>7} {fm:>7} {sum(r['mine'] for r in rs)/len(rs):>9,.0f} {sum(refg[(name, r['seed'])][0] for r in rs)/len(rs):>9,.0f} {sum(r['theirs'] for r in rs)/len(rs):>8,.0f}/{sum(refg[(name, r['seed'])][1] for r in rs)/len(rs):>8,.0f}")
    lines.append(f"{'TOTAL vs 14':16} {f'{tw}-{tl}-{tt}':>10} {f'{rw}-{rl}-{rt}':>10} {tot_flip_p:>7} {tot_flip_m:>7}   crashes {crashes}")
    # per-seed flips
    flips = [(r["opp"], r["seed"], r["seat"], r["mine"], r["theirs"], refg[(r["opp"], r["seed"])][0], refg[(r["opp"], r["seed"])][1]) for r in res if r["opp"] != ref
             and (r["mine"] > r["theirs"]) != (refg[(r["opp"], r["seed"])][0] > refg[(r["opp"], r["seed"])][1])]
    lines.append(f"\nflipped games ({len(flips)}): opponent, seed, seat, new (mine vs opp), ref (mine vs opp)")
    for f in sorted(flips, key=lambda x: (x[1], x[0])):
        lines.append(f"  {f[0]:16} {f[1]:>7} s{f[2]}  new {f[3]:>9,.0f} vs {f[4]:>9,.0f} {'W' if f[3] > f[4] else 'L'}   ref {f[5]:>9,.0f} vs {f[6]:>9,.0f} {'W' if f[5] > f[6] else 'L'}")
    text = "\n".join(lines); print(text, flush=True)
    open(os.path.join(args.out, "challenger.txt"), "w").write(text + "\n")
    json.dump(dict(program=args.program, reference=ref, games=res, wall_s=wall), open(os.path.join(args.out, "challenger.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
