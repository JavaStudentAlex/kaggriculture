#!/usr/bin/env python3
"""Round-robin tournament of the whole champion pool on FRESH seeds.

Protocol per pairing = evaluate.py's (20 seeds with A on seat 0 + 20 disjoint seeds
with B on seat 0), but on seed blocks no evolution run or tie-break ever used, so a
champion that over-fitted the evaluator's fixed seeds gets no credit for it.

Stage 1: every pair of the pool, 40 games -> 15 champions = 105 pairs = 4,200 games.
Stage 2: the top PLAYOFF champions by stage-1 Bradley-Terry rating replay each
         other on a third fresh block (30 + 30 per pair) so the podium rests on
         100 games per pairing instead of 40.
Ranking: Bradley-Terry (ties = half a win) over every game; the bootstrap resamples
each pairing's games to give P(rank 1). Games are logged in evaluate.py's
champion_vs_champion record format so curate_pool.py can consume them.

Run under the CUDA venv with the eval_once.sh environment (run.sh does that).
"""
import os, sys, json, time, random, itertools, collections, math, argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import numpy as np

REPO = os.environ.get("KAGG_TASK_DIR", os.path.expanduser("~/kaggriculture"))
EVO = os.path.join(REPO, "shinka", "evolution")
sys.path.insert(0, EVO)
import evaluate  # noqa: E402  (needs KAGG_* env set by the caller)

POOL = os.path.join(REPO, "shinka", "champions", "pool")
CODENAMES = os.path.join(REPO, "shinka", "champions", "CODENAMES.json")

# Fresh, disjoint from evaluate.py (101..2020 / 70102..72021) and from the run-2
# tie-break (5104..7023 / 90108..92027).
STAGE1_SEAT0 = [11003 + 101 * i for i in range(1, 21)]      # 11104 .. 13023
STAGE1_SEAT1 = [130007 + 101 * i for i in range(1, 21)]     # 130108 .. 132027
STAGE2_SEAT0 = [23011 + 101 * i for i in range(1, 31)]      # 23112 .. 26041
STAGE2_SEAT1 = [150013 + 101 * i for i in range(1, 31)]     # 150114 .. 153043
_RUN2_TIEBREAK = {5003 + 101 * i for i in range(1, 21)} | {90007 + 101 * i for i in range(1, 21)}
_ALL = STAGE1_SEAT0 + STAGE1_SEAT1 + STAGE2_SEAT0 + STAGE2_SEAT1
assert len(set(_ALL)) == len(_ALL), "seed blocks overlap"
assert not set(_ALL) & (set(evaluate.SEAT0_SEEDS) | set(evaluate.SEAT1_SEEDS) | _RUN2_TIEBREAK), "seeds already used"


def load_champions():
    """[(codename, pool file basename, path, era)] for every active pool member."""
    reg = json.load(open(CODENAMES))
    champs = []
    for c in reg["champions"]:
        if not c.get("status", "").startswith("active pool member"):
            continue
        base = os.path.basename(c["file"])
        path = os.path.join(POOL, base)
        if not os.path.exists(path):
            raise SystemExit(f"missing pool file for {c['codename']}: {path}")
        era = "oracle" if ("run 1" in c["origin"] or "run 2" in c["origin"] or "run 3" in c["origin"]) else "pre-oracle"
        champs.append((c["codename"], base, path, era))
    return champs


def abbrev(names):
    """Two-letter initials for the head-to-head matrix (must be unique)."""
    ab = {n: "".join(w[0] for w in n.split()) for n in names}
    assert len(set(ab.values())) == len(ab), ab
    return ab


def build_tasks(champs, seat0_seeds, seat1_seeds, stage):
    """One task per game: (path0, path1, seed) + meta (a, b, a_seat, stage)."""
    tasks, meta = [], []
    for (na, _, pa, _), (nb, _, pb, _) in itertools.combinations(champs, 2):
        for s in seat0_seeds:
            tasks.append((pa, pb, s)); meta.append((na, nb, 0, stage))
        for s in seat1_seeds:
            tasks.append((pb, pa, s)); meta.append((na, nb, 1, stage))
    return tasks, meta


def play(tasks, meta, workers, log_path, by_name, label):
    """Run the games (spawned workers, model loaded once per worker), append every
    result to log_path in evaluate.py's champion_vs_champion format."""
    order = list(range(len(tasks)))
    random.Random(2026_09_14).shuffle(order)          # spread heavy pairings over time
    t0 = time.time()
    games = [None] * len(tasks)
    done = 0
    ctx = multiprocessing.get_context("spawn")
    stamp = time.strftime("%Y%m%dT%H%M%S")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool, open(log_path, "a") as log:
        futs = {pool.submit(evaluate.run_single_match_worker, tasks[i]): i for i in order}
        for fut in as_completed(futs):
            i = futs[fut]
            r0, r1, c0, c1 = fut.result()
            a, b, a_seat, stage = meta[i]
            ra, rb = (r0, r1) if a_seat == 0 else (r1, r0)
            ca, cb = (c0, c1) if a_seat == 0 else (c1, c0)
            games[i] = dict(a=a, b=b, a_seat=a_seat, seed=tasks[i][2], ra=ra, rb=rb,
                            crash_a=ca, crash_b=cb, stage=stage)
            p0, p1, seed = tasks[i]
            log.write(json.dumps({"ts": stamp, "kind": "champion_vs_champion", "stage": stage,
                                  "a": os.path.basename(p0), "b": os.path.basename(p1), "seed": seed,
                                  "ra": r0, "rb": r1, "crash_a": c0, "crash_b": c1}) + "\n")
            done += 1
            if done % 100 == 0 or done == len(tasks):
                el = time.time() - t0
                print(f"[{label}] {done}/{len(tasks)} games  {el:,.0f}s elapsed  "
                      f"ETA {el / done * (len(tasks) - done):,.0f}s", flush=True)
    return games, time.time() - t0


def bt_fit(names, games, iters=500):
    """Bradley-Terry by MM (Hunter 2004); ties count half. Returns Elo-like scale
    with the leader at 0, plus the W/N matrices."""
    idx = {n: k for k, n in enumerate(names)}
    n = len(names)
    W = np.zeros((n, n)); N = np.zeros((n, n))
    for g in games:
        if g["crash_a"] or g["crash_b"]:
            continue
        i, j = idx[g["a"]], idx[g["b"]]
        N[i, j] += 1; N[j, i] += 1
        if g["ra"] > g["rb"]: W[i, j] += 1
        elif g["rb"] > g["ra"]: W[j, i] += 1
        else: W[i, j] += 0.5; W[j, i] += 0.5
    return bt_from_matrix(W, N, iters), W, N


def bt_from_matrix(W, N, iters=500):
    n = W.shape[0]
    p = np.ones(n)
    wins = W.sum(1)
    for _ in range(iters):
        den = (N / (p[:, None] + p[None, :])).sum(1)
        new = np.where((den > 0) & (wins > 0), wins / np.maximum(den, 1e-12), 1e-9)
        p = new / new.mean()
    return 400.0 * np.log10(np.maximum(p, 1e-12) / p.max())


def bootstrap_rank1(names, games, n_boot=2000, seed=0):
    """Resample each pairing's games with replacement, refit, count who leads."""
    idx = {n: k for k, n in enumerate(names)}
    n = len(names)
    per_pair = collections.defaultdict(list)     # (i, j) -> outcomes for i: 1 / 0.5 / 0
    for g in games:
        if g["crash_a"] or g["crash_b"]:
            continue
        i, j = idx[g["a"]], idx[g["b"]]
        per_pair[(i, j)].append(1.0 if g["ra"] > g["rb"] else 0.0 if g["rb"] > g["ra"] else 0.5)
    pairs = [(k, np.array(v)) for k, v in per_pair.items()]
    rng = np.random.default_rng(seed)
    top = np.zeros(n); rank_sum = np.zeros(n)
    for _ in range(n_boot):
        W = np.zeros((n, n)); N = np.zeros((n, n))
        for (i, j), v in pairs:
            s = rng.choice(v, size=len(v), replace=True)
            W[i, j] += s.sum(); W[j, i] += len(s) - s.sum()
            N[i, j] += len(s); N[j, i] += len(s)
        elo = bt_from_matrix(W, N, iters=200)
        order = np.argsort(-elo)
        top[order[0]] += 1
        ranks = np.empty(n); ranks[order] = np.arange(1, n + 1)
        rank_sum += ranks
    return top / n_boot, rank_sum / n_boot


def aggregate(names, games, restrict=None):
    """Per-champion W/L/T, seat split, cash over the given games (optionally only
    games where both sides are in `restrict`). Pair records are keyed in `names`
    order whichever side was "a" in the game (stage 2 builds its pairings in
    finalist order, stage 1 in pool order)."""
    agg = {n: dict(w=0, l=0, t=0, g=0, cash=0.0, w0=0, g0=0, w1=0, g1=0, crash=0) for n in names}
    pair = collections.defaultdict(lambda: dict(w=0, l=0, t=0))
    rank = {n: k for k, n in enumerate(names)}
    for g in games:
        a, b = g["a"], g["b"]
        if restrict is not None and (a not in restrict or b not in restrict):
            continue
        if rank[a] > rank[b]:
            g = dict(g, a=b, b=a, ra=g["rb"], rb=g["ra"], crash_a=g["crash_b"], crash_b=g["crash_a"],
                     a_seat=1 - g["a_seat"])
            a, b = b, a
        agg[a]["crash"] += g["crash_a"]; agg[b]["crash"] += g["crash_b"]
        if g["crash_a"] or g["crash_b"]:
            continue
        for x, rx, ry, seat in ((a, g["ra"], g["rb"], g["a_seat"]), (b, g["rb"], g["ra"], 1 - g["a_seat"])):
            s = agg[x]
            s["g"] += 1; s["cash"] += rx; s["g0" if seat == 0 else "g1"] += 1
            if rx > ry:
                s["w"] += 1; s["w0" if seat == 0 else "w1"] += 1
            elif rx < ry:
                s["l"] += 1
            else:
                s["t"] += 1
        d = pair[(a, b)]
        if g["ra"] > g["rb"]: d["w"] += 1
        elif g["ra"] < g["rb"]: d["l"] += 1
        else: d["t"] += 1
    return agg, pair


def wilson(w, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = w / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def table(title, names, agg, elo, p_top=None, mean_rank=None, note=None):
    lines = [title]
    if note:
        lines.append(note)
    hdr = f"{'#':>2} {'champion':16} {'W':>4} {'L':>4} {'T':>3} {'WR%':>6} {'95% CI':>13} {'seat0%':>7} {'seat1%':>7} {'avg cash':>9} {'BT-Elo':>7}"
    if p_top is not None:
        hdr += f" {'P(#1)':>6} {'E[rank]':>7}"
    hdr += f" {'crash':>5}"
    lines.append(hdr)
    order = sorted(range(len(names)), key=lambda k: -elo[k])
    for rank, k in enumerate(order, 1):
        n = names[k]; s = agg[n]
        lo, hi = wilson(s["w"], max(1, s["g"]))
        row = (f"{rank:>2} {n:16} {s['w']:>4} {s['l']:>4} {s['t']:>3} {100 * s['w'] / max(1, s['g']):>6.1f} "
               f"{100 * lo:>5.1f}-{100 * hi:<5.1f}  {100 * s['w0'] / max(1, s['g0']):>7.1f} {100 * s['w1'] / max(1, s['g1']):>7.1f} "
               f"{s['cash'] / max(1, s['g']):>9,.0f} {elo[k]:>7.0f}")
        if p_top is not None:
            row += f" {100 * p_top[k]:>5.1f}% {mean_rank[k]:>7.2f}"
        row += f" {s['crash']:>5}"
        lines.append(row)
    return lines


def matrix(names, ab, pair, order):
    lines = ["head-to-head (row vs column, W-L-T from the row's side; games of both stages):",
             " " * 19 + "".join(f"{ab[n]:>9}" for n in order)]
    for a in order:
        row = f"{a:16} {ab[a]:>2}"
        for b in order:
            if a == b:
                row += f"{'-':>9}"; continue
            if (a, b) in pair:
                d = pair[(a, b)]; w, l, t = d["w"], d["l"], d["t"]
            elif (b, a) in pair:
                d = pair[(b, a)]; w, l, t = d["l"], d["w"], d["t"]
            else:
                row += f"{'':>9}"; continue
            row += f"{f'{w}-{l}' + (f'-{t}' if t else ''):>9}"
        lines.append(row)
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/results/kagg/tournament_2026-09-14")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("KAGG_WORKERS", "108")))
    ap.add_argument("--playoff", type=int, default=5, help="champions in stage 2 (0 = skip)")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--from-json", help="re-render the report from a saved tournament.json (no games played)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, "games.jsonl")

    champs = load_champions()
    names = [c[0] for c in champs]
    by_name = {c[0]: c for c in champs}
    ab = abbrev(names)
    print(f"{len(champs)} champions: " + ", ".join(f"{n} ({ab[n]})" for n in names), flush=True)

    if args.from_json:
        saved = json.load(open(args.from_json))
        games1 = [g for g in saved["games"] if g["stage"] == 1]
        games2 = [g for g in saved["games"] if g["stage"] == 2]
        wall1, wall2 = saved["wall_s"]["stage1"], saved["wall_s"]["stage2"]
        finalists = saved["finalists"]
    else:
        # ---- stage 1: full round-robin ---------------------------------------------
        tasks, meta = build_tasks(champs, STAGE1_SEAT0, STAGE1_SEAT1, 1)
        print(f"stage 1: {len(tasks)} games over {len(tasks) // 40} pairings, {args.workers} workers", flush=True)
        games1, wall1 = play(tasks, meta, args.workers, log_path, by_name, "stage 1")
        elo1, _, _ = bt_fit(names, games1)
        agg1, pair1 = aggregate(names, games1)
        json.dump(dict(names=names, games=games1, wall_s=wall1, elo=dict(zip(names, elo1.tolist()))),
                  open(os.path.join(args.out, "stage1.json"), "w"))
        order1 = [names[k] for k in np.argsort(-elo1)]
        print("\n".join(table(f"stage 1: {len(games1)} games, {wall1:,.0f} s wall", names, agg1, elo1)), flush=True)

        # ---- stage 2: playoff among the top ----------------------------------------
        games2, wall2, finalists = [], 0.0, []
        if args.playoff and args.playoff >= 2:
            finalists = order1[:args.playoff]
            tasks2, meta2 = build_tasks([by_name[n] for n in finalists], STAGE2_SEAT0, STAGE2_SEAT1, 2)
            print(f"\nstage 2: {', '.join(finalists)} -> {len(tasks2)} games", flush=True)
            games2, wall2 = play(tasks2, meta2, min(args.workers, len(tasks2)), log_path, by_name, "stage 2")

    # ---- ranking over everything ---------------------------------------------------
    games = games1 + games2
    elo, W, N = bt_fit(names, games)
    p_top, mean_rank = bootstrap_rank1(names, games, n_boot=args.boot)
    agg, pair = aggregate(names, games)
    order = [names[k] for k in np.argsort(-elo)]
    oracle = [n for n in names if by_name[n][3] == "oracle"]
    agg_o, _ = aggregate(names, games, restrict=set(oracle))
    elo_o, _, _ = bt_fit(oracle, [g for g in games if g["a"] in oracle and g["b"] in oracle])

    lines = [f"Champion tournament on fresh seeds, {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}",
             f"stage 1: {len(games1)} games ({len(names)} champions, 40 per pairing: seat-0 seeds "
             f"{STAGE1_SEAT0[0]}..{STAGE1_SEAT0[-1]}, seat-1 seeds {STAGE1_SEAT1[0]}..{STAGE1_SEAT1[-1]}), {wall1:,.0f} s wall"]
    if games2:
        lines.append(f"stage 2: {len(games2)} games ({len(finalists)} finalists, 60 more per pairing: seat-0 seeds "
                     f"{STAGE2_SEAT0[0]}..{STAGE2_SEAT0[-1]}, seat-1 seeds {STAGE2_SEAT1[0]}..{STAGE2_SEAT1[-1]}), {wall2:,.0f} s wall")
    crashes = sum(g["crash_a"] + g["crash_b"] for g in games)
    lines += [f"crashes: {crashes}", ""]
    lines += table("FINAL RANKING (Bradley-Terry over every game; ties = half a win; bootstrap "
                   f"{args.boot} resamples per pairing)", names, agg, elo, p_top, mean_rank)
    lines.append("")
    lines += table(f"among the {len(oracle)} oracle-era champions only (games between them)", oracle,
                   agg_o, elo_o)
    if games2:
        lines.append("")
        agg_f, _ = aggregate(names, games, restrict=set(finalists))
        elo_f, _, _ = bt_fit(finalists, [g for g in games if g["a"] in finalists and g["b"] in finalists])
        lines += table(f"finalists head-to-head only (100 games per pairing)", finalists, agg_f, elo_f)
    lines.append("")
    lines += matrix(names, ab, pair, order)
    best = order[0]
    runner = order[1]
    k0, k1 = names.index(best), names.index(runner)
    lines += ["", f"BEST: {best} -- BT-Elo lead of {elo[k0] - elo[k1]:.0f} over {runner}, "
                  f"P(#1) = {100 * p_top[k0]:.1f}% (bootstrap), "
                  f"{agg[best]['w']}W-{agg[best]['l']}L-{agg[best]['t']}T overall"]
    text = "\n".join(lines)
    print("\n" + text, flush=True)
    open(os.path.join(args.out, "tournament.txt"), "w").write(text + "\n")
    json.dump(dict(champions=[dict(codename=c[0], file=c[1], era=c[3]) for c in champs],
                   stage1_seeds=dict(seat0=STAGE1_SEAT0, seat1=STAGE1_SEAT1),
                   stage2_seeds=dict(seat0=STAGE2_SEAT0, seat1=STAGE2_SEAT1),
                   finalists=finalists, wall_s=dict(stage1=wall1, stage2=wall2),
                   ranking=order, bt_elo=dict(zip(names, elo.tolist())),
                   p_rank1=dict(zip(names, p_top.tolist())), mean_rank=dict(zip(names, mean_rank.tolist())),
                   aggregate=agg, pairs={f"{a}|{b}": v for (a, b), v in pair.items()},
                   games=games, text=text),
              open(os.path.join(args.out, "tournament.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
