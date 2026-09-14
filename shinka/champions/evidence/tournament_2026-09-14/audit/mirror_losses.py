"""Replay every game Copper Weir lost with the seats swapped. Identical cash both ways = the
world is seat-symmetric and the loss is genuine; a flipped result with shifted cash = the
seed hands one seat a better farm and the loss is a seat artefact of that seed."""
import os, sys, json, time, multiprocessing
from concurrent.futures import ProcessPoolExecutor
REPO = "/home/jovyan/kaggriculture"
sys.path.insert(0, f"{REPO}/shinka/evolution")
import evaluate
POOL = f"{REPO}/shinka/champions/pool"
reg = json.load(open(f"{REPO}/shinka/champions/CODENAMES.json"))
PATH = {c["codename"]: f"{POOL}/{os.path.basename(c['file'])}" for c in reg["champions"]}
losses = json.load(open(sys.argv[1]))
def main():
    tasks = []
    for r in losses:   # mirrored: CW takes the other seat
        cw, op = PATH["Copper Weir"], PATH[r["opp"]]
        tasks.append((op, cw, r["seed"]) if r["seat"] == 0 else (cw, op, r["seed"]))
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=min(104, len(tasks)), mp_context=multiprocessing.get_context("spawn")) as pool:
        res = list(pool.map(evaluate.run_single_match_worker, tasks))
    out = []
    for r, (r0, r1, c0, c1) in zip(losses, res):
        m_cw, m_op = (r1, r0) if r["seat"] == 0 else (r0, r1)   # CW now on the other seat
        out.append(dict(r, mirror_cw=m_cw, mirror_opp=m_op, mirror_seat=1 - r["seat"],
                        identical=(abs(m_cw - r["mine"]) < 0.5 and abs(m_op - r["theirs"]) < 0.5),
                        mirror_result="W" if m_cw > m_op else "L" if m_cw < m_op else "T", crash=c0 or c1))
    print(f"{len(tasks)} mirrored games in {time.time()-t0:.0f}s, crashes {sum(o['crash'] for o in out)}")
    json.dump(out, open(sys.argv[2], "w"), indent=1)
if __name__ == "__main__":
    main()
