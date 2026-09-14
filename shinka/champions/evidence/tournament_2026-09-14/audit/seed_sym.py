"""Is a seed seat-symmetric? Play two deterministic, seat-agnostic policies both ways."""
import sys, json, multiprocessing
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, "/home/jovyan/kaggriculture/shinka/evolution")
import evaluate
POOL = "/home/jovyan/kaggriculture/shinka/champions/pool"
A = f"{POOL}/champ_20260908_191956_avg91958.py"   # Cider Ridge
B = f"{POOL}/champ_20260914_152040_avg82067.py"   # Quiet Barley
SEEDS = [11205, 12821, 152841, 12316, 11609, 130310]
def main():
    tasks = [(A, B, s) for s in SEEDS] + [(B, A, s) for s in SEEDS]
    with ProcessPoolExecutor(max_workers=len(tasks), mp_context=multiprocessing.get_context("spawn")) as pool:
        res = list(pool.map(evaluate.run_single_match_worker, tasks))
    n = len(SEEDS)
    for i, s in enumerate(SEEDS):
        r0, r1, _, _ = res[i]; m0, m1, _, _ = res[n + i]      # mirrored: B on seat 0
        print(f"seed {s:>6}: CR@0 {r0:>9,.0f} QB@1 {r1:>9,.0f} | mirrored QB@0 {m0:>9,.0f} CR@1 {m1:>9,.0f} | "
              f"{'symmetric' if abs(r0 - m1) < 0.5 and abs(r1 - m0) < 0.5 else f'SEAT-ASYMMETRIC (CR {m1 - r0:+,.0f}, QB {m0 - r1:+,.0f} when moved)'}")
if __name__ == "__main__":
    main()
