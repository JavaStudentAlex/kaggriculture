"""Replay chosen games with the full step record kept (env.toJSON) for trace diffs."""
import os, sys, json, time, multiprocessing, importlib.util
from concurrent.futures import ProcessPoolExecutor
REPO = "/home/jovyan/kaggriculture"
POOL = f"{REPO}/shinka/champions/pool"
OUT = "/results/kagg/tournament_2026-09-14/traces"
reg = json.load(open(f"{REPO}/shinka/champions/CODENAMES.json"))
PATH = {c["codename"]: f"{POOL}/{os.path.basename(c['file'])}" for c in reg["champions"]}

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod.agent

def worker(args):
    seed, p0, p1, tag = args
    from kaggle_environments import make
    a0, a1 = load(PATH[p0], "mod0"), load(PATH[p1], "mod1")
    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed})
    env.run([a0, a1])
    j = env.toJSON()
    out = f"{OUT}/{tag}.json"
    json.dump(j, open(out, "w"))
    s = env.steps[-1]
    return tag, float(s[0].reward), float(s[1].reward), os.path.getsize(out)

def main():
    jobs = []
    for seed in (12316, 12720):
        for opp in ("Cider Ridge", "Quiet Barley"):
            for me in ("Copper Weir", "Open Sluice"):
                jobs.append((seed, opp, me, f"{seed}_{me.replace(' ', '')}_vs_{opp.replace(' ', '')}"))
        jobs.append((seed, "Open Sluice", "Copper Weir", f"{seed}_CopperWeir_vs_OpenSluice_direct"))
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=len(jobs), mp_context=multiprocessing.get_context("spawn")) as pool:
        for tag, r0, r1, sz in pool.map(worker, jobs):
            print(f"{tag:45} seat0 {r0:>9,.0f}  seat1 {r1:>9,.0f}  ({sz/1e6:.1f} MB)")
    print(f"{len(jobs)} games in {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
