"""Exact per-unit market ledger: rerun a game with the engine's _commit_unit wrapped so every
committed unit (step, player, op, item, price) is logged. Output: ledger JSON per game."""
import os, sys, json, importlib.util, multiprocessing
from concurrent.futures import ProcessPoolExecutor
REPO = "/home/jovyan/kaggriculture"; POOL = f"{REPO}/shinka/champions/pool"
OUT = "/results/kagg/tournament_2026-09-14/traces"
reg = json.load(open(f"{REPO}/shinka/champions/CODENAMES.json"))
PATH = {c["codename"]: f"{POOL}/{os.path.basename(c['file'])}" for c in reg["champions"]}

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod.agent

def worker(args):
    seed, p0, p1, tag = args
    import kaggle_environments.envs.kaggriculture.kaggriculture as K
    from kaggle_environments import make
    ledger = []; cur = {"step": 0}
    orig_commit = K._commit_unit
    farms_ref = {}
    def commit(op, item, price, farm, private, market, shed_capacity):
        ok = orig_commit(op, item, price, farm, private, market, shed_capacity)
        if ok:
            pid = farms_ref.get(id(farm), -1)
            ledger.append((cur["step"], pid, op, item, price))
        return ok
    K._commit_unit = commit
    orig_market = K._process_market
    def market(state, env):
        farms = state[0].observation.farms
        farms_ref.clear()
        for i, f in enumerate(farms): farms_ref[id(f)] = i
        cur["step"] = int(state[0].observation.get("step", 0))
        return orig_market(state, env)
    K._process_market = market
    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed})
    env.run([load(PATH[p0], "mod0"), load(PATH[p1], "mod1")])
    s = env.steps[-1]
    json.dump(ledger, open(f"{OUT}/{tag}_ledger.json", "w"))
    return tag, float(s[0].reward), float(s[1].reward), len(ledger)

def main():
    jobs = []
    for seed in (12316, 12720):
        for opp in ("Cider Ridge", "Quiet Barley"):
            for me in ("Copper Weir", "Open Sluice"):
                jobs.append((seed, opp, me, f"{seed}_{me.replace(' ', '')}_vs_{opp.replace(' ', '')}"))
    with ProcessPoolExecutor(max_workers=len(jobs), mp_context=multiprocessing.get_context("spawn")) as pool:
        for tag, r0, r1, n in pool.map(worker, jobs):
            print(f"{tag:40} seat0 {r0:>9,.0f} seat1 {r1:>9,.0f} ledger {n} units")

if __name__ == "__main__":
    main()
