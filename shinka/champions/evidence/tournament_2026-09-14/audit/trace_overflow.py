"""Count units discarded by a full shed (DROP and end-of-day drop) per player, per product."""
import os, sys, json, importlib.util, multiprocessing, collections
from concurrent.futures import ProcessPoolExecutor
REPO = "/home/jovyan/kaggriculture"; POOL = f"{REPO}/shinka/champions/pool"
reg = json.load(open(f"{REPO}/shinka/champions/CODENAMES.json"))
PATH = {c["codename"]: f"{POOL}/{os.path.basename(c['file'])}" for c in reg["champions"]}
def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod.agent
def worker(args):
    seed, p0, p1, tag = args
    import kaggle_environments.envs.kaggriculture.kaggriculture as K
    from kaggle_environments import make
    lost = [collections.Counter(), collections.Counter()]; events = []
    privs = {}; cur = {"step": 0}
    orig_interp = K.interpreter
    orig_apply = K._apply_unit_action
    def apply(farm, private, idx, action, board_size, day, turns_per_day, shed_capacity=100):
        pid = privs.get(id(private), -1)
        if isinstance(action, list) and action and action[0] == "DROP" and pid >= 0:
            pos = tuple(farm["farmer"]) if idx == 0 else tuple(farm["hands"][idx - 1])
            if K._is_shed_adjacent(pos, board_size):
                inv = private["inventories"][idx]; room = max(0, shed_capacity - sum(private["shed"].values()))
                for item, n in inv.items():
                    take = min(n, room); room -= take
                    if n - take > 0: lost[pid][item] += n - take; events.append((cur["step"], pid, item, n - take, "drop"))
        return orig_apply(farm, private, idx, action, board_size, day, turns_per_day, shed_capacity)
    K._apply_unit_action = apply
    orig_eod = K._drop_inventories_to_shed
    def eod(private, capacity):
        pid = privs.get(id(private), -1); room = max(0, capacity - sum(private["shed"].values()))
        for inv in private["inventories"]:
            for item, n in inv.items():
                take = min(n, room); room -= take
                if n - take > 0 and pid >= 0: lost[pid][item] += n - take; events.append((cur["step"], pid, item, n - take, "eod"))
        return orig_eod(private, capacity)
    K._drop_inventories_to_shed = eod
    orig_market = K._process_market
    def market(state, env):
        return orig_market(state, env)
    def tag_privs(state):
        for i, s in enumerate(state): privs[id(s.observation.private)] = i
        cur["step"] = int(state[0].observation.get("step", 0))
    # the interpreter is bound at registration, so tag through _process_market's caller: wrap _apply via state? simplest: tag at each market call AND at first apply via env hook
    K._process_market = lambda state, env: (tag_privs(state), orig_market(state, env))[1]
    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed})
    # tag before the first step too
    agents = [load(PATH[p0], "mod0"), load(PATH[p1], "mod1")]
    env.run(agents)
    s = env.steps[-1]
    return tag, float(s[0].reward), float(s[1].reward), [dict(l) for l in lost], events
def main():
    jobs = []
    for seed in (12316, 12720):
        for opp in ("Cider Ridge", "Quiet Barley"):
            for me in ("Copper Weir", "Open Sluice"):
                jobs.append((seed, opp, me, f"{seed}_{me.replace(' ', '')}_vs_{opp.replace(' ', '')}"))
    with ProcessPoolExecutor(max_workers=len(jobs), mp_context=multiprocessing.get_context("spawn")) as pool:
        for tag, r0, r1, lost, ev in pool.map(worker, jobs):
            print(f"{tag:40} seat1 {r1:>9,.0f} | discarded by a full shed: ours {lost[1]} | opp {lost[0]}")
            for e in ev:
                if e[1] == 1: print(f"     step {e[0]} (d{e[0]//24}h{e[0]%24}) {e[2]} x{e[3]} [{e[4]}]")
if __name__ == "__main__":
    main()
