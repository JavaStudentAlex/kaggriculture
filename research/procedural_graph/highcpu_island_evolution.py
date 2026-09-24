"""High-CPU island evolution of the executable policy graph (edit-based SIFT loop).

Built for 64-128 core boxes (e.g. a Brev n2d-highcpu-80), used locally or over ssh.
Each iteration works on one island:

1. SIFT stage 1: three LLMs picked by a UCB1 bandit each propose an *edit* of the
   executable controls (graph_edits.py: champion parameters on chain nodes, market
   stage toggles, dispatch order, surgical switches). The edit is applied to the island
   champion, type-checked, refused when its normalized settings were already evaluated,
   and validated by building the runtime and playing the opening turns in a child
   process. Every failure goes back to the model (up to three attempts).
2. SIFT stage 2: a pairwise LLM judge compares the viable edits as diffs and a
   Bradley-Terry fit ranks them; only the winner is played.
3. Paired gauntlet (graph_gauntlet.py): the same seeds against Hazel, Copper, Orchard,
   Mohui and Mohui13, plus head-to-head games against the island champion. The winner
   replaces the champion only if the exact sign test over the changed games is
   significant and the mean change is positive.

Every --supervisor_interval iterations the meta-supervisor turns recent results into
guidance for the next prompts. Everything is written under --run_dir (checkpoint.json,
candidates.jsonl, best_graph.json, status.json, games and logs); policy_graph.json is
read once as the seed and never written: promoting a result is a reviewed commit.
"""
from __future__ import annotations

import os

# One BLAS/OpenMP thread per process on many-core boxes.
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import copy
import json
import logging
from pathlib import Path
import signal
import sys
import time
from typing import Any, Callable, Dict, List, Optional

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import graph_edits  # noqa: E402
from bam_graph_mutator import BAMGraphMutator  # noqa: E402
from graph_gauntlet import Gauntlet, LocalExecutor, SSHExecutor, compare  # noqa: E402
from pool_upgrade_state import atomic_json  # noqa: E402
from shinka_graph_mab import UCB1Bandit  # noqa: E402
from shinka_graph_supervisor import run_meta_supervisor  # noqa: E402
from sift_graph_judge import SIFTGraphJudge  # noqa: E402

CHECKPOINT_VERSION = "edit-1"
SEED_GRAPH = CURRENT_DIR / "policy_graph.json"
KNOWLEDGE = CURRENT_DIR / "evolution_knowledge.md"

# SIFT parameters (from MIT+Sakana AI paper arXiv:2609.19526)
SIFT_CANDIDATES_PER_ITERATION = 3   # Generate N candidates, judge pairwise, play only the winner
SIFT_MUTATION_RETRIES = 3           # Self-healing: retry failed mutations with error feedback

# (name, focus, market stages it centres on)
ISLANDS = (
    ("Island-Opening", "Day-0/1 cash and early spending: the step-0/1 wheat scalp, the day-1 hire "
     "step (step 24), the wage reserve, the investment freeze, the optional cash floor and "
     "livestock top-up.",
     ("opening_scalp", "wage_liquidity", "investment_freeze", "optional_cash_floor", "optional_livestock")),
    ("Island-Town", "Town shop sales: cadence phase relative to the step % 4 consumption, batch "
     "sizes, price thresholds, sell priority, fertilizer sales.", ("town_and_fertilizer",)),
    ("Island-Shed", "Shed capacity and inventory flow: overflow relief thresholds, pre-drop "
     "headroom, feed reserves, deferred sells.",
     ("feed_reserve", "shed_pressure", "predrop_headroom", "deferred_sales")),
    ("Island-Oracle", "Opponent-forecast levers: selling ahead of a predicted dump, score "
     "thresholds, batches, price floors, the priority horizon.", ("oracle_frontrun",)),
    ("Island-Endgame", "Endgame liquidation: when and how fast stock is sold before turn 720.",
     ("early_liquidation", "final_liquidation")),
    ("Island-Dispatch", "Order of market orders within a turn (lockstep clearing), worker rescue "
     "thresholds, the surgical switches.", ("routine_dispatch", "market_setup")),
)

_RUNNING = True


def _stop(signum, frame):
    global _RUNNING
    print("\n[SIGTERM/SIGINT] Stopping after the current iteration...", flush=True)
    _RUNNING = False


def _result_line(record: Dict[str, Any]) -> str:
    verdict = record.get("verdict") or {}
    if not verdict:
        return f"{'; '.join(record['changes'])} | not played: {record.get('cause', '')[:160]}"
    outcome = "PROMOTED" if verdict.get("promote") else "rejected"
    return (f"{'; '.join(record['changes'])} | changed games {verdict['wins']}W-{verdict['losses']}L "
            f"({verdict['ties']} unchanged), mean change ${verdict['mean_change']:+,.0f}, "
            f"p={verdict['p']:.2g} -> {outcome}")


class EditEvolution:
    """The iteration loop with its dependencies injected (tests pass fakes)."""

    def __init__(self, run_dir: Path, gauntlet: Gauntlet, mutator: Any, judge: Any, bandit: Any,
                 validate: Callable[[Path], Any], supervisor: Callable[..., List[str]],
                 knowledge: str, seed_graph: Optional[Dict[str, Any]] = None,
                 candidates: int = SIFT_CANDIDATES_PER_ITERATION,
                 retries: int = SIFT_MUTATION_RETRIES, supervisor_interval: int = 8):
        self.run_dir = Path(run_dir)
        self.gauntlet, self.mutator, self.judge, self.bandit = gauntlet, mutator, judge, bandit
        self.validate, self.supervisor, self.knowledge = validate, supervisor, knowledge
        self.candidates, self.retries, self.supervisor_interval = candidates, retries, supervisor_interval
        self.constants = graph_edits.catalog()
        self.checkpoint = self.run_dir / "checkpoint.json"
        self.state = self._load() if self.checkpoint.exists() else self._fresh(seed_graph)

    # ------------------------------------------------------------------ state
    def _fresh(self, seed_graph):
        seed = copy.deepcopy(seed_graph if seed_graph is not None else json.loads(SEED_GRAPH.read_text()))
        if seed.get("runtime") != "hazel_merged_v1":
            raise ValueError("the seed graph must declare runtime hazel_merged_v1")
        return {"version": CHECKPOINT_VERSION, "next_iteration": 1, "seed_graph": seed,
                "islands": [{"name": n, "focus": f, "stages": list(s), "graph": copy.deepcopy(seed),
                             "history": [], "rejections": []} for n, f, s in ISLANDS],
                "seen": [graph_edits.settings_key(seed, self.constants)], "guidance": [], "best": None}

    def _load(self):
        state = json.loads(self.checkpoint.read_text())
        if state.get("version") != CHECKPOINT_VERSION:
            raise ValueError(f"unsupported checkpoint {self.checkpoint} (version {state.get('version')!r}); "
                             "start a new --run_dir")
        for island in state["islands"]:
            if island["graph"].get("runtime") != "hazel_merged_v1":
                raise ValueError(f"island {island['name']} holds a graph without the merged runtime")
        return state

    def persist(self):
        atomic_json(self.checkpoint, self.state)

    def _log_candidate(self, record):
        with (self.run_dir / "candidates.jsonl").open("a") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

    # ------------------------------------------------------------------ prompts
    def results_text(self, graph) -> str:
        baseline = self.gauntlet.baseline(graph)
        per = Gauntlet.summary(baseline["margins"])
        lines = [f"{tag}: {t['wins']}W-{t['losses']}L-{t['ties']}T, mean margin ${t['mean_margin']:+,.0f}"
                 for tag, t in sorted(per.items())]
        worst = sorted(baseline["margins"].items(), key=lambda kv: kv[1])[:6]
        lines.append("worst games (opponent|seed|champion seat: margin): " +
                     ", ".join(f"{k}: ${m:+,.0f}" for k, m in worst))
        return "\n".join(lines)

    def _history(self, island) -> List[str]:
        return [_result_line(r) for r in (island["history"] + island["rejections"])[-12:]]

    # ------------------------------------------------------------------ one iteration
    def propose(self, island, iteration, results, models_used) -> List[Dict[str, Any]]:
        proposals = []
        champion = island["graph"]
        controls = graph_edits.describe_controls(champion, self.constants)
        focus = f"{island['focus']} Stages: {', '.join(island['stages'])}."
        for index in range(self.candidates):
            model = self.bandit.select_arm(exclude=models_used)
            models_used.append(model)
            error = None
            for attempt in range(1, self.retries + 1):
                try:
                    edit, rationale = self.mutator.mutate_edit(
                        model, controls, focus, results, self._history(island),
                        self.state["guidance"], self.knowledge, error)
                    graph = graph_edits.apply_edit(champion, edit, self.constants)
                    key = graph_edits.settings_key(graph, self.constants)
                    changes = graph_edits.diff(champion, graph, self.constants)
                    if key in self.state["seen"] or any(p["key"] == key for p in proposals):
                        raise ValueError("these settings were already evaluated: " + "; ".join(changes))
                    path = self.run_dir / "candidates" / f"iter{iteration:04d}_{index}.json"
                    atomic_json(path, graph)
                    self.validate(path)
                    proposals.append({"graph": graph, "edit": edit, "rationale": rationale, "model": model,
                                      "key": key, "changes": changes, "path": str(path)})
                    logging.info("  [%s] candidate %d (attempt %d): %s", model, index + 1, attempt, "; ".join(changes))
                    break
                except Exception as exc:
                    error = str(exc)[:1200]
                    logging.warning("  [%s] attempt %d/%d failed: %s", model, attempt, self.retries, error[:200])
                    self._log_candidate({"iteration": iteration, "island": island["name"], "model": model,
                                         "attempt": attempt, "stage": "proposal", "error": error})
            else:
                self.bandit.update(model, reward=0.0, crowned=False)
        return proposals

    def iteration(self, iteration: int):
        island = self.state["islands"][(iteration - 1) % len(self.state["islands"])]
        logging.info("-" * 65)
        logging.info("ITERATION %d | %s", iteration, island["name"])
        results = self.results_text(island["graph"])
        proposals = self.propose(island, iteration, results, [])
        if not proposals:
            logging.warning("  no viable candidate this iteration")
            return None
        context = f"Island focus: {island['focus']}\n{results}"
        ranked, votes = self.judge.rank_edits(proposals, self.knowledge, context)
        winner = ranked[0][0]
        logging.info("  [SIFT] %d valid votes; winner %s: %s", votes, winner["model"], "; ".join(winner["changes"]))
        for proposal, _ in ranked[1:]:
            self.bandit.update(proposal["model"], reward=0.15, crowned=False)
        baseline = self.gauntlet.baseline(island["graph"])
        evaluation = self.gauntlet.evaluate(winner["graph"], island["graph"])
        verdict = compare(evaluation, baseline, self.gauntlet.alpha)
        self.state["seen"].append(winner["key"])
        record = {"iteration": iteration, "island": island["name"], "model": winner["model"],
                  "rationale": winner["rationale"], "edit": winner["edit"], "changes": winner["changes"],
                  "bundle": evaluation["bundle"], "verdict": verdict, "judge_votes": votes,
                  "errors": evaluation["errors"][:5], "fallbacks": evaluation["fallbacks"]}
        self._log_candidate(dict(record, stage="gauntlet"))
        logging.info("  gauntlet: %s", _result_line(record))
        if verdict["promote"]:
            island["graph"] = winner["graph"]
            island["history"].append({k: v for k, v in record.items() if k != "errors"})
            self.bandit.update(winner["model"], reward=1.0, crowned=True)
            logging.info("  [CROWNED] new champion of %s", island["name"])
        else:
            island["rejections"].append({k: v for k, v in record.items() if k != "errors"})
            decided = verdict["wins"] + verdict["losses"]
            share = (verdict["wins"] - verdict["losses"]) / decided if decided and verdict["valid"] else 0.0
            self.bandit.update(winner["model"], reward=0.5 * max(0.0, share), crowned=False)
        self.update_best()
        return record

    def update_best(self):
        """The island champion with the largest mean gain over the seed on the pool games."""
        seed_baseline = self.gauntlet.baseline(self.state["seed_graph"])
        best = None
        for island in self.state["islands"]:
            if not island["history"]:
                continue
            stats = compare(self.gauntlet.baseline(island["graph"]), seed_baseline, self.gauntlet.alpha)
            if best is None or stats["mean_change"] > best["vs_seed"]["mean_change"]:
                best = {"island": island["name"], "vs_seed": stats,
                        "changes": graph_edits.diff(self.state["seed_graph"], island["graph"], self.constants)}
                atomic_json(self.run_dir / "best_graph.json", island["graph"])
        self.state["best"] = best
        if best:
            atomic_json(self.run_dir / "best.json", best)

    def status(self, iteration, event):
        atomic_json(self.run_dir / "status.json", {
            "updated_unix": time.time(), "iteration": iteration, "event": event,
            "islands": [{"name": i["name"], "promotions": len(i["history"]),
                         "rejections": len(i["rejections"]),
                         "changes_vs_seed": graph_edits.diff(self.state["seed_graph"], i["graph"], self.constants)}
                        for i in self.state["islands"]],
            "best": self.state.get("best"), "bandit": self.bandit.summary()})

    def run(self, iterations: int):
        seed_path = self.run_dir / "seed_graph.json"
        atomic_json(seed_path, self.state["seed_graph"])
        self.validate(seed_path)
        while _RUNNING and self.state["next_iteration"] <= iterations:
            n = self.state["next_iteration"]
            record = self.iteration(n)
            if n % self.supervisor_interval == 0:
                try:
                    recent = [r["verdict"] for i in self.state["islands"] for r in i["history"][-2:]]
                    rejected = [{"changes": r["changes"], "result": _result_line(r)}
                                for i in self.state["islands"] for r in i["rejections"][-2:]]
                    best = max(self.state["islands"], key=lambda i: len(i["history"]))["graph"]
                    self.state["guidance"] = self.supervisor(
                        best, recent, rejected, n, controls=graph_edits.describe_controls(best, self.constants))
                    logging.info("Meta-Supervisor guidance: %s", self.state["guidance"])
                except Exception as exc:
                    logging.error("Meta-supervisor failed: %s", exc)
            self.state["next_iteration"] = n + 1
            self.persist()
            self.status(n, "promoted" if record and record["verdict"]["promote"] else "done")
        logging.info("Evolution stopped at iteration %d", self.state["next_iteration"] - 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--run_dir", type=Path, default=CURRENT_DIR / "runs" / "evolution")
    parser.add_argument("--executor", choices=("local", "ssh"), default="local")
    parser.add_argument("--host", default="kagg-arena-80", help="ssh alias of the game box (--executor ssh)")
    parser.add_argument("--remote_dir", default="~/evolution")
    parser.add_argument("--workers", type=int, default=60, help="concurrent games (~1 GB RAM each)")
    parser.add_argument("--seeds_per_opponent", type=int, default=40,
                        help="seeds per opponent block; the candidate's seat alternates, so 40 = 20 per seat")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--candidates", type=int, default=SIFT_CANDIDATES_PER_ITERATION)
    parser.add_argument("--supervisor_interval", type=int, default=8)
    args = parser.parse_args()
    if args.seeds_per_opponent < 40 or args.seeds_per_opponent % 2:
        parser.error("--seeds_per_opponent must be even and at least 40 (20 games per seat)")

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.FileHandler(args.run_dir / "evolution.log", encoding="utf-8"),
                                  logging.StreamHandler(sys.stdout)])
    executor = (LocalExecutor(args.workers) if args.executor == "local"
                else SSHExecutor(args.host, args.remote_dir, args.workers))
    gauntlet = Gauntlet(args.run_dir, executor, args.seeds_per_opponent, alpha=args.alpha)
    fingerprint = gauntlet.prepare()
    evolution = EditEvolution(
        args.run_dir, gauntlet, BAMGraphMutator(), SIFTGraphJudge(),
        UCB1Bandit(args.run_dir / "bandit_state.json"), graph_edits.validate_graph, run_meta_supervisor,
        KNOWLEDGE.read_text(), candidates=args.candidates, supervisor_interval=args.supervisor_interval)
    logging.info("=" * 70)
    logging.info("EDIT-BASED ISLAND EVOLUTION | %d islands | %s | %d seeds x %d opponents + head-to-head | "
                 "evaluation %s", len(evolution.state["islands"]), executor.describe(),
                 args.seeds_per_opponent, len(gauntlet.opponents), fingerprint[:12])
    logging.info("=" * 70)
    evolution.run(args.iterations)


if __name__ == "__main__":
    main()
