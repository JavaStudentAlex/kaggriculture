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
3. Paired gauntlet (graph_gauntlet.py): the same seeds against the opponent pool
   (--opponents, default Mohui13, Mohui, Hazel and Willow), plus head-to-head games
   against the island champion. The winner replaces the champion only if the exact sign
   test over the changed games is significant and the mean change is positive.

Every --supervisor_interval iterations the meta-supervisor turns recent results into
guidance for the next prompts. Every --mix_interval iterations the islands mix: the island
whose champion gains most over the seed is the donor, and every other island is offered its
champion plus the donor's changes (a setting the island changed itself keeps its own value).
The same paired gauntlet decides; pool games a graph already played are reused, so a
champion moving to another island costs only the head-to-head block. The prompts cannot do
this: an edit whose settings were evaluated on any island is refused as a repeat.
A --queue file lets a person test a specific edit: an island's next iteration plays its first
queued edit not played yet (model "queue"), with the same gauntlet, instead of the proposals.
Everything is written under --run_dir (checkpoint.json, candidates.jsonl, best_graph.json,
status.json, games and logs); policy_graph.json is read once as the seed and never written:
promoting a result is a reviewed commit.
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
import urllib.error
import urllib.request

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import graph_edits  # noqa: E402
from bam_graph_mutator import BAMGraphMutator  # noqa: E402
from graph_gauntlet import ColabPoolExecutor, Gauntlet, LocalExecutor, SSHExecutor, compare  # noqa: E402
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
QUEUE = "queue"                     # the "model" of an edit played from the --queue file

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

# Islands of a graph whose backbone is a ladder engine (make_ladder_graph.py): the engine's own
# constants grouped by what they steer, plus our layers (channels) on top of it.
LADDER_ISLANDS = (
    ("Island-Herd", "Which animals the farm keeps for which town: the shop-pair route tables (_R110_OLD_SHOPS, "
     "_V92_TABLE), the herd swaps (V9_HERD_*, _HD2_*, _CS_*: cows/sheep/geese by quotes and shop counts), the "
     "yarn-town sheep/geese purchase (_Y_CFG) and the livestock cap (_V231_CAP).",
     ("engine: herd and routes",)),
    ("Island-Crops", "Crop inputs and timing: the carrot planner (V9_CARROT_*, _CA_*), the tomato investment "
     "(CROP_MIN_PRICE, _V219_FERTILIZE), wheat/carrot input planning (_R51_INPUT_*), fertilizer use (V9_FERT_*), "
     "watering skips (V13V_SKIP_DAYS).",
     ("engine: crops",)),
    ("Island-Market", "The engine's sale timing against the rival: race reservation and gates (V9_RACE_*, "
     "V9_RACEPX_*, V9_RACEGATE_*, _RACE_*), rival-flow lead selling (_FX_*; _FX_FLOW_MIN 999 keeps it off), "
     "hour-window lead sells (_EV_*, _DP_*, _MP_*, _ADV_*), the model-based lead seller (_MPX_*), its "
     "stream-library predictor (_V92_P_*, _V92_Q_* when present), order-slot priority (_OR2_*), quote "
     "reordering (_R37_*), FRONT_RUN_ITEMS and _SETTINGS.",
     ("engine: market",)),
    ("Island-Oracle", "Our opponent-supply predictor (TinyTimeMixer oracle) on top of the engine: switch the "
     "oracle_guard channel on and tune its _OG_* front-run thresholds, or switch our full market channel on "
     "(the champion's oracle_frontrun, town cadence and liquidation stages) and tune those.",
     ("channel: oracle_guard", "channel: market")),
    ("Island-Opening", "Day-0/1 cash and early spending: the step-0 wheat trade (V9_OPENING_STEP0, "
     "V9_OPENING_TAPE, _R42_OPENING), the productive opening (_PIPE_MODE / _ALT_MODE), early building "
     "(_E343_*), wheat buy-the-dip (_BD_*), feed/fertilizer economics (_R85_*, _R88_*), the cash reserve for "
     "fixed purchases (_CXD_*, _V7_CXD_*) and the milk-shop purchase shield (_SM_MILK_SHOP). An opening that "
     "leaves too little cash after step 1 starves the herd when the rival also buys wheat at step 0.",
     ("engine: opening and cash",)),
    ("Island-Endgame", "Shed room and the last days: night shed guard (_SR_*), capacity harvest/sell (_CH_*), "
     "the courier (V9_COURIER_*), yarn-town reorder hours (_Y_HOURS, _Y_ITEMS, _Y_MARGIN, _Y_MIN_DAY), and the "
     "experimental extension stages (capacity_liquidation, order_arbitration, ...).",
     ("engine: endgame and shed", "experimental")),
)

_RUNNING = True
PROXY_MODELS_URL = "http://localhost:8317/v1/models"
MAX_PROXY_RETRIES = 20  # proxy outages per proposal before it counts as a failed attempt


def wait_for_proxy(url: str = PROXY_MODELS_URL, poll: float = 60.0) -> bool:
    """Block until the LLM proxy answers. An unreachable proxy (a dropped tunnel, a
    sleeping laptop) must pause the run, not spend its iterations on refused calls."""
    minutes = 0
    while _RUNNING:
        try:
            with urllib.request.urlopen(url, timeout=15) as response:
                if response.status == 200:
                    if minutes:
                        logging.info("LLM proxy reachable again after ~%d min", minutes)
                    return True
        except Exception as exc:
            if minutes % 10 == 0:
                logging.warning("LLM proxy unreachable (%s); waiting", exc)
        time.sleep(poll)
        minutes += max(1, round(poll / 60))
    return False


def served_models(url: str = PROXY_MODELS_URL) -> List[str]:
    """Model ids the LLM proxy serves now (the bandit only pulls these)."""
    wait_for_proxy(url)
    with urllib.request.urlopen(url, timeout=30) as response:
        return [m["id"] for m in json.loads(response.read().decode()).get("data", [])]


def _proxy_outage(exc: Exception) -> bool:
    """Connection-level failure to reach the proxy (not an HTTP error from a model)."""
    return isinstance(exc, (urllib.error.URLError, ConnectionError, TimeoutError)) and \
        not isinstance(exc, urllib.error.HTTPError)


def _stop(signum, frame):
    global _RUNNING
    print("\n[SIGTERM/SIGINT] Stopping after the current iteration...", flush=True)
    _RUNNING = False


def _result_line(record: Dict[str, Any]) -> str:
    verdict = record.get("verdict") or {}
    if not verdict:
        return f"{'; '.join(record['changes'])} | not played: {record.get('cause', '')[:160]}"
    outcome = "PROMOTED" if verdict.get("promote") else "rejected"
    per = "; ".join(f"{tag} {s['wins']}-{s['losses']}-{s['ties']} ${s['mean']:+,.0f}"
                    for tag, s in sorted((verdict.get("per_opponent") or {}).items()))
    return (f"{'; '.join(record['changes'])} | changed games {verdict['wins']}W-{verdict['losses']}L "
            f"({verdict['ties']} unchanged), mean change ${verdict['mean_change']:+,.0f}, "
            f"p={verdict['p']:.2g} -> {outcome}"
            + (f" | per opponent (W-L-T, mean change): {per}" if per else ""))


class EditEvolution:
    """The iteration loop with its dependencies injected (tests pass fakes)."""

    def __init__(self, run_dir: Path, gauntlet: Gauntlet, mutator: Any, judge: Any, bandit: Any,
                 validate: Callable[[Path], Any], supervisor: Callable[..., List[str]],
                 knowledge: str, seed_graph: Optional[Dict[str, Any]] = None,
                 candidates: int = SIFT_CANDIDATES_PER_ITERATION,
                 retries: int = SIFT_MUTATION_RETRIES, supervisor_interval: int = 8,
                 ideas_path: Optional[Path] = None, proxy_ready: Callable[[], bool] = lambda: True,
                 islands=None, knowledge_path: Optional[Path] = None, mix_interval: int = 0,
                 queue_path: Optional[Path] = None):
        self.run_dir = Path(run_dir)
        self.queue_path = Path(queue_path) if queue_path else None
        self.islands = islands
        self.mix_interval = mix_interval
        self.proxy_ready = proxy_ready
        self.ideas_path = Path(ideas_path) if ideas_path else None
        self.knowledge_path = Path(knowledge_path) if knowledge_path else None
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
        islands = self.islands or ISLANDS
        return {"version": CHECKPOINT_VERSION, "next_iteration": 1, "seed_graph": seed,
                "islands": [{"name": n, "focus": f, "stages": list(s), "graph": copy.deepcopy(seed),
                             "history": [], "rejections": []} for n, f, s in islands],
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

    def ideas(self) -> List[str]:
        """Bullet lines of the --ideas file, re-read every iteration so a run can be steered."""
        if not self.ideas_path or not self.ideas_path.exists():
            return []
        return [line.strip()[2:].strip() for line in self.ideas_path.read_text().splitlines()
                if line.strip().startswith(("- ", "* "))]

    def current_knowledge(self) -> str:
        """The --knowledge file, re-read every iteration like the ideas (corrections need no restart)."""
        if self.knowledge_path and self.knowledge_path.exists():
            self.knowledge = self.knowledge_path.read_text()
        return self.knowledge

    def queued(self, island, iteration) -> Optional[Dict[str, Any]]:
        """The first --queue entry for this island whose settings were not played yet, as the
        iteration's candidate: it skips the models' proposals and the judge. The queue file (a JSON
        list of {"island", "edit", "rationale"}) is re-read every iteration, like the ideas."""
        if not self.queue_path or not self.queue_path.exists():
            return None
        for entry in json.loads(self.queue_path.read_text()):
            if entry.get("island") != island["name"]:
                continue
            try:
                graph = graph_edits.apply_edit(island["graph"], entry["edit"], self.constants)
                key = graph_edits.settings_key(graph, self.constants)
                if key in self.state["seen"]:
                    continue
                path = self.run_dir / "candidates" / f"iter{iteration:04d}_queue.json"
                atomic_json(path, graph)
                self.validate(path)
            except Exception as exc:
                logging.warning("  [QUEUE] %s refused: %s", json.dumps(entry.get("edit")), str(exc)[:200])
                self._log_candidate({"iteration": iteration, "island": island["name"], "model": QUEUE,
                                     "stage": "proposal", "edit": entry.get("edit"), "error": str(exc)[:1200]})
                continue
            return {"graph": graph, "edit": entry["edit"], "rationale": entry.get("rationale", ""), "model": QUEUE,
                    "key": key, "changes": graph_edits.diff(island["graph"], graph, self.constants), "path": str(path)}
        return None

    def _history(self, island) -> List[str]:
        """Every edit played in this run on any island, oldest first; this island's own marked."""
        played = sorted(((r.get("iteration", 0), i["name"], r) for i in self.state["islands"]
                         for r in i["history"] + i["rejections"]), key=lambda x: x[0])
        return [f"[{'this island' if name == island['name'] else name}] {_result_line(r)}"
                for _, name, r in played[-12:]]

    # ------------------------------------------------------------------ one iteration
    def propose(self, island, iteration, results, models_used) -> List[Dict[str, Any]]:
        proposals = []
        champion = island["graph"]
        controls = graph_edits.describe_controls(champion, self.constants)
        focus = f"{island['focus']} Areas: {', '.join(island['stages'])}."
        ideas = self.ideas()
        knowledge = self.current_knowledge()
        for index in range(self.candidates):
            model = self.bandit.select_arm(exclude=models_used)
            models_used.append(model)
            error = None
            attempt, outages = 0, 0
            while attempt < self.retries:
                attempt += 1
                # the edits other models already proposed in this iteration: a repeat is refused
                pending = [f"[this iteration, proposed by another model, not played yet] {'; '.join(p['changes'])}"
                           for p in proposals]
                try:
                    try:
                        edit, rationale = self.mutator.mutate_edit(
                            model, controls, focus, results, self._history(island) + pending,
                            self.state["guidance"], knowledge, error, ideas=ideas)
                    except Exception as exc:
                        if _proxy_outage(exc) and outages < MAX_PROXY_RETRIES:
                            outages += 1
                            attempt -= 1  # the model was never reached: not its failed attempt
                            logging.warning("  [%s] proxy unreachable (%s); waiting", model, exc)
                            if not self.proxy_ready():
                                return proposals
                            continue
                        raise
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
        winner, votes = self.queued(island, iteration), 0
        if winner:
            logging.info("  [QUEUE] candidate: %s", "; ".join(winner["changes"]))
        else:
            results = self.results_text(island["graph"])
            if not self.proxy_ready():
                return None
            proposals = self.propose(island, iteration, results, [])
            if not proposals:
                logging.warning("  no viable candidate this iteration")
                return None
            context = f"Island focus: {island['focus']}\n{results}"
            ranked, votes = self.judge.rank_edits(proposals, self.current_knowledge(), context)
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
            if winner["model"] != QUEUE:
                self.bandit.update(winner["model"], reward=1.0, crowned=True)
            logging.info("  [CROWNED] new champion of %s", island["name"])
        else:
            island["rejections"].append({k: v for k, v in record.items() if k != "errors"})
            decided = verdict["wins"] + verdict["losses"]
            share = (verdict["wins"] - verdict["losses"]) / decided if decided and verdict["valid"] else 0.0
            if winner["model"] != QUEUE:
                self.bandit.update(winner["model"], reward=0.5 * max(0.0, share), crowned=False)
        self.update_best()
        return record

    # ------------------------------------------------------------------ island mixing
    def mix_due(self, after: int) -> bool:
        """One mixing per block of --mix_interval iterations, after the block's last iteration (a
        run that starts mixing mid-block mixes at once)."""
        return bool(self.mix_interval and after >= self.mix_interval and
                    after // self.mix_interval > self.state.get("mixed_after", 0) // self.mix_interval)

    def mix(self, after: int) -> bool:
        """Offer every island the donor's changes. The donor is fixed when the mixing starts and
        saved, so a stopped mixing resumes with the islands not yet offered. False if a stop
        arrived before every island was offered them."""
        mixing = self.state.get("mixing")
        if not mixing or mixing.get("after") != after:
            self.update_best()
            best = self.state.get("best")
            if not best:
                logging.info("MIXING after iteration %d: no island champion beats the seed yet", after)
                return True
            donor = next(i for i in self.state["islands"] if i["name"] == best["island"])
            mixing = {"after": after, "donor": donor["name"], "graph": copy.deepcopy(donor["graph"]), "done": []}
            self.state["mixing"] = mixing
            self.persist()
        logging.info("-" * 65)
        logging.info("MIXING after iteration %d | donor %s: %s", after, mixing["donor"],
                     "; ".join(graph_edits.diff(self.state["seed_graph"], mixing["graph"], self.constants)))
        for island in self.state["islands"]:
            if island["name"] == mixing["donor"] or island["name"] in mixing["done"]:
                continue
            if not _RUNNING:
                return False
            self.migrate(island, mixing)
            mixing["done"].append(island["name"])
            self.persist()
        self.status(after, "mixed")
        return True

    def migrate(self, island, mixing):
        """One island offered the donor's changes; the gauntlet decides as for an edit."""
        after, donor = mixing["after"], mixing["donor"]
        edit = graph_edits.migration_edit(island["graph"], mixing["graph"], self.state["seed_graph"], self.constants)
        if edit is None:
            logging.info("  [MIX] %s already has every change of %s", island["name"], donor)
            return None
        try:
            graph = graph_edits.apply_edit(island["graph"], edit, self.constants)
            path = self.run_dir / "candidates" / f"mix{after:04d}_{island['name']}.json"
            atomic_json(path, graph)
            self.validate(path)
        except Exception as exc:
            logging.warning("  [MIX] %s cannot take the changes of %s: %s", island["name"], donor, str(exc)[:300])
            self._log_candidate({"iteration": after, "island": island["name"], "model": "mixing", "donor": donor,
                                 "stage": "mixing", "error": str(exc)[:1200]})
            return None
        changes = graph_edits.diff(island["graph"], graph, self.constants)
        baseline = self.gauntlet.baseline(island["graph"])
        evaluation = self.gauntlet.evaluate(graph, island["graph"])
        verdict = compare(evaluation, baseline, self.gauntlet.alpha)
        key = graph_edits.settings_key(graph, self.constants)
        if key not in self.state["seen"]:
            self.state["seen"].append(key)
        record = {"iteration": after, "island": island["name"], "model": "mixing",
                  "rationale": f"island mixing: the changes of {donor}", "edit": edit, "changes": changes,
                  "donor": donor, "bundle": evaluation["bundle"], "verdict": verdict,
                  "errors": evaluation["errors"][:5], "fallbacks": evaluation["fallbacks"]}
        self._log_candidate(dict(record, stage="mixing"))
        logging.info("  [MIX] %s: %s", island["name"], _result_line(record))
        if verdict["promote"]:
            island["graph"] = graph
            island["history"].append({k: v for k, v in record.items() if k != "errors"})
            logging.info("  [CROWNED] new champion of %s (from %s)", island["name"], donor)
        else:
            island["rejections"].append({k: v for k, v in record.items() if k != "errors"})
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
            if self.mix_due(n - 1):
                if not self.mix(n - 1):
                    break  # stopped during the mixing: it resumes with the islands not yet offered
                self.state["mixed_after"] = n - 1
                self.persist()
                if not _RUNNING:
                    break
            record = self.iteration(n)
            if not _RUNNING and record is None:
                break  # stopped before a game was played: repeat this iteration on resume
            # a stop that arrives during the gauntlet ends the run after this iteration is saved
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
    parser.add_argument("--executor", choices=("local", "ssh", "colab-pool"), default="local")
    parser.add_argument("--pool_dir", type=Path, default=None,
                        help="--executor colab-pool: the pool directory of an arena/colab_pool.py on this host")
    parser.add_argument("--plan", type=Path, default=None,
                        help="job plan (ladder_seed_plan.py): lost ladder seeds + random seeds per opponent")
    parser.add_argument("--islands", choices=("hazel", "ladder"), default="hazel",
                        help="ladder: islands over a ladder engine's constants and our channels")
    parser.add_argument("--knowledge", type=Path, default=KNOWLEDGE, help="facts shown to every mutating model")
    parser.add_argument("--models", default="", help="comma list of bandit models (default: all served ones)")
    parser.add_argument("--judge_model", default="gpt-6-astra")
    parser.add_argument("--host", default="kagg-arena-80", help="ssh alias of the game box (--executor ssh)")
    parser.add_argument("--remote_dir", default="~/evolution")
    parser.add_argument("--workers", type=int, default=60, help="concurrent games (~1 GB RAM each)")
    parser.add_argument("--seeds_per_opponent", type=int, default=40,
                        help="seeds per opponent block; the candidate's seat alternates, so 40 = 20 per seat")
    parser.add_argument("--opponents", default="mohui13,mohui,hazel,willow",
                        help="opponent bundles (arena/payload.py names); default: the ladder-like pool")
    parser.add_argument("--seed_graph", type=Path, default=SEED_GRAPH,
                        help="graph a fresh run starts from (e.g. a previous run's best_graph.json)")
    parser.add_argument("--ideas", type=Path, default=None,
                        help="markdown bullet list injected into every mutation prompt; re-read each iteration")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--candidates", type=int, default=SIFT_CANDIDATES_PER_ITERATION)
    parser.add_argument("--supervisor_interval", type=int, default=8)
    parser.add_argument("--mix_interval", type=int, default=12,
                        help="islands mix after every N iterations (0: never); see the module docstring")
    parser.add_argument("--queue", type=Path, default=None,
                        help="JSON list of {island, edit, rationale}: an island's next iteration plays its first "
                             "queued edit not played yet instead of the models' proposals; re-read each iteration")
    args = parser.parse_args()
    if args.seeds_per_opponent % 2 or args.seeds_per_opponent < (10 if args.plan else 40):
        parser.error("--seeds_per_opponent must be even and at least 40 (20 games per seat); "
                     "with --plan it only sizes the head-to-head block (at least 10)")

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    args.run_dir = args.run_dir.resolve()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.FileHandler(args.run_dir / "evolution.log", encoding="utf-8"),
                                  logging.StreamHandler(sys.stdout)])
    if args.executor == "local":
        executor = LocalExecutor(args.workers)
    elif args.executor == "ssh":
        executor = SSHExecutor(args.host, args.remote_dir, args.workers)
    else:
        if args.pool_dir is None:
            parser.error("--executor colab-pool needs --pool_dir")
        executor = ColabPoolExecutor(args.pool_dir)
    plan = json.loads(args.plan.read_text()) if args.plan else None
    gauntlet = Gauntlet(args.run_dir, executor, args.seeds_per_opponent,
                        opponents=tuple(o for o in args.opponents.split(",") if o), alpha=args.alpha, plan=plan)
    fingerprint = gauntlet.prepare()
    served = served_models()
    wanted = [m for m in args.models.split(",") if m] or served
    arms = [m for m in wanted if m in served]
    logging.info("bandit models: %s (served by the proxy: %s)", arms, served)
    evolution = EditEvolution(
        args.run_dir, gauntlet, BAMGraphMutator(), SIFTGraphJudge(judge_model=args.judge_model),
        UCB1Bandit(args.run_dir / "bandit_state.json", arms=arms), graph_edits.validate_graph, run_meta_supervisor,
        args.knowledge.read_text(), seed_graph=json.loads(args.seed_graph.read_text()),
        candidates=args.candidates, supervisor_interval=args.supervisor_interval, ideas_path=args.ideas,
        proxy_ready=wait_for_proxy, islands=LADDER_ISLANDS if args.islands == "ladder" else None,
        knowledge_path=args.knowledge, mix_interval=args.mix_interval, queue_path=args.queue)
    logging.info("=" * 70)
    logging.info("EDIT-BASED ISLAND EVOLUTION | %d islands | %s | %d seeds (%d per seat) x %s + head-to-head | "
                 "evaluation %s", len(evolution.state["islands"]), executor.describe(), args.seeds_per_opponent,
                 args.seeds_per_opponent // 2, ",".join(gauntlet.opponents), fingerprint[:12])
    logging.info("=" * 70)
    evolution.run(args.iterations)


if __name__ == "__main__":
    main()
