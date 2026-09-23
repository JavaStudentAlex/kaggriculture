"""Distributed Kaggle-CPU Multi-Island Procedural Graph Evolutionary Pipeline.

Architecture:
- Cloud Workers: 10 Concurrent Kaggle CPU Kernels (40 vCPUs, 300 GB RAM, 0 GPU hours used).
  * Shard 0-9: Each evaluates Candidate vs 1 Champion (100 matches: 50 Seat 0 + 50 Seat 1)
  * Total per generation: 1,000 matches across the 10 champions simultaneously in ~35-45 minutes.
- Local Master (Hermes):
  * 8 Evolutionary Islands exploring specialized game dimensions.
  * UCB1 Multi-Armed Bandit (MAB) dynamically selecting frontier LLMs.
  * Two-Level Novelty Filter (MiniLM-L6-v2 embeddings + Gemini 3.1 Pro Preview judge).
  * Meta-Supervisor (gpt-6-astra, xhigh reasoning) cross-island synthesis every 8 iterations.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import json
import logging
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

CURRENT_DIR = Path(__file__).resolve().parent
REPO_DIR = CURRENT_DIR.parent.parent
EVO_DIR = REPO_DIR / "shinka" / "evolution"
POOL_DIR = REPO_DIR / "shinka" / "champions" / "pool"
SHARDS_DIR = REPO_DIR / "kaggle_shards"

for p in [str(CURRENT_DIR), str(REPO_DIR), str(EVO_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from bam_graph_mutator import BAMGraphMutator
from call_telemetry import merge_worker_metrics, reset as reset_call_telemetry, snapshot as call_telemetry_snapshot
from champion_pool import (
    aggregate_shard_results,
    build_bundle_archive,
    graph_fingerprint,
    induct_graph_champion,
    list_champions,
    plan_shard_split,
    publish_bundle,
)
from kaggriculture_domain_knowledge import build_jev_knowledge_summary
from sift_graph_judge import SIFTGraphJudge
from shinka_graph_mab import MODEL_SPECS, UCB1Bandit
from shinka_graph_novelty import cosine_similarity, get_graph_embedding, judge_novelty_with_llm
from shinka_graph_supervisor import run_meta_supervisor

# SIFT parameters (from MIT+Sakana AI paper arXiv:2609.19526)
SIFT_CANDIDATES_PER_ITERATION = 3   # Generate N candidates, judge pairwise, push only winner
SIFT_MUTATION_RETRIES = 3           # Self-healing: retry failed mutations with error feedback

# --- Growing champion pool (Red Queen ratchet) ---------------------------------
# Kaggle allows a limited number of concurrently running notebooks; the swarm
# width is chosen per generation from the live pool size, never above this.
MAX_CONCURRENT_SHARDS = int(os.environ.get("KAGG_MAX_SHARDS", "5"))
# Keep each notebook's runtime bounded as the pool grows.
CHAMPIONS_PER_SHARD = int(os.environ.get("KAGG_CHAMPIONS_PER_SHARD", "2"))
# A crowned graph only becomes a permanent opponent if it is genuinely strong;
# inducting weak agents just dilutes the gauntlet and wastes notebook time.
INDUCTION_WIN_RATE_GATE = float(os.environ.get("KAGG_INDUCTION_GATE", "0.75"))
BUNDLE_DIR = REPO_DIR / "kaggle_datasets" / "kaggriculture-champions-bundle"

_RUNNING = True


def sigterm_handler(signum, frame):
    global _RUNNING
    print("\n[SIGTERM/SIGINT] Gracefully finishing current iteration before exiting...", flush=True)
    _RUNNING = False


class EvolutionaryIsland:
    def __init__(self, island_id: int, name: str, focus_theme: str, initial_graph: Dict[str, Any]):
        self.island_id = island_id
        self.name = name
        self.focus_theme = focus_theme
        self.champion_graph = copy.deepcopy(initial_graph)
        self.champion_score: Dict[str, Any] = {}
        self.history: List[Dict[str, Any]] = []
        self.rejections: List[Dict[str, Any]] = []
        self.embeddings_archive: List[List[float]] = [get_graph_embedding(initial_graph)]


def _load_typesafe_key() -> str:
    """Load TYPESAFE_API_KEY from env or .env file (failsafe for tmux sessions)."""
    key = os.environ.get("TYPESAFE_API_KEY", "")
    if key:
        return key
    # Fallback: read from repo .env
    env_path = REPO_DIR / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("export "):
                line = line[len("export "):]
            if line.startswith("TYPESAFE_API_KEY=") and not line.startswith("#"):
                val = line.split("=", 1)[1].strip().strip("'\"")
                if val:
                    os.environ["TYPESAFE_API_KEY"] = val
                    return val
    return ""


def _load_tunnel_credentials() -> tuple:
    """Load the restricted SSH tunnel key + this host's public IP.

    The key is authorized server-side with permitopen="127.0.0.1:8317" and a
    forced command, so it can ONLY forward to the LLM proxy — no shell access.
    Returns (private_key_text, public_ip). Either may be "" if unavailable.
    """
    key_path = Path.home() / ".ssh" / "colab_tunnel_ed25519"
    key_text = ""
    if key_path.exists():
        key_text = key_path.read_text(encoding="utf-8")

    # Resolve this VM's external IP from GCP metadata (authoritative).
    public_ip = os.environ.get("KAGG_TUNNEL_HOST", "")
    if not public_ip:
        try:
            import urllib.request
            req = urllib.request.Request(
                "http://metadata.google.internal/computeMetadata/v1/"
                "instance/network-interfaces/0/access-configs/0/external-ip",
                headers={"Metadata-Flavor": "Google"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                public_ip = resp.read().decode("utf-8").strip()
        except Exception:
            public_ip = ""

    return key_text, public_ip


def prepare_and_push_shards(
    cand_graph: Dict[str, Any],
    evaluation_id: str,
    num_shards: int = 5,
) -> bool:
    """Embed a graph + unique evaluation ID and push shard notebooks to Kaggle."""
    template_path = SHARDS_DIR / "template" / "run_shard.py"
    template_code = template_path.read_text(encoding="utf-8")
    graph_json_str = json.dumps(cand_graph)

    # Jev disabled by user direction: zero Jev API calls, pure Procedural Graph + TTM Oracle predictor
    typesafe_key = ""
    logging.info("ℹ️  Jev API disabled by user configuration — using pure Procedural Graph + TTM Oracle predictor ($0.00).")

    # Load SSH tunnel credentials (enables inner SIFT evolution in the kernels)
    tunnel_key, tunnel_host = _load_tunnel_credentials()
    if tunnel_key and tunnel_host:
        logging.info(f"✓ SSH tunnel credentials loaded (host={tunnel_host}) — inner SIFT will be active.")
    else:
        logging.warning("⚠️  SSH tunnel credentials unavailable — shards run WITHOUT inner SIFT evolution.")

    def push_single_shard(shard_id: int) -> bool:
        shard_dir = SHARDS_DIR / f"shard_{shard_id}"
        shard_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "id": f"sunshinethroughfog/kagg-eval-shard-{shard_id}",
            "title": f"kagg-eval-shard-{shard_id}",
            "code_file": "run_shard.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "false",
            "enable_tpu": "false",
            "enable_internet": "true",
            "dataset_sources": ["sunshinethroughfog/kaggriculture-champions-bundle"],
            "competition_sources": ["kaggriculture"]
        }
        (shard_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        # Embed graph and API key into code
        code = template_code.replace('SHARD_ID = int(os.environ.get("SHARD_ID", "0"))', f'SHARD_ID = {shard_id}')
        code = code.replace(
            'TOTAL_SHARDS = int(os.environ.get("TOTAL_SHARDS", "5"))',
            f'TOTAL_SHARDS = {num_shards}'
        )
        code = code.replace(
            'EVALUATION_ID = os.environ.get("EVALUATION_ID", "unbound-evaluation")',
            f'EVALUATION_ID = {evaluation_id!r}'
        )
        code = code.replace(
            'TYPESAFE_KEY_PLACEHOLDER = ""',
            f'TYPESAFE_KEY_PLACEHOLDER = {repr(typesafe_key)}'
        )
        code = code.replace(
            'SSH_KEY_PLACEHOLDER = ""',
            f'SSH_KEY_PLACEHOLDER = {repr(tunnel_key)}'
        )
        code = code.replace(
            'SSH_HOST_PLACEHOLDER = ""',
            f'SSH_HOST_PLACEHOLDER = {repr(tunnel_host)}'
        )
        # Inner SIFT roster is derived from MODEL_SPECS (single source of truth)
        # so the in-kernel loop always draws from the same arms as the outer MAB.
        code = re.sub(
            r"INNER_SIFT_MODELS = \[[^\]]*\]",
            f"INNER_SIFT_MODELS = {repr(list(MODEL_SPECS.keys()))}",
            code,
            count=1,
        )
        code = code.replace(
            'cand_path = WORKING_DIR / "candidate_graph.json"',
            f'(WORKING_DIR / "candidate_graph.json").write_text({repr(graph_json_str)}, encoding="utf-8")\n    cand_path = WORKING_DIR / "candidate_graph.json"'
        )
        (shard_dir / "run_shard.py").write_text(code, encoding="utf-8")

        for attempt in range(5):
            cmd = ["kaggle", "kernels", "push", "-p", str(shard_dir)]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode == 0:
                logging.info(f"   ✓ [Shard {shard_id}] Successfully pushed to Kaggle cloud.")
                return True
            if "429" in res.stderr:
                delay = 4 + attempt * 2
                logging.warning(f"   ⚠️ [Shard {shard_id}] 429 Rate limit hit, retrying in {delay}s...")
                time.sleep(delay)
            else:
                logging.error(f"   ✗ [Shard {shard_id}] Push failed: {res.stderr.strip()}")
                time.sleep(2)
        return False

    logging.info(f"Pushing candidate graph to {num_shards} Kaggle CPU kernels with paced cadence...")
    results = []
    for s in range(num_shards):
        ok = push_single_shard(s)
        results.append(ok)
        time.sleep(2.0)

    return all(results)


def wait_and_collect_shards(
    evaluation_id: str,
    num_shards: int = 5,
    poll_interval: int = 30,
    max_wait_seconds: int = 7200,
    status_cb: Optional[Any] = None
) -> Dict[str, Any]:
    """Wait for this evaluation's Kaggle notebooks and reject stale outputs.

    Kaggle can briefly report the *previous* kernel version as COMPLETE just
    after a new push. A shard must first be observed QUEUED/RUNNING, and its
    downloaded JSON must carry this exact ``evaluation_id`` before it counts.
    """
    t0 = time.time()
    completed_shards = set()
    active_shards = set()
    stale_status_logged = set()

    logging.info(f"Monitoring {num_shards} Kaggle CPU kernels (300 GB cloud RAM)...")

    while len(completed_shards) < num_shards and time.time() - t0 < max_wait_seconds:
        for s in range(num_shards):
            if s in completed_shards:
                continue
            slug = f"sunshinethroughfog/kagg-eval-shard-{s}"
            cmd = ["kaggle", "kernels", "status", slug]
            try:
                out = subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout
                state = out.upper()
                if any(marker in state for marker in ("QUEUED", "RUNNING", "PENDING")):
                    if s not in active_shards:
                        active_shards.add(s)
                        logging.info(f"   ↻ [Shard {s}] New evaluation {evaluation_id} is active.")
                elif "COMPLETE" in state and s in active_shards:
                    completed_shards.add(s)
                    logging.info(f"   ✓ [Shard {s}] Complete! ({len(completed_shards)}/{num_shards} finished)")
                elif "COMPLETE" in state:
                    if s not in stale_status_logged:
                        stale_status_logged.add(s)
                        logging.info(f"   ⏳ [Shard {s}] Ignoring stale COMPLETE until the new evaluation becomes active.")
                elif ("FAILED" in state or "CANCELLED" in state) and s in active_shards:
                    logging.error(f"   ✗ [Shard {s}] Error/Failed on Kaggle: {out.strip()}")
                    completed_shards.add(s)
            except Exception:
                pass

        el = time.time() - t0
        if status_cb:
            status_cb(len(completed_shards), num_shards, el)

        if len(completed_shards) == num_shards:
            break

        logging.info(f"   Polling Kaggle Swarm: {len(completed_shards)}/{num_shards} kernels completed ({el:.0f}s elapsed)...")
        time.sleep(poll_interval)

    # Download and aggregate all shard results
    shard_results: List[Dict[str, Any]] = []

    temp_out_dir = CURRENT_DIR / "kaggle_downloads"
    temp_out_dir.mkdir(parents=True, exist_ok=True)

    for s in range(num_shards):
        slug = f"sunshinethroughfog/kagg-eval-shard-{s}"
        dest = temp_out_dir / f"shard_{s}"
        dest.mkdir(parents=True, exist_ok=True)
        subprocess.run(["kaggle", "kernels", "output", slug, "-p", str(dest)], capture_output=True, text=True)
        # Rule: delete completed Kaggle notebook immediately after collecting output
        subprocess.run(["kaggle", "kernels", "delete", "-y", slug], capture_output=True, text=True)

        res_file = dest / "shard_results.json"
        if res_file.exists():
            try:
                data = json.loads(res_file.read_text(encoding="utf-8"))
                if data.get("evaluation_id") != evaluation_id:
                    logging.error(
                        f"Rejecting shard {s} output with evaluation_id "
                        f"{data.get('evaluation_id')!r}; expected {evaluation_id!r}."
                    )
                    continue
                shard_results.append(data)
            except Exception as e:
                logging.error(f"Failed to read shard {s} result: {e}")

    shutil.rmtree(temp_out_dir, ignore_errors=True)

    if len(shard_results) != num_shards:
        raise RuntimeError(
            f"Only {len(shard_results)}/{num_shards} shards produced verified results "
            f"for evaluation {evaluation_id}; refusing to score a partial or stale gauntlet."
        )

    # Match-weighted aggregation: shards may hold different numbers of
    # champions once the pool grows, so a plain per-shard average would
    # over-weight the smallest notebook.
    agg = aggregate_shard_results(shard_results)
    for shard in shard_results:
        merge_worker_metrics(shard.get("remote_call_metrics"))
    agg["elapsed_sec"] = round(time.time() - t0, 1)
    return agg


def refresh_champion_bundle(reason: str) -> bool:
    """Rebuild + republish the champions dataset so the cloud sees new champions.

    A champion that exists only on the local disk is invisible to the Kaggle
    notebooks, so induction without publishing would silently keep the gauntlet
    frozen. Returns True when the new version is live.
    """
    try:
        archive = build_bundle_archive(REPO_DIR, BUNDLE_DIR)
        size_mb = archive.stat().st_size / (1024 * 1024)
        logging.info(f"   Rebuilt champions bundle ({size_mb:.1f} MB) — publishing new version...")
    except Exception as exc:
        logging.error(f"   Bundle rebuild failed: {exc}")
        return False

    ok, out = publish_bundle(BUNDLE_DIR, reason)
    if ok:
        logging.info("   ✓ Champions dataset version published — cloud gauntlet updated.")
    else:
        logging.error(f"   ✗ Champions dataset publish FAILED: {out[:300]}")
    return ok


def main():
    parser = argparse.ArgumentParser(description="Kaggle-Distributed Multi-Island Procedural Graph Evolution")
    parser.add_argument("--iterations", type=int, default=200, help="Total evolutionary iterations")
    parser.add_argument("--supervisor_interval", type=int, default=8, help="Iterations between cross-island merges")
    parser.add_argument("--resume_baseline_id", type=str, default=None, help="Resume waiting for an existing in-flight baseline evaluation")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, sigterm_handler)
    signal.signal(signal.SIGTERM, sigterm_handler)

    status_file = CURRENT_DIR / "differential_evo_status.json"
    log_file = CURRENT_DIR / "differential_evo.log"
    telemetry_file = CURRENT_DIR / "differential_evo_call_telemetry.json"
    if args.resume_baseline_id:
        baseline_evaluation_id = args.resume_baseline_id
        run_id = args.resume_baseline_id.replace("-baseline", "")
    else:
        run_id = f"island-evo-{int(time.time())}"
        baseline_evaluation_id = f"{run_id}-baseline"
    os.environ["KAGG_CALL_TELEMETRY_PATH"] = str(telemetry_file)
    reset_call_telemetry(telemetry_file, run_id)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler(sys.stdout)]
    )

    logging.info("=" * 70)
    logging.info(f"LAUNCHING KAGGLE-CPU DISTRIBUTED 8-ISLAND EVOLUTION ({args.iterations} ITERATIONS)")
    logging.info("Cloud Swarm: 5 Kaggle CPU Kernels (20 vCPUs, 150 GB RAM, 0 GPU hours used)")
    logging.info("Gauntlet: 1,000 Matches/Evaluation (100 matches per champion across 10 champions)")
    logging.info("=" * 70)

    base_graph_path = CURRENT_DIR / "policy_graph.json"
    master_graph = json.loads(base_graph_path.read_text(encoding="utf-8"))

    # The pool grows during the run, so the swarm width is recomputed each time.
    pool_size = len(list_champions(POOL_DIR))
    if pool_size == 0:
        logging.error(f"Champion pool at {POOL_DIR} is empty — cannot evaluate. Aborting.")
        return
    num_shards = plan_shard_split(pool_size, MAX_CONCURRENT_SHARDS, CHAMPIONS_PER_SHARD)
    logging.info(f"Champion pool: {pool_size} champions → {num_shards} Kaggle notebook(s)")

    # Initial baseline tournament on Kaggle Swarm
    if args.resume_baseline_id:
        logging.info(f"Resuming wait for active baseline evaluation (eval={baseline_evaluation_id})...")
    else:
        logging.info(f"Evaluating baseline graph across {num_shards} Kaggle CPU kernels (eval={baseline_evaluation_id})...")
        prepare_and_push_shards(
            master_graph, evaluation_id=baseline_evaluation_id, num_shards=num_shards)

    def baseline_cb(done, total, el):
        payload = {
            "status": "running",
            "current_iteration": 0,
            "total_iterations": args.iterations,
            "active_island": "Baseline Cloud Gauntlet",
            "champion_pool_size": pool_size,
            "num_shards": num_shards,
            "call_telemetry": call_telemetry_snapshot(telemetry_file),
            "live_gauntlet_progress": f"Kaggle Swarm: {done}/{total} kernels complete ({el:.0f}s elapsed)",
            "last_updated_unix": time.time()
        }
        status_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    base_eval = wait_and_collect_shards(
        evaluation_id=baseline_evaluation_id, num_shards=num_shards, status_cb=baseline_cb)
    logging.info(
        f"Baseline Cloud Gauntlet Results (1,000 matches):\n"
        f"  • Overall Win Rate: {base_eval['overall_win_rate']*100:.1f}% ({base_eval['total_wins']}/{base_eval['total_matches']})\n"
        f"  • Mean Cash: ${base_eval['overall_cand_cash']:,.2f} vs Champs: ${base_eval['overall_opp_cash']:,.2f} "
        f"in {base_eval['elapsed_sec']}s"
    )

    islands_def = [
        (0, "Island-Capital", "Days 0-10 Workforce Sizing, Day-0 Hires, and Initial Seed Ratios"),
        (1, "Island-Arbitrage", "Town Shop Preemption, Contract Fulfillment & Scarcity Capture"),
        (2, "Island-Watering", "Closed-Loop Strawberry Hydration, Moisture Maintenance & Zero Crop Waste"),
        (3, "Island-Liquidation", "Terminal Horizon Fire Sale (Steps 696-720) & Shed Inventory Zeroing"),
        (4, "Island-Solvency", "Midnight Wage Liability Defense & Hour >= 18 Cash Reserve Enforcement"),
        (5, "Island-OracleFrontrun", "Predictive Opponent Counter-Play & TTM 96-Step Glut Frontrunning"),
        (6, "Island-OrderCap", "10-Order Ceiling Dynamic Queue Arbitration & Slot Congestion Defense"),
        (7, "Island-Livestock", "Animal Compounding, 8 Cow / 6 Sheep Acquisition Curves & Feed Balance")
    ]
    islands = [EvolutionaryIsland(i[0], i[1], i[2], master_graph) for i in islands_def]
    for isl in islands:
        isl.champion_score = base_eval

    bandit = UCB1Bandit(CURRENT_DIR / "shinka_evo_run" / "bandit_state.json")
    mutator = BAMGraphMutator()
    sift_judge = SIFTGraphJudge()
    domain_knowledge = build_jev_knowledge_summary()

    def write_status(current_iter: int, active_isl: str, extra: Dict[str, Any] | None = None):
        payload = {
            "status": "running" if _RUNNING else "stopped",
            "current_iteration": current_iter,
            "total_iterations": args.iterations,
            "active_island": active_isl,
            "engine": "Kaggle CPU Cloud Swarm (dynamic width, growing champion pool)",
            "champion_pool_size": pool_size,
            "num_shards": num_shards,
            "call_telemetry": call_telemetry_snapshot(telemetry_file),
            "baseline_score": {
                "win_rate": base_eval["overall_win_rate"],
                "cand_cash": base_eval["overall_cand_cash"],
                "opp_cash": base_eval["overall_opp_cash"],
                "total_matches": base_eval["total_matches"]
            },
            "islands": [
                {
                    "name": isl.name,
                    "focus": isl.focus_theme,
                    "champion_win_rate": isl.champion_score.get("overall_win_rate", 0.0),
                    "champion_cash": isl.champion_score.get("overall_cand_cash", 0.0),
                    "accepted_count": len(isl.history),
                    "rejected_count": len(isl.rejections)
                } for isl in islands
            ],
            "bandit_arms": {
                m: {
                    "calls": bandit.state.get(m, {}).get("pulls", 0),
                    "avg_reward": round(bandit.state.get(m, {}).get("total_reward", 0.0) / max(1, bandit.state.get(m, {}).get("pulls", 0)), 3),
                    "crowns": bandit.state.get(m, {}).get("crowns", 0)
                } for m in bandit.arms
            },
            "last_updated_unix": time.time()
        }
        if extra:
            payload["latest_event"] = extra
        status_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    write_status(0, "Initialization", {"event": "Baseline cloud gauntlet complete", "score": base_eval})

    iteration = 1
    while _RUNNING and iteration <= args.iterations:
        isl_idx = (iteration - 1) % len(islands)
        island = islands[isl_idx]

        logging.info("-" * 65)
        logging.info(f"ITERATION {iteration}/{args.iterations} | {island.name} ({island.focus_theme})")
        logging.info(f"[SIFT] Generating {SIFT_CANDIDATES_PER_ITERATION} candidate mutations for pairwise judging...")

        # 1. Extract Diagnostics from cloud losses
        diagnostics = {
            "island_id": island.island_id,
            "island_name": island.name,
            "island_focus": island.focus_theme,
            "current_champ_win_rate": island.champion_score.get("overall_win_rate", 0.0),
            "current_champ_cash": island.champion_score.get("overall_cand_cash", 0.0),
            "worst_cloud_losses": island.champion_score.get("worst_losses", []),
            "mined_loss_traces": island.champion_score.get("mined_loss_traces", [])
        }

        # 2. Generate N candidate mutations from different MAB arms (with self-healing retry)
        viable_candidates: list = []  # [(graph, rationale, model_name)]
        selected_models: list = []

        for cand_idx in range(SIFT_CANDIDATES_PER_ITERATION):
            selected_model = bandit.select_arm()
            # Avoid re-selecting same model within same iteration if possible
            safety_counter = 0
            while selected_model in selected_models and safety_counter < 5:
                selected_model = bandit.select_arm()
                safety_counter += 1
            selected_models.append(selected_model)

            logging.info(f"  [Candidate {cand_idx+1}/{SIFT_CANDIDATES_PER_ITERATION}] UCB1 selected: [{selected_model}]")

            # Self-healing retry loop (SIFT paper: capture error, feed back, retry)
            last_error = None
            cand_graph = None
            cand_rationale = ""
            for attempt in range(SIFT_MUTATION_RETRIES):
                t_mut0 = time.time()
                try:
                    error_context = ""
                    if last_error:
                        error_context = f"\n\n### PREVIOUS MUTATION ATTEMPT FAILED:\n{last_error}\nPlease fix the issue and produce a valid graph mutation."

                    cand_graph, cand_rationale = mutator.mutate_graph(
                        model_name=selected_model,
                        current_graph=island.champion_graph,
                        match_diagnostics=diagnostics,
                        decision_trace=diagnostics["mined_loss_traces"] if diagnostics["mined_loss_traces"] else diagnostics["worst_cloud_losses"],
                        domain_knowledge=domain_knowledge + error_context,
                        rejections=[r.get("cause", "") for r in island.rejections[-5:]]
                    )
                    mut_dt = time.time() - t_mut0
                    logging.info(f"    ✓ [{selected_model}] synthesized candidate in {mut_dt:.1f}s (attempt {attempt+1})")
                    logging.info(f"      Rationale: {cand_rationale[:120]}...")
                    last_error = None
                    break
                except Exception as exc:
                    last_error = str(exc)[:500]
                    logging.warning(f"    ✗ [{selected_model}] attempt {attempt+1}/{SIFT_MUTATION_RETRIES} failed: {last_error[:120]}")

            if cand_graph is None:
                logging.error(f"    [EXHAUSTED] All {SIFT_MUTATION_RETRIES} attempts failed for [{selected_model}]")
                bandit.update(selected_model, reward=0.0, crowned=False)
                continue

            # Novelty check on each candidate
            cand_emb = get_graph_embedding(cand_graph)
            max_sim = -1.0
            collided_idx = -1
            for idx, p_emb in enumerate(island.embeddings_archive):
                sim = cosine_similarity(cand_emb, p_emb)
                if sim > max_sim:
                    max_sim = sim
                    collided_idx = idx

            is_novel = True
            novelty_reason = "Structurally novel"
            if max_sim >= 0.985:
                collided_g = island.history[collided_idx]["graph"] if collided_idx < len(island.history) else island.champion_graph
                is_novel, novelty_reason = judge_novelty_with_llm(cand_graph, collided_g, max_sim)

            if not is_novel:
                logging.info(f"    [Candidate {cand_idx+1}] Novelty rejected ({max_sim:.4f}): {novelty_reason[:80]}")
                island.rejections.append({"model": selected_model, "cause": novelty_reason, "rationale": cand_rationale})
                bandit.update(selected_model, reward=0.0, crowned=False)
                continue

            island.embeddings_archive.append(cand_emb)
            viable_candidates.append((cand_graph, cand_rationale, selected_model))
            logging.info(f"    [Candidate {cand_idx+1}] ✓ Novel (sim={max_sim:.4f}), added to SIFT judge pool")

        # 3. SIFT Pairwise Judge → Bradley-Terry ranking
        if not viable_candidates:
            logging.warning(f"  [SIFT] No viable candidates produced this iteration. Skipping.")
            write_status(iteration, island.name, {"event": "All candidates failed/rejected"})
            iteration += 1
            continue

        if len(viable_candidates) == 1:
            best_graph, best_rationale, best_model = viable_candidates[0]
            best_bt_score = 1.0
            logging.info(f"  [SIFT] Single viable candidate from [{best_model}], skipping judge.")
        else:
            logging.info(f"  [SIFT] Judging {len(viable_candidates)} candidates pairwise via [{sift_judge.judge_model}]...")
            ranked = sift_judge.rank_candidates(viable_candidates, domain_knowledge, diagnostics)
            best_graph, best_rationale, best_model, best_bt_score = ranked[0]
            logging.info(f"  [SIFT] 🏅 Winner: [{best_model}] (BT θ={best_bt_score:.3f})")
            # Give partial reward to non-winners for producing novel candidates
            for _, _, m, _ in ranked[1:]:
                bandit.update(m, reward=0.15, crowned=False)

        # 4. Push SIFT-selected candidate to Kaggle Cloud Swarm
        candidate_evaluation_id = f"{run_id}-iter-{iteration}-{int(time.time())}"
        logging.info(
            f"   Deploying candidate graph to {num_shards} Kaggle CPU kernels "
            f"(eval={candidate_evaluation_id})..."
        )
        prepare_and_push_shards(
            best_graph, evaluation_id=candidate_evaluation_id, num_shards=num_shards)

        def cand_cb(done, total, el):
            write_status(iteration, island.name, {
                "event": f"Kaggle Swarm: {done}/{total} kernels complete ({el:.0f}s elapsed)",
                "model": best_model
            })

        cand_eval = wait_and_collect_shards(
            evaluation_id=candidate_evaluation_id, num_shards=num_shards, status_cb=cand_cb)

        win_rate = cand_eval["overall_win_rate"]
        cand_cash = cand_eval["overall_cand_cash"]
        prev_win_rate = island.champion_score.get("overall_win_rate", 0.0)
        prev_cash = island.champion_score.get("overall_cand_cash", 0.0)

        logging.info(
            f"   Cloud Gauntlet Result: Overall Win Rate {win_rate*100:.1f}% ({cand_eval['total_wins']}/{cand_eval['total_matches']}) | "
            f"Cash: ${cand_cash:,.2f} vs Opp: ${cand_eval['overall_opp_cash']:,.2f} in {cand_eval['elapsed_sec']}s"
        )

        # 5. Gating & Crowning
        is_promoted = False
        if win_rate > prev_win_rate:
            is_promoted = True
        elif win_rate == prev_win_rate and cand_cash > prev_cash + 500.0:
            is_promoted = True

        if is_promoted:
            logging.info(f"🏆 [CROWNED] Promoted as new champion of {island.name}!")
            island.champion_graph = best_graph
            island.champion_score = cand_eval
            island.history.append({
                "iteration": iteration,
                "model": best_model,
                "graph": best_graph,
                "eval": cand_eval,
                "rationale": best_rationale
            })
            bandit.update(best_model, reward=1.0, crowned=True)
            event_type = "Promoted"

            # --- Red Queen ratchet -------------------------------------------
            # Induct the crowned graph into the opponent pool so every LATER
            # candidate (on every island) must also beat it, then republish the
            # bundle and re-plan the swarm width for the enlarged pool.
            if win_rate >= INDUCTION_WIN_RATE_GATE:
                try:
                    champ_path, induct_status = induct_graph_champion(
                        best_graph, POOL_DIR,
                        iteration=iteration, island=island.name,
                        win_rate=win_rate, cash=cand_cash,
                    )
                except Exception as exc:
                    logging.error(f"   Induction failed: {exc}")
                    champ_path, induct_status = None, "error"

                if induct_status == "created" and champ_path is not None:
                    new_pool_size = len(list_champions(POOL_DIR))
                    logging.info(
                        f"   ⚔️  INDUCTED into champion pool as {champ_path.name} "
                        f"(pool {pool_size} → {new_pool_size})"
                    )
                    published = refresh_champion_bundle(
                        f"iter {iteration} {island.name}: +{champ_path.name} "
                        f"(wr={win_rate:.3f}, cash={cand_cash:.0f})"
                    )
                    if published:
                        pool_size = new_pool_size
                        new_shards = plan_shard_split(
                            pool_size, MAX_CONCURRENT_SHARDS, CHAMPIONS_PER_SHARD)
                        if new_shards != num_shards:
                            logging.info(
                                f"   Swarm width re-planned: {num_shards} → {new_shards} notebooks "
                                f"for {pool_size} champions"
                            )
                            num_shards = new_shards
                        # Every island's recorded score was measured against the
                        # SMALLER pool and is no longer comparable. Clear the
                        # cached benchmark so the next evaluation on each island
                        # re-establishes it against the new opponent set.
                        for isl in islands:
                            isl.champion_score = dict(isl.champion_score)
                            isl.champion_score["stale_vs_pool"] = True
                    else:
                        logging.warning(
                            "   Champion inducted locally but publish failed — "
                            "cloud still runs the previous pool this generation."
                        )
                elif induct_status == "duplicate":
                    logging.info("   Graph already present in champion pool — no induction needed.")
        else:
            logging.info(f"   [NOT PROMOTED] Did not beat benchmark ({win_rate*100:.1f}% vs {prev_win_rate*100:.1f}%).")
            reward = 0.2 if cand_cash > prev_cash else 0.05
            bandit.update(best_model, reward=reward, crowned=False)
            event_type = "Evaluated (Not Promoted)"

        write_status(iteration, island.name, {
            "event": event_type,
            "model": best_model,
            "win_rate": win_rate,
            "cand_cash": cand_cash,
            "promoted": is_promoted
        })

        # 7. Cross-Island Supervisor Merge Cadence
        if iteration % args.supervisor_interval == 0:
            logging.info("=" * 70)
            logging.info(f"🏛️  INVOKING META-SUPERVISOR (gpt-6-astra) MERGE AT ITERATION {iteration}")
            logging.info("=" * 70)
            try:
                supervisor_recs = run_meta_supervisor(
                    best_graph=islands[0].champion_graph,
                    recent_evals=[{"island": isl.name, "score": isl.champion_score} for isl in islands],
                    recent_rejections=[],
                    current_gen=iteration
                )
                logging.info(f"Meta-Supervisor Guidance: {supervisor_recs[:3]}")

                best_island = max(islands, key=lambda isl: (isl.champion_score.get("overall_win_rate", 0), isl.champion_score.get("overall_cand_cash", 0)))
                base_graph_path.write_text(json.dumps(best_island.champion_graph, indent=2), encoding="utf-8")
                logging.info(f"Updated global master policy_graph.json from {best_island.name} (Win Rate: {best_island.champion_score.get('overall_win_rate', 0)*100:.1f}%)")
            except Exception as e:
                logging.error(f"Supervisor merge encountered error: {e}")

        iteration += 1

    logging.info("Distributed Kaggle evolution run finished.")


if __name__ == "__main__":
    main()
