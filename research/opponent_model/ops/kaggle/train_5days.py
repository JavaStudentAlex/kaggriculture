import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

print("=" * 70)
print("Kaggriculture Opponent Model (TTM) 5-Day Dual-GPU Fine-Tuning")
print("=" * 70)

# -------------------------------------------------------------------------
# Step 1: Check Hardware & Accelerator Setup
# -------------------------------------------------------------------------
import torch

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available:  {torch.cuda.is_available()}")
num_gpus = torch.cuda.device_count()
print(f"GPU Count:       {num_gpus}")
for i in range(num_gpus):
    print(f"  • GPU {i}: {torch.cuda.get_device_name(i)}")

# -------------------------------------------------------------------------
# Step 2: Install Dependencies
# -------------------------------------------------------------------------
print("\n[1/6] Installing dependencies (granite-tsfm, kaggle-environments)...")
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "granite-tsfm", "kaggle-environments==1.32.7"],
    check=True,
)

# -------------------------------------------------------------------------
# Step 3: Locate Base Checkpoint & Stage Pipeline Code
# -------------------------------------------------------------------------
print("\n[2/6] Locating base model & staging pipeline code...")
base_model_candidates = list(Path("/kaggle/input").glob("**/checkpoint/model.safetensors"))
if not base_model_candidates:
    base_model_candidates = list(Path("/kaggle/input").glob("**/model.safetensors"))

assert base_model_candidates, "Could not find model.safetensors in /kaggle/input!"
base_model_dir = base_model_candidates[0].parent
print(f"  Base Model Checkpoint: {base_model_dir}")
assert (base_model_dir / "scaler.npz").exists(), f"scaler.npz missing in {base_model_dir}!"

# Locate opponent_model scripts in input
opponent_script_cands = list(Path("/kaggle/input").glob("**/opponent_model/train_ttm.py"))
assert opponent_script_cands, "Could not find train_ttm.py in /kaggle/input!"
src_scripts_dir = opponent_script_cands[0].parent

# Copy to /tmp/opponent_model so we can patch for Tesla T4 GPU compatibility
work_scripts_dir = Path("/tmp/opponent_model")
if work_scripts_dir.exists():
    shutil.rmtree(work_scripts_dir)
shutil.copytree(src_scripts_dir, work_scripts_dir)

# Patch train_ttm.py for Turing GPU (Tesla T4) compatibility:
# T4 (sm_75) does not support native bf16; use fp16 on T4, bf16 on Ampere+
train_py_path = work_scripts_dir / "train_ttm.py"
train_code = train_py_path.read_text()

# Fix bf16 -> check compute capability
old_bf16_line = "bf16=torch.cuda.is_available(),"
new_bf16_line = (
    "bf16=(torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8),\n"
    "        fp16=(torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8),"
)
if old_bf16_line in train_code:
    train_code = train_code.replace(old_bf16_line, new_bf16_line)

# Tune dataloader workers for Kaggle's 4 vCPUs
train_code = train_code.replace("dataloader_num_workers=8,", "dataloader_num_workers=2,")
train_py_path.write_text(train_code)
print(f"  Pipeline code staged & patched for Tesla T4 at: {work_scripts_dir}")

# Add to Python path so we can import extract & mechanics
sys.path.insert(0, str(work_scripts_dir))
from extract import episode_rows, write_labels_marker  # noqa: E402
from features import feature_names  # noqa: E402
from mechanics import PRODUCTS  # noqa: E402
import numpy as np  # noqa: E402

# -------------------------------------------------------------------------
# Step 4: Extract 5-Day Shards Directly from Mounted JSONs
# -------------------------------------------------------------------------
print("\n[3/6] Extracting 5 days of replay episodes into .npz feature shards...")
target_dates = [
    "2026-09-13",
    "2026-09-14",
    "2026-09-15",
    "2026-09-16",
    "2026-09-17",
    "2026-09-18",
]
shards_dir = Path("/tmp/shards")
shards_dir.mkdir(parents=True, exist_ok=True)
write_labels_marker(shards_dir, "next_action")


def process_one_day(date_str):
    dest_npz = shards_dir / f"kaggriculture-episodes-{date_str}.npz"
    if dest_npz.exists() and dest_npz.stat().st_size > 1000:
        return date_str, 0, f"already cached ({dest_npz.stat().st_size / 1e6:.1f} MB)"

    # Find the day's folder in /kaggle/input
    cands = list(Path("/kaggle/input").glob(f"**/*{date_str}*"))
    cands = [c for c in cands if c.is_dir()]
    if not cands:
        return date_str, 0, "directory not found in /kaggle/input"

    day_dir = cands[0]
    json_files = sorted(list(day_dir.glob("*.json")))
    if not json_files:
        return date_str, 0, "no json files found"

    feat_names = feature_names()
    xs, ys, ds = [], [], []
    episodes = 0

    t0 = time.time()
    for jf in json_files:
        try:
            with open(jf, encoding="utf-8") as f:
                doc = json.load(f)
        except Exception:
            continue
        for x, y, d in episode_rows(doc, feat_names, stride=1, alignment="next_action"):
            xs.append(x)
            ys.append(y)
            ds.append(d)
        episodes += 1

    if not xs:
        return date_str, 0, "no valid rows extracted"

    np.savez_compressed(
        dest_npz,
        X=np.stack(xs),
        Y=np.stack(ys),
        D=np.stack(ds),
        feature_names=np.array(feat_names),
        products=np.array(PRODUCTS),
    )
    dur = time.time() - t0
    size_mb = dest_npz.stat().st_size / (1024 * 1024)
    return date_str, len(xs), f"{episodes} episodes -> {len(xs):,} rows -> {size_mb:.1f} MB ({dur:.1f}s)"


# Run extraction across the 5 days in parallel
with ProcessPoolExecutor(max_workers=min(4, len(target_dates))) as pool:
    results = list(pool.map(process_one_day, target_dates))

for d, rows, msg in results:
    print(f"  • {d}: {msg}")

# -------------------------------------------------------------------------
# Step 5: Stage Dataset Split for Recency-Weighted Training
# -------------------------------------------------------------------------
print("\n[4/6] Staging dataset for recency-weighted fine-tuning...")
newest_day = target_dates[-1]  # 2026-09-15
dataset_stage = Path("/tmp/dataset_stage")
day_dir = dataset_stage / "day"
prev_dir = dataset_stage / "prev"
day_dir.mkdir(parents=True, exist_ok=True)
prev_dir.mkdir(parents=True, exist_ok=True)

newest_shard = shards_dir / f"kaggriculture-episodes-{newest_day}.npz"
shutil.copy(newest_shard, day_dir / newest_shard.name)

for d in target_dates[:-1]:
    p_shard = shards_dir / f"kaggriculture-episodes-{d}.npz"
    if p_shard.exists():
        shutil.copy(p_shard, prev_dir / p_shard.name)

labels_src = shards_dir / "labels.json"
if labels_src.exists():
    shutil.copy(labels_src, day_dir / "labels.json")
    shutil.copy(labels_src, prev_dir / "labels.json")

print(f"  Target day (split 90/10 train/val): {newest_day} -> {day_dir}")
print(f"  Previous days (100% extra-train with decay=2.0): {[d for d in target_dates[:-1]]} -> {prev_dir}")

# -------------------------------------------------------------------------
# Step 6: Launch Dual-GPU Training via torchrun DDP
# -------------------------------------------------------------------------
print("\n[5/6] Launching TTM Fine-Tuning on Dual T4 GPUs...")
out_run = Path(f"/kaggle/working/ttm_c256_h96_ft_{newest_day}")
out_run.mkdir(parents=True, exist_ok=True)

train_script = str(work_scripts_dir / "train_ttm.py")

if num_gpus >= 2:
    launch_cmd = ["torchrun", f"--nproc_per_node={num_gpus}", "--master_port=29581", train_script]
    lr = "4e-5"  # scaled for 2 GPUs
else:
    launch_cmd = [sys.executable, train_script]
    lr = "2e-5"

train_args = [
    "--dataset",
    str(day_dir),
    "--extra-train",
    str(prev_dir),
    "--extra-train-decay",
    "2.0",
    "--out",
    str(out_run),
    "--init-from",
    str(base_model_dir),
    "--scaler",
    str(base_model_dir / "scaler.npz"),
    "--split",
    "episode",
    "--prediction-filter-length",
    "96",
    "--metric-horizon",
    "all",
    "--eval-on-start",
    "--max-episodes",
    "100000",
    "--epochs",
    "30",
    "--patience",
    "5",
    "--plateau",
    "--plateau-factor",
    "0.5",
    "--plateau-patience",
    "2",
    "--batch-size",
    "64",
    "--window-stride",
    "4",
    "--lr",
    lr,
]

full_cmd = launch_cmd + train_args
print("  Running command:\n ", " ".join(full_cmd))
sys.stdout.flush()

t_train_start = time.time()
res_train = subprocess.run(full_cmd)
train_duration = time.time() - t_train_start
print(f"\nTraining process exited with code {res_train.returncode} in {train_duration / 60:.1f} minutes.")

if res_train.returncode != 0:
    print("ERROR: Training failed!")
    sys.exit(res_train.returncode)

# -------------------------------------------------------------------------
# Step 7: Export Best Checkpoint to /kaggle/working/
# -------------------------------------------------------------------------
print("\n[6/6] Exporting best model artifacts to /kaggle/working/...")
best_dir = out_run / "best"
working_dir = Path("/kaggle/working")

if best_dir.exists():
    for f in best_dir.iterdir():
        if f.is_file():
            shutil.copy(f, working_dir / f.name)
            print(f"  • Exported: {f.name} ({f.stat().st_size / 1e3:.1f} KB)")

shutil.copy(base_model_dir / "scaler.npz", working_dir / "scaler.npz")
if (out_run / "scores.json").exists():
    shutil.copy(out_run / "scores.json", working_dir / "scores.json")

print("\n" + "=" * 70)
print(f"SUCCESS: Fine-tuning finished! New model: ttm_c256_h96_ft_{newest_day}")
print("=" * 70)
