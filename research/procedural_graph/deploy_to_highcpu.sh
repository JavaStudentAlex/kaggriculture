#!/usr/bin/env bash
set -euo pipefail

INSTANCE_NAME="evo-80"
INSTANCE_TYPE="n2d-highcpu-80"

echo "[1/6] Checking Brev instance $INSTANCE_NAME..."
if ! brev ls | grep -q "$INSTANCE_NAME"; then
    echo "Creating instance $INSTANCE_NAME ($INSTANCE_TYPE)..."
    brev create "$INSTANCE_NAME" --type "$INSTANCE_TYPE"
else
    echo "Instance $INSTANCE_NAME already exists."
fi

echo "[2/6] Waiting for instance to be ready..."
brev refresh

echo "[3/6] Setting up SSH connection..."
# Get SSH hostname/port for instance
SSH_HOST=$(brev ls | grep "$INSTANCE_NAME" | awk '{print $1}')
echo "Target host: $SSH_HOST"

echo "[4/6] Syncing code and state..."
brev copy /home/alex/kaggriculture "$INSTANCE_NAME:/home/shadeform/"

echo "[5/6] Remote environment setup..."
# Run setup script on remote machine
brev exec "$INSTANCE_NAME" -- bash -c '
    set -euo pipefail
    sudo apt-get update -y && sudo apt-get install -y tmux curl rsync htop
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    cd /home/shadeform/kaggriculture
    uv venv --python 3.12 .venv
    source .venv/bin/activate
    uv pip install kaggle-environments kaggle numpy pandas scipy pyyaml requests sentence-transformers
'

echo "[6/6] Verifying remote LLM access and launching..."
echo "Ready for tunnel forward and execution."
