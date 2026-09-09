#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

cd "$ROOT_DIR"
uv sync --group dev
"$ROOT_DIR/.github/scripts/install_aps_tools.sh"
