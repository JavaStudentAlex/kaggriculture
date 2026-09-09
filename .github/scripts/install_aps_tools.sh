#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
APS_REPO_URL="${APS_REPO_URL:-https://github.com/unclebob/Acceptance-Pipeline-Specification.git}"
APS_TOOLS_REF="${APS_TOOLS_REF:-accaa33d503340c56513ef387258f8da929ba902}"
APS_TOOLS_CACHE_DIR="${APS_TOOLS_CACHE_DIR:-$ROOT_DIR/build/aps-tools}"
APS_TOOLS_INSTALL_DIR="${APS_TOOLS_INSTALL_DIR:-$ROOT_DIR/.venv/bin}"
APS_TOOLS_SRC_DIR="$APS_TOOLS_CACHE_DIR/src"

if ! command -v go >/dev/null 2>&1; then
  echo "Go is required to install APS fallback command binaries." >&2
  echo "Install Go 1.22+ or use a CI job with actions/setup-go before running this script." >&2
  exit 1
fi

mkdir -p "$APS_TOOLS_CACHE_DIR" "$APS_TOOLS_INSTALL_DIR"

if [ ! -d "$APS_TOOLS_SRC_DIR/.git" ]; then
  git clone --no-checkout "$APS_REPO_URL" "$APS_TOOLS_SRC_DIR"
fi

if ! git -C "$APS_TOOLS_SRC_DIR" fetch --depth 1 origin "$APS_TOOLS_REF"; then
  git -C "$APS_TOOLS_SRC_DIR" fetch origin
fi
git -C "$APS_TOOLS_SRC_DIR" checkout --detach "$APS_TOOLS_REF"

for tool in gherkin-parser gherkin-ir-dry-checker gherkin-mutator; do
  go -C "$APS_TOOLS_SRC_DIR" build -o "$APS_TOOLS_INSTALL_DIR/$tool" "./cmd/$tool"
done

printf '%s\n' "$APS_TOOLS_REF" > "$APS_TOOLS_INSTALL_DIR/.aps-tools-ref"

if [ -n "${GITHUB_PATH:-}" ]; then
  printf '%s\n' "$APS_TOOLS_INSTALL_DIR" >> "$GITHUB_PATH"
fi

printf 'Installed APS tools at %s from %s\n' "$APS_TOOLS_INSTALL_DIR" "$APS_TOOLS_REF"
