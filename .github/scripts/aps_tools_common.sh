#!/usr/bin/env bash

ensure_aps_tools() {
  if [ "$#" -eq 0 ]; then
    echo "ensure_aps_tools requires at least one command name." >&2
    return 2
  fi

  local root_dir
  root_dir="${APS_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
  export PATH="$root_dir/.venv/bin:$PATH"

  local missing=()
  local tool
  for tool in "$@"; do
    if ! command -v "$tool" >/dev/null 2>&1; then
      missing+=("$tool")
    fi
  done

  if [ "${#missing[@]}" -gt 0 ]; then
    echo "APS tools missing: ${missing[*]}; running .github/scripts/install_aps_tools.sh." >&2
    "$root_dir/.github/scripts/install_aps_tools.sh"
    export PATH="$root_dir/.venv/bin:$PATH"
  fi

  local failed=0
  for tool in "$@"; do
    if ! command -v "$tool" >/dev/null 2>&1; then
      echo "$tool is required but is still unavailable after APS tool bootstrap." >&2
      failed=1
    fi
  done

  if [ "$failed" -ne 0 ]; then
    echo "Run .github/scripts/bootstrap_dev_env.sh or .github/scripts/install_aps_tools.sh and confirm Go 1.22+ is available." >&2
  fi

  return "$failed"
}
