#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=.github/scripts/aps_tools_common.sh
. "$ROOT_DIR/.github/scripts/aps_tools_common.sh"
cd "$ROOT_DIR"

mapfile -t features < <(git ls-files '*.feature' | grep -Ev '(^|/)(\.agents|\.codex|\.gsd|\.skills|skills)/' || true)
if [ "${#features[@]}" -eq 0 ]; then
  echo "No .feature files found; skipping APS/Gherkin parse gate."
  exit 0
fi

ensure_aps_tools gherkin-parser gherkin-ir-dry-checker

mkdir -p build/pre-commit/acceptance/ir build/pre-commit/acceptance/dry
for feature in "${features[@]}"; do
  slug="${feature//\//__}"
  slug="${slug// /__}"
  ir="build/pre-commit/acceptance/ir/${slug%.feature}.json"
  dry="build/pre-commit/acceptance/dry/${slug%.feature}.json"
  gherkin-parser "$feature" "$ir"
  gherkin-ir-dry-checker --include-exact "$ir" "$dry"
done
