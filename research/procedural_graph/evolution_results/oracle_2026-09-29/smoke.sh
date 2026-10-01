#!/bin/bash
set -euo pipefail
BASE=/home/alex/kagg-evo/oracle-20260929
PG=$BASE/repo/research/procedural_graph
PY=/home/alex/kagg-evo/venv/bin/python
ART=$PG/evolution_results/oracle_2026-09-29
trap 'code=$?; echo "ORACLE_SMOKE_EXIT=$code"' EXIT
BUNDLE=$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["bundle"])' "$ART/deployment.json")
for seat in 0 1; do
    "$PY" "$ART/smoke_oracle.py" --bundle "$BASE/run/bundles/$BUNDLE" \
        --opponent "$BASE/run/bundles/tetsutani_demand_0927" \
        --rpc "$BASE/run/bundle_agent.py" --seat "$seat" --out "$BASE/smoke_seat$seat.json"
done
