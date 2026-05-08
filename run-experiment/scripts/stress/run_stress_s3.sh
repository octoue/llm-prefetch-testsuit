#!/bin/bash
# E4: mixed-background stress (isolation experiment).
#
# Usage: ./run_stress_s3.sh [results_root]
#
# Tunables:
#   MIX_RATIO_LIST="0.25 1.0 4.0"
#   QPS=1.0
#   DURATION_SEC=300   REPEAT=3

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_stress_common.sh"

ROOT="${1:-$RUN_EXP_DIR/results/stress/s3_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$ROOT"

MIX_RATIO_LIST="${MIX_RATIO_LIST:-0.25 1.0 4.0}"
REPEAT="${REPEAT:-3}"

for mix in $MIX_RATIO_LIST; do
  for rep in $(seq 1 "$REPEAT"); do
    unit="$ROOT/mix${mix}_rep${rep}"
    MIX_RATIO="$mix" \
      run_one_unit "$unit" s3
  done
done

echo "Done. Aggregate with:"
echo "  python3 result-analysis/analyze_stress.py --root $ROOT --scenario s3"
