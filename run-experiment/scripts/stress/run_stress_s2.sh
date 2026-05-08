#!/bin/bash
# E3: misclick-burst stress.
# Sweeps the burst size N (number of prefetches per real-request window) and
# the TTL knob (with TTL_LIST="0 60000" for the on/off ablation).
#
# Usage: ./run_stress_s2.sh [results_root]
#
# Tunables:
#   BURST_SIZE_LIST="2 5 10"
#   TTL_LIST="60000"
#   QPS=1.0   BURST_INTERVAL_MS=50
#   DURATION_SEC=300   REPEAT=3

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_stress_common.sh"

ROOT="${1:-$RUN_EXP_DIR/results/stress/s2_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$ROOT"

BURST_SIZE_LIST="${BURST_SIZE_LIST:-2 5 10}"
TTL_LIST="${TTL_LIST:-60000}"
REPEAT="${REPEAT:-3}"

for ttl in $TTL_LIST; do
  for n in $BURST_SIZE_LIST; do
    for rep in $(seq 1 "$REPEAT"); do
      unit="$ROOT/ttl${ttl}_burst${n}_rep${rep}"
      BURST_SIZE="$n" \
      TTL_OVERRIDE="$ttl" \
        run_one_unit "$unit" s2
    done
  done
done

echo "Done. Aggregate with:"
echo "  python3 result-analysis/analyze_stress.py --root $ROOT --scenario s2"
