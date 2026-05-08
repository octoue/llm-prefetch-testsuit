#!/bin/bash
# E1+E2: malicious-flood stress.
# Sweeps prefetch QPS and (optionally) prefetch quota ratio.
#
# Required env: a vLLM server already started by start_vllm_stress.sh.
# Usage:
#   ./run_stress_s1.sh [results_root]
#
# Tunables via env:
#   PREFETCH_QPS_LIST="50 200 800"         attack QPS sweep
#   RATIO_LIST="0.3"                        prefetch quota ratio (set to
#                                          "0 0.1 0.3 0.5" for E2)
#   DURATION_SEC=300
#   REPEAT=3
#   ATTACK_PREFIX_FILE=...                  override default attack prefixes

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_stress_common.sh"

ROOT="${1:-$RUN_EXP_DIR/results/stress/s1_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$ROOT"

PREFETCH_QPS_LIST="${PREFETCH_QPS_LIST:-50 200 800}"
RATIO_LIST="${RATIO_LIST:-0.3}"
REPEAT="${REPEAT:-3}"

echo "S1 sweep: PREFETCH_QPS=[$PREFETCH_QPS_LIST] RATIO=[$RATIO_LIST] REPEAT=$REPEAT"
echo "Server is expected to honour --max-prefetch-block-ratio=<ratio>;"
echo "if RATIO_LIST has more than one value, please restart the server"
echo "between values OR send each ratio via the engine config endpoint."
echo ""

for ratio in $RATIO_LIST; do
  for qps in $PREFETCH_QPS_LIST; do
    for rep in $(seq 1 "$REPEAT"); do
      unit="$ROOT/ratio${ratio}_qps${qps}_rep${rep}"
      PREFETCH_QPS="$qps" \
      RATIO_OVERRIDE="$ratio" \
        run_one_unit "$unit" s1
    done
  done
done

echo "Done. Aggregate with:"
echo "  python3 result-analysis/analyze_stress.py --root $ROOT --scenario s1"
