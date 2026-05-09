#!/bin/bash
# run_stress_all.sh — Complete extreme-prefetch-failure experiment.
#
# Answers the reviewer's question:
#   "Under malicious flood or high-frequency misclick, are PCIe wake-ups,
#    wasted I/O, power, and backend bandwidth bounded by admission control
#    and TTL reclaim?"
#
# Phase 1  PROTECTED  (ratio=0.3, TTL=60s — thesis default)
#   S1 QPS sweep {50, 200, 800} x3   overhead saturates with quota
#   S2 burst=5 x3                    TTL reclaim handles misclick waste
#   S3 mix=1.0 x3                    legitimate users unaffected
#
# Phase 2  UNPROTECTED  (ratio=0, TTL=0 — no protection)
#   S1 QPS=200 x3                    comparison: quota is necessary
#   S2 burst=5 x3                    comparison: TTL is necessary
#
# Total: 21 runs x ~6 min = ~2.5 hours.
#
# Usage:
#   cd llm-prefetch-testsuit/run-experiment
#   bash scripts/stress/run_stress_all.sh [results_root]
#
# Env overrides (all optional):
#   DURATION_SEC=300        per-unit wall time (default 5 min)
#   REPEAT=3                repeats per config point
#   S1_QPS="50 200 800"    QPS sweep for S1

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_stress_common.sh"

ROOT="${1:-$RUN_EXP_DIR/results/stress/all_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$ROOT"

REPEAT="${REPEAT:-3}"
S1_QPS="${S1_QPS:-50 200 800}"
S2_BURST="${S2_BURST:-5}"
S3_MIX="${S3_MIX:-1.0}"
export DURATION_SEC="${DURATION_SEC:-300}"

SERVER_PID=""

start_server() {
  local ratio="$1" ttl_ms="$2"
  local log="$ROOT/vllm_ratio${ratio}_ttl${ttl_ms}.log"

  echo ""
  echo "================================================================"
  echo "  Starting vLLM   ratio=$ratio  ttl_ms=$ttl_ms"
  echo "================================================================"

  bash "$SCRIPT_DIR/start_vllm_stress.sh" \
       --ratio "$ratio" --ttl-ms "$ttl_ms" --log "$log" &
  SERVER_PID=$!

  local w=0
  until curl -sf "http://localhost:${API_PORT}/health" >/dev/null 2>&1; do
    sleep 5; w=$((w + 5))
    if [ "$w" -ge 300 ]; then
      echo "FATAL: server not ready after ${w}s" >&2; exit 1
    fi
  done
  echo "  Ready (${w}s)."

  local n
  n=$(curl -s "http://localhost:${API_PORT}/metrics" \
       | grep -c "vllm:prefetch_" || true)
  if [ "$n" -lt 5 ]; then
    echo "FATAL: $n/5 prefetch metrics found — wrong vLLM version?" >&2
    exit 1
  fi
  echo "  Metrics verified ($n counters)."
}

stop_server() {
  echo ""
  echo "  Stopping server..."
  pkill -f 'vllm serve' 2>/dev/null || true
  sleep 3
  [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null || true
  [ -n "$SERVER_PID" ] && wait "$SERVER_PID" 2>/dev/null || true
  SERVER_PID=""
  sleep 10
  echo "  Server stopped."
}

cleanup() { [ -n "$SERVER_PID" ] && stop_server || true; }
trap cleanup EXIT

echo ""
echo "=========================================="
echo "  Extreme prefetch failure experiment"
echo "  $(date)"
echo "  Results -> $ROOT"
echo "=========================================="

# ───────────────────────────────────────────────────────
#  Phase 1: Protected (thesis default config)
# ───────────────────────────────────────────────────────
start_server 0.3 60000

echo ""
echo "===== Phase 1: PROTECTED (ratio=0.3, TTL=60s) ====="

# S1: malicious flood — QPS sweep
for qps in $S1_QPS; do
  for r in $(seq 1 "$REPEAT"); do
    PREFETCH_QPS="$qps" \
      run_one_unit "$ROOT/protected_s1_qps${qps}_rep${r}" s1
  done
done

# S2: user misclick burst
for r in $(seq 1 "$REPEAT"); do
  BURST_SIZE="$S2_BURST" \
    run_one_unit "$ROOT/protected_s2_burst${S2_BURST}_rep${r}" s2
done

# S3: mixed — isolation test
for r in $(seq 1 "$REPEAT"); do
  MIX_RATIO="$S3_MIX" \
    run_one_unit "$ROOT/protected_s3_mix${S3_MIX}_rep${r}" s3
done

stop_server

# ───────────────────────────────────────────────────────
#  Phase 2: Unprotected (no quota, no TTL)
# ───────────────────────────────────────────────────────
start_server 0 0

echo ""
echo "===== Phase 2: UNPROTECTED (ratio=0, TTL=0) ====="

# S1 at QPS=200 for direct comparison with Phase 1
for r in $(seq 1 "$REPEAT"); do
  PREFETCH_QPS=200 \
    run_one_unit "$ROOT/unprotected_s1_qps200_rep${r}" s1
done

# S2 at same burst for direct comparison
for r in $(seq 1 "$REPEAT"); do
  BURST_SIZE="$S2_BURST" \
    run_one_unit "$ROOT/unprotected_s2_burst${S2_BURST}_rep${r}" s2
done

stop_server

# ───────────────────────────────────────────────────────
#  Analysis
# ───────────────────────────────────────────────────────
echo ""
echo "===== Analysis ====="
python3 "$TESTSUIT_ROOT/result-analysis/analyze_stress.py" --root "$ROOT"

echo ""
echo "=========================================="
echo "  Done.  $(date)"
echo "  Results:    $ROOT/aggregate.csv"
echo "  Figures:    $ROOT/figs/"
echo "=========================================="
echo ""
echo "Key comparisons for the thesis:"
echo "  1. protected s1 qps 50/200/800  ->  overhead saturates"
echo "  2. protected vs unprotected s1 qps200  ->  quota bounds overhead"
echo "  3. protected vs unprotected s2  ->  TTL prevents waste accumulation"
echo "  4. protected s3  ->  real TTFT p95 degradation < 20%"
