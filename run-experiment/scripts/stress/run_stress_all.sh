#!/bin/bash
# run_stress_all.sh — Three-phase stress defense experiment for thesis.
#
# Proves the three-layer defense (scheduler NO_HIT discard + quota/TTL +
# API rate limiter) bounds overhead from extreme prefetch scenarios.
#
# Phase 1  PROTECTED  (ratio=0.3, TTL=60s)        — thesis default
#   S1 QPS=200 × REPEAT    scheduler handles flood via NO_HIT discard
#   S2 burst=5 × REPEAT    TTL reclaim + quota bound misclick overhead
#
# Phase 2  UNPROTECTED (ratio=0, TTL=0)            — defense disabled
#   S1 QPS=200 × REPEAT    comparison baseline (for S1 NO_HIT, same as P1)
#   S2 burst=5 × REPEAT    no TTL reclaim: prefetch blocks not reclaimed
#
# Phase 3  PROTECTED + RATE LIMIT (ratio=0.3, TTL=60s, rate_limit=5)
#   S1 QPS=200 × REPEAT    full defense: rate limiter absorbs flood
#
# Server is restarted between phases (different configs). Within a phase,
# prefix cache is reset between units for isolation (see _stress_common.sh).
#
# Total: 5 × REPEAT runs. Default REPEAT=3 → 15 runs × ~6 min ≈ 90 min
# plus ~5 min server restart per phase → ~2 hours total.
#
# Usage:
#   cd llm-prefetch-testsuit/run-experiment
#   bash scripts/stress/run_stress_all.sh [results_root]
#
# Env overrides (all optional):
#   DURATION_SEC=300             per-unit wall time (default 5 min)
#   REPEAT=3                    repeats per config point
#   PREFETCH_RATE_LIMIT=5       Phase 3 rate limit (req/s)
#   RESTART_PER_UNIT=0          set 1 to restart vLLM between every unit

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_stress_common.sh"

ROOT="${1:-$RUN_EXP_DIR/results/stress/all_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$ROOT"

REPEAT="${REPEAT:-3}"
S2_BURST="${S2_BURST:-5}"
RATE_LIMIT="${PREFETCH_RATE_LIMIT:-5}"
RESTART_PER_UNIT="${RESTART_PER_UNIT:-0}"
export DURATION_SEC="${DURATION_SEC:-300}"

TOTAL_UNITS=$(( REPEAT * 5 ))
CURRENT_UNIT=0

SERVER_PID=""
CURRENT_RATIO=""
CURRENT_TTL=""
CURRENT_RL=""

start_server() {
  local ratio="$1" ttl_ms="$2" rate_limit="${3:-0}"
  local log="$ROOT/vllm_ratio${ratio}_ttl${ttl_ms}_rl${rate_limit}.log"

  echo ""
  echo "================================================================"
  echo "  Starting vLLM   ratio=$ratio  ttl_ms=$ttl_ms  rate_limit=$rate_limit"
  echo "================================================================"

  bash "$SCRIPT_DIR/start_vllm_stress.sh" \
       --ratio "$ratio" --ttl-ms "$ttl_ms" --rate-limit "$rate_limit" \
       --log "$log" &
  SERVER_PID=$!
  CURRENT_RATIO="$ratio"
  CURRENT_TTL="$ttl_ms"
  CURRENT_RL="$rate_limit"

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

run_unit() {
  CURRENT_UNIT=$((CURRENT_UNIT + 1))
  echo ""
  echo "────────────────────────────────────────"
  echo "  [$CURRENT_UNIT/$TOTAL_UNITS] $1"
  echo "────────────────────────────────────────"

  if [ "$RESTART_PER_UNIT" = "1" ] && [ "$CURRENT_UNIT" -gt 1 ]; then
    stop_server
    start_server "$CURRENT_RATIO" "$CURRENT_TTL" "$CURRENT_RL"
  fi

  shift
  run_one_unit "$@"
}

cleanup() { [ -n "$SERVER_PID" ] && stop_server || true; }
trap cleanup EXIT

START_TIME=$(date +%s)

echo ""
echo "=========================================="
echo "  Extreme prefetch failure experiment"
echo "  $(date)"
echo "  Results -> $ROOT"
echo "  Runs: $TOTAL_UNITS (${DURATION_SEC}s each, REPEAT=$REPEAT)"
echo "=========================================="

# ───────────────────────────────────────────────────────
#  Phase 1: Protected (thesis default config)
# ───────────────────────────────────────────────────────
start_server 0.3 60000

echo ""
echo "===== Phase 1: PROTECTED (ratio=0.3, TTL=60s) ====="

for r in $(seq 1 "$REPEAT"); do
  PREFETCH_QPS=200 \
    run_unit "P1 S1 qps200 rep$r" "$ROOT/protected_s1_qps200_rep${r}" s1
done

for r in $(seq 1 "$REPEAT"); do
  BURST_SIZE="$S2_BURST" \
    run_unit "P1 S2 burst$S2_BURST rep$r" "$ROOT/protected_s2_burst${S2_BURST}_rep${r}" s2
done

stop_server

# ───────────────────────────────────────────────────────
#  Phase 2: Unprotected (no quota, no TTL)
# ───────────────────────────────────────────────────────
start_server 0 0

echo ""
echo "===== Phase 2: UNPROTECTED (ratio=0, TTL=0) ====="

for r in $(seq 1 "$REPEAT"); do
  PREFETCH_QPS=200 \
    run_unit "P2 S1 qps200 rep$r" "$ROOT/unprotected_s1_qps200_rep${r}" s1
done

for r in $(seq 1 "$REPEAT"); do
  BURST_SIZE="$S2_BURST" \
    run_unit "P2 S2 burst$S2_BURST rep$r" "$ROOT/unprotected_s2_burst${S2_BURST}_rep${r}" s2
done

stop_server

# ───────────────────────────────────────────────────────
#  Phase 3: Protected + API rate limit (full defense)
# ───────────────────────────────────────────────────────
start_server 0.3 60000 "$RATE_LIMIT"

echo ""
echo "===== Phase 3: PROTECTED + RATE LIMIT (${RATE_LIMIT} req/s) ====="

for r in $(seq 1 "$REPEAT"); do
  PREFETCH_QPS=200 \
    run_unit "P3 S1 qps200 rep$r" "$ROOT/ratelimit_s1_qps200_rep${r}" s1
done

stop_server

# ───────────────────────────────────────────────────────
#  Post-processing
# ───────────────────────────────────────────────────────
echo ""
echo "===== Post-processing ====="

python3 "$TESTSUIT_ROOT/result-analysis/analyze_stress.py" --root "$ROOT" || true

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
ELAPSED_MIN=$(( ELAPSED / 60 ))

echo ""
echo "=========================================="
echo "  Done.  $(date)"
echo "  Wall time: ${ELAPSED_MIN} min"
echo "  Results:   $ROOT"
echo "=========================================="
echo ""
echo "Key comparisons:"
echo "  1. protected vs ratelimit S1  →  rate limit eliminates scheduler flood overhead"
echo "  2. protected vs unprotected S2  →  TTL+quota bounds misclick overhead"
echo "  3. All configs: PCIe events  →  NO_HIT path = zero wasted PCIe"
echo ""
echo "Generate thesis figures:"
echo "  python3 paper-figs/prefetch/fig_stress_defense.py --root $ROOT"
