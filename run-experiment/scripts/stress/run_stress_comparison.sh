#!/bin/bash
# run_stress_comparison.sh — 补测：vLLM 基线 vs LLM-Prefetch+S2 对照
#
# 目的：证明在 S2 误触压力下，LLM-Prefetch 的真实推理性能不劣于无预取的 vLLM 基线。
#
# Phase 1  BASELINE (ratio=0, TTL=0, mode=vanilla)
#   在不同 QPS 下运行纯推理负载（无任何预取），作为 vLLM 基线。
#
# Phase 2  PREFETCH + S2 BURST (ratio=0.3, TTL=60s)
#   在相同 QPS 下运行推理+预取负载，叠加 S2 burst 压力。
#
# 每组 REPEAT=1，QPS 测 3 个点。总计 6 runs × 5 min ≈ 35 min。
#
# Usage:
#   cd llm-prefetch-testsuit/run-experiment
#   bash scripts/stress/run_stress_comparison.sh [results_root]
#
# Env overrides:
#   DURATION_SEC=300        per-unit wall time
#   QPS_LIST="0.5 1.0"     space-separated QPS levels to test
#   S2_BURST=5              burst size for S2 overlay

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_stress_common.sh"

ROOT="${1:-$RUN_EXP_DIR/results/stress/comparison_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$ROOT"

QPS_LIST="${QPS_LIST:-0.5 1.0}"
S2_BURST="${S2_BURST:-5}"
export DURATION_SEC="${DURATION_SEC:-300}"

# Count total units
N_QPS=$(echo $QPS_LIST | wc -w | tr -d ' ')
TOTAL_UNITS=$(( N_QPS * 2 ))
CURRENT_UNIT=0

SERVER_PID=""
CURRENT_RATIO=""
CURRENT_TTL=""
CURRENT_RL=""

start_server() {
  local ratio="$1" ttl_ms="$2" rate_limit="${3:-0}"
  local log="$ROOT/vllm_ratio${ratio}_ttl${ttl_ms}.log"

  echo ""
  echo "================================================================"
  echo "  Starting vLLM   ratio=$ratio  ttl_ms=$ttl_ms"
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
}

stop_server() {
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

# ─────────────────────────────────────────────────
# Baseline runner: vanilla mode, real requests only
# ─────────────────────────────────────────────────
run_baseline_unit() {
  local results_dir="$1"
  local qps="$2"
  local duration="${DURATION_SEC:-300}"

  CURRENT_UNIT=$((CURRENT_UNIT + 1))
  echo ""
  echo "────────────────────────────────────────"
  echo "  [$CURRENT_UNIT/$TOTAL_UNITS] baseline qps=$qps"
  echo "────────────────────────────────────────"

  mkdir -p "$results_dir"

  # Reset prefix cache
  curl -s -X POST "http://localhost:${API_PORT}/reset_prefix_cache?reset_external=true" \
       > /dev/null || true
  sleep 3

  # Read GPU IDs for power sampling
  local stress_gpu_ids=""
  if [ -f "$RUN_EXP_DIR/.stress_gpus" ]; then
    stress_gpu_ids=$(cat "$RUN_EXP_DIR/.stress_gpus" 2>/dev/null || true)
  fi

  # Power sampler
  bash "$UTILS_DIR/sample_power.sh" \
        "$results_dir/power.csv" "$duration" "$stress_gpu_ids" &
  local power_pid=$!

  # Run vanilla (no-prefetch) workload
  echo "    launching vanilla runner at qps=$qps"
  python3 "$RUN_EXP_DIR/prefetch_ab_runner.py" \
      --trace-file "$DEFAULT_BG_TRACE" \
      --mode vanilla \
      --qps "$qps" \
      --num-multi-turn "${BASELINE_NUM_CONV:-40}" \
      --model "$MODEL" \
      --api-base "$API_BASE" \
      --output "$results_dir/requests.jsonl" \
      --timeout "$duration" \
      --schedule-mode "${SCHEDULE_MODE:-scaled-timestamp}" \
      &> "$results_dir/runner.log" || true

  # Stop power sampler
  kill -TERM "$power_pid" 2>/dev/null || true
  wait "$power_pid" 2>/dev/null || true

  # Extract TTFT/TPOT from requests.jsonl
  python3 -c "
import json, sys, statistics
ttfts, tpots = [], []
with open('$results_dir/requests.jsonl') as f:
    for line in f:
        r = json.loads(line.strip())
        if r.get('success') and r.get('ttft_ms') is not None:
            ttfts.append(r['ttft_ms'])
        if r.get('success') and r.get('tpot_ms') is not None:
            tpots.append(r['tpot_ms'])
summary = {
    'config': 'baseline', 'qps': $qps,
    'real_count': len(ttfts),
    'ttft_mean': statistics.mean(ttfts) if ttfts else None,
    'ttft_p50': sorted(ttfts)[len(ttfts)//2] if ttfts else None,
    'ttft_p95': sorted(ttfts)[int(len(ttfts)*0.95)] if ttfts else None,
    'tpot_mean': statistics.mean(tpots) if tpots else None,
}
with open('$results_dir/summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f'    baseline qps={$qps}: n={len(ttfts)}, TTFT_mean={summary[\"ttft_mean\"]:.1f}ms, TPOT_mean={summary[\"tpot_mean\"]:.1f}ms')
" || echo "    WARN: summary extraction failed"

  sleep 10
}

# ─────────────────────────────────────────────────
# Prefetch + S2 runner: reuse run_one_unit from _stress_common.sh
# ─────────────────────────────────────────────────
run_prefetch_s2_unit() {
  local results_dir="$1"
  local qps="$2"

  CURRENT_UNIT=$((CURRENT_UNIT + 1))
  echo ""
  echo "────────────────────────────────────────"
  echo "  [$CURRENT_UNIT/$TOTAL_UNITS] prefetch+S2 qps=$qps burst=$S2_BURST"
  echo "────────────────────────────────────────"

  QPS="$qps" BURST_SIZE="$S2_BURST" BASELINE_QPS="$qps" \
    run_one_unit "$results_dir" s2
}

START_TIME=$(date +%s)

echo ""
echo "=========================================="
echo "  Baseline vs Prefetch+S2 comparison"
echo "  $(date)"
echo "  Results -> $ROOT"
echo "  QPS levels: $QPS_LIST"
echo "  Duration: ${DURATION_SEC}s per unit, $TOTAL_UNITS units total"
echo "=========================================="

# ───────────────────────────────────────────────────────
#  Phase 1: vLLM Baseline (no prefetch)
# ───────────────────────────────────────────────────────
start_server 0 0

echo ""
echo "===== Phase 1: vLLM BASELINE (ratio=0, TTL=0, vanilla mode) ====="

for qps in $QPS_LIST; do
  run_baseline_unit "$ROOT/baseline_qps${qps}_rep1" "$qps"
done

stop_server

# ───────────────────────────────────────────────────────
#  Phase 2: LLM-Prefetch + S2 burst
# ───────────────────────────────────────────────────────
start_server 0.3 60000

echo ""
echo "===== Phase 2: LLM-Prefetch + S2 BURST (ratio=0.3, TTL=60s, burst=$S2_BURST) ====="

for qps in $QPS_LIST; do
  run_prefetch_s2_unit "$ROOT/prefetch_s2_qps${qps}_rep1" "$qps"
done

stop_server

# ───────────────────────────────────────────────────────
#  Summary
# ───────────────────────────────────────────────────────
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
echo "Comparison:"
echo ""
printf "  %-35s  %-12s  %-12s\n" "Config" "TPOT (ms)" "TTFT (ms)"
echo "  -----------------------------------------------------------"
for qps in $QPS_LIST; do
  for d in "$ROOT/baseline_qps${qps}_rep1" "$ROOT/prefetch_s2_qps${qps}_rep1"; do
    if [ -f "$d/summary.json" ]; then
      python3 -c "
import json
with open('$d/summary.json') as f: s = json.load(f)
tpot = s.get('tpot_mean') or s.get('real_latency_ms',{}).get('tpot_mean')
ttft = s.get('ttft_mean') or s.get('real_latency_ms',{}).get('ttft_mean')
name = '$(basename $d)'
print(f'  {name:<35s}  {tpot:>10.1f}ms  {ttft:>10.1f}ms')
" 2>/dev/null || echo "  $(basename $d)  (parse error)"
    fi
  done
done
echo ""
echo "后台运行:"
echo "  nohup bash scripts/stress/run_stress_comparison.sh > results/stress/comparison.log 2>&1 &"
