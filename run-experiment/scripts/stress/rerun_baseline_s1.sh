#!/bin/bash
# rerun_baseline_s1.sh — 补跑 baseline_s1_rep1
#
# 自动启动 vLLM（ratio=0, TTL=0），运行 baseline S1
# （baseline 模式背景负载 + 200 QPS S1 泛洪），提取指标后关闭服务器。
#
# Usage:
#   cd llm-prefetch-testsuit/run-experiment
#   bash scripts/stress/rerun_baseline_s1.sh [results_dir]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_stress_common.sh"

if [ -n "$1" ]; then
  ROOT="$1"
else
  ROOT=$(ls -dt "$RUN_EXP_DIR/results/stress/3way_"* 2>/dev/null | head -1)
  [ -z "$ROOT" ] && { echo "FATAL: 找不到 3way_* 结果目录，请手动指定" >&2; exit 1; }
fi

OUT="$ROOT/baseline_s1_rep1"
echo "结果目录: $OUT"

export DURATION_SEC="${DURATION_SEC:-300}"
S1_QPS="${S1_QPS:-200}"
BG_QPS="${BASELINE_QPS:-1.0}"
BG_CONV="${BASELINE_NUM_CONV:-40}"

SERVER_PID=""

start_server() {
  local log="$ROOT/vllm_rerun_baseline_s1.log"

  echo ""
  echo "================================================================"
  echo "  Starting vLLM   ratio=0  ttl_ms=0  (baseline config)"
  echo "================================================================"

  bash "$SCRIPT_DIR/start_vllm_stress.sh" \
       --ratio 0 --ttl-ms 0 --rate-limit 0 \
       --log "$log" &
  SERVER_PID=$!

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

# ── 启动服务器 ────────────────────────────────────────
start_server

# ── 运行 baseline S1 ─────────────────────────────────
echo ""
echo "────────────────────────────────────────"
echo "  baseline S1 (baseline mode + flood qps=$S1_QPS)"
echo "────────────────────────────────────────"

mkdir -p "$OUT"

curl -s -X POST "http://localhost:${API_PORT}/reset_prefix_cache?reset_external=true" \
     > /dev/null || true
sleep 3

curl -s -X POST "http://localhost:${API_PORT}/start_profile" >/dev/null || true

stress_gpu_ids=""
[ -f "$RUN_EXP_DIR/.stress_gpus" ] && \
  stress_gpu_ids=$(cat "$RUN_EXP_DIR/.stress_gpus" 2>/dev/null || true)

bash "$UTILS_DIR/sample_power.sh" \
      "$OUT/power.csv" "$DURATION_SEC" "$stress_gpu_ids" &
power_pid=$!
bash "$UTILS_DIR/sample_perf.sh" \
      "$OUT/perf_pkg.txt" "$DURATION_SEC" &
perf_pid=$!

echo "    launching baseline workload at qps=$BG_QPS, conv=$BG_CONV, duration=${DURATION_SEC}s"
python3 "$RUN_EXP_DIR/prefetch_ab_runner.py" \
    --trace-file "$DEFAULT_BG_TRACE" \
    --mode baseline \
    --qps "$BG_QPS" \
    --num-multi-turn "$BG_CONV" \
    --model "$MODEL" \
    --api-base "$API_BASE" \
    --output "$OUT/baseline.jsonl" \
    --timeout "$DURATION_SEC" \
    --schedule-mode "${SCHEDULE_MODE:-scaled-timestamp}" \
    &> "$OUT/baseline.log" &
baseline_pid=$!

head_start="${BASELINE_HEAD_START:-15}"
echo "    baseline head start: ${head_start}s"
sleep "$head_start"

overlay_dur=$(( DURATION_SEC - head_start ))
[ "$overlay_dur" -lt 30 ] && overlay_dur=30
echo "    launching S1 flood for ${overlay_dur}s at qps=$S1_QPS"
rc=0
python3 "$RUN_EXP_DIR/prefetch_stress_runner.py" \
    --scenario s1 \
    --api-base "$API_BASE" \
    --model "$MODEL" \
    --duration-sec "$overlay_dur" \
    --output-jsonl "$OUT/requests.jsonl" \
    --output-summary "$OUT/summary.json" \
    --prefetch-qps "$S1_QPS" \
    --attack-prefix-file "${ATTACK_PREFIX_FILE:-$DEFAULT_ATTACK_PREFIX}" \
    &> "$OUT/runner.log" || rc=$?
echo "$rc" > "$OUT/runner.exit_code"
[ "$rc" -ne 0 ] && echo "    WARN: overlay exited with code $rc" >&2

wait "$baseline_pid" 2>/dev/null || true

kill -TERM "$power_pid" 2>/dev/null || true
kill -TERM "$perf_pid" 2>/dev/null || true
wait "$power_pid" 2>/dev/null || true
wait "$perf_pid" 2>/dev/null || true

curl -s -X POST "http://localhost:${API_PORT}/stop_profile" >/dev/null || true
sleep 2
bash "$UTILS_DIR/sample_pcie.sh" "$OUT/pcie_events.json" \
     "${PCIE_PROFILER_DIR:-$RUN_EXP_DIR/profiler_output}" || true

for required in baseline.jsonl requests.jsonl summary.json power.csv; do
  [ ! -s "$OUT/$required" ] && \
    echo "    WARN: $OUT/$required is empty or missing" >&2
done

# ── 提取指标 ──────────────────────────────────────────
python3 -c "
import json, statistics, sys
ttfts, tpots = [], []
with open('$OUT/baseline.jsonl') as f:
    for line in f:
        line = line.strip()
        if not line: continue
        r = json.loads(line)
        if r.get('success') and r.get('ttft_ms') is not None:
            ttfts.append(r['ttft_ms'])
        if r.get('success') and r.get('tpot_ms') is not None:
            tpots.append(r['tpot_ms'])
if not ttfts:
    print('    WARN: no successful requests found', file=sys.stderr)
    json.dump({'config':'baseline','scenario':'s1','real_count':0},
              open('$OUT/real_latency.json','w'), indent=2)
    sys.exit(0)
s_ttfts = sorted(ttfts)
info = {
    'config': 'baseline', 'scenario': 's1',
    'real_count': len(ttfts),
    'ttft_mean': round(statistics.mean(ttfts), 1),
    'ttft_p50':  round(s_ttfts[len(s_ttfts)//2], 1),
    'ttft_p95':  round(s_ttfts[int(len(s_ttfts)*0.95)], 1),
    'tpot_mean': round(statistics.mean(tpots), 1) if tpots else None,
}
with open('$OUT/real_latency.json', 'w') as f:
    json.dump(info, f, indent=2)
print(f'    n={len(ttfts)}  TTFT mean={info[\"ttft_mean\"]}ms  p95={info[\"ttft_p95\"]}ms  TPOT={info[\"tpot_mean\"]}ms')
" || echo "    WARN: metric extraction failed"

# ── 关闭服务器 ────────────────────────────────────────
stop_server

echo ""
echo "=========================================="
echo "  Done. Results: $OUT"
echo "  real_latency.json + baseline.jsonl + summary.json + power.csv + pcie_events.json"
echo "=========================================="
