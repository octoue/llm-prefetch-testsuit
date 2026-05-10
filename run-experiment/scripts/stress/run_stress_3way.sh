#!/bin/bash
# run_stress_3way.sh — 三组对照：baseline vs unprotected vs protected
#
# 在 S1（泛洪攻击）和 S2（用户误触）两种极端场景下，对比三种配置的
# 真实推理性能（TTFT / TPOT / GPU 功耗）：
#
#   Config A  BASELINE      vanilla 模式，无任何预取
#   Config B  UNPROTECTED   预取开启，ratio=0, TTL=0（无保护）
#   Config C  PROTECTED     预取开启，ratio=0.3, TTL=60s, RL=5（完整防御）
#
# S1: 三种配置均叠加相同的 200 QPS 泛洪攻击，衡量攻击对真实推理的影响。
#     Baseline 用 vanilla 背景负载；B/C 用 prefetch 背景负载。
# S2: A 仅运行 vanilla 推理（纯净基线）；B/C 叠加 burst=5 误触。
#
# 服务器在 baseline + unprotected 之间共享（ratio=0, TTL=0），
# protected 阶段重启（ratio=0.3, TTL=60s）。
#
# 默认 REPEAT=1，共 6 个实验单元，约 40 分钟。
#
# Usage:
#   cd llm-prefetch-testsuit/run-experiment
#   bash scripts/stress/run_stress_3way.sh [results_root]
#
# Env overrides:
#   DURATION_SEC=300        每个实验单元持续时间
#   REPEAT=1                重复次数
#   S1_QPS=200              S1 泛洪 QPS
#   S2_BURST=5              S2 burst 大小
#   BASELINE_QPS=1.0        背景推理负载 QPS
#   BASELINE_NUM_CONV=40    背景推理对话数

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_stress_common.sh"

ROOT="${1:-$RUN_EXP_DIR/results/stress/3way_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$ROOT"

export DURATION_SEC="${DURATION_SEC:-300}"
REPEAT="${REPEAT:-1}"
S1_QPS="${S1_QPS:-200}"
S2_BURST="${S2_BURST:-5}"
BG_QPS="${BASELINE_QPS:-1.0}"
BG_CONV="${BASELINE_NUM_CONV:-40}"

TOTAL_UNITS=$(( REPEAT * 6 ))
CURRENT_UNIT=0
SERVER_PID=""

# ── Server lifecycle ──────────────────────────────────

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

# ── Latency extraction helper ─────────────────────────
# Reads a JSONL of request records and writes real_latency.json with
# standardized metrics, so the final summary can compare all 6 runs.
extract_real_latency() {
  local jsonl_file="$1" output_dir="$2" config="$3" scenario="$4"
  python3 -c "
import json, statistics, sys
ttfts, tpots = [], []
with open('${jsonl_file}') as f:
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
    json.dump({'config':'${config}','scenario':'${scenario}','real_count':0},
              open('${output_dir}/real_latency.json','w'), indent=2)
    sys.exit(0)
s_ttfts = sorted(ttfts)
info = {
    'config': '${config}', 'scenario': '${scenario}',
    'real_count': len(ttfts),
    'ttft_mean': round(statistics.mean(ttfts), 1),
    'ttft_p50':  round(s_ttfts[len(s_ttfts)//2], 1),
    'ttft_p95':  round(s_ttfts[int(len(s_ttfts)*0.95)], 1),
    'tpot_mean': round(statistics.mean(tpots), 1) if tpots else None,
}
with open('${output_dir}/real_latency.json', 'w') as f:
    json.dump(info, f, indent=2)
print(f'    n={len(ttfts)}  TTFT mean={info[\"ttft_mean\"]}ms  p95={info[\"ttft_p95\"]}ms  TPOT={info[\"tpot_mean\"]}ms')
" || echo "    WARN: metric extraction failed"
}

# ── Baseline S1: vanilla workload + S1 flood ─────────
# 背景用 vanilla 模式（不发预取），叠加 S1 泛洪攻击。
# 泛洪请求在 NO_HIT 路径被丢弃，衡量调度器 hash 查找的 CPU 开销
# 是否影响真实推理。

run_baseline_s1() {
  local results_dir="$1"
  local duration="${DURATION_SEC:-300}"

  CURRENT_UNIT=$((CURRENT_UNIT + 1))
  echo ""
  echo "────────────────────────────────────────"
  echo "  [$CURRENT_UNIT/$TOTAL_UNITS] baseline S1 (vanilla + flood qps=$S1_QPS)"
  echo "────────────────────────────────────────"

  mkdir -p "$results_dir"

  curl -s -X POST "http://localhost:${API_PORT}/reset_prefix_cache?reset_external=true" \
       > /dev/null || true
  sleep 3

  curl -s -X POST "http://localhost:${API_PORT}/start_profile" >/dev/null || true

  local stress_gpu_ids=""
  [ -f "$RUN_EXP_DIR/.stress_gpus" ] && \
    stress_gpu_ids=$(cat "$RUN_EXP_DIR/.stress_gpus" 2>/dev/null || true)

  bash "$UTILS_DIR/sample_power.sh" \
        "$results_dir/power.csv" "$duration" "$stress_gpu_ids" &
  local power_pid=$!
  bash "$UTILS_DIR/sample_perf.sh" \
        "$results_dir/perf_pkg.txt" "$duration" &
  local perf_pid=$!

  echo "    launching vanilla baseline at qps=$BG_QPS, conv=$BG_CONV"
  python3 "$RUN_EXP_DIR/prefetch_ab_runner.py" \
      --trace-file "$DEFAULT_BG_TRACE" \
      --mode vanilla \
      --qps "$BG_QPS" \
      --num-multi-turn "$BG_CONV" \
      --model "$MODEL" \
      --api-base "$API_BASE" \
      --output "$results_dir/baseline.jsonl" \
      --timeout "$duration" \
      --schedule-mode "${SCHEDULE_MODE:-scaled-timestamp}" \
      &> "$results_dir/baseline.log" &
  local baseline_pid=$!

  local head_start="${BASELINE_HEAD_START:-15}"
  echo "    baseline head start: ${head_start}s"
  sleep "$head_start"

  local overlay_dur=$(( duration - head_start ))
  [ "$overlay_dur" -lt 30 ] && overlay_dur=30
  echo "    launching S1 flood for ${overlay_dur}s at qps=$S1_QPS"
  local rc=0
  python3 "$RUN_EXP_DIR/prefetch_stress_runner.py" \
      --scenario s1 \
      --api-base "$API_BASE" \
      --model "$MODEL" \
      --duration-sec "$overlay_dur" \
      --output-jsonl "$results_dir/requests.jsonl" \
      --output-summary "$results_dir/summary.json" \
      --prefetch-qps "$S1_QPS" \
      --attack-prefix-file "${ATTACK_PREFIX_FILE:-$DEFAULT_ATTACK_PREFIX}" \
      &> "$results_dir/runner.log" || rc=$?
  echo "$rc" > "$results_dir/runner.exit_code"
  [ "$rc" -ne 0 ] && echo "    WARN: overlay exited with code $rc" >&2

  wait "$baseline_pid" 2>/dev/null || true

  kill -TERM "$power_pid" 2>/dev/null || true
  kill -TERM "$perf_pid" 2>/dev/null || true
  wait "$power_pid" 2>/dev/null || true
  wait "$perf_pid" 2>/dev/null || true

  curl -s -X POST "http://localhost:${API_PORT}/stop_profile" >/dev/null || true
  sleep 2
  bash "$UTILS_DIR/sample_pcie.sh" "$results_dir/pcie_events.json" \
       "${PCIE_PROFILER_DIR:-$RUN_EXP_DIR/profiler_output}" || true

  for required in baseline.jsonl requests.jsonl summary.json power.csv; do
    [ ! -s "$results_dir/$required" ] && \
      echo "    WARN: $results_dir/$required is empty or missing" >&2
  done

  extract_real_latency "$results_dir/baseline.jsonl" "$results_dir" baseline s1
  sleep 30
}

# ── Baseline S2: vanilla workload only ────────────────
# 纯 vanilla 推理（无预取、无 burst），作为 S2 的纯净性能基线。

run_baseline_s2() {
  local results_dir="$1"
  local duration="${DURATION_SEC:-300}"

  CURRENT_UNIT=$((CURRENT_UNIT + 1))
  echo ""
  echo "────────────────────────────────────────"
  echo "  [$CURRENT_UNIT/$TOTAL_UNITS] baseline S2 (vanilla only, no burst)"
  echo "────────────────────────────────────────"

  mkdir -p "$results_dir"

  curl -s -X POST "http://localhost:${API_PORT}/reset_prefix_cache?reset_external=true" \
       > /dev/null || true
  sleep 3

  local stress_gpu_ids=""
  [ -f "$RUN_EXP_DIR/.stress_gpus" ] && \
    stress_gpu_ids=$(cat "$RUN_EXP_DIR/.stress_gpus" 2>/dev/null || true)

  bash "$UTILS_DIR/sample_power.sh" \
        "$results_dir/power.csv" "$duration" "$stress_gpu_ids" &
  local power_pid=$!
  bash "$UTILS_DIR/sample_perf.sh" \
        "$results_dir/perf_pkg.txt" "$duration" &
  local perf_pid=$!

  echo "    launching vanilla workload at qps=$BG_QPS, conv=$BG_CONV"
  python3 "$RUN_EXP_DIR/prefetch_ab_runner.py" \
      --trace-file "$DEFAULT_BG_TRACE" \
      --mode vanilla \
      --qps "$BG_QPS" \
      --num-multi-turn "$BG_CONV" \
      --model "$MODEL" \
      --api-base "$API_BASE" \
      --output "$results_dir/requests.jsonl" \
      --timeout "$duration" \
      --schedule-mode "${SCHEDULE_MODE:-scaled-timestamp}" \
      &> "$results_dir/runner.log" || true

  kill -TERM "$power_pid" 2>/dev/null || true
  kill -TERM "$perf_pid" 2>/dev/null || true
  wait "$power_pid" 2>/dev/null || true
  wait "$perf_pid" 2>/dev/null || true

  [ ! -s "$results_dir/requests.jsonl" ] && \
    echo "    WARN: $results_dir/requests.jsonl is empty or missing" >&2

  extract_real_latency "$results_dir/requests.jsonl" "$results_dir" baseline s2
  sleep 10
}

# ── Prefetch S1/S2: delegates to run_one_unit ─────────
# run_one_unit (from _stress_common.sh) runs:
#   background: prefetch_ab_runner.py --mode prefetch
#   overlay:    prefetch_stress_runner.py --scenario s1|s2
# 真实推理指标从 baseline.jsonl（背景负载输出）提取。

run_prefetch_s1() {
  local config_label="$1" results_dir="$2"
  CURRENT_UNIT=$((CURRENT_UNIT + 1))
  echo ""
  echo "────────────────────────────────────────"
  echo "  [$CURRENT_UNIT/$TOTAL_UNITS] $config_label S1 (prefetch + flood qps=$S1_QPS)"
  echo "────────────────────────────────────────"
  PREFETCH_QPS="$S1_QPS" BASELINE_QPS="$BG_QPS" \
    run_one_unit "$results_dir" s1
  extract_real_latency "$results_dir/baseline.jsonl" "$results_dir" "$config_label" s1
}

run_prefetch_s2() {
  local config_label="$1" results_dir="$2"
  CURRENT_UNIT=$((CURRENT_UNIT + 1))
  echo ""
  echo "────────────────────────────────────────"
  echo "  [$CURRENT_UNIT/$TOTAL_UNITS] $config_label S2 (prefetch + burst=$S2_BURST)"
  echo "────────────────────────────────────────"
  BURST_SIZE="$S2_BURST" QPS="$BG_QPS" BASELINE_QPS="$BG_QPS" \
    run_one_unit "$results_dir" s2
  extract_real_latency "$results_dir/baseline.jsonl" "$results_dir" "$config_label" s2
}

# ═════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════

START_TIME=$(date +%s)

echo ""
echo "=========================================="
echo "  3-way stress comparison"
echo "  $(date)"
echo "  Results -> $ROOT"
echo "  Duration: ${DURATION_SEC}s × $TOTAL_UNITS units (REPEAT=$REPEAT)"
echo "  S1 flood QPS: $S1_QPS"
echo "  S2 burst size: $S2_BURST"
echo "  Background: qps=$BG_QPS, conv=$BG_CONV"
echo "=========================================="

# ───────────────────────────────────────────────────────
#  Phase 1 + 2: server ratio=0, TTL=0
#  Baseline (vanilla) 和 Unprotected (prefetch) 共享同一服务器，
#  因为 ratio=0, TTL=0 对 vanilla 模式无影响。
# ───────────────────────────────────────────────────────
start_server 0 0

echo ""
echo "===== Phase 1: BASELINE (vanilla, no prefetch) ====="

for r in $(seq 1 "$REPEAT"); do
  run_baseline_s1 "$ROOT/baseline_s1_rep${r}"
  run_baseline_s2 "$ROOT/baseline_s2_rep${r}"
done

echo ""
echo "===== Phase 2: UNPROTECTED (prefetch, ratio=0, TTL=0) ====="

for r in $(seq 1 "$REPEAT"); do
  run_prefetch_s1 unprotected "$ROOT/unprotected_s1_rep${r}"
  run_prefetch_s2 unprotected "$ROOT/unprotected_s2_rep${r}"
done

stop_server

# ───────────────────────────────────────────────────────
#  Phase 3: server ratio=0.3, TTL=60s
# ───────────────────────────────────────────────────────
RATE_LIMIT="${PREFETCH_RATE_LIMIT:-5}"
start_server 0.3 60000 "$RATE_LIMIT"

echo ""
echo "===== Phase 3: PROTECTED (prefetch, ratio=0.3, TTL=60s, RL=${RATE_LIMIT}) ====="

for r in $(seq 1 "$REPEAT"); do
  run_prefetch_s1 protected "$ROOT/protected_s1_rep${r}"
  run_prefetch_s2 protected "$ROOT/protected_s2_rep${r}"
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

python3 - "$ROOT" "$REPEAT" <<'PYEOF'
import json, os, sys, statistics

root = sys.argv[1]
repeat = int(sys.argv[2])

configs = [
    ("baseline",    "Baseline (no prefetch)"),
    ("unprotected", "Unprotected (prefetch, no defense)"),
    ("protected",   "Protected (prefetch + quota/TTL/RL)"),
]
scenarios = [("s1", "S1 flood attack"), ("s2", "S2 misclick burst")]

def load_latency(dirname):
    fp = os.path.join(root, dirname, "real_latency.json")
    if not os.path.isfile(fp):
        return None
    with open(fp) as f:
        return json.load(f)

def load_power(dirname):
    """Read power.csv → mean watts."""
    import csv
    fp = os.path.join(root, dirname, "power.csv")
    if not os.path.isfile(fp):
        return None
    with open(fp) as f:
        reader = csv.DictReader(f)
        vals = []
        for row in reader:
            for k in ("power_w", "power", "gpu_power_w"):
                if k in row:
                    try: vals.append(float(row[k]))
                    except: pass
                    break
    return round(statistics.mean(vals), 1) if vals else None

for scenario_key, scenario_label in scenarios:
    print(f"  {scenario_label}:")
    hdr = f"  {'Config':<42s}  {'TTFT mean':>10s}  {'TTFT p95':>10s}  {'TPOT':>8s}  {'GPU W':>7s}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for cfg_key, cfg_label in configs:
        all_lat, all_pwr = [], []
        for r in range(1, repeat + 1):
            dirname = f"{cfg_key}_{scenario_key}_rep{r}"
            lat = load_latency(dirname)
            if lat and lat.get("ttft_mean") is not None:
                all_lat.append(lat)
            pwr = load_power(dirname)
            if pwr is not None:
                all_pwr.append(pwr)
        if all_lat:
            ttft_m = statistics.mean([l["ttft_mean"] for l in all_lat])
            ttft_95 = statistics.mean([l["ttft_p95"] for l in all_lat])
            tpot = statistics.mean([l["tpot_mean"] for l in all_lat if l.get("tpot_mean")])
            pwr_str = f"{statistics.mean(all_pwr):.0f}W" if all_pwr else "n/a"
            print(f"  {cfg_label:<42s}  {ttft_m:>8.1f}ms  {ttft_95:>8.1f}ms  {tpot:>6.1f}ms  {pwr_str:>7s}")
        else:
            print(f"  {cfg_label:<42s}  {'(no data)':>10s}")
    print()

# Compute deltas vs baseline
print("  Relative to baseline:")
for scenario_key, scenario_label in scenarios:
    base = load_latency(f"baseline_{scenario_key}_rep1")
    if not base or not base.get("ttft_mean"):
        continue
    print(f"    {scenario_label}:")
    for cfg_key, cfg_label in configs:
        if cfg_key == "baseline":
            continue
        lat = load_latency(f"{cfg_key}_{scenario_key}_rep1")
        if lat and lat.get("ttft_mean"):
            d_ttft = (lat["ttft_mean"] - base["ttft_mean"]) / base["ttft_mean"] * 100
            d_tpot = ((lat["tpot_mean"] - base["tpot_mean"]) / base["tpot_mean"] * 100
                      if lat.get("tpot_mean") and base.get("tpot_mean") else float("nan"))
            print(f"      {cfg_label:<40s}  TTFT {d_ttft:+.1f}%  TPOT {d_tpot:+.1f}%")
print()
PYEOF

echo "后台运行:"
echo "  nohup bash scripts/stress/run_stress_3way.sh > results/stress/3way.log 2>&1 &"
