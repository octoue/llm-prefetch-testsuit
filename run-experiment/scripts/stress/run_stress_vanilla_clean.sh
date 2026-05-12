#!/bin/bash
# run_stress_vanilla_clean.sh —— 纯净的 Vanilla 基线（无任何 overlay 攻击/误触）
#
# 设计思路：
#   语义上，「Vanilla = 系统未暴露预取接口」。在这种系统上，无论攻击者
#   想发 S1 泛洪还是 S2 burst-then-abandon，请求都不会被处理，合法用户
#   感受到的端到端行为与「无攻击」完全一致。因此纯净 Vanilla 的正确做法是：
#
#       只跑合法用户后台流量，不发起任何 overlay。
#
#   旧的 vanilla 用 ratio=0 / TTL=1ms / RL=0.01 模拟「接口被压死」，
#   但服务端仍在处理预取 HTTP 请求、产生 KV 分配/TTL 立即过期的二次 PCIe，
#   因此 Vanilla 的 PCIe/TTFT 都被人造攻击污染了。这个脚本绕开那个问题。
#
#   服务端参数取 Protected 默认值即可——因为 prefetch 接口根本不会被调用，
#   ratio/TTL/RL 的具体值在本实验中观察不到。
#
# 输出目录结构与 run_stress_robustness.sh 完全对齐，可直接覆盖原图脚本
# 中的 vanilla_s{1,2}_rep* 数据：
#
#       <root>/vanilla_s1_rep<r>/   ←  合法负载 only（标 S1 仅为命名）
#       <root>/vanilla_s2_rep<r>/   ←  合法负载 only（标 S2 仅为命名）
#
# 这两组在语义上是等价的（同样的合法流量、零攻击），跑两次只是为了
# 与现有 6 格矩阵保持结构一致；后续画图脚本不需要任何改动。
#
# 默认 REPEAT=1，2 个单元，约 12 分钟（含 1 次服务器启停）。
#
# Usage:
#   cd llm-prefetch-testsuit/run-experiment
#   bash scripts/stress/run_stress_vanilla_clean.sh [results_root]
#
# Env overrides:
#   DURATION_SEC=300            每个单元持续时间
#   REPEAT=1                    每场景的重复次数
#   BASELINE_QPS=1.0            合法用户 QPS
#   BASELINE_NUM_CONV=40        合法用户对话数
#   PROT_RATIO=0.3              （服务端 ratio，反正不会被触发）
#   PROT_TTL_MS=60000
#   PROT_RL=5

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_stress_common.sh"

ROOT="${1:-$RUN_EXP_DIR/results/stress/vanilla_clean_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$ROOT"

DURATION_SEC="${DURATION_SEC:-300}"
REPEAT="${REPEAT:-1}"

BG_QPS="${BASELINE_QPS:-1.0}"
BG_CONV="${BASELINE_NUM_CONV:-40}"

# 服务端 prefetch 参数（这里只是给 vllm 一个合法的命令行；prefetch 路径
# 在本脚本下不会被任何客户端触达，因此具体取值不会影响测量结果）。
PROT_RATIO="${PROT_RATIO:-0.3}"
PROT_TTL_MS="${PROT_TTL_MS:-60000}"
PROT_RL="${PROT_RL:-5}"

TOTAL_UNITS=$(( REPEAT * 2 ))
CURRENT_UNIT=0
SERVER_PID=""

# ── Server lifecycle ──────────────────────────────────────────

start_server() {
  local log="$ROOT/vllm_vanilla_clean.log"

  echo ""
  echo "================================================================"
  echo "  Starting vLLM (vanilla-clean: prefetch endpoint exists but"
  echo "                 will never be invoked by any client)"
  echo "    ratio=$PROT_RATIO  ttl_ms=$PROT_TTL_MS  rl=$PROT_RL req/s"
  echo "================================================================"

  bash "$SCRIPT_DIR/start_vllm_stress.sh" \
       --ratio "$PROT_RATIO" --ttl-ms "$PROT_TTL_MS" --rate-limit "$PROT_RL" \
       --log "$log" &
  SERVER_PID=$!

  local w=0
  until curl -sf "http://localhost:${API_PORT}/health" >/dev/null 2>&1; do
    sleep 5; w=$((w + 5))
    if [ "$w" -ge 300 ]; then
      echo "FATAL: server not ready after ${w}s" >&2
      exit 1
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

# ── 从 baseline.jsonl 中抽取合法用户的 TTFT/TPOT ──

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
s = sorted(ttfts)
info = {
    'config': '${config}', 'scenario': '${scenario}',
    'real_count': len(ttfts),
    'ttft_mean': round(statistics.mean(ttfts), 1),
    'ttft_p50':  round(s[len(s)//2], 1),
    'ttft_p95':  round(s[int(len(s)*0.95)], 1),
    'tpot_mean': round(statistics.mean(tpots), 1) if tpots else None,
}
with open('${output_dir}/real_latency.json', 'w') as f:
    json.dump(info, f, indent=2)
print(f'    n={len(ttfts)}  TTFT mean={info[\"ttft_mean\"]}ms  p95={info[\"ttft_p95\"]}ms  TPOT={info[\"tpot_mean\"]}ms')
" || echo "    WARN: metric extraction failed"
}

# ── 单元执行：仅合法用户后台流量，无任何 overlay ──

run_unit() {
  local scenario="$1" rep="$2"
  local label="vanilla"
  local results_dir="$ROOT/${label}_${scenario}_rep${rep}"
  local duration="$DURATION_SEC"

  CURRENT_UNIT=$((CURRENT_UNIT + 1))
  echo ""
  echo "────────────────────────────────────────"
  echo "  [$CURRENT_UNIT/$TOTAL_UNITS] $label $scenario rep$rep"
  echo "    (legit-only; no S1/S2 overlay — pure Vanilla semantics)"
  echo "────────────────────────────────────────"

  mkdir -p "$results_dir"

  # 清空缓存，保证每个单元从干净状态开始。
  curl -s -X POST "http://localhost:${API_PORT}/reset_prefix_cache?reset_external=true" \
       > /dev/null || true
  sleep 3

  # 启动 PCIe tracer。
  curl -s -X POST "http://localhost:${API_PORT}/start_profile" >/dev/null || true

  local stress_gpu_ids=""
  [ -f "$RUN_EXP_DIR/.stress_gpus" ] && \
    stress_gpu_ids=$(cat "$RUN_EXP_DIR/.stress_gpus" 2>/dev/null || true)

  # 功耗/perf 采样。
  bash "$UTILS_DIR/sample_power.sh" \
        "$results_dir/power.csv" "$duration" "$stress_gpu_ids" &
  local power_pid=$!
  bash "$UTILS_DIR/sample_perf.sh" \
        "$results_dir/perf_pkg.txt" "$duration" &
  local perf_pid=$!

  # ───── 合法用户流量（与 robustness 脚本完全一致的参数）─────
  echo "    legit baseline: --mode baseline qps=$BG_QPS conv=$BG_CONV  for ${duration}s"
  python3 "$RUN_EXP_DIR/prefetch_ab_runner.py" \
      --trace-file "$DEFAULT_BG_TRACE" \
      --mode baseline \
      --qps "$BG_QPS" \
      --num-multi-turn "$BG_CONV" \
      --model "$MODEL" \
      --api-base "$API_BASE" \
      --output "$results_dir/baseline.jsonl" \
      --timeout "$duration" \
      --schedule-mode "${SCHEDULE_MODE:-scaled-timestamp}" \
      &> "$results_dir/baseline.log"
  local bg_rc=$?
  echo "$bg_rc" > "$results_dir/baseline.exit_code"

  # 没有 overlay；为了与 robustness 脚本输出目录结构对齐，
  # 写一个空的 requests.jsonl + 合法 summary.json，画图脚本不会因缺文件挂掉。
  : > "$results_dir/requests.jsonl"
  cat > "$results_dir/summary.json" <<JSON
{
  "scenario": "${scenario}",
  "config": "vanilla",
  "mode": "vanilla-clean (legit-only, no overlay)",
  "overlay_requests_sent": 0,
  "overlay_requests_ack": 0
}
JSON

  # 停止采样。
  kill -TERM "$power_pid" 2>/dev/null || true
  kill -TERM "$perf_pid" 2>/dev/null || true
  wait "$power_pid" 2>/dev/null || true
  wait "$perf_pid" 2>/dev/null || true

  # 把 PCIe 事件落盘。
  curl -s -X POST "http://localhost:${API_PORT}/stop_profile" >/dev/null || true
  sleep 2
  bash "$UTILS_DIR/sample_pcie.sh" "$results_dir/pcie_events.json" \
       "${PCIE_PROFILER_DIR:-$RUN_EXP_DIR/profiler_output}" || true

  # 完整性检查。
  for required in baseline.jsonl summary.json power.csv pcie_events.json; do
    [ ! -s "$results_dir/$required" ] && \
      echo "    WARN: $results_dir/$required is empty or missing" >&2
  done

  extract_real_latency "$results_dir/baseline.jsonl" "$results_dir" "$label" "$scenario"
  sleep 30
}

# ════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════

START_TIME=$(date +%s)

echo ""
echo "=========================================="
echo "  Vanilla-clean baseline (no overlay)"
echo "  $(date)"
echo "  Results -> $ROOT"
echo "  Duration: ${DURATION_SEC}s × $TOTAL_UNITS units (REPEAT=$REPEAT)"
echo "  Legit traffic: --mode baseline  qps=$BG_QPS  conv=$BG_CONV"
echo "  (S1/S2 attackers DISABLED — prefetch endpoint never invoked)"
echo "=========================================="

start_server
for r in $(seq 1 "$REPEAT"); do
  run_unit s1 "$r"
  run_unit s2 "$r"
done
stop_server

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
ELAPSED_MIN=$(( ELAPSED / 60 ))

echo ""
echo "=========================================="
echo "  Done.  $(date)  (wall time: ${ELAPSED_MIN} min)"
echo "  Results: $ROOT"
echo "=========================================="

python3 - "$ROOT" "$REPEAT" <<'PYEOF'
import json, os, sys, statistics, csv

root = sys.argv[1]
repeat = int(sys.argv[2])

def load_latency(dirname):
    fp = os.path.join(root, dirname, "real_latency.json")
    if not os.path.isfile(fp):
        return None
    with open(fp) as f:
        return json.load(f)

def load_power(dirname):
    fp = os.path.join(root, dirname, "power.csv")
    if not os.path.isfile(fp):
        return None
    vals = []
    with open(fp) as f:
        for row in csv.DictReader(f):
            for k in ("power_w", "power", "gpu_power_w"):
                if k in row:
                    try: vals.append(float(row[k]))
                    except: pass
                    break
    return round(statistics.mean(vals), 1) if vals else None

print("\n  Vanilla-clean (legit-only) summary:")
hdr = f"  {'Scenario':<10s}  {'TTFT mean':>10s}  {'TTFT p50':>10s}  {'TTFT p95':>10s}  {'TPOT':>8s}  {'GPU W':>7s}"
print(hdr)
print("  " + "─" * (len(hdr) - 2))
for scn in ("s1", "s2"):
    lats, pwrs = [], []
    for r in range(1, repeat + 1):
        d = f"vanilla_{scn}_rep{r}"
        lat = load_latency(d)
        if lat and lat.get("ttft_mean") is not None: lats.append(lat)
        pwr = load_power(d)
        if pwr is not None: pwrs.append(pwr)
    if lats:
        ttft_m = statistics.mean([l["ttft_mean"] for l in lats])
        ttft_50 = statistics.mean([l["ttft_p50"] for l in lats])
        ttft_95 = statistics.mean([l["ttft_p95"] for l in lats])
        tpots = [l["tpot_mean"] for l in lats if l.get("tpot_mean")]
        tpot = statistics.mean(tpots) if tpots else float("nan")
        pwr_str = f"{statistics.mean(pwrs):.0f}W" if pwrs else "n/a"
        print(f"  {scn:<10s}  {ttft_m:>8.1f}ms  {ttft_50:>8.1f}ms  {ttft_95:>8.1f}ms  {tpot:>6.1f}ms  {pwr_str:>7s}")
    else:
        print(f"  {scn:<10s}  (no data)")
print("\n  注：s1/s2 两行在本脚本下语义等价（同样的合法负载、零攻击）；")
print("      若两行数值偏差明显，说明实验自身存在抖动，可考虑 REPEAT=3 取均值。")
PYEOF

echo ""
echo "如需替换原 robustness 结果中的 vanilla 数据，把目录:"
echo "  $ROOT/vanilla_s1_rep*/  ,  $ROOT/vanilla_s2_rep*/"
echo "拷贝到原 results/stress/robustness_XXX/ 下覆盖同名目录即可，"
echo "画图/PCIe 脚本无需改动。"
