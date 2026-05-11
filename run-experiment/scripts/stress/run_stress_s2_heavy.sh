#!/bin/bash
# run_stress_s2_heavy.sh — 只跑 S2 的加强版鲁棒性补充实验
#
# 背景：robustness_20260511_123439 的 S2 实测预取速率仅 ≈4.25 QPS，
#       恰好低于 protected 的 rate_limit=5 req/s，限流没踢到；
#       三组 TTFT 差别 <4%、吞吐都跌 ~38%，protected vs unprotected
#       在 S2 上没拉开差距。
#
# 本脚本：保持 S2 的「burst-then-abandon」语义不变，把 overlay 预取速率
#         从 ~5 QPS 提升到 ~15 QPS（3× protected rate_limit），同时保持
#         baseline 配置（qps=1.0、conv=40、--mode baseline）与原版完全一致，
#         所以新结果可与 robustness_20260511_123439 的 S2 三组直接对比。
#
# 三组配置（与 run_stress_robustness.sh 完全一致）：
#   vanilla     ratio=0.0  ttl=1ms     rl=0.01 req/s
#   unprotected ratio=1.0  ttl=24h     rl=0
#   protected   ratio=0.3  ttl=60s     rl=5 req/s
#
# Overlay 参数（vs 原 S2）：
#   S2_BURST=10                 ← 原 5
#   S2_QPS=1.5                  ← 原 1.0
#   BURST_INTERVAL_MS=50        （不变）
#   S2_ABANDON_PROB=1.0         （不变，保持纯 abandon 语义）
#   ⇒ 预取速率 ≈ 1.5 × 10 = 15 QPS（5min 累积 ~4500 个）
#
# 预期：
#   - protected vs unprotected 在 baseline 吞吐 / TTFT 上拉开差距
#   - unprotected GPU 仍能撑住（不至于 OOM），表现为合法用户被持续挤压
#   - 若 unprotected 反而出现 OOM/拒新请求，那是另一种鲁棒性结论
#
# 总耗时：3 单元 × 5min + 2 次 server restart ≈ 22 min
#
# Usage:
#   cd llm-prefetch-testsuit/run-experiment
#   bash scripts/stress/run_stress_s2_heavy.sh [results_root]
#
# Env overrides:
#   DURATION_SEC=300   REPEAT=1
#   S2_BURST=10        S2_QPS=1.5   S2_ABANDON_PROB=1.0
#   BURST_INTERVAL_MS=50
#   BASELINE_QPS=1.0   BASELINE_NUM_CONV=40   BASELINE_HEAD_START=15
#   VANILLA_*  /  UNPROT_*  /  PROT_*  防御参数

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_stress_common.sh"

ROOT="${1:-$RUN_EXP_DIR/results/stress/robustness_s2heavy_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$ROOT"

DURATION_SEC="${DURATION_SEC:-300}"
REPEAT="${REPEAT:-1}"

# Overlay 参数（加强版默认值——原 S2 是 burst=5 / qps=1.0）。
S2_BURST="${S2_BURST:-10}"
S2_QPS="${S2_QPS:-1.5}"
S2_ABANDON_PROB="${S2_ABANDON_PROB:-1.0}"
BURST_INTERVAL_MS="${BURST_INTERVAL_MS:-50}"

BG_QPS="${BASELINE_QPS:-1.0}"
BG_CONV="${BASELINE_NUM_CONV:-40}"
BG_HEAD_START="${BASELINE_HEAD_START:-15}"

# 三组防御参数（与 run_stress_robustness.sh 完全一致，接受 env 覆盖）。
VANILLA_RATIO="${VANILLA_RATIO:-0.0}"
VANILLA_TTL_MS="${VANILLA_TTL_MS:-1}"
VANILLA_RL="${VANILLA_RL:-0.01}"

UNPROT_RATIO="${UNPROT_RATIO:-1.0}"
UNPROT_TTL_MS="${UNPROT_TTL_MS:-86400000}"   # 24h ≈ 永不过期
UNPROT_RL="${UNPROT_RL:-0}"                   # 0 = off

PROT_RATIO="${PROT_RATIO:-0.3}"
PROT_TTL_MS="${PROT_TTL_MS:-60000}"
PROT_RL="${PROT_RL:-5}"

TOTAL_UNITS=$(( REPEAT * 3 ))   # 3 组 × 1 场景 × REPEAT
CURRENT_UNIT=0
SERVER_PID=""

# ── Server lifecycle ──────────────────────────────────────────

start_server() {
  local label="$1" ratio="$2" ttl_ms="$3" rate_limit="$4"
  local log="$ROOT/vllm_${label}.log"

  echo ""
  echo "================================================================"
  echo "  Starting vLLM   group=$label  ratio=$ratio  ttl_ms=$ttl_ms  rl=$rate_limit req/s"
  echo "================================================================"

  bash "$SCRIPT_DIR/start_vllm_stress.sh" \
       --ratio "$ratio" --ttl-ms "$ttl_ms" --rate-limit "$rate_limit" \
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

# ── 单元执行：合法用户（无预取） + S2 overlay ──

run_unit() {
  local label="$1" rep="$2"
  local scenario="s2"
  local results_dir="$ROOT/${label}_${scenario}_rep${rep}"
  local duration="$DURATION_SEC"

  CURRENT_UNIT=$((CURRENT_UNIT + 1))
  echo ""
  echo "────────────────────────────────────────"
  echo "  [$CURRENT_UNIT/$TOTAL_UNITS] $label $scenario rep$rep"
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

  # ───── 合法用户后台流量（不发预取，三组一致；与原版完全相同）─────
  echo "    legit background: --mode baseline qps=$BG_QPS conv=$BG_CONV"
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
      &> "$results_dir/baseline.log" &
  local bg_pid=$!

  echo "    baseline head start: ${BG_HEAD_START}s"
  sleep "$BG_HEAD_START"

  # ───── S2 overlay：加强版 burst-then-abandon ─────
  local overlay_dur=$(( duration - BG_HEAD_START ))
  [ "$overlay_dur" -lt 30 ] && overlay_dur=30

  echo "    S2 heavy burst-then-abandon: burst=$S2_BURST chain_qps=$S2_QPS abandon=$S2_ABANDON_PROB (≈$(awk "BEGIN{printf \"%.1f\", $S2_BURST * $S2_QPS}") prefetch/s) for ${overlay_dur}s"
  local rc=0
  python3 "$RUN_EXP_DIR/prefetch_stress_runner.py" \
      --scenario s2 \
      --api-base "$API_BASE" \
      --model "$MODEL" \
      --duration-sec "$overlay_dur" \
      --output-jsonl "$results_dir/requests.jsonl" \
      --output-summary "$results_dir/summary.json" \
      --bg-trace-file "${BG_TRACE_FILE:-$DEFAULT_BG_TRACE}" \
      --qps "$S2_QPS" \
      --burst-size "$S2_BURST" \
      --burst-interval-ms "$BURST_INTERVAL_MS" \
      --abandon-prob "$S2_ABANDON_PROB" \
      &> "$results_dir/runner.log" || rc=$?

  echo "$rc" > "$results_dir/runner.exit_code"
  [ "$rc" -ne 0 ] && echo "    WARN: overlay exited rc=$rc; see $results_dir/runner.log" >&2

  wait "$bg_pid" 2>/dev/null || true

  kill -TERM "$power_pid" 2>/dev/null || true
  kill -TERM "$perf_pid" 2>/dev/null || true
  wait "$power_pid" 2>/dev/null || true
  wait "$perf_pid" 2>/dev/null || true

  curl -s -X POST "http://localhost:${API_PORT}/stop_profile" >/dev/null || true
  sleep 2
  bash "$UTILS_DIR/sample_pcie.sh" "$results_dir/pcie_events.json" \
       "${PCIE_PROFILER_DIR:-$RUN_EXP_DIR/profiler_output}" || true

  for required in baseline.jsonl requests.jsonl summary.json power.csv pcie_events.json; do
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
echo "  S2 heavy补充实验: vanilla / unprotected / protected"
echo "  $(date)"
echo "  Results -> $ROOT"
echo "  Duration: ${DURATION_SEC}s × $TOTAL_UNITS units (REPEAT=$REPEAT)"
echo "  S2 heavy:  burst=$S2_BURST  chain_qps=$S2_QPS  abandon=$S2_ABANDON_PROB"
echo "             ≈ $(awk "BEGIN{printf \"%.1f\", $S2_BURST * $S2_QPS}") prefetch/s (vs protected rate_limit=$PROT_RL)"
echo "  Legit background: --mode baseline  qps=$BG_QPS  conv=$BG_CONV"
echo "=========================================="

# ─── Group 1/3: VANILLA ──────────────────────────────────────
echo ""
echo "===== Group 1/3: VANILLA (prefetch 三道关卡全开到死) ====="
echo "  ratio=$VANILLA_RATIO  ttl_ms=$VANILLA_TTL_MS  rl=$VANILLA_RL req/s"
start_server vanilla "$VANILLA_RATIO" "$VANILLA_TTL_MS" "$VANILLA_RL"
for r in $(seq 1 "$REPEAT"); do
  run_unit vanilla "$r"
done
stop_server

# ─── Group 2/3: UNPROTECTED ──────────────────────────────────
echo ""
echo "===== Group 2/3: UNPROTECTED (prefetch 配额/TTL/限流 全部撤掉) ====="
echo "  ratio=$UNPROT_RATIO  ttl_ms=$UNPROT_TTL_MS  rl=$UNPROT_RL (0=off)"
start_server unprotected "$UNPROT_RATIO" "$UNPROT_TTL_MS" "$UNPROT_RL"
for r in $(seq 1 "$REPEAT"); do
  run_unit unprotected "$r"
done
stop_server

# ─── Group 3/3: PROTECTED ────────────────────────────────────
echo ""
echo "===== Group 3/3: PROTECTED (生产默认值) ====="
echo "  ratio=$PROT_RATIO  ttl_ms=$PROT_TTL_MS  rl=$PROT_RL req/s"
start_server protected "$PROT_RATIO" "$PROT_TTL_MS" "$PROT_RL"
for r in $(seq 1 "$REPEAT"); do
  run_unit protected "$r"
done
stop_server

# ─── Summary ─────────────────────────────────────────────────

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

configs = [("vanilla",     "Vanilla     (prefetch 全部拒绝)"),
           ("unprotected", "Unprotected (无任何防御)"),
           ("protected",   "Protected   (生产默认值)")]

def load_latency(dirname):
    fp = os.path.join(root, dirname, "real_latency.json")
    if not os.path.isfile(fp):
        return None
    with open(fp) as f:
        return json.load(f)

def load_summary(dirname):
    fp = os.path.join(root, dirname, "summary.json")
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

print("\n  S2 heavy (burst-then-abandon, 加强版):")
hdr = f"  {'Config':<40s}  {'TTFT mean':>10s}  {'TTFT p95':>10s}  {'TPOT':>8s}  {'GPU W':>7s}  {'thrpt':>9s}"
print(hdr)
print("  " + "─" * (len(hdr) - 2))
for cfg_key, cfg_label in configs:
    all_lat, all_pwr, all_thrpt = [], [], []
    for r in range(1, repeat + 1):
        dirname = f"{cfg_key}_s2_rep{r}"
        lat = load_latency(dirname)
        smy = load_summary(dirname)
        if lat and lat.get("ttft_mean") is not None:
            all_lat.append(lat)
            if smy and smy.get("duration_sec"):
                all_thrpt.append(lat["real_count"] / smy["duration_sec"])
        pwr = load_power(dirname)
        if pwr is not None:
            all_pwr.append(pwr)
    if all_lat:
        ttft_m = statistics.mean([l["ttft_mean"] for l in all_lat])
        ttft_95 = statistics.mean([l["ttft_p95"] for l in all_lat])
        tpots = [l["tpot_mean"] for l in all_lat if l.get("tpot_mean")]
        tpot = statistics.mean(tpots) if tpots else float("nan")
        pwr_str = f"{statistics.mean(all_pwr):.0f}W" if all_pwr else "n/a"
        thrpt_str = f"{statistics.mean(all_thrpt):.3f}/s" if all_thrpt else "n/a"
        print(f"  {cfg_label:<40s}  {ttft_m:>8.1f}ms  {ttft_95:>8.1f}ms  {tpot:>6.1f}ms  {pwr_str:>7s}  {thrpt_str:>9s}")
    else:
        print(f"  {cfg_label:<40s}  (no data)")

print("\n  Relative to Vanilla (参考线):")
base = load_latency("vanilla_s2_rep1")
if base and base.get("ttft_mean"):
    for cfg_key, cfg_label in configs:
        if cfg_key == "vanilla":
            continue
        lat = load_latency(f"{cfg_key}_s2_rep1")
        if lat and lat.get("ttft_mean"):
            d_ttft = (lat["ttft_mean"] - base["ttft_mean"]) / base["ttft_mean"] * 100
            d_tpot = (
                (lat["tpot_mean"] - base["tpot_mean"]) / base["tpot_mean"] * 100
                if lat.get("tpot_mean") and base.get("tpot_mean") else float("nan")
            )
            print(f"    {cfg_label:<40s}  TTFT {d_ttft:+.1f}%  TPOT {d_tpot:+.1f}%")

print("\n  Protected vs Unprotected (核心鲁棒性结论):")
u = load_latency("unprotected_s2_rep1")
p = load_latency("protected_s2_rep1")
if u and p and u.get("ttft_mean") and p.get("ttft_mean"):
    d_ttft = (p["ttft_mean"] - u["ttft_mean"]) / u["ttft_mean"] * 100
    d_tpot = (
        (p["tpot_mean"] - u["tpot_mean"]) / u["tpot_mean"] * 100
        if p.get("tpot_mean") and u.get("tpot_mean") else float("nan")
    )
    print(f"    TTFT {d_ttft:+.1f}%  TPOT {d_tpot:+.1f}%")

# 吞吐对比（S2 上吞吐才是真正能拉开差距的指标）
print("\n  Baseline 吞吐 (req/s, S2 上更敏感的指标):")
for cfg_key, cfg_label in configs:
    lat = load_latency(f"{cfg_key}_s2_rep1")
    smy = load_summary(f"{cfg_key}_s2_rep1")
    if lat and smy and lat.get("real_count") and smy.get("duration_sec"):
        thrpt = lat["real_count"] / smy["duration_sec"]
        print(f"    {cfg_label:<40s}  {thrpt:>7.3f} req/s  ({lat['real_count']} / {smy['duration_sec']:.0f}s)")
PYEOF

echo ""
echo "结果与原 robustness_20260511_123439 的 S2 三组直接可比 (baseline 配置完全一致)。"
echo "后台运行示例:"
echo "  nohup bash scripts/stress/run_stress_s2_heavy.sh \\"
echo "        > results/stress/s2_heavy.log 2>&1 &"
