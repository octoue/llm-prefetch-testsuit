#!/bin/bash
# run_stress_robustness.sh — 鲁棒性评估：vanilla / unprotected / protected 三组对照
#
# 设计要点（与 run_stress_3way.sh 的关键差别）：
#
#   1) 三组共用同一份混合输入流量：
#        - 合法用户：prefetch_ab_runner.py --mode baseline （**legit 用户不发预取**，
#          这一点是本设计的关键——保证三组之间是 within-config 对比，
#          差异只反映系统对极端流量的应对，不带入 §3.5 的「预取收益」混淆）
#        - 极端流量 overlay：S1（200 QPS prefetch-only 泛洪）
#                       或 S2（burst=5 + abandon-prob=1.0，burst-then-abandon，
#                              对齐盲审「大量预取被触发但**最终未提交真实请求**」）
#
#   2) 三组的区别只在服务端的预取防御参数：
#
#        vanilla     ratio=0.0  ttl=1ms     rl=0.01 req/s
#                    预取在三道关卡（配额/TTL/限流）下被叠加压死，
#                    overlay 几乎都被拒；模拟「没有预取接口」的对照基线。
#
#        unprotected ratio=1.0  ttl=24h     rl=0
#                    预取配额无上限、TTL 几乎不过期、API 层不限流——
#                    "预取接口存在且完全无保护" 的最坏情形。
#
#        protected   ratio=0.3  ttl=60s     rl=5 req/s
#                    第 3 章的生产默认值。
#
# 预期结果（参考用，跑出来不一致再讨论原因）：
#   S1: vanilla ≈ protected ≤ unprotected   —— NO_HIT 架构上几乎免费，
#       三组很可能接近；unprotected 的额外开销主要在调度器 CPU，
#       不一定能体现在合法用户 TTFT 上（这是个好结果，
#       证明 NO_HIT 路径本身是免疫的，限流只是为节省 CPU）。
#   S2: vanilla ≈ protected << unprotected  —— unprotected 在 burst-then-abandon
#       下持续往 GPU 缓存塞 prefetch 块且不回收，合法用户被挤；
#       protected 通过配额+TTL 把污染挡住。
#
# 关键鲁棒性结论：Protected vs Unprotected 的差距 = 防御机制本身的价值；
#                Protected vs Vanilla 的差距    = 接口暴露后的剩余开销。
#
# 默认 REPEAT=1，6 个单元，约 35–40 分钟（含 3 次服务器重启）。
# REPEAT=3 约 100 分钟。
#
# Usage:
#   cd llm-prefetch-testsuit/run-experiment
#   bash scripts/stress/run_stress_robustness.sh [results_root]
#
# Env overrides:
#   DURATION_SEC=300            每个单元持续时间
#   REPEAT=1                    每组×每场景的重复次数
#   S1_QPS=200                  泛洪 QPS
#   S2_BURST=5                  S2 每对话 burst 大小
#   S2_QPS=1.0                  S2 对话链启动速率
#   S2_ABANDON_PROB=1.0         1.0 = burst-then-abandon（盲审字面）；0 = burst-then-commit
#   BASELINE_QPS=1.0            合法用户 QPS
#   BASELINE_NUM_CONV=40        合法用户对话数
#   BASELINE_HEAD_START=15      overlay 启动前给合法用户的预热时间（秒）
#   VANILLA_RATIO / VANILLA_TTL_MS / VANILLA_RL   三道防御参数（默认 0.0 / 1 / 0.01）
#   UNPROT_RATIO  / UNPROT_TTL_MS  / UNPROT_RL    （默认 1.0 / 86400000 / 0）
#   PROT_RATIO    / PROT_TTL_MS    / PROT_RL      （默认 0.3 / 60000   / 5）

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_stress_common.sh"

ROOT="${1:-$RUN_EXP_DIR/results/stress/robustness_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$ROOT"

DURATION_SEC="${DURATION_SEC:-300}"
REPEAT="${REPEAT:-1}"

S1_QPS="${S1_QPS:-200}"
S2_BURST="${S2_BURST:-5}"
S2_QPS="${S2_QPS:-1.0}"
S2_ABANDON_PROB="${S2_ABANDON_PROB:-1.0}"

BG_QPS="${BASELINE_QPS:-1.0}"
BG_CONV="${BASELINE_NUM_CONV:-40}"
BG_HEAD_START="${BASELINE_HEAD_START:-15}"

# 三组防御参数（都接受 env 覆盖）。
VANILLA_RATIO="${VANILLA_RATIO:-0.0}"
VANILLA_TTL_MS="${VANILLA_TTL_MS:-1}"
VANILLA_RL="${VANILLA_RL:-0.01}"

UNPROT_RATIO="${UNPROT_RATIO:-1.0}"
UNPROT_TTL_MS="${UNPROT_TTL_MS:-86400000}"   # 24h ≈ 永不过期
UNPROT_RL="${UNPROT_RL:-0}"                   # 0 = off

PROT_RATIO="${PROT_RATIO:-0.3}"
PROT_TTL_MS="${PROT_TTL_MS:-60000}"
PROT_RL="${PROT_RL:-5}"

TOTAL_UNITS=$(( REPEAT * 6 ))   # 3 组 × 2 场景 × REPEAT
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

# ── 单元执行：合法用户（无预取） + 极端流量 overlay ──

run_unit() {
  local label="$1" scenario="$2" rep="$3"
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

  # ───── 合法用户后台流量（不发预取，三组一致）─────
  # 与 run_stress_3way.sh 的最关键区别：--mode baseline
  # ⇒ 合法用户不使用预取接口，三组的 legit 流量完全一致；
  #    这样组间差异仅反映系统对 overlay 的应对，不带入 §3.5 的预取收益混淆。
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

  # 让合法用户先跑一段，给缓存预热。
  echo "    baseline head start: ${BG_HEAD_START}s"
  sleep "$BG_HEAD_START"

  # ───── Overlay（极端流量）─────
  local overlay_dur=$(( duration - BG_HEAD_START ))
  [ "$overlay_dur" -lt 30 ] && overlay_dur=30

  local rc=0
  if [ "$scenario" = "s1" ]; then
    echo "    S1 flood: 200 QPS prefetch-only, all NO_HIT, for ${overlay_dur}s"
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
  else  # s2
    echo "    S2 burst-then-abandon: burst=$S2_BURST chain_qps=$S2_QPS abandon=$S2_ABANDON_PROB for ${overlay_dur}s"
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
        --burst-interval-ms "${BURST_INTERVAL_MS:-50}" \
        --abandon-prob "$S2_ABANDON_PROB" \
        &> "$results_dir/runner.log" || rc=$?
  fi

  echo "$rc" > "$results_dir/runner.exit_code"
  [ "$rc" -ne 0 ] && echo "    WARN: overlay exited rc=$rc; see $results_dir/runner.log" >&2

  # 等合法用户后台流量结束（它有自己的 --timeout）。
  wait "$bg_pid" 2>/dev/null || true

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
  for required in baseline.jsonl requests.jsonl summary.json power.csv pcie_events.json; do
    [ ! -s "$results_dir/$required" ] && \
      echo "    WARN: $results_dir/$required is empty or missing" >&2
  done

  # 提取合法用户的延迟指标——overlay 自身不产生真实推理请求
  # （S1 是 prefetch-only，S2 是 burst-then-abandon），所以「真实 TTFT」
  # 只能从合法用户后台 baseline.jsonl 中取。
  extract_real_latency "$results_dir/baseline.jsonl" "$results_dir" "$label" "$scenario"
  sleep 30
}

# ════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════

START_TIME=$(date +%s)

echo ""
echo "=========================================="
echo "  Robustness eval: vanilla / unprotected / protected"
echo "  $(date)"
echo "  Results -> $ROOT"
echo "  Duration: ${DURATION_SEC}s × $TOTAL_UNITS units (REPEAT=$REPEAT)"
echo "  S1 flood QPS: $S1_QPS"
echo "  S2 burst=$S2_BURST  abandon-prob=$S2_ABANDON_PROB  (1.0 = burst-then-abandon)"
echo "  Legit background: --mode baseline  qps=$BG_QPS  conv=$BG_CONV"
echo "  (legit 用户不发预取——保证三组 within-config 对比)"
echo "=========================================="

# ─── Group 1/3: VANILLA ──────────────────────────────────────
echo ""
echo "===== Group 1/3: VANILLA (prefetch 三道关卡全开到死) ====="
echo "  ratio=$VANILLA_RATIO  ttl_ms=$VANILLA_TTL_MS  rl=$VANILLA_RL req/s"
start_server vanilla "$VANILLA_RATIO" "$VANILLA_TTL_MS" "$VANILLA_RL"
for r in $(seq 1 "$REPEAT"); do
  run_unit vanilla s1 "$r"
  run_unit vanilla s2 "$r"
done
stop_server

# ─── Group 2/3: UNPROTECTED ──────────────────────────────────
echo ""
echo "===== Group 2/3: UNPROTECTED (prefetch 配额/TTL/限流 全部撤掉) ====="
echo "  ratio=$UNPROT_RATIO  ttl_ms=$UNPROT_TTL_MS  rl=$UNPROT_RL (0=off)"
start_server unprotected "$UNPROT_RATIO" "$UNPROT_TTL_MS" "$UNPROT_RL"
for r in $(seq 1 "$REPEAT"); do
  run_unit unprotected s1 "$r"
  run_unit unprotected s2 "$r"
done
stop_server

# ─── Group 3/3: PROTECTED ────────────────────────────────────
echo ""
echo "===== Group 3/3: PROTECTED (生产默认值) ====="
echo "  ratio=$PROT_RATIO  ttl_ms=$PROT_TTL_MS  rl=$PROT_RL req/s"
start_server protected "$PROT_RATIO" "$PROT_TTL_MS" "$PROT_RL"
for r in $(seq 1 "$REPEAT"); do
  run_unit protected s1 "$r"
  run_unit protected s2 "$r"
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
scenarios = [("s1", "S1 flood (200 QPS prefetch-only)"),
             ("s2", "S2 burst-then-abandon (burst=5)")]

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

for scn_key, scn_label in scenarios:
    print(f"\n  {scn_label}:")
    hdr = f"  {'Config':<40s}  {'TTFT mean':>10s}  {'TTFT p95':>10s}  {'TPOT':>8s}  {'GPU W':>7s}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for cfg_key, cfg_label in configs:
        all_lat, all_pwr = [], []
        for r in range(1, repeat + 1):
            dirname = f"{cfg_key}_{scn_key}_rep{r}"
            lat = load_latency(dirname)
            if lat and lat.get("ttft_mean") is not None:
                all_lat.append(lat)
            pwr = load_power(dirname)
            if pwr is not None:
                all_pwr.append(pwr)
        if all_lat:
            ttft_m = statistics.mean([l["ttft_mean"] for l in all_lat])
            ttft_95 = statistics.mean([l["ttft_p95"] for l in all_lat])
            tpots = [l["tpot_mean"] for l in all_lat if l.get("tpot_mean")]
            tpot = statistics.mean(tpots) if tpots else float("nan")
            pwr_str = f"{statistics.mean(all_pwr):.0f}W" if all_pwr else "n/a"
            print(f"  {cfg_label:<40s}  {ttft_m:>8.1f}ms  {ttft_95:>8.1f}ms  {tpot:>6.1f}ms  {pwr_str:>7s}")
        else:
            print(f"  {cfg_label:<40s}  (no data)")

print("\n  Relative to Vanilla (参考线):")
for scn_key, scn_label in scenarios:
    base = load_latency(f"vanilla_{scn_key}_rep1")
    if not base or not base.get("ttft_mean"):
        continue
    print(f"    {scn_label}:")
    for cfg_key, cfg_label in configs:
        if cfg_key == "vanilla":
            continue
        lat = load_latency(f"{cfg_key}_{scn_key}_rep1")
        if lat and lat.get("ttft_mean"):
            d_ttft = (lat["ttft_mean"] - base["ttft_mean"]) / base["ttft_mean"] * 100
            d_tpot = (
                (lat["tpot_mean"] - base["tpot_mean"]) / base["tpot_mean"] * 100
                if lat.get("tpot_mean") and base.get("tpot_mean") else float("nan")
            )
            print(f"      {cfg_label:<40s}  TTFT {d_ttft:+.1f}%  TPOT {d_tpot:+.1f}%")

print("\n  Protected vs Unprotected (核心鲁棒性结论):")
for scn_key, scn_label in scenarios:
    u = load_latency(f"unprotected_{scn_key}_rep1")
    p = load_latency(f"protected_{scn_key}_rep1")
    if u and p and u.get("ttft_mean") and p.get("ttft_mean"):
        d_ttft = (p["ttft_mean"] - u["ttft_mean"]) / u["ttft_mean"] * 100
        d_tpot = (
            (p["tpot_mean"] - u["tpot_mean"]) / u["tpot_mean"] * 100
            if p.get("tpot_mean") and u.get("tpot_mean") else float("nan")
        )
        print(f"    {scn_label}:  TTFT {d_ttft:+.1f}%  TPOT {d_tpot:+.1f}%")
PYEOF

echo ""
echo "后台运行示例:"
echo "  cd llm-prefetch-testsuit/run-experiment"
echo "  nohup bash scripts/stress/run_stress_robustness.sh \\"
echo "        > results/stress/robustness.log 2>&1 &"
echo ""
echo "更多 rep:"
echo "  REPEAT=3 bash scripts/stress/run_stress_robustness.sh"
