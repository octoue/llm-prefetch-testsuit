#!/bin/bash
# run_stress_s2_coalesce.sh —— S2 场景下验证 CPU-hit prefetch 去抖合并机制
#
# 背景：论文 §3.6 原实验显示 S2 下 Protected 在 TTFT 维度几乎没能把
# Unprotected 拉回 Vanilla（60s vs 58s，只回收 3.9%），原因在于预取请求
# 必须进入调度器、走到 CPU_HIT 才能被识别，预取配额/TTL 都是「事后才生效」
# 的机制，对已放过去的那部分预取无能为力。
#
# 本次新增的合并机制（在 CPU_HIT 准入前加 80ms 去抖窗口）应当把
# burst-then-abandon 里 10 连预取压成 1 个真正生效的预取，从而在
# PCIe 和 TTFT 两个维度都显著改善。
#
# 对比四组：
#   1) vanilla-clean     无预取接口基线（不发 overlay）
#   2) unprotected       预取接口无保护（ratio=1, ttl=24h, rl=0,
#                                          coalesce=0）
#   3) protected         生产默认（ratio=0.3, ttl=60s, rl=5,
#                                  coalesce=0 ← 关掉去抖，与旧实验对齐）
#   4) protected-coal    在 3) 基础上打开 80ms 合并窗口
#                        （ratio=0.3, ttl=60s, rl=5, coalesce=80）
#
# 每组只跑 S2（S1 用原结果即可，合并机制对 NO_HIT 无影响）。
# 默认 REPEAT=1，4 个单元 × 300s ≈ 25–30 分钟（含 4 次服务器启停）。
#
# Usage:
#   cd llm-prefetch-testsuit/run-experiment
#   bash scripts/stress/run_stress_s2_coalesce.sh [results_root]
#
# Env overrides（与 run_stress_robustness.sh 保持一致）：
#   DURATION_SEC=300            每单元持续时间
#   REPEAT=1                    重复次数
#   S2_BURST=5                  S2 每对话 burst 大小
#   S2_QPS=1.0                  S2 对话链启动速率
#   S2_ABANDON_PROB=1.0         1.0 = burst-then-abandon
#   BASELINE_QPS=1.0            合法用户 QPS
#   BASELINE_NUM_CONV=40        合法用户对话数
#   BASELINE_HEAD_START=15      overlay 启动前预热秒数
#   COALESCE_MS=80              Protected-Coalesce 的合并窗口（毫秒）
#   PROT_RATIO / PROT_TTL_MS / PROT_RL  生产默认值

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_stress_common.sh"

ROOT="${1:-$RUN_EXP_DIR/results/stress/s2_coalesce_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$ROOT"

DURATION_SEC="${DURATION_SEC:-300}"
REPEAT="${REPEAT:-1}"

S2_BURST="${S2_BURST:-5}"
S2_QPS="${S2_QPS:-1.0}"
S2_ABANDON_PROB="${S2_ABANDON_PROB:-1.0}"

BG_QPS="${BASELINE_QPS:-1.0}"
BG_CONV="${BASELINE_NUM_CONV:-40}"
BG_HEAD_START="${BASELINE_HEAD_START:-15}"

COALESCE_MS="${COALESCE_MS:-80}"

UNPROT_RATIO="${UNPROT_RATIO:-1.0}"
UNPROT_TTL_MS="${UNPROT_TTL_MS:-86400000}"
UNPROT_RL="${UNPROT_RL:-0}"

PROT_RATIO="${PROT_RATIO:-0.3}"
PROT_TTL_MS="${PROT_TTL_MS:-60000}"
PROT_RL="${PROT_RL:-5}"

TOTAL_UNITS=$(( REPEAT * 4 ))   # vanilla + unprotected + protected + protected-coal
CURRENT_UNIT=0
SERVER_PID=""

# ── Server lifecycle ──────────────────────────────────────────

start_server() {
  local label="$1" ratio="$2" ttl_ms="$3" rate_limit="$4" coalesce_ms="$5"
  local log="$ROOT/vllm_${label}.log"

  echo ""
  echo "================================================================"
  echo "  Starting vLLM   group=$label"
  echo "    ratio=$ratio  ttl_ms=$ttl_ms  rl=$rate_limit req/s"
  echo "    coalesce_window=${coalesce_ms}ms"
  echo "================================================================"

  # start_vllm_stress.sh 通过 STRESS_PREFETCH_COALESCE_MS 环境变量
  # 把去抖窗口传给 vllm serve 的 --prefetch-coalesce-window-ms。
  STRESS_PREFETCH_COALESCE_MS="$coalesce_ms" \
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

run_s2_unit() {
  local label="$1" rep="$2" legit_only="${3:-0}"
  local results_dir="$ROOT/${label}_s2_rep${rep}"
  local duration="$DURATION_SEC"

  CURRENT_UNIT=$((CURRENT_UNIT + 1))
  echo ""
  echo "────────────────────────────────────────"
  echo "  [$CURRENT_UNIT/$TOTAL_UNITS] $label s2 rep$rep  legit_only=$legit_only"
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

  echo "    legit background: qps=$BG_QPS conv=$BG_CONV"
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

  if [ "$legit_only" = "1" ]; then
    # vanilla-clean：不发 S2 overlay
    : > "$results_dir/requests.jsonl"
    cat > "$results_dir/summary.json" <<JSON
{
  "scenario": "s2",
  "config": "${label}",
  "mode": "legit-only (no S2 overlay)"
}
JSON
    # 等 legit 自己跑完
    wait "$bg_pid" 2>/dev/null || true
  else
    echo "    baseline head start: ${BG_HEAD_START}s"
    sleep "$BG_HEAD_START"
    local overlay_dur=$(( duration - BG_HEAD_START ))
    [ "$overlay_dur" -lt 30 ] && overlay_dur=30

    echo "    S2 burst-then-abandon: burst=$S2_BURST chain_qps=$S2_QPS abandon=$S2_ABANDON_PROB for ${overlay_dur}s"
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
        --burst-interval-ms "${BURST_INTERVAL_MS:-50}" \
        --abandon-prob "$S2_ABANDON_PROB" \
        &> "$results_dir/runner.log" || rc=$?
    echo "$rc" > "$results_dir/runner.exit_code"
    [ "$rc" -ne 0 ] && echo "    WARN: overlay exited rc=$rc" >&2
    wait "$bg_pid" 2>/dev/null || true
  fi

  kill -TERM "$power_pid" 2>/dev/null || true
  kill -TERM "$perf_pid" 2>/dev/null || true
  wait "$power_pid" 2>/dev/null || true
  wait "$perf_pid" 2>/dev/null || true

  curl -s -X POST "http://localhost:${API_PORT}/stop_profile" >/dev/null || true
  sleep 2
  bash "$UTILS_DIR/sample_pcie.sh" "$results_dir/pcie_events.json" \
       "${PCIE_PROFILER_DIR:-$RUN_EXP_DIR/profiler_output}" || true

  extract_real_latency "$results_dir/baseline.jsonl" "$results_dir" "$label" "s2"
  sleep 30
}

# ════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════

START_TIME=$(date +%s)

echo ""
echo "=========================================="
echo "  S2 coalesce eval: vanilla / unprotected / protected / protected+coalesce(${COALESCE_MS}ms)"
echo "  $(date)"
echo "  Results -> $ROOT"
echo "  Duration: ${DURATION_SEC}s × $TOTAL_UNITS units (REPEAT=$REPEAT)"
echo "  S2 burst=$S2_BURST  abandon-prob=$S2_ABANDON_PROB"
echo "  Legit background: qps=$BG_QPS  conv=$BG_CONV"
echo "=========================================="

# ── Group 1: VANILLA (legit-only, no overlay) ───────────────
echo ""
echo "===== Group 1/4: VANILLA (legit-only) ====="
start_server vanilla "$PROT_RATIO" "$PROT_TTL_MS" "$PROT_RL" 0
for r in $(seq 1 "$REPEAT"); do
  run_s2_unit vanilla "$r" 1
done
stop_server

# ── Group 2: UNPROTECTED ────────────────────────────────────
echo ""
echo "===== Group 2/4: UNPROTECTED ====="
start_server unprotected "$UNPROT_RATIO" "$UNPROT_TTL_MS" "$UNPROT_RL" 0
for r in $(seq 1 "$REPEAT"); do
  run_s2_unit unprotected "$r" 0
done
stop_server

# ── Group 3: PROTECTED (coalesce OFF, 对齐旧实验基线) ─────────
echo ""
echo "===== Group 3/4: PROTECTED (coalesce=0) ====="
start_server protected "$PROT_RATIO" "$PROT_TTL_MS" "$PROT_RL" 0
for r in $(seq 1 "$REPEAT"); do
  run_s2_unit protected "$r" 0
done
stop_server

# ── Group 4: PROTECTED + COALESCE ───────────────────────────
echo ""
echo "===== Group 4/4: PROTECTED + COALESCE=${COALESCE_MS}ms ====="
start_server protected-coal "$PROT_RATIO" "$PROT_TTL_MS" "$PROT_RL" "$COALESCE_MS"
for r in $(seq 1 "$REPEAT"); do
  run_s2_unit protected-coal "$r" 0
done
stop_server

END_TIME=$(date +%s)
ELAPSED_MIN=$(( (END_TIME - START_TIME) / 60 ))

echo ""
echo "=========================================="
echo "  Done.  $(date)  (wall time: ${ELAPSED_MIN} min)"
echo "  Results: $ROOT"
echo "=========================================="

python3 - "$ROOT" "$REPEAT" <<'PYEOF'
import json, os, sys, statistics

root = sys.argv[1]; repeat = int(sys.argv[2])
configs = [('vanilla',        'Vanilla (legit-only)'),
           ('unprotected',    'Unprotected'),
           ('protected',      'Protected (coalesce=0)'),
           ('protected-coal', 'Protected + coalesce')]

def load(d):
    fp = os.path.join(root, d, 'real_latency.json')
    return json.load(open(fp)) if os.path.isfile(fp) else None

print("\n  S2 latency:")
print(f"  {'Config':<28s} {'TTFT_mean':>10s} {'P50':>8s} {'P95':>8s} {'TPOT':>7s}")
print("  " + "─"*66)
for k,lbl in configs:
    lats=[load(f'{k}_s2_rep{r}') for r in range(1,repeat+1)]
    lats=[l for l in lats if l and l.get('ttft_mean') is not None]
    if not lats: print(f"  {lbl:<28s} (no data)"); continue
    ttft=statistics.mean(l['ttft_mean'] for l in lats)
    p50=statistics.mean(l['ttft_p50'] for l in lats)
    p95=statistics.mean(l['ttft_p95'] for l in lats)
    tp=statistics.mean(l['tpot_mean'] for l in lats if l.get('tpot_mean'))
    print(f"  {lbl:<28s} {ttft:>8.1f}ms {p50:>6.1f}ms {p95:>6.1f}ms {tp:>5.1f}ms")

# PCIe (取各目录 pcie_events.json 的 last-cluster 作为本次实验的数据)
print("\n  S2 PCIe (last-cluster, KV only):")
print(f"  {'Config':<28s} {'Evict':>8s} {'Restore':>8s} {'Prefetch':>9s} {'Total':>8s}")
print("  " + "─"*64)
def pcie(d):
    fp = os.path.join(root, d, 'pcie_events.json')
    if not os.path.isfile(fp): return None
    ev = json.load(open(fp))
    kv = sorted([e for e in ev if e['op_type'] in ('Evict','Restore','Prefetch')],
                key=lambda x:x['start_us'])
    if not kv: return None
    cur=[kv[0]]; cl=[]
    for e in kv[1:]:
        if e['start_us']-cur[-1]['start_us']>60e6: cl.append(cur); cur=[e]
        else: cur.append(e)
    cl.append(cur)
    c = cl[-1]
    span_min = (c[-1]['start_us']-c[0]['start_us'])/1e6/60
    if span_min<=0: return None
    b = {'Evict':0,'Restore':0,'Prefetch':0}
    for e in c: b[e['op_type']] = b.get(e['op_type'],0) + (e.get('size_bytes',0) or 0)
    return {k: v/1e9/span_min for k,v in b.items()}

for k,lbl in configs:
    rates = [pcie(f'{k}_s2_rep{r}') for r in range(1,repeat+1)]
    rates = [r for r in rates if r]
    if not rates: print(f"  {lbl:<28s} (no pcie)"); continue
    ev=statistics.mean(r['Evict'] for r in rates)
    re=statistics.mean(r['Restore'] for r in rates)
    pf=statistics.mean(r['Prefetch'] for r in rates)
    tot=ev+re+pf
    print(f"  {lbl:<28s} {ev:>6.2f}GB/m {re:>6.2f}GB/m {pf:>7.2f}GB/m {tot:>6.2f}GB/m")
PYEOF
