#!/bin/bash
#
# 轻量化 Prefetch A/B 测试 + PCIe Profiling
# 在 run_lite_test.sh 基础上：在 Phase 1 (Prefetch) 期间触发 profiler，采集后生成甘特图
#
# 用法: ./run_lite_test_pcie.sh [qps] [--no-tensorboard]
# 前提: 使用 start_vllm_pcie.sh 启动 vLLM（而非 start_vllm.sh）
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SCRIPT_DIR"

[ -f "$SCRIPT_DIR/config.env" ] || { echo "错误: 缺少 config.env"; exit 1; }
set -a && source "$SCRIPT_DIR/config.env" && set +a

[[ "$VLLM_SRC" != /* ]] && VLLM_SRC="$SCRIPT_DIR/$VLLM_SRC"
[[ "$PCIE_PROFILER_DIR" != /* ]] && PCIE_PROFILER_DIR="$SCRIPT_DIR/$PCIE_PROFILER_DIR"

# 解析命令行参数
QPS="$LITE_QPS"
NO_TENSORBOARD=""
for arg in "$@"; do
  if [ "$arg" = "--no-tensorboard" ]; then
    NO_TENSORBOARD=1
  elif [[ "$arg" =~ ^[0-9.]+$ ]]; then
    QPS="$arg"
  fi
done

NUM_CONV="$LITE_NUM_CONV"
TIMEOUT="$LITE_TIMEOUT"
TRACE="$LITE_TRACE"
FULL_TRACE="$LITE_FULL_TRACE"
[[ "$TRACE" != /* ]] && TRACE="$PROJECT_ROOT/$TRACE"
[[ "$FULL_TRACE" != /* ]] && FULL_TRACE="$PROJECT_ROOT/$FULL_TRACE"
[[ "$VLLM_LOG" != /* ]] && VLLM_LOG="$SCRIPT_DIR/$VLLM_LOG"

# 若 lite_dataset.jsonl 不存在，先生成
if [ ! -f "$TRACE" ]; then
  echo "生成轻量化数据集: $TRACE"
  python3 "$PROJECT_ROOT/data/prepare_lite_dataset.py" \
    --trace-file "$FULL_TRACE" \
    --output "$TRACE" \
    --max-input-length 3000 \
    --short 3 --medium 8 --long 7 \
    --seed "$SEED"
fi

RESULTS_DIR="$PROJECT_ROOT/results/lite_qps${QPS}_pcie"
mkdir -p "$RESULTS_DIR"
[ -z "$NO_TENSORBOARD" ] && TB_DIR="$RESULTS_DIR/tensorboard" || TB_DIR=""

TOTAL_REQUESTS=$(python3 -c "
import json
n=0
with open('$TRACE') as f:
    for line in f:
        if line.strip(): n+=1
print(n)
" 2>/dev/null || echo "55")

PP_SIZE="${VLLM_PIPELINE_PARALLEL_SIZE:-2}"
echo "============================================"
echo "轻量化 Prefetch A/B 测试 + PCIe Profiling"
echo "============================================"
echo "Trace: $TRACE (约 $TOTAL_REQUESTS 请求)"
echo "QPS: $QPS, 对话数: $NUM_CONV"
echo "Pipeline Parallel: $PP_SIZE 卡"
echo "结果目录: $RESULTS_DIR"
echo "Profiler 输出: $PCIE_PROFILER_DIR"
[ -n "$TB_DIR" ] && echo "TensorBoard: $TB_DIR" || echo "TensorBoard: 已禁用"
echo "============================================"

# 启动 TensorBoard（可选）
if [ -z "$NO_TENSORBOARD" ] && [ -n "$TB_DIR" ]; then
  if command -v tensorboard &>/dev/null; then
    tensorboard --logdir "$TB_DIR" --port "$TB_PORT" &
    TB_PID=$!
    echo "TensorBoard 已启动 (PID=$TB_PID), http://localhost:$TB_PORT"
  fi
fi

# 清空 cache
echo ""
echo "清空 prefix cache 和 connector cache..."
curl -s -X POST "http://${API_HOST}/reset_prefix_cache?reset_external=true" || true
sleep 5
echo "Cache 已重置。"

# Phase 1: Prefetch（带 Profiling）
TB_PREFETCH_ARGS=()
[ -n "$TB_DIR" ] && TB_PREFETCH_ARGS=(--tensorboard-dir "${TB_DIR}/prefetch")

echo ""
echo "[Phase 1] 启动 Profiler，运行 Prefetch..."
curl -s -X POST "http://${API_HOST}/start_profile" || { echo "Warning: start_profile 失败，请确认 vLLM 由 start_vllm_pcie.sh 启动"; }

python3 -u "$SCRIPT_DIR/prefetch_ab_runner.py" \
  --trace-file "$TRACE" \
  --mode prefetch \
  --qps "$QPS" \
  --num-multi-turn "$NUM_CONV" \
  --model "$MODEL_PATH" \
  --api-base "$API_BASE" \
  --output "$RESULTS_DIR/prefetch.jsonl" \
  --seed "$SEED" \
  --timeout "$TIMEOUT" \
  --request-timeout "$REQUEST_TIMEOUT" \
  --prefetch-lead-time "${PREFETCH_LEAD_TIME:-3.0}" \
  "${TB_PREFETCH_ARGS[@]}" \
  &> "$RESULTS_DIR/prefetch.log"

echo ""
echo "停止 Profiler..."
curl -s -X POST "http://${API_HOST}/stop_profile" || true
sleep 3

grep "Avg prompt throughput" "$VLLM_LOG" >> "$RESULTS_DIR/prefetch.log" 2>/dev/null || true

echo ""
echo "等待 60 秒让 server 冷却..."
sleep 60

# 清空 prefix cache
echo ""
echo "清空 prefix cache 和 connector cache..."
curl -s -X POST "http://${API_HOST}/reset_prefix_cache?reset_external=true" || true
sleep 5

# Phase 2: Baseline（无 profiling）
TB_BASELINE_ARGS=()
[ -n "$TB_DIR" ] && TB_BASELINE_ARGS=(--tensorboard-dir "${TB_DIR}/baseline")

echo ""
echo "[Phase 2] 运行 Baseline (无 prefetch)..."
python3 -u "$SCRIPT_DIR/prefetch_ab_runner.py" \
  --trace-file "$TRACE" \
  --mode baseline \
  --qps "$QPS" \
  --num-multi-turn "$NUM_CONV" \
  --model "$MODEL_PATH" \
  --api-base "$API_BASE" \
  --output "$RESULTS_DIR/baseline.jsonl" \
  --seed "$SEED" \
  --timeout "$TIMEOUT" \
  --request-timeout "$REQUEST_TIMEOUT" \
  "${TB_BASELINE_ARGS[@]}" \
  &> "$RESULTS_DIR/baseline.log"

grep "Avg prompt throughput" "$VLLM_LOG" >> "$RESULTS_DIR/baseline.log" 2>/dev/null || true

# Phase 3: 生成报告 + PCIe 甘特图
echo ""
echo "[Phase 3] 生成报告与 PCIe 甘特图..."

CONFIG_STR="QPS=$QPS, NUM_CONV=$NUM_CONV, SEED=$SEED, LITE=1, PCIE_PROFILING=1, PP=$PP_SIZE"
[ -n "$KV_OFFLOADING_SIZE" ] && CONFIG_STR="$CONFIG_STR, KV_OFFLOADING_SIZE=$KV_OFFLOADING_SIZE"

python3 -u "$PROJECT_ROOT/result-analysis/generate_report.py" \
  --baseline "$RESULTS_DIR/baseline.jsonl" \
  --prefetch "$RESULTS_DIR/prefetch.jsonl" \
  --output "$RESULTS_DIR/report.md" \
  --config "$CONFIG_STR" \
  --vllm-config "MODEL_PATH=$MODEL_PATH, VLLM_LOG=$VLLM_LOG, KV_OFFLOADING_SIZE=$KV_OFFLOADING_SIZE, GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION, NUM_GPU_BLOCKS_OVERRIDE=$NUM_GPU_BLOCKS_OVERRIDE, SWAP_SPACE=$SWAP_SPACE" \
  --test-config "TRACE=$TRACE, FULL_TRACE=$FULL_TRACE, TIMEOUT=$TIMEOUT, REQUEST_TIMEOUT=$REQUEST_TIMEOUT, API_BASE=$API_BASE"

# 生成 PCIe 甘特图（单卡为 pcie_events_0.json，多卡 PP 时合并 pcie_events_*.json）
PCIE_MERGED="$RESULTS_DIR/pcie_events_merged.json"
GANTT_HTML="$RESULTS_DIR/pcie_gantt.html"

# 合并多卡 pcie_events_*.json（若存在多个）
if [ -d "$PCIE_PROFILER_DIR" ]; then
  PCIE_FILES=$(ls "$PCIE_PROFILER_DIR"/pcie_events_*.json 2>/dev/null | sort -V)
  if [ -n "$PCIE_FILES" ]; then
    PCIE_COUNT=$(echo "$PCIE_FILES" | wc -l | tr -d ' ')
    if [ "$PCIE_COUNT" -gt 1 ]; then
      echo "合并 $PCIE_COUNT 个 PCIe 事件文件..."
      python3 -c "
import json, glob
events = []
for f in sorted(glob.glob('$PCIE_PROFILER_DIR/pcie_events_*.json')):
    with open(f) as fp:
        events.extend(json.load(fp))
with open('$PCIE_MERGED', 'w') as fp:
    json.dump(events, fp, indent=2)
print(f'Merged {len(events)} events')
" 2>/dev/null && PCIE_EVENTS="$PCIE_MERGED" || PCIE_EVENTS="$PCIE_PROFILER_DIR/pcie_events_0.json"
    else
      PCIE_EVENTS="$PCIE_PROFILER_DIR/pcie_events_0.json"
    fi
  else
    PCIE_EVENTS="$PCIE_PROFILER_DIR/pcie_events_0.json"
  fi
else
  PCIE_EVENTS="$PCIE_PROFILER_DIR/pcie_events_0.json"
fi

if [ -f "${PCIE_EVENTS:-}" ]; then
  if [ -f "$VLLM_SRC/tools/profiler/visualize_pcie_gantt.py" ]; then
    echo "生成 PCIe 甘特图: $GANTT_HTML"
    python3 "$VLLM_SRC/tools/profiler/visualize_pcie_gantt.py" \
      --input "$PCIE_EVENTS" \
      --output "$GANTT_HTML" 2>/dev/null || echo "Warning: 需安装 plotly (pip install plotly) 以生成甘特图"
  else
    echo "Warning: visualize_pcie_gantt.py 未找到，跳过甘特图"
  fi
  [ -f "$PCIE_EVENTS" ] && cp "$PCIE_EVENTS" "$RESULTS_DIR/" 2>/dev/null || true
else
  echo "Warning: PCIe 事件文件不存在，请确认 VLLM_PCIE_TRACE=1 且 start_vllm_pcie.sh 已正确启动"
fi

[ -f "$VLLM_LOG" ] && cp "$VLLM_LOG" "$RESULTS_DIR/vllm_state.log" 2>/dev/null || true
[ -d "$PCIE_PROFILER_DIR" ] && cp -r "$PCIE_PROFILER_DIR" "$RESULTS_DIR/profiler_output" 2>/dev/null || true

echo ""
echo "============================================"
echo "完成!"
echo "报告: $RESULTS_DIR/report.md"
echo "PCIe 甘特图: $GANTT_HTML"
[ -n "${TB_PID:-}" ] && echo "TensorBoard: http://localhost:$TB_PORT"
echo "============================================"
