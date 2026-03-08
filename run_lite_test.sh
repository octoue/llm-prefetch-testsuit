#!/bin/bash
#
# 轻量化 Prefetch A/B 测试（对应 test.sh 的替代链路）
#
# 用法: ./run_lite_test.sh
#
# 与 run_ab_test.sh 的区别:
# - 使用 lite_dataset.jsonl（9 个对话，约 55 请求）
# - 默认 TIMEOUT=300, REQUEST_TIMEOUT=120
# - 启用 TensorBoard 实时监控
#
# 前提: vLLM server 已启动
#   ./start_vllm.sh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

[ -f "$SCRIPT_DIR/prefetch_config.sh" ] && source "$SCRIPT_DIR/prefetch_config.sh"

# 轻量化测试默认参数
QPS="${QPS:-0.8}"
NUM_CONV="${NUM_CONV:-9}"
TIMEOUT="${TIMEOUT:-300}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-120}"
TRACE="${TRACE:-lite_dataset.jsonl}"
FULL_TRACE="${FULL_TRACE:-qwen_traceA_blksz_16.jsonl}"
MODEL="${MODEL_PATH:-/lpai/models/Qwen__Qwen3-8B/25-07-26-0349}"
API_BASE="${API_BASE:-http://localhost:8000/v1}"
API_HOST="${API_HOST:-localhost:8000}"
VLLM_LOG="${VLLM_LOG:-./vllm_state.log}"
SEED="${SEED:-42}"
TB_PORT="${TB_PORT:-6006}"

# 若 lite_dataset.jsonl 不存在，先生成
if [ ! -f "$TRACE" ]; then
    echo "生成轻量化数据集: $TRACE"
    python3 prepare_lite_dataset.py --trace-file "$FULL_TRACE" --output "$TRACE"
fi

# 结果目录
RESULTS_DIR="results/lite_$(date +%Y%m%d_%H%M%S)_qps${QPS}"
mkdir -p "$RESULTS_DIR"
TB_DIR="$SCRIPT_DIR/runs/lite_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$TB_DIR"

# 预估运行时间（调度跨度 + 缓冲）
TOTAL_REQUESTS=$(python3 -c "
import json
n=0
with open('$TRACE') as f:
    for line in f:
        if line.strip(): n+=1
print(n)
" 2>/dev/null || echo "55")
EST_SCHEDULE=$((TOTAL_REQUESTS * 125 / 100))  # 1/0.8 ≈ 1.25s per req
echo "============================================"
echo "轻量化 Prefetch A/B 测试"
echo "============================================"
echo "Trace: $TRACE (约 $TOTAL_REQUESTS 请求)"
echo "QPS: $QPS, 对话数: $NUM_CONV"
echo "TIMEOUT: $TIMEOUT s, REQUEST_TIMEOUT: $REQUEST_TIMEOUT s"
echo "预估调度跨度: ~${EST_SCHEDULE}s"
echo "结果目录: $RESULTS_DIR"
echo "TensorBoard: $TB_DIR"
echo "============================================"

# 启动 TensorBoard（后台）
if command -v tensorboard &>/dev/null; then
    tensorboard --logdir "$(dirname "$TB_DIR")" --port "$TB_PORT" &
    TB_PID=$!
    echo "TensorBoard 已启动 (PID=$TB_PID), http://localhost:$TB_PORT"
else
    echo "未找到 tensorboard 命令，跳过 TensorBoard"
fi

# Phase 1: Prefetch
echo ""
echo "[Phase 1] 运行 Prefetch..."
python3 -u prefetch_ab_runner.py \
  --trace-file "$TRACE" \
  --mode prefetch \
  --qps "$QPS" \
  --num-multi-turn "$NUM_CONV" \
  --model "$MODEL" \
  --api-base "$API_BASE" \
  --output "$RESULTS_DIR/prefetch.jsonl" \
  --seed "$SEED" \
  --timeout "$TIMEOUT" \
  --request-timeout "$REQUEST_TIMEOUT" \
  --tensorboard-dir "${TB_DIR}_prefetch" \
  &> "$RESULTS_DIR/prefetch.log"
grep "Avg prompt throughput" "$VLLM_LOG" >> "$RESULTS_DIR/prefetch.log" 2>/dev/null || true

echo ""
echo "等待 60 秒让 server 冷却..."
sleep 60

# 清空 prefix cache
echo ""
echo "清空 prefix cache 和 connector cache..."
curl -s -X POST "http://${API_HOST}/reset_prefix_cache?reset_external=true" || true
sleep 5
echo "Cache 已重置。"

# Phase 2: Baseline
echo ""
echo "[Phase 2] 运行 Baseline (无 prefetch)..."
python3 -u prefetch_ab_runner.py \
  --trace-file "$TRACE" \
  --mode baseline \
  --qps "$QPS" \
  --num-multi-turn "$NUM_CONV" \
  --model "$MODEL" \
  --api-base "$API_BASE" \
  --output "$RESULTS_DIR/baseline.jsonl" \
  --seed "$SEED" \
  --timeout "$TIMEOUT" \
  --request-timeout "$REQUEST_TIMEOUT" \
  --tensorboard-dir "${TB_DIR}_baseline" \
  &> "$RESULTS_DIR/baseline.log"
grep "Avg prompt throughput" "$VLLM_LOG" >> "$RESULTS_DIR/baseline.log" 2>/dev/null || true

# Phase 3: Report
echo ""
echo "[Phase 3] 生成报告..."
CONFIG_STR="QPS=$QPS, NUM_CONV=$NUM_CONV, SEED=$SEED, LITE=1"
[ -n "${KV_OFFLOADING_SIZE:-}" ] && CONFIG_STR="$CONFIG_STR, KV_OFFLOADING_SIZE=$KV_OFFLOADING_SIZE"
python3 -u generate_report.py \
  --baseline "$RESULTS_DIR/baseline.jsonl" \
  --prefetch "$RESULTS_DIR/prefetch.jsonl" \
  --output "$RESULTS_DIR/report.html" \
  --config "$CONFIG_STR"

echo ""
echo "============================================"
echo "完成! 报告: $RESULTS_DIR/report.html"
[ -n "${TB_PID:-}" ] && echo "TensorBoard: http://localhost:$TB_PORT"
echo "============================================"
