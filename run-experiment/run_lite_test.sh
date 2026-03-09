#!/bin/bash
#
# 轻量化 Prefetch A/B 测试（对应 test.sh 的替代链路）
#
# 用法: ./run_lite_test.sh
#
# 参数来自 config.env（使用 LITE_* 配置项）
#
# 前提: vLLM server 已启动
#   ./start_vllm.sh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SCRIPT_DIR"

[ -f "$SCRIPT_DIR/config.env" ] || { echo "错误: 缺少 config.env"; exit 1; }
set -a && source "$SCRIPT_DIR/config.env" && set +a

# run_lite_test 使用 LITE_* 参数
QPS="$LITE_QPS"
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
    python3 "$PROJECT_ROOT/data/prepare_lite_dataset.py" --trace-file "$FULL_TRACE" --output "$TRACE"
fi

# 结果目录：项目根 results/xxxx，TensorBoard 放在同一目录下
RESULTS_DIR="$PROJECT_ROOT/results/lite_$(date +%Y%m%d_%H%M%S)_qps${QPS}"
TB_DIR="$RESULTS_DIR/tensorboard"
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

# 启动 TensorBoard（后台，logdir 指向当前 run 的 tensorboard 目录）
if command -v tensorboard &>/dev/null; then
    tensorboard --logdir "$TB_DIR" --port "$TB_PORT" &
    TB_PID=$!
    echo "TensorBoard 已启动 (PID=$TB_PID), http://localhost:$TB_PORT"
else
    echo "未找到 tensorboard 命令，跳过 TensorBoard"
fi

# Phase 1: Prefetch
echo ""
echo "[Phase 1] 运行 Prefetch..."
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
  --tensorboard-dir "${TB_DIR}/prefetch" \
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
  --tensorboard-dir "${TB_DIR}/baseline" \
  &> "$RESULTS_DIR/baseline.log"
grep "Avg prompt throughput" "$VLLM_LOG" >> "$RESULTS_DIR/baseline.log" 2>/dev/null || true

# Phase 3: Report
echo ""
echo "[Phase 3] 生成报告..."
CONFIG_STR="QPS=$QPS, NUM_CONV=$NUM_CONV, SEED=$SEED, LITE=1"
[ -n "$KV_OFFLOADING_SIZE" ] && CONFIG_STR="$CONFIG_STR, KV_OFFLOADING_SIZE=$KV_OFFLOADING_SIZE"
python3 -u "$PROJECT_ROOT/result-analysis/generate_report.py" \
  --baseline "$RESULTS_DIR/baseline.jsonl" \
  --prefetch "$RESULTS_DIR/prefetch.jsonl" \
  --output "$RESULTS_DIR/report.md" \
  --config "$CONFIG_STR" \
  --vllm-config "MODEL_PATH=$MODEL_PATH, VLLM_LOG=$VLLM_LOG, KV_OFFLOADING_SIZE=$KV_OFFLOADING_SIZE, GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION, NUM_GPU_BLOCKS_OVERRIDE=$NUM_GPU_BLOCKS_OVERRIDE, SWAP_SPACE=$SWAP_SPACE" \
  --test-config "TRACE=$TRACE, FULL_TRACE=$FULL_TRACE, TIMEOUT=$TIMEOUT, REQUEST_TIMEOUT=$REQUEST_TIMEOUT, API_BASE=$API_BASE"

# 复制 vLLM 日志到结果目录
[ -f "$VLLM_LOG" ] && cp "$VLLM_LOG" "$RESULTS_DIR/vllm_state.log" 2>/dev/null || true

echo ""
echo "============================================"
echo "完成! 报告: $RESULTS_DIR/report.md"
[ -n "${TB_PID:-}" ] && echo "TensorBoard: http://localhost:$TB_PORT (logdir: $TB_DIR)"
echo "============================================"
