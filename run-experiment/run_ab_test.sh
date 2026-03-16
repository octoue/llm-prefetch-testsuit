#!/bin/bash
#
# Prefetch A/B 实验编排脚本（单 QPS 模式）
#
# 用法: ./run_ab_test.sh
# 参数来自 config.env
#
# 前提: vLLM server 已启动（使用本地模型、离线模式）:
#   ./start_vllm.sh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SCRIPT_DIR"

[ -f "$SCRIPT_DIR/config.env" ] || { echo "错误: 缺少 config.env"; exit 1; }
set -a && source "$SCRIPT_DIR/config.env" && set +a

[[ "$TRACE" != /* ]] && TRACE="$PROJECT_ROOT/$TRACE"
[[ "$VLLM_LOG" != /* ]] && VLLM_LOG="$SCRIPT_DIR/$VLLM_LOG"

# 结果目录：项目根 results/xxxx
RESULTS_DIR="$PROJECT_ROOT/results/$(date +%Y%m%d_%H%M%S)_qps${QPS}"
mkdir -p "$RESULTS_DIR"

# 实时将 vLLM 日志写入本结果目录
VLLM_TAIL_PID=""
if [ -f "${VLLM_LOG:-}" ]; then
  tail -f "$VLLM_LOG" >> "$RESULTS_DIR/vllm_live.log" 2>/dev/null &
  VLLM_TAIL_PID=$!
  trap '[ -n "${VLLM_TAIL_PID:-}" ] && kill $VLLM_TAIL_PID 2>/dev/null || true' EXIT
fi
# 写入完整配置到结果目录
bash "$SCRIPT_DIR/dump_config.sh" > "$RESULTS_DIR/config.env" 2>/dev/null || true

echo "============================================"
echo "Prefetch A/B 实验"
echo "============================================"
echo "QPS: $QPS"
echo "多轮对话数: $NUM_CONV"
echo "Trace: $TRACE"
echo "Model: $MODEL_PATH"
echo "结果目录: $RESULTS_DIR"
echo "============================================"

# 构建 timeout 参数
TIMEOUT_ARGS=()
if [ -n "$TIMEOUT" ]; then
  TIMEOUT_ARGS=(--timeout "$TIMEOUT")
  echo "Timeout: $TIMEOUT 秒"
fi

# 每次测试开始前清空 cache，确保 Prefetch 从干净状态启动
echo ""
echo "清空 prefix cache 和 connector cache（确保 Prefetch 从干净状态开始）..."
curl -s -X POST "http://${API_HOST}/reset_prefix_cache?reset_external=true" || true
sleep 5
echo "Cache 已重置。"

# Phase 1: Prefetch（与旧 full-test.sh 顺序一致：prefetch first）
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
  "${TIMEOUT_ARGS[@]}" \
  &> "$RESULTS_DIR/prefetch.log"
grep "Avg prompt throughput" "$VLLM_LOG" >> "$RESULTS_DIR/prefetch.log" 2>/dev/null || true

echo ""
echo "等待 60 秒让 server 冷却..."
sleep 60

# 清空 prefix cache 和 CPU offload，确保 baseline 不受 prefetch 实验残留影响
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
  "${TIMEOUT_ARGS[@]}" \
  &> "$RESULTS_DIR/baseline.log"
grep "Avg prompt throughput" "$VLLM_LOG" >> "$RESULTS_DIR/baseline.log" 2>/dev/null || true

# Phase 3: Report
echo ""
echo "[Phase 3] 生成报告..."
CONFIG_STR="QPS=$QPS, NUM_CONV=$NUM_CONV, SEED=$SEED"
[ -n "$KV_OFFLOADING_SIZE" ] && CONFIG_STR="$CONFIG_STR, KV_OFFLOADING_SIZE=$KV_OFFLOADING_SIZE"
python3 -u "$PROJECT_ROOT/result-analysis/generate_report.py" \
  --baseline "$RESULTS_DIR/baseline.jsonl" \
  --prefetch "$RESULTS_DIR/prefetch.jsonl" \
  --output "$RESULTS_DIR/report.md" \
  --config "$CONFIG_STR" \
  --vllm-config "MODEL_PATH=$MODEL_PATH, VLLM_LOG=$VLLM_LOG, KV_OFFLOADING_SIZE=$KV_OFFLOADING_SIZE, GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION, NUM_GPU_BLOCKS_OVERRIDE=$NUM_GPU_BLOCKS_OVERRIDE, SWAP_SPACE=$SWAP_SPACE" \
  --test-config "TRACE=$TRACE, TIMEOUT=$TIMEOUT, API_BASE=$API_BASE"

# 复制 vLLM 日志到结果目录（完整快照，vllm_live.log 为实时追加）
[ -f "$VLLM_LOG" ] && cp "$VLLM_LOG" "$RESULTS_DIR/vllm_state.log" 2>/dev/null || true
[ -n "${VLLM_TAIL_PID:-}" ] && kill $VLLM_TAIL_PID 2>/dev/null || true

echo ""
echo "============================================"
echo "完成! 报告: $RESULTS_DIR/report.md"
echo "============================================"
