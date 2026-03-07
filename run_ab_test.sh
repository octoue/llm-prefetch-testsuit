#!/bin/bash
#
# Prefetch A/B 实验编排脚本
#
# 用法: ./run_ab_test.sh
#   或: QPS=0.3 NUM_CONV=30 ./run_ab_test.sh
#
# 前提: vLLM server 已启动（如通过 vllm-launch-script/start_vllm_simple.sh 或 start_vllm_with_offload.sh）
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

[ -f "$SCRIPT_DIR/prefetch_config.sh" ] && source "$SCRIPT_DIR/prefetch_config.sh"

QPS="${QPS:-0.5}"
NUM_CONV="${NUM_CONV:-50}"
TRACE="${TRACE:-qwen_traceA_blksz_16.jsonl}"
MODEL="${MODEL_PATH:-/lpai/models/Qwen__Qwen3-8B/25-07-26-0349}"
API_BASE="${API_BASE:-http://localhost:8000/v1}"

RESULTS_DIR="results/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "============================================"
echo "Prefetch A/B 实验"
echo "============================================"
echo "QPS: $QPS"
echo "多轮对话数: $NUM_CONV"
echo "Trace: $TRACE"
echo "Model: $MODEL"
echo "结果目录: $RESULTS_DIR"
echo "============================================"

# Phase 1: Baseline
echo ""
echo "[Phase 1] 运行 Baseline (无 prefetch)..."
python -u prefetch_ab_runner.py \
  --trace-file "$TRACE" \
  --mode baseline \
  --qps "$QPS" \
  --num-multi-turn "$NUM_CONV" \
  --model "$MODEL" \
  --api-base "$API_BASE" \
  --output "$RESULTS_DIR/baseline.jsonl"

echo ""
echo "等待 30 秒让 server 冷却..."
sleep 30

# Phase 2: Prefetch
echo ""
echo "[Phase 2] 运行 Prefetch..."
python -u prefetch_ab_runner.py \
  --trace-file "$TRACE" \
  --mode prefetch \
  --qps "$QPS" \
  --num-multi-turn "$NUM_CONV" \
  --model "$MODEL" \
  --api-base "$API_BASE" \
  --output "$RESULTS_DIR/prefetch.jsonl"

# Phase 3: Report
echo ""
echo "[Phase 3] 生成报告..."
python -u generate_report.py \
  --baseline "$RESULTS_DIR/baseline.jsonl" \
  --prefetch "$RESULTS_DIR/prefetch.jsonl" \
  --output "$RESULTS_DIR/report.html"

echo ""
echo "============================================"
echo "完成! 报告: $RESULTS_DIR/report.html"
echo "============================================"
