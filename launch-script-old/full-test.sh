#!/bin/bash

# 加载配置（模型路径、日志路径等）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$SCRIPT_DIR/prefetch_config.sh" ] && source "$SCRIPT_DIR/prefetch_config.sh"

# 定义要测试的 QPS 数组
QPS_LIST=(0.8 0.6 0.4 0.2 1.0)
TRACE_FILE="qwen_traceA_blksz_16.jsonl"
MODEL_PATH="${MODEL_PATH:-/lpai/models/Qwen__Qwen3-8B/25-07-26-0349}"
VLLM_LOG="${VLLM_LOG:-./vllm_state.log}"

# 确保 logs 目录存在
mkdir -p logs

for qps in "${QPS_LIST[@]}"
do
    echo "=========================================="
    echo "Processing QPS: $qps"
    echo "=========================================="

    # 1. 测试 With Prefetch
    echo "-> Running With Prefetch..."
    python -u test.py \
        --trace-file $TRACE_FILE \
        --api-base http://localhost:8000/v1 \
        --num-single-turn 0 \
        --num-multi-turn 50 \
        --qps $qps \
        --enable-prefetch \
        --prefetch-lead-time 0.2 \
        --model $MODEL_PATH \
        --output "results_qps${qps}_prefetch.json"  &> "./logs/prefetch_results_qps${qps}_prefetch.log"

    grep "Avg prompt throughput" "$VLLM_LOG" >> "./logs/prefetch_results_qps${qps}_prefetch.log" 2>/dev/null || true

    sleep 60

    # 2. 测试 No Prefetch (去掉 --enable-prefetch 参数)
    echo "-> Running Without Prefetch..."
    python -u test.py \
        --trace-file $TRACE_FILE \
        --api-base http://localhost:8000/v1 \
        --num-single-turn 0 \
        --num-multi-turn 50 \
        --qps $qps \
        --model $MODEL_PATH \
        --output "results_qps${qps}_no_prefetch.json" &> "./logs/no_prefetch_results_qps${qps}.log"
        
    echo "Done with QPS $qps"
    echo ""

    grep "Avg prompt throughput" "$VLLM_LOG" >> "./logs/no_prefetch_results_qps${qps}.log" 2>/dev/null || true

    sleep 60
done
