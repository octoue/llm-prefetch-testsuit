#!/bin/bash

# 定义要测试的 QPS 数组
QPS_LIST=(0.8 0.6 0.4 0.2 1.0)
TRACE_FILE="qwen_traceA_blksz_16.jsonl"
MODEL_PATH="Qwen/Qwen2.5-72B-Instruct"

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

    grep "Avg prompt throughput" /workspace/vllm_offload_test/vllm_state.log >> "./logs/prefetch_results_qps${qps}_prefetch.log"

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

    grep "Avg prompt throughput" /workspace/vllm_offload_test/vllm_state.log >> "./logs/no_prefetch_results_qps${qps}.log"

    sleep 60
done
