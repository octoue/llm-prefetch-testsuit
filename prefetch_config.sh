#!/bin/bash
# Prefetch 测试用配置 - 可被 full-test.sh、test.sh 等脚本 source
# 用法: source prefetch_config.sh 或 . prefetch_config.sh

# 本地模型路径（需配合 HF_HUB_OFFLINE=1 启动 vLLM）
export MODEL_PATH="${MODEL_PATH:-/lpai/models/Qwen__Qwen3-8B/25-07-26-0349}"

# vLLM 日志路径（start_vllm.sh 输出）
export VLLM_LOG="${VLLM_LOG:-./vllm_state.log}"

# KV Offloading 大小 (GiB)，用于 prefetch 从 CPU 加载 KV 到 GPU
# 默认 5（与 start_vllm.sh 同步，等比缩放自 H20-96GB + 14B 的 8 GiB）
# 禁用: export KV_OFFLOADING_SIZE=0
export KV_OFFLOADING_SIZE="${KV_OFFLOADING_SIZE:-5}"
