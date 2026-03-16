#!/bin/bash
#
# 中等重度单次测试（仅 Prefetch 阶段，无 Baseline）
# 使用 max_input_length=2500 数据，需配合 NUM_GPU_BLOCKS_OVERRIDE=600~800 启动 vLLM
#
# 用法: ./run_lite_test_pcie_medium.sh [qps] [--no-tensorboard]
# 前提: ./start_vllm_pcie.sh medium  （或 NUM_GPU_BLOCKS_OVERRIDE=700 ./start_vllm_pcie.sh）
#

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
[ -f config.env ] || { echo "缺少 config.env"; exit 1; }
set -a && source config.env && set +a

# 使用中等重度配置
export LITE_PCIE_TRACE="${LITE_PCIE_MEDIUM_TRACE:-data/medium_lite_2500.jsonl}"
export LITE_PCIE_MAX_INPUT_LENGTH="${LITE_PCIE_MEDIUM_MAX_INPUT_LENGTH:-2500}"
export LITE_PCIE_TIER_SHORT="${LITE_PCIE_MEDIUM_TIER_SHORT:-2}"
export LITE_PCIE_TIER_MEDIUM="${LITE_PCIE_MEDIUM_TIER_MEDIUM:-6}"
export LITE_PCIE_TIER_LONG="${LITE_PCIE_MEDIUM_TIER_LONG:-8}"
export LITE_PCIE_NUM_CONV="${LITE_PCIE_MEDIUM_NUM_CONV:-16}"
export RESULTS_PREFIX="medium_lite"

echo "============================================"
echo "中等重度单次测试 (max_input=2500, prefetch-only)"
echo "============================================"

./run_lite_test_pcie.sh "$@" --prefetch-only
