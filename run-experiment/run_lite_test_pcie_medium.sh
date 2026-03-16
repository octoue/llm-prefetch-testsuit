#!/bin/bash
#
# 中等重度单次测试（仅 Prefetch 阶段，无 Baseline）
# 使用 max_input_length=2500 数据，与 heavy 共用 config.env 中的 NUM_GPU_BLOCKS_OVERRIDE
#
# 用法: ./run_lite_test_pcie_medium.sh [qps] [--no-tensorboard]
# 前提: ./start_vllm_pcie.sh medium
#

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
[ -f config.env ] || { echo "缺少 config.env"; exit 1; }
set -a && source config.env && set +a

# 使用中等重度配置（tier/num_conv 与 heavy 共用 config.env 中的值）
export LITE_PCIE_TRACE="$LITE_PCIE_MEDIUM_TRACE"
export LITE_PCIE_MAX_INPUT_LENGTH="$LITE_PCIE_MEDIUM_MAX_INPUT_LENGTH"
export RESULTS_PREFIX="medium_lite"

echo "============================================"
echo "中等重度单次测试 (max_input=2500, prefetch-only)"
echo "============================================"

./run_lite_test_pcie.sh "$@" --prefetch-only
