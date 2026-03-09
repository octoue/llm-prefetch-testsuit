#!/bin/bash
#
# 多 QPS 扫描：依次对 0.2 / 0.4 / 0.6 / 0.8 运行 run_lite_test
#
# 用法: ./run_lite_sweep.sh [--no-tensorboard]
#   --no-tensorboard  可选，禁用 TensorBoard
#
# 前提: vLLM server 已启动
#   ./start_vllm.sh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 透传 --no-tensorboard 给 run_lite_test.sh
EXTRA_ARGS=()
[[ " $* " =~ " --no-tensorboard " ]] && EXTRA_ARGS=(--no-tensorboard)

for qps in 0.2 0.4 0.6 0.8; do
    echo ""
    echo "============================================"
    echo "Running lite test with QPS=$qps"
    echo "============================================"
    ./run_lite_test.sh "$qps" "${EXTRA_ARGS[@]}"
done

echo ""
echo "============================================"
echo "Sweep 完成! 结果目录: results/lite_qps{0.2,0.4,0.6,0.8}"
echo "============================================"
