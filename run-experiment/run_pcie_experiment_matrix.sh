#!/bin/bash
#
# PCIe 竞争实验矩阵（需在远端 GPU 环境手动执行）
#
# 用法: 先 start_vllm_pcie.sh，再本脚本
#   ./run_pcie_experiment_matrix.sh
#
# 矩阵: QPS x lead_time，固定 heavy-lite + scaled-timestamp
# 目标: 找到最容易打出 PCIe 竞争的参数区间
#

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
[ -f config.env ] || { echo "缺少 config.env"; exit 1; }
set -a && source config.env && set +a

echo "============================================"
echo "PCIe 实验矩阵 (heavy-lite + scaled-timestamp)"
echo "============================================"

for QPS in 0.8 1.2 1.6 2.0; do
  for LEAD in 0.5 1.0 1.5; do
    echo ""
    echo ">>> QPS=$QPS, LEAD_TIME=$LEAD"
    LITE_PCIE_PREFETCH_LEAD_TIME=$LEAD ./run_lite_test_pcie.sh "$QPS" --no-tensorboard || true
  done
done

echo ""
echo "完成。结果见 results/lite_qps*_pcie/"
