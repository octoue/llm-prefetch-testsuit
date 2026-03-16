#!/bin/bash
#
# PCIe 竞争实验矩阵（需在远端 GPU 环境手动执行）
#
# 用法: 先 start_vllm_pcie.sh，再本脚本
#   ./run_pcie_experiment_matrix.sh
#
# 矩阵: QPS x lead_time x dataset_shape，固定 heavy-lite + scaled-timestamp
#   QPS: 0.6 / 0.8 / 1.0 / 1.2
#   lead_time: 0.5 / 1.0
#   dataset shape 两档:
#     - max_input=4000, conv=16, tiers=2/6/8
#     - max_input=4500, conv=18, tiers=2/6/10
#

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
[ -f config.env ] || { echo "缺少 config.env"; exit 1; }
set -a && source config.env && set +a

echo "============================================"
echo "PCIe 实验矩阵 (heavy-lite + scaled-timestamp)"
echo "QPS: 0.6/0.8/1.0/1.2, lead_time: 0.5/1.0"
echo "dataset: max4000/conv16/tiers2-6-8 | max4500/conv18/tiers2-6-10"
echo "============================================"

# 两档 dataset shape
# Shape 1: max_input=4000, conv=16, tiers=2/6/8
# Shape 2: max_input=4500, conv=18, tiers=2/6/10
SHAPES=(
  "4000:16:2:6:8:data/heavy_lite_4000.jsonl"
  "4500:18:2:6:10:data/heavy_lite_4500.jsonl"
)

for SHAPE_SPEC in "${SHAPES[@]}"; do
  IFS=':' read -r MAX_INP NUM_CONV TIER_S TIER_M TIER_L TRACE_FILE <<< "$SHAPE_SPEC"
  echo ""
  echo "========== Dataset: max_input=$MAX_INP, conv=$NUM_CONV, tiers=$TIER_S/$TIER_M/$TIER_L =========="

  for QPS in 0.6 0.8 1.0 1.2; do
    for LEAD in 0.5 1.0; do
      echo ""
      echo ">>> QPS=$QPS, LEAD_TIME=$LEAD, max_input=$MAX_INP"
      LITE_PCIE_TRACE="$TRACE_FILE" \
      LITE_PCIE_MAX_INPUT_LENGTH="$MAX_INP" \
      LITE_PCIE_NUM_CONV="$NUM_CONV" \
      LITE_PCIE_TIER_SHORT="$TIER_S" \
      LITE_PCIE_TIER_MEDIUM="$TIER_M" \
      LITE_PCIE_TIER_LONG="$TIER_L" \
      LITE_PCIE_PREFETCH_LEAD_TIME="$LEAD" \
        ./run_lite_test_pcie.sh "$QPS" --no-tensorboard || true
    done
  done
done

echo ""
echo "完成。结果见 results/heavy_lite_*"
