#!/bin/bash
# 批量跑不同 QPS 的 PCIe Ablation 实验
# 用法: nohup bash scripts/pcie/batch_run_qps.sh pcie-heavy > batch_qps.log 2>&1 &
#
# 合上电脑也不会断。查看进度: tail -f batch_qps.log

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET="${1:-pcie-heavy}"

QPS_LIST=(0.5 1.0 1.5 2.5 3.0 3.5 4.0)

TOTAL=${#QPS_LIST[@]}
FAILED=()

echo "========================================"
echo "Batch QPS Experiment"
echo "Dataset: $DATASET"
echo "QPS values: ${QPS_LIST[*]}"
echo "Total runs: $TOTAL"
echo "Start time: $(date)"
echo "========================================"

for i in "${!QPS_LIST[@]}"; do
    QPS="${QPS_LIST[$i]}"
    RUN_NUM=$((i + 1))

    echo ""
    echo "========================================"
    echo "[$RUN_NUM/$TOTAL] Starting QPS=$QPS  ($(date))"
    echo "========================================"

    if bash "$SCRIPT_DIR/auto_run_pcie_ablation_ab.sh" "$DATASET" --qps "$QPS"; then
        echo "[$RUN_NUM/$TOTAL] QPS=$QPS completed successfully  ($(date))"
    else
        echo "[$RUN_NUM/$TOTAL] QPS=$QPS FAILED  ($(date))"
        FAILED+=("$QPS")
        # 继续跑下一个，不要因为一个失败而全部停掉
    fi
done

echo ""
echo "========================================"
echo "All done!  ($(date))"
echo "Total: $TOTAL, Failed: ${#FAILED[@]}"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "Failed QPS values: ${FAILED[*]}"
fi
echo "========================================"
