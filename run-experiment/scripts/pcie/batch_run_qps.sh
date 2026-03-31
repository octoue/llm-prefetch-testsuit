#!/bin/bash
# 批量跑不同 QPS 的 PCIe Ablation 实验
#
# 用法:
#   nohup bash scripts/pcie/batch_run_qps.sh [dataset] [options] > batch_qps.log 2>&1 &
#
# 选项:
#   --rounds N         全部 QPS 跑 N 轮（默认 1），每轮按 QPS 从小到大依次执行
#   --gpu-blocks N     覆盖 GPU blocks
#   --qps-list "..."   自定义 QPS 列表（空格分隔，需引号括起来）
#
# 示例:
#   # 每个 QPS 跑 1 次
#   nohup bash scripts/pcie/batch_run_qps.sh pcie-heavy --gpu-blocks 1000 > batch.log 2>&1 &
#
#   # 全部 QPS 跑 3 轮（用于论文 mean ± std）
#   nohup bash scripts/pcie/batch_run_qps.sh pcie-heavy --rounds 3 --gpu-blocks 1000 > batch.log 2>&1 &
#
#   # 自定义 QPS 列表 + 3 轮
#   nohup bash scripts/pcie/batch_run_qps.sh pcie-heavy --rounds 3 --qps-list "0.5 1.0 1.5 2.0 2.5" --gpu-blocks 1000 > batch.log 2>&1 &
#
# 合上电脑也不会断。查看进度: tail -f batch.log / batch_qps.log

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_EXP_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$RUN_EXP_DIR/scripts/utils/common.sh"

DATASET="${1:-pcie-heavy}"
shift 2>/dev/null || true

ROUNDS=1
QPS_LIST=(0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0)
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --rounds)
            ROUNDS="$2"; shift 2 ;;
        --repeats)
            # 兼容旧参数名
            ROUNDS="$2"; shift 2 ;;
        --qps-list)
            IFS=' ' read -r -a QPS_LIST <<< "$2"; shift 2 ;;
        *)
            EXTRA_ARGS+=("$1"); shift ;;
    esac
done

NUM_QPS=${#QPS_LIST[@]}
TOTAL_RUNS=$((NUM_QPS * ROUNDS))
FAILED=()

echo "========================================"
echo "Batch QPS Experiment"
echo "========================================"
echo "Dataset:    $DATASET"
echo "QPS values: ${QPS_LIST[*]}"
echo "Rounds:     $ROUNDS"
echo "Extra args: ${EXTRA_ARGS[*]}"
echo "Total runs: $NUM_QPS QPS × $ROUNDS rounds = $TOTAL_RUNS ablation runs"
echo "Start time: $(date)"
echo "========================================"

RUN_COUNT=0

for ((round = 1; round <= ROUNDS; round++)); do
    echo ""
    echo "########################################"
    echo "# Round $round / $ROUNDS  ($(date))"
    echo "########################################"

    for i in "${!QPS_LIST[@]}"; do
        QPS="${QPS_LIST[$i]}"
        RUN_COUNT=$((RUN_COUNT + 1))

        echo ""
        echo "========================================"
        echo "[Run $RUN_COUNT/$TOTAL_RUNS] Round $round, QPS=$QPS  ($(date))"
        echo "========================================"

        # 等待 GPU 空闲后再启动
        wait_for_idle_gpus 2 100 60

        if bash "$SCRIPT_DIR/auto_run_pcie_ablation_ab.sh" "$DATASET" --qps "$QPS" "${EXTRA_ARGS[@]}"; then
            echo "[Run $RUN_COUNT/$TOTAL_RUNS] Round $round, QPS=$QPS completed successfully  ($(date))"
        else
            echo "[Run $RUN_COUNT/$TOTAL_RUNS] Round $round, QPS=$QPS FAILED  ($(date))"
            FAILED+=("r${round}_q${QPS}")
        fi
    done
done

echo ""
echo "========================================"
echo "All done!  ($(date))"
echo "Total runs: $TOTAL_RUNS, Failed: ${#FAILED[@]}"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "Failed runs: ${FAILED[*]}"
fi
echo "========================================"
