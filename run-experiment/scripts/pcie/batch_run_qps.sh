#!/bin/bash
# 批量跑不同 QPS 的 PCIe Ablation 实验
#
# 用法:
#   nohup bash scripts/pcie/batch_run_qps.sh [dataset] [options] > batch_qps.log 2>&1 &
#
# 选项:
#   --repeats N        每个 QPS 重复 N 次（默认 1 = 不重复，等同旧行为）
#   --gpu-blocks N     覆盖 GPU blocks
#   --qps-list "..."   自定义 QPS 列表（空格分隔，需引号括起来）
#
# 示例:
#   # 旧行为：每个 QPS 跑 1 次消融
#   nohup bash scripts/pcie/batch_run_qps.sh pcie-heavy > batch.log 2>&1 &
#
#   # 每个 QPS 跑 3 次消融（用于论文 mean ± std）
#   nohup bash scripts/pcie/batch_run_qps.sh pcie-heavy --repeats 3 > batch.log 2>&1 &
#
#   # 自定义 QPS 列表 + 3 次重复
#   nohup bash scripts/pcie/batch_run_qps.sh pcie-heavy --repeats 3 --qps-list "1.5 3.0 4.0" > batch.log 2>&1 &
#
# 合上电脑也不会断。查看进度: tail -f batch.log / batch_qps.log

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_EXP_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$RUN_EXP_DIR/scripts/utils/common.sh"

DATASET="${1:-pcie-heavy}"
shift 2>/dev/null || true

REPEATS=1
QPS_LIST=(0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0)
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repeats)
            REPEATS="$2"; shift 2 ;;
        --qps-list)
            IFS=' ' read -r -a QPS_LIST <<< "$2"; shift 2 ;;
        *)
            EXTRA_ARGS+=("$1"); shift ;;
    esac
done

TOTAL=${#QPS_LIST[@]}
FAILED=()

echo "========================================"
echo "Batch QPS Experiment"
echo "========================================"
echo "Dataset:    $DATASET"
echo "QPS values: ${QPS_LIST[*]}"
echo "Repeats:    $REPEATS (per QPS)"
echo "Extra args: ${EXTRA_ARGS[*]}"
echo "Total runs: $TOTAL QPS × $REPEATS repeats = $((TOTAL * REPEATS)) ablation runs"
echo "Start time: $(date)"
echo "========================================"

for i in "${!QPS_LIST[@]}"; do
    QPS="${QPS_LIST[$i]}"
    RUN_NUM=$((i + 1))

    echo ""
    echo "========================================"
    echo "[$RUN_NUM/$TOTAL] QPS=$QPS × $REPEATS repeats  ($(date))"
    echo "========================================"

    # 等待 GPU 空闲后再启动
    wait_for_idle_gpus 2 100 60

    if [[ "$REPEATS" -gt 1 ]]; then
        # 使用重复脚本
        if bash "$SCRIPT_DIR/run_repeated_ablation.sh" "$DATASET" --qps "$QPS" --repeats "$REPEATS" "${EXTRA_ARGS[@]}"; then
            echo "[$RUN_NUM/$TOTAL] QPS=$QPS completed successfully  ($(date))"
        else
            echo "[$RUN_NUM/$TOTAL] QPS=$QPS FAILED  ($(date))"
            FAILED+=("$QPS")
        fi
    else
        # 单次运行，直接调用 auto_run（旧行为）
        if bash "$SCRIPT_DIR/auto_run_pcie_ablation_ab.sh" "$DATASET" --qps "$QPS" "${EXTRA_ARGS[@]}"; then
            echo "[$RUN_NUM/$TOTAL] QPS=$QPS completed successfully  ($(date))"
        else
            echo "[$RUN_NUM/$TOTAL] QPS=$QPS FAILED  ($(date))"
            FAILED+=("$QPS")
        fi
    fi
done

echo ""
echo "========================================"
echo "All done!  ($(date))"
echo "Total QPS points: $TOTAL, Failed: ${#FAILED[@]}"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "Failed QPS values: ${FAILED[*]}"
fi
echo "========================================"
