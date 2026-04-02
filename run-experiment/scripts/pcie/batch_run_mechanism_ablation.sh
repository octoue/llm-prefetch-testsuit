#!/bin/bash
# 批量跑不同 QPS 的机制消融实验
#
# 汇总表（与 batch_run_qps + auto_run_pcie_ablation_ab 类似）：
#   <repo>/results/pcie_mechanism_ablation_experiments.txt（每次完整 auto_run 按 --groups 追加多行）
#
# 用法:
#   nohup bash scripts/pcie/batch_run_mechanism_ablation.sh [dataset] [options] > batch_mechanism_ablation.log 2>&1 &
#
# 选项:
#   --rounds N         全部 QPS 跑 N 轮（默认 1）
#   --gpu-blocks N     覆盖 GPU blocks
#   --qps-list "..."   自定义 QPS 列表（空格分隔）
#   --groups "..."     只跑指定 group（逗号分隔，或 all）
#
# 示例:
#   # C1 核心消融：仅 3 个新消融组，3 轮
#   nohup bash scripts/pcie/batch_run_mechanism_ablation.sh pcie-heavy \
#       --rounds 3 --gpu-blocks 1000 --qps-list "1.0 1.5 2.0 2.5" \
#       --groups "no-pq,no-ef,no-cc" > batch_mechanism_ablation.log 2>&1 &
#
#   # C2 完整性重跑：6 组全跑
#   nohup bash scripts/pcie/batch_run_mechanism_ablation.sh pcie-heavy \
#       --rounds 3 --gpu-blocks 1000 --qps-list "1.0 1.5 2.0 2.5" \
#       --groups all > batch_mechanism_ablation.log 2>&1 &
#
#   # C3 gpu_blocks 敏感性：full vs g1，不同 blocks
#   nohup bash scripts/pcie/batch_run_mechanism_ablation.sh pcie-heavy \
#       --rounds 3 --gpu-blocks 500 --qps-list "1.0 1.5 2.0 2.5" \
#       --groups "full,g1" > batch_mech_blk500.log 2>&1 &
#
#   # 补跑单个组
#   nohup bash scripts/pcie/batch_run_mechanism_ablation.sh pcie-heavy \
#       --rounds 1 --gpu-blocks 1000 --qps-list "2.5" \
#       --groups no-ef > batch_mech_fix.log 2>&1 &

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_EXP_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$RUN_EXP_DIR/scripts/utils/common.sh"

DATASET="${1:-pcie-heavy}"
shift 2>/dev/null || true

ROUNDS=1
GPU_BLOCKS=""
QPS_LIST=(1.0 1.5 2.0 2.5)
TARGET_GROUPS=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --rounds)
            ROUNDS="$2"; shift 2 ;;
        --gpu-blocks)
            GPU_BLOCKS="$2"; shift 2 ;;
        --qps-list)
            IFS=' ' read -r -a QPS_LIST <<< "$2"; shift 2 ;;
        --groups)
            TARGET_GROUPS="$2"; shift 2 ;;
        *)
            EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [[ -n "$GPU_BLOCKS" ]]; then
    EXTRA_ARGS+=(--gpu-blocks "$GPU_BLOCKS")
fi
if [[ -n "$TARGET_GROUPS" ]]; then
    EXTRA_ARGS+=(--groups "$TARGET_GROUPS")
fi

NUM_QPS=${#QPS_LIST[@]}
TOTAL_RUNS=$((NUM_QPS * ROUNDS))
FAILED=()

echo "========================================"
echo "Batch Mechanism Ablation Experiment"
echo "========================================"
echo "Dataset:    $DATASET"
echo "QPS values: ${QPS_LIST[*]}"
echo "Rounds:     $ROUNDS"
echo "GPU blocks: ${GPU_BLOCKS:-<from config>}"
echo "Groups:     ${TARGET_GROUPS:-no-pq,no-ef,no-cc (default)}"
echo "Extra args: ${EXTRA_ARGS[*]}"
echo "Total runs: $NUM_QPS QPS x $ROUNDS rounds = $TOTAL_RUNS ablation runs"
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

        wait_for_idle_gpus 2 100 60

        if bash "$SCRIPT_DIR/auto_run_mechanism_ablation.sh" "$DATASET" --qps "$QPS" "${EXTRA_ARGS[@]}"; then
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
