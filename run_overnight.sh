#!/bin/bash
# 挂机实验脚本 - 串行运行多组实验
# 用法: nohup bash run_overnight.sh > overnight.log 2>&1 &

set -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_PCIE="$REPO_ROOT/run-experiment/scripts/pcie/run_pcie.sh"
RESULTS_DIR="$REPO_ROOT/results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$REPO_ROOT/overnight_${TIMESTAMP}.log"

# 创建结果目录
mkdir -p "$RESULTS_DIR/32b"
mkdir -p "$RESULTS_DIR/72b"

# ============================================================
# 资源清理函数
# ============================================================
API_PORT="${API_PORT:-8000}"

cleanup_resources() {
    echo "[$(date)] Cleaning up resources..."

    # 杀掉残留的端口占用进程
    local pids
    pids=$(lsof -ti :"$API_PORT" 2>/dev/null) || true
    if [[ -n "$pids" ]]; then
        echo "  Killing residual processes on port $API_PORT: $pids"
        echo "$pids" | xargs kill -9 2>/dev/null || true
        sleep 2
    fi

    # 清理可能残留的 vllm 进程
    pkill -f "vllm.entrypoints" 2>/dev/null || true
    sleep 3

    # 清理 GPU 锁
    if [[ -f "$REPO_ROOT/run-experiment/scripts/utils/gpu_lock.sh" ]]; then
        (source "$REPO_ROOT/run-experiment/scripts/utils/gpu_lock.sh" 2>/dev/null && \
            _clean_stale_locks 2>/dev/null) || true
    fi

    echo "[$(date)] Cleanup done."
}

# 捕获退出信号，确保清理
trap cleanup_resources EXIT INT TERM

# ============================================================
# 快照 & 移动结果
# ============================================================
# 记录 results/ 下已有的子目录，实验结束后把新增的目录移到目标文件夹
snapshot_results() {
    # 输出 results/ 下直接子目录名（不含 32b/72b 等目标目录）
    ls -1d "$RESULTS_DIR"/*/ 2>/dev/null | xargs -I{} basename {} | sort
}

move_new_results() {
    local target_dir="$1"
    local before_file="$2"
    local after
    after=$(snapshot_results)

    # 找出新增的目录
    local new_dirs
    new_dirs=$(comm -13 "$before_file" <(echo "$after"))

    if [[ -n "$new_dirs" ]]; then
        while IFS= read -r dir; do
            # 跳过目标目录本身
            if [[ "$dir" == "32b" || "$dir" == "72b" ]]; then
                continue
            fi
            echo "  Moving results/$dir -> $target_dir/$dir"
            mv "$RESULTS_DIR/$dir" "$target_dir/"
        done <<< "$new_dirs"
    else
        echo "  No new result directories found."
    fi
}

# ============================================================
# 实验运行函数
# ============================================================
TOTAL_EXPERIMENTS=4
CURRENT_EXP=0
FAILED_LIST=()

run_experiment() {
    local exp_name="$1"
    local target_result_dir="$2"
    shift 2
    local args=("$@")

    CURRENT_EXP=$((CURRENT_EXP + 1))

    echo ""
    echo "================================================================"
    echo "[$(date)] [$CURRENT_EXP/$TOTAL_EXPERIMENTS] Starting: $exp_name"
    echo "  Args: ${args[*]}"
    echo "  Target results -> $target_result_dir"
    echo "================================================================"

    # 运行前清理残留资源
    cleanup_resources

    # 快照当前结果目录
    local snap_file
    snap_file=$(mktemp)
    snapshot_results > "$snap_file"

    # 运行实验，捕获退出码，单个实验失败不终止整体
    if bash "$RUN_PCIE" "${args[@]}"; then
        echo "[$(date)] Experiment $exp_name: SUCCESS"
    else
        local rc=$?
        echo "[$(date)] Experiment $exp_name: FAILED (exit code $rc)"
        FAILED_LIST+=("$exp_name")
    fi

    # 移动新产生的结果到目标目录
    move_new_results "$target_result_dir" "$snap_file"
    rm -f "$snap_file"

    # 实验结束后清理资源，为下一个实验做准备
    cleanup_resources
}

# ============================================================
# 开始实验
# ============================================================
{
echo "================================================================"
echo "[$(date)] Overnight experiment batch started"
echo "  Results: $RESULTS_DIR/{32b,72b}/"
echo "  Log: $LOG_FILE"
echo "================================================================"

# ----------------------------------------------------------
# 实验1: 32B 模型
# ----------------------------------------------------------
run_experiment "32b_pcie_heavy" "$RESULTS_DIR/32b" \
    --model 32b \
    --qps 0.5,1.0,1.5,2.0,2.5 \
    --rounds 3 \
    --num-blocks 750 \
    --pp 2 \
    --dataset pcie-heavy \
    --closed-loop

# ----------------------------------------------------------
# 实验2: 72B 模型
# ----------------------------------------------------------

# 2.1: num-blocks 750, qps 1.0,1.5,2.0
run_experiment "72b_blk750" "$RESULTS_DIR/72b" \
    --model 72b \
    --qps 1.0,1.5,2.0 \
    --rounds 1 \
    --num-blocks 750 \
    --pp 4 \
    --dataset pcie-heavy \
    --closed-loop

# 2.2: num-blocks 500, qps 1.0,1.5,2.0
run_experiment "72b_blk500" "$RESULTS_DIR/72b" \
    --model 72b \
    --qps 1.0,1.5,2.0 \
    --rounds 1 \
    --num-blocks 500 \
    --pp 4 \
    --dataset pcie-heavy \
    --closed-loop

# 2.3: max-num-seqs 128, num-blocks 500, qps 0.5,1.0,1.5
run_experiment "72b_blk500_seqs128" "$RESULTS_DIR/72b" \
    --model 72b \
    --qps 0.5,1.0,1.5 \
    --rounds 1 \
    --num-blocks 500 \
    --pp 4 \
    --dataset pcie-heavy \
    --max-num-seqs 128 \
    --closed-loop

# ============================================================
# 总结
# ============================================================
echo ""
echo "================================================================"
echo "[$(date)] All experiments finished!"
echo "  32B results: $RESULTS_DIR/32b/"
echo "  72B results: $RESULTS_DIR/72b/"
if [[ ${#FAILED_LIST[@]} -gt 0 ]]; then
    echo "  FAILED experiments: ${FAILED_LIST[*]}"
else
    echo "  All experiments succeeded."
fi
echo "================================================================"
} 2>&1 | tee "$LOG_FILE"
