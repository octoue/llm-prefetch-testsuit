#!/bin/bash
# 批量运行 mechanism ablation 实验
# num_blocks=1000, qps={0.5,1.0,1.5,2.0,2.5}, groups=g0,g1,full
# num_blocks=750,  qps={0.5,1.5,2.5},           groups=g0,g1,full

set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ABLATION_SCRIPT="$SCRIPT_DIR/auto_run_mechanism_ablation.sh"

DATASET="pcie-heavy"
ABLATION_GROUPS="g0,g1,full"
MAX_REQUESTS_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --max-requests) MAX_REQUESTS_ARGS=(--max-requests "$2"); shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# 当前正在运行的子实验 PID
CHILD_PID=""

cleanup() {
    echo ""
    echo "Batch script interrupted, cleaning up..."
    if [[ -n "$CHILD_PID" ]] && kill -0 "$CHILD_PID" 2>/dev/null; then
        # 向子进程的整个进程组发 TERM，让子脚本的 trap 有机会清理 vLLM
        kill -TERM -- -"$CHILD_PID" 2>/dev/null || kill -TERM "$CHILD_PID" 2>/dev/null
        # 等够子脚本 stop_vllm 的 15s + 余量
        local waited=0
        while kill -0 "$CHILD_PID" 2>/dev/null && [[ $waited -lt 25 ]]; do
            sleep 1; waited=$((waited + 1))
        done
        if kill -0 "$CHILD_PID" 2>/dev/null; then
            echo "Force killing child process group..."
            kill -9 -- -"$CHILD_PID" 2>/dev/null || true
        fi
    fi
    # 兜底：杀掉本脚本占用端口的残留进程
    local port="${API_PORT:-8000}"
    local pids
    pids=$(lsof -ti :"$port" 2>/dev/null) || true
    if [[ -n "$pids" ]]; then
        echo "Killing residual processes on port $port: $pids"
        echo "$pids" | xargs kill -9 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM HUP

FAILED=()
SUCCEEDED=()

run_one() {
    local blocks="$1"
    local qps="$2"
    local label="blk${blocks}_qps${qps}"

    echo ""
    echo "============================================"
    echo "  Running: blocks=$blocks qps=$qps groups=$GROUPS"
    echo "  $(date)"
    echo "============================================"

    # setsid 让子实验在新进程组中运行，方便整组清理
    setsid bash "$ABLATION_SCRIPT" "$DATASET" \
        --gpu-blocks "$blocks" \
        --qps "$qps" \
        --open-loop \
        --groups "$ABLATION_GROUPS" \
        "${MAX_REQUESTS_ARGS[@]}" &
    CHILD_PID=$!
    wait "$CHILD_PID"
    local rc=$?
    CHILD_PID=""

    if [[ $rc -eq 0 ]]; then
        SUCCEEDED+=("$label")
        echo "SUCCESS: $label"
    else
        FAILED+=("$label")
        echo "FAILED: $label (skipping to next)"
    fi
}

# --- num_blocks=1000 ---
for qps in 0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 2.0; do
    run_one 1000 "$qps"
done

# --- num_blocks=750 ---
for qps in 0.4 0.8 1.2; do
    run_one 750 "$qps"
done

echo ""
echo "============================================"
echo "  Batch Complete! $(date)"
echo "============================================"
echo "  Succeeded: ${#SUCCEEDED[@]} - ${SUCCEEDED[*]}"
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "  Failed:    ${#FAILED[@]} - ${FAILED[*]}"
fi
echo "============================================"
