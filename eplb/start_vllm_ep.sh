#!/bin/bash
# vLLM 启动脚本: Expert Parallelism + EPLB + PCIe-only
# 用法:
#   ./start_vllm_ep.sh [--model PATH] [--ep-size N] [--eplb] [--port PORT] [--log-file PATH]
#
# 默认: DeepSeek-V2-Lite-Chat, EP=2, PCIe-only (NVLink disabled)
#
# 进程管理:
#   启动后将 vLLM 主进程 PID 写入 $PIDFILE (默认 /tmp/vllm_ep_<port>.pid)
#   调用方可通过 stop_vllm_ep() 或直接 kill 该 PID 来停止服务

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================
# Defaults
# ============================================================
MODEL_PATH="${MODEL_PATH:-/lpai/models/deepseek-ai__deepseek-v2-lite-chat/24-05-17-0658}"
EP_SIZE=2
ENABLE_EPLB=0
API_PORT="${API_PORT:-8000}"
LOG_FILE=""
NUM_REDUNDANT_EXPERTS=0
EPLB_STEP_INTERVAL=3000
ALL2ALL_BACKEND="allgather_reducescatter"
GPU_MEM_UTIL=0.85
MAX_NUM_SEQS=32
MAX_MODEL_LEN=4096

# ============================================================
# Parse args
# ============================================================
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)         MODEL_PATH="$2";              shift 2 ;;
        --ep-size)       EP_SIZE="$2";                 shift 2 ;;
        --eplb)          ENABLE_EPLB=1;                shift ;;
        --port)          API_PORT="$2";                shift 2 ;;
        --log-file)      LOG_FILE="$2";                shift 2 ;;
        --redundant)     NUM_REDUNDANT_EXPERTS="$2";   shift 2 ;;
        --step-interval) EPLB_STEP_INTERVAL="$2";      shift 2 ;;
        --all2all)       ALL2ALL_BACKEND="$2";         shift 2 ;;
        --gpu-mem-util)  GPU_MEM_UTIL="$2";            shift 2 ;;
        --max-num-seqs)  MAX_NUM_SEQS="$2";            shift 2 ;;
        --max-model-len) MAX_MODEL_LEN="$2";           shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--model PATH] [--ep-size N] [--eplb] [--port PORT]"
            exit 0 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

# ============================================================
# PID file for process management
# ============================================================
PIDFILE="/tmp/vllm_ep_${API_PORT}.pid"
export VLLM_EP_PIDFILE="$PIDFILE"

# ============================================================
# Force PCIe: disable NVLink
# ============================================================
export NCCL_P2P_DISABLE=1
export NCCL_NVLS_ENABLE=0

# GPU selection: use first EP_SIZE GPUs (or respect CUDA_VISIBLE_DEVICES)
if [[ -z "$CUDA_VISIBLE_DEVICES" ]]; then
    GPUS=$(seq -s, 0 $((EP_SIZE - 1)))
    export CUDA_VISIBLE_DEVICES="$GPUS"
fi

echo "============================================"
echo "vLLM EP Mode (PCIe-only)"
echo "============================================"
echo "Model:        $MODEL_PATH"
echo "EP size:      $EP_SIZE (GPUs: $CUDA_VISIBLE_DEVICES)"
echo "All2All:      $ALL2ALL_BACKEND"
echo "EPLB:         $([ $ENABLE_EPLB -eq 1 ] && echo 'ON' || echo 'OFF')"
echo "NCCL_P2P_DISABLE=1  NCCL_NVLS_ENABLE=0"
echo "Port:         $API_PORT"
echo "Max Seqs:     $MAX_NUM_SEQS"
echo "Max Model Len:$MAX_MODEL_LEN"
echo "GPU Mem Util: $GPU_MEM_UTIL"
echo "PID file:     $PIDFILE"
echo "============================================"

# ============================================================
# Build vLLM command
# ============================================================
CMD_ARGS=(
    --model "$MODEL_PATH"
    --host 0.0.0.0
    --port "$API_PORT"
    --dtype float16
    --tensor-parallel-size "$EP_SIZE"
    --gpu-memory-utilization "$GPU_MEM_UTIL"
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-model-len "$MAX_MODEL_LEN"
    --trust-remote-code
    --enforce-eager
    --enable-expert-parallel
    --all2all-backend "$ALL2ALL_BACKEND"
)

if [[ $ENABLE_EPLB -eq 1 ]]; then
    # EPLB uses --eplb-config JSON string
    EPLB_JSON="{\"num_redundant_experts\": $NUM_REDUNDANT_EXPERTS, \"step_interval\": $EPLB_STEP_INTERVAL}"
    CMD_ARGS+=(
        --enable-eplb
        --eplb-config "$EPLB_JSON"
    )
    echo "EPLB config: $EPLB_JSON"
fi

# ============================================================
# Launch (write PID for callers to manage)
# ============================================================
if [[ -n "$LOG_FILE" ]]; then
    [[ "$LOG_FILE" != /* ]] && LOG_FILE="$(pwd)/$LOG_FILE"
    mkdir -p "$(dirname "$LOG_FILE")"
    echo "Log: $LOG_FILE"
    VLLM_TEST_ENABLE_EP=1 HF_HUB_OFFLINE=1 vllm serve "${CMD_ARGS[@]}" > "$LOG_FILE" 2>&1 &
else
    VLLM_TEST_ENABLE_EP=1 HF_HUB_OFFLINE=1 vllm serve "${CMD_ARGS[@]}" &
fi

VLLM_PID=$!
echo "$VLLM_PID" > "$PIDFILE"
echo "vLLM PID: $VLLM_PID (saved to $PIDFILE)"

# Forward signals to vLLM process
trap 'kill -TERM $VLLM_PID 2>/dev/null; wait $VLLM_PID 2>/dev/null; rm -f "$PIDFILE"' EXIT INT TERM

wait $VLLM_PID
EXIT_CODE=$?
rm -f "$PIDFILE"
exit $EXIT_CODE
