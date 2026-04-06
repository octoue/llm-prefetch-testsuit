#!/bin/bash
# vLLM 启动脚本: Mixtral-8x7B EP=4 + EPLB + PCIe-only
#
# 用法:
#   ./start_vllm_mixtral_ep4.sh [--eplb] [--pcie-sched] [--eplb-phase] [--port PORT]
#
# 参数:
#   --model PATH           模型路径 (默认: 需手动填写)
#   --ep-size N            EP 大小 (默认: 4)
#   --eplb                 启用 EPLB
#   --pcie-sched           启用 PCIe 调度器
#   --eplb-phase           启用 EPLB-Phase-Aware 调度
#   --async-eplb           使用异步 EPLB
#   --port PORT            API 端口 (默认: 8000)
#   --gpu-mem-util FLOAT   GPU 显存利用率 (默认: 0.5, 低值以触发 offloading)
#   --step-interval N      EPLB step interval (默认: 100)
#   --log-file PATH        日志文件

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================
# Defaults — MODEL_PATH 需要手动填写
# ============================================================
MODEL_PATH="${MODEL_PATH:-/lpai/models/mistralai__mixtral-8x7b-instruct-v0_1/24-08-19-1318}"
EP_SIZE=4
ENABLE_EPLB=0
ENABLE_PCIE_SCHED=0
ENABLE_EPLB_PHASE=0
USE_ASYNC_EPLB=0
API_PORT="${API_PORT:-8000}"
LOG_FILE=""
NUM_REDUNDANT_EXPERTS=0
EPLB_STEP_INTERVAL=100
GPU_MEM_UTIL=0.5
MAX_NUM_SEQS=32
MAX_MODEL_LEN=4096
KV_OFFLOADING_SIZE=20

# ============================================================
# Parse args
# ============================================================
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)           MODEL_PATH="$2";              shift 2 ;;
        --ep-size)         EP_SIZE="$2";                 shift 2 ;;
        --eplb)            ENABLE_EPLB=1;                shift ;;
        --pcie-sched)      ENABLE_PCIE_SCHED=1;          shift ;;
        --eplb-phase)      ENABLE_EPLB_PHASE=1;          shift ;;
        --async-eplb)      USE_ASYNC_EPLB=1;             shift ;;
        --port)            API_PORT="$2";                shift 2 ;;
        --log-file)        LOG_FILE="$2";                shift 2 ;;
        --redundant)       NUM_REDUNDANT_EXPERTS="$2";   shift 2 ;;
        --step-interval)   EPLB_STEP_INTERVAL="$2";      shift 2 ;;
        --gpu-mem-util)    GPU_MEM_UTIL="$2";            shift 2 ;;
        --max-num-seqs)    MAX_NUM_SEQS="$2";            shift 2 ;;
        --max-model-len)   MAX_MODEL_LEN="$2";           shift 2 ;;
        --kv-offloading)   KV_OFFLOADING_SIZE="$2";      shift 2 ;;
        -h|--help)
            sed -n '2,/^$/p' "$0" | grep '^#' | sed 's/^# \?//'
            exit 0 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

# ============================================================
# Validate
# ============================================================
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "Error: 模型路径不存在: $MODEL_PATH"
    exit 1
fi

# ============================================================
# PID file & PCIe-only
# ============================================================
PIDFILE="/tmp/vllm_mixtral_ep_${API_PORT}.pid"
export VLLM_EP_PIDFILE="$PIDFILE"
export NCCL_P2P_DISABLE=1
export NCCL_NVLS_ENABLE=0

if [[ -z "$CUDA_VISIBLE_DEVICES" ]]; then
    GPUS=$(seq -s, 0 $((EP_SIZE - 1)))
    export CUDA_VISIBLE_DEVICES="$GPUS"
fi

# PCIe scheduler env vars
if [[ $ENABLE_PCIE_SCHED -eq 1 ]]; then
    export VLLM_PCIE_SCHEDULER=1
fi
if [[ $ENABLE_EPLB_PHASE -eq 1 ]]; then
    export VLLM_EPLB_PHASE_AWARE=1
fi

echo "============================================"
echo "vLLM Mixtral-8x7B EP Mode (PCIe-only)"
echo "============================================"
echo "Model:          $MODEL_PATH"
echo "EP size:        $EP_SIZE (GPUs: $CUDA_VISIBLE_DEVICES)"
echo "EPLB:           $([ $ENABLE_EPLB -eq 1 ] && echo 'ON' || echo 'OFF')"
echo "  Async:        $([ $USE_ASYNC_EPLB -eq 1 ] && echo 'ON' || echo 'OFF')"
echo "  Step interval:$EPLB_STEP_INTERVAL"
echo "PCIe Scheduler: $([ $ENABLE_PCIE_SCHED -eq 1 ] && echo 'ON' || echo 'OFF')"
echo "EPLB Phase:     $([ $ENABLE_EPLB_PHASE -eq 1 ] && echo 'ON' || echo 'OFF')"
echo "Offloading:     ${KV_OFFLOADING_SIZE}GiB"
echo "GPU Mem Util:   $GPU_MEM_UTIL"
echo "Port:           $API_PORT"
echo "PID file:       $PIDFILE"
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
    --kv-offloading-size "$KV_OFFLOADING_SIZE"
    --kv-offloading-backend native
    --swap-space 64
    --enable-prefix-caching
    --disable-hybrid-kv-cache-manager
)

if [[ $ENABLE_EPLB -eq 1 ]]; then
    EPLB_JSON="{\"num_redundant_experts\": $NUM_REDUNDANT_EXPERTS, \"step_interval\": $EPLB_STEP_INTERVAL"
    if [[ $USE_ASYNC_EPLB -eq 1 ]]; then
        EPLB_JSON="$EPLB_JSON, \"use_async\": true"
    fi
    EPLB_JSON="$EPLB_JSON}"
    CMD_ARGS+=(
        --enable-eplb
        --eplb-config "$EPLB_JSON"
    )
    echo "EPLB config: $EPLB_JSON"
fi

# ============================================================
# Launch
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

trap 'kill -TERM $VLLM_PID 2>/dev/null; wait $VLLM_PID 2>/dev/null; rm -f "$PIDFILE"' EXIT INT TERM

wait $VLLM_PID
EXIT_CODE=$?
rm -f "$PIDFILE"
exit $EXIT_CODE
