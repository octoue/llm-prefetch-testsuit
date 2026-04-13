#!/bin/bash
# ============================================================================
# Phi-3.5-MoE boot smoke test
#
# 目的: 在跑完整实验之前, 先确认 Phi-3.5-MoE 能在当前硬件+参数下成功启动
# (weight load, profile run 含 EPLB dummy rearrange, 首个 health check)
#
# 不跑任何真实 workload. boot 成功后立即清理.
#
# 用法:
#   ./smoke_phi.sh
#   ./smoke_phi.sh --mode sync    # 默认, 测 sync EPLB
#   ./smoke_phi.sh --mode async   # 测 async EPLB
#   ./smoke_phi.sh --gpu-mem-util 0.7 --num-gpu-blocks 8000   # 调参重试
# ============================================================================

set -o pipefail

# ---- Config ---------------------------------------------------------------
PHI_MODEL="/lpai/models/microsoft__phi-3_5-moe-instruct/24-08-30-0107"
GPU_LIST="0,1,2,3"
API_PORT=8000
EP_SIZE=4
GPU_MEM_UTIL=0.65
NUM_GPU_BLOCKS=6000
MAX_NUM_SEQS=32
MAX_MODEL_LEN=4096
KV_OFFLOADING_SIZE=20
EPLB_STEP_INTERVAL=100
MODE="sync"
HEALTH_TIMEOUT=600   # 10 分钟封顶

# ---- Parse args -----------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)           PHI_MODEL="$2";        shift 2 ;;
        --gpus)            GPU_LIST="$2";         shift 2 ;;
        --gpu-mem-util)    GPU_MEM_UTIL="$2";     shift 2 ;;
        --num-gpu-blocks)  NUM_GPU_BLOCKS="$2";   shift 2 ;;
        --step-interval)   EPLB_STEP_INTERVAL="$2"; shift 2 ;;
        --mode)            MODE="$2";             shift 2 ;;
        --timeout)         HEALTH_TIMEOUT="$2";   shift 2 ;;
        -h|--help)
            sed -n '2,/^$/p' "$0" | grep '^#' | sed 's/^# \?//'
            exit 0 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

if [[ "$MODE" != "sync" && "$MODE" != "async" ]]; then
    echo "ERROR: --mode must be sync or async"; exit 1
fi
if [[ ! -d "$PHI_MODEL" ]]; then
    echo "ERROR: model not found: $PHI_MODEL"; exit 1
fi

USE_ASYNC="false"
[[ "$MODE" == "async" ]] && USE_ASYNC="true"

LOG_FILE="/tmp/smoke_phi_${MODE}_$(date +%Y%m%d_%H%M%S).log"
PIDFILE="/tmp/smoke_phi_${API_PORT}_$$.pid"
VLLM_PGID=""

# ---- Logging --------------------------------------------------------------
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log_section() {
    echo ""
    echo "================================================================"
    log "$*"
    echo "================================================================"
}

# ---- Cleanup --------------------------------------------------------------
stop_vllm() {
    log "Cleaning up..."
    if [[ -n "$VLLM_PGID" ]]; then
        kill -TERM -"$VLLM_PGID" 2>/dev/null || true
        sleep 3
        kill -9 -"$VLLM_PGID" 2>/dev/null || true
    fi
    if [[ -f "$PIDFILE" ]]; then
        local pid
        pid=$(cat "$PIDFILE" 2>/dev/null)
        [[ -n "$pid" ]] && kill -9 "$pid" 2>/dev/null || true
        rm -f "$PIDFILE"
    fi
    local port_pids
    port_pids=$(lsof -ti :"$API_PORT" 2>/dev/null) || true
    if [[ -n "$port_pids" ]]; then
        echo "$port_pids" | xargs kill -9 2>/dev/null || true
    fi
    sleep 3
    log "Cleanup done."
}

trap 'log "Interrupted"; stop_vllm; exit 130' INT TERM
trap 'stop_vllm' EXIT

# ---- Start vLLM -----------------------------------------------------------
log_section "Phi-3.5-MoE boot smoke test"
log "Model:           $PHI_MODEL"
log "Mode:            $MODE (use_async=$USE_ASYNC)"
log "GPUs:            $GPU_LIST"
log "gpu_mem_util:    $GPU_MEM_UTIL"
log "num_gpu_blocks:  $NUM_GPU_BLOCKS"
log "step_interval:   $EPLB_STEP_INTERVAL"
log "Log file:        $LOG_FILE"

export CUDA_VISIBLE_DEVICES="$GPU_LIST"
export NCCL_P2P_DISABLE=1
export NCCL_NVLS_ENABLE=0
export VLLM_TEST_ENABLE_EP=1
export HF_HUB_OFFLINE=1
export VLLM_PCIE_PP_PHASE_AWARE=0

log "Starting vLLM..."

setsid vllm serve "$PHI_MODEL" \
    --host 0.0.0.0 \
    --port "$API_PORT" \
    --dtype float16 \
    --tensor-parallel-size "$EP_SIZE" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-model-len "$MAX_MODEL_LEN" \
    --trust-remote-code \
    --enforce-eager \
    --enable-expert-parallel \
    --kv-offloading-size "$KV_OFFLOADING_SIZE" \
    --kv-offloading-backend native \
    --swap-space 64 \
    --enable-prefix-caching \
    --disable-hybrid-kv-cache-manager \
    --num-gpu-blocks-override "$NUM_GPU_BLOCKS" \
    --enable-eplb \
    --eplb-config "{\"step_interval\": $EPLB_STEP_INTERVAL, \"num_redundant_experts\": 0, \"use_async\": $USE_ASYNC, \"log_balancedness\": true, \"log_balancedness_interval\": 10}" \
    > "$LOG_FILE" 2>&1 &

VLLM_PID=$!
VLLM_PGID=$VLLM_PID
echo "$VLLM_PID" > "$PIDFILE"
log "vLLM PID=$VLLM_PID PGID=$VLLM_PGID"
log "Log tail: tail -f $LOG_FILE"

# ---- Wait for health ------------------------------------------------------
log "Waiting for /health (timeout ${HEALTH_TIMEOUT}s)..."
waited=0
ticks=0
while [[ $waited -lt $HEALTH_TIMEOUT ]]; do
    if curl -s -m 2 "http://localhost:$API_PORT/health" &>/dev/null; then
        log_section "✓ BOOT SUCCESS"
        log "Server up after ${waited}s."
        log "Key checkpoints from vLLM log:"
        grep -E "Rearranging experts|Rearranged experts|GPU blocks|num_gpu_blocks|Profile|out of memory|OOM|ERROR|FATAL" "$LOG_FILE" \
            | head -30 | sed 's/^/    /'
        log ""
        log "Phi-3.5-MoE 可以在这组参数下启动. 可以放心跑完整实验."
        log ""
        log "配置确认:"
        log "  --mode $MODE"
        log "  --gpu-mem-util $GPU_MEM_UTIL"
        log "  --num-gpu-blocks $NUM_GPU_BLOCKS"
        log "  --step-interval $EPLB_STEP_INTERVAL"
        exit 0
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        log_section "✗ BOOT FAILED — vLLM process died"
        log "Last 50 lines of log:"
        tail -50 "$LOG_FILE" | sed 's/^/    /'
        log ""
        log "常见原因诊断:"
        if grep -q "out of memory\|CUDA out of memory\|OOM" "$LOG_FILE"; then
            log "  → OOM: 降低 --gpu-mem-util (当前 $GPU_MEM_UTIL) 或 --num-gpu-blocks (当前 $NUM_GPU_BLOCKS)"
        fi
        if grep -q "No module named\|ImportError" "$LOG_FILE"; then
            log "  → 模块缺失: 检查 vLLM 版本或 python 环境"
        fi
        if grep -q "FileNotFoundError\|No such file" "$LOG_FILE"; then
            log "  → 模型文件问题: 确认 $PHI_MODEL 路径存在且有权限"
        fi
        if grep -q "AssertionError" "$LOG_FILE"; then
            log "  → 参数/架构断言失败: 可能 Phi-3.5-MoE 与 --enable-expert-parallel 不兼容"
        fi
        exit 1
    fi
    if [[ $((ticks % 10)) -eq 0 ]]; then
        log "  still booting... (${waited}s elapsed)"
    fi
    sleep 3
    waited=$((waited + 3))
    ticks=$((ticks + 1))
done

log_section "✗ BOOT TIMEOUT after ${HEALTH_TIMEOUT}s"
log "Last 50 lines of log:"
tail -50 "$LOG_FILE" | sed 's/^/    /'
exit 2
