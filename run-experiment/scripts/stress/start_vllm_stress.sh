#!/bin/bash
# Start vLLM with prefetch + PCIeTracer enabled and TP=2/4 on A100.
# Default targets Qwen2.5-32B on A100 x 2.
#
# Usage:
#   ./start_vllm_stress.sh [--tp N] [--ratio R] [--ttl-ms M] [--port P] [--log PATH]
#
# Notable env / flags:
#   STRESS_MODEL                override MODEL_PATH from system.env
#   STRESS_TP                   tensor-parallel size (default 2; use 4 for 72B)
#   STRESS_GPU_MEM_UTIL         GPU memory utilisation (default 0.85 for A100)
#   STRESS_BLOCK_RATIO          --max-prefetch-block-ratio (default 0.3)
#   STRESS_PREFETCH_TTL_MS      --prefetch-ttl-ms (default 60000; 0 disables)
#
# Notes
#   * Uses TP rather than PP so single-node 32B fits on 2x A100-80GB.
#   * VLLM_PCIE_TRACE=1 is forced so the stress runner can dump events.
#   * NVLink is left enabled (unlike start_vllm_pcie.sh) since this experiment
#     measures Prefetch-related H2D, not P2P.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_EXP_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

[ -f "$RUN_EXP_DIR/config/system.env" ] || { echo "missing config/system.env"; exit 1; }
set -a
source "$RUN_EXP_DIR/config/system.env"
[ -f "$RUN_EXP_DIR/config/experiments.env" ] && source "$RUN_EXP_DIR/config/experiments.env"
set +a

TP="${STRESS_TP:-2}"
RATIO="${STRESS_BLOCK_RATIO:-0.3}"
TTL_MS="${STRESS_PREFETCH_TTL_MS:-60000}"
PORT="${API_PORT:-8000}"
GPU_MEM_UTIL="${STRESS_GPU_MEM_UTIL:-0.85}"
LOG_FILE="${RUN_EXP_DIR}/results/stress/vllm_stress.log"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tp) TP="$2"; shift 2 ;;
    --ratio) RATIO="$2"; shift 2 ;;
    --ttl-ms) TTL_MS="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --log) LOG_FILE="$2"; shift 2 ;;
    *) shift ;;
  esac
done

mkdir -p "$(dirname "$LOG_FILE")"
mkdir -p "$RUN_EXP_DIR/profiler_output"

MODEL="${STRESS_MODEL:-$MODEL_PATH}"
[[ -z "$MODEL" ]] && { echo "MODEL_PATH not set; export STRESS_MODEL"; exit 1; }

echo "============================================"
echo "vLLM stress server"
echo "  model     = $MODEL"
echo "  tp        = $TP"
echo "  ratio     = $RATIO"
echo "  ttl_ms    = $TTL_MS"
echo "  port      = $PORT"
echo "============================================"

export VLLM_PCIE_TRACE=1
export VLLM_LOGGING_LEVEL="${VLLM_LOG_LEVEL:-INFO}"

PROFILER_DIR="${PCIE_PROFILER_DIR:-$RUN_EXP_DIR/profiler_output}"
mkdir -p "$PROFILER_DIR"

CMD_ARGS=(
  --model "$MODEL"
  --host "${VLLM_HOST:-0.0.0.0}"
  --port "$PORT"
  --tensor-parallel-size "$TP"
  --gpu-memory-utilization "$GPU_MEM_UTIL"
  --max-num-seqs "${VLLM_MAX_NUM_SEQS:-128}"
  --block-size "${VLLM_BLOCK_SIZE:-16}"
  --enable-prefix-caching
  --enable-prompt-tokens-details
  --trust-remote-code
  --disable-hybrid-kv-cache-manager
  --max-prefetch-block-ratio "$RATIO"
  --prefetch-ttl-ms "$TTL_MS"
  --profiler-config "{\"profiler\":\"pcie\",\"torch_profiler_dir\":\"$PROFILER_DIR\"}"
)

if [ -n "${KV_OFFLOADING_SIZE:-}" ] && [ "${KV_OFFLOADING_SIZE}" != "0" ]; then
  CMD_ARGS+=( --kv-offloading-size "$KV_OFFLOADING_SIZE"
              --kv-offloading-backend native
              --swap-space "${SWAP_SPACE:-128}" )
fi

VLLM_SERVER_DEV_MODE=1 HF_HUB_OFFLINE=1 \
  vllm serve "${CMD_ARGS[@]}" 2>&1 | tee "$LOG_FILE"
