#!/bin/bash
# Start vLLM for the prefetch stress experiments (S1/S2/S3).
#
# Mirrors run-experiment/start_vllm.sh in all engine-side knobs (model, GPU
# memory utilisation, max sequences, block size, num_gpu_blocks override,
# KV offloading size, swap space) so the stress server matches the baseline
# experiments config exactly. The only stress-specific additions are:
#
#   --max-prefetch-block-ratio   (chapter-3 quota knob; default 0.3)
#   --prefetch-ttl-ms            (chapter-3 TTL knob; default 60s)
#   --profiler-config pcie       (so /start_profile + /stop_profile flush
#                                 PCIe events to PROFILER_DIR for the
#                                 stress runner to ingest)
#   VLLM_PCIE_TRACE=1            (turns on PCIeTracer recording)
#
# GPU selection reuses scripts/utils/gpu_lock.sh (same path
# start_vllm_pcie.sh uses), so multiple stress runs on the same node won't
# fight for cards.
#
# Usage:
#   ./start_vllm_stress.sh [--tp N] [--ratio R] [--ttl-ms M] [--port P] [--log PATH]
#
# Common env overrides (otherwise read from config/system.env):
#   STRESS_MODEL                 path to the served model (overrides MODEL_PATH)
#   STRESS_TP                    tensor-parallel size (default 2)
#   STRESS_BLOCK_RATIO           --max-prefetch-block-ratio (default 0.3)
#   STRESS_PREFETCH_TTL_MS       --prefetch-ttl-ms          (default 60000)
#   GPU_WAIT_TIMEOUT             seconds to wait for free GPUs (0 = forever)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_EXP_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

[ -f "$RUN_EXP_DIR/config/system.env" ] || { echo "missing config/system.env"; exit 1; }
[ -f "$RUN_EXP_DIR/config/experiments.env" ] || { echo "missing config/experiments.env"; exit 1; }

set -a
source "$RUN_EXP_DIR/config/system.env"
source "$RUN_EXP_DIR/config/experiments.env"
set +a

# Stress-specific knobs (env-overridable, CLI takes precedence below).
TP="${STRESS_TP:-2}"
RATIO="${STRESS_BLOCK_RATIO:-0.3}"
TTL_MS="${STRESS_PREFETCH_TTL_MS:-60000}"
PORT="${API_PORT:-8000}"
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

PROFILER_DIR="${PCIE_PROFILER_DIR:-$RUN_EXP_DIR/profiler_output}"
mkdir -p "$PROFILER_DIR"

MODEL="${STRESS_MODEL:-$MODEL_PATH}"
[[ -z "$MODEL" ]] && { echo "MODEL_PATH not set; export STRESS_MODEL"; exit 1; }

# Acquire $TP free GPUs via the same lock manager as start_vllm_pcie.sh.
# This sets ACQUIRED_GPUS and exports CUDA_VISIBLE_DEVICES on success.
source "$RUN_EXP_DIR/scripts/utils/gpu_lock.sh"
if ! wait_for_free_gpus "$TP" "${GPU_WAIT_TIMEOUT:-0}"; then
  echo "ERROR: 无法获取 $TP 张空闲 GPU，退出"
  exit 1
fi
FREE_GPUS="$ACQUIRED_GPUS"
trap 'release_gpus' EXIT INT TERM

echo "============================================"
echo "vLLM stress server"
echo "  model     = $MODEL"
echo "  GPUs      = $FREE_GPUS  (tp=$TP)"
echo "  port      = $PORT"
echo "  ratio     = $RATIO"
echo "  ttl_ms    = $TTL_MS"
echo "  GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION"
echo "  NUM_GPU_BLOCKS_OVERRIDE=$NUM_GPU_BLOCKS_OVERRIDE"
echo "  KV_OFFLOADING_SIZE=${KV_OFFLOADING_SIZE}, SWAP_SPACE=${SWAP_SPACE}"
echo "============================================"

export VLLM_PCIE_TRACE=1
export VLLM_LOGGING_LEVEL="${VLLM_LOG_LEVEL:-INFO}"

CMD_ARGS=(
  --model "$MODEL"
  --host "${VLLM_HOST:-0.0.0.0}"
  --port "$PORT"
  --tensor-parallel-size "$TP"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --max-num-seqs "$VLLM_MAX_NUM_SEQS"
  --block-size "$VLLM_BLOCK_SIZE"
  --enable-prefix-caching
  --enable-prompt-tokens-details
  --trust-remote-code
  --disable-hybrid-kv-cache-manager
  --max-prefetch-block-ratio "$RATIO"
  --prefetch-ttl-ms "$TTL_MS"
  --profiler-config "{\"profiler\":\"pcie\",\"torch_profiler_dir\":\"$PROFILER_DIR\"}"
)

# GPU blocks override (matches start_vllm.sh).
if [ -n "$NUM_GPU_BLOCKS_OVERRIDE" ] && [ "$NUM_GPU_BLOCKS_OVERRIDE" != "auto" ]; then
  CMD_ARGS+=( --num-gpu-blocks-override "$NUM_GPU_BLOCKS_OVERRIDE" )
fi

# KV offloading (matches start_vllm.sh). With offload enabled, S1/S2/S3
# attacks can additionally produce CPU_HIT prefetches that go through the
# H2D path, which is exactly the traffic the reviewer is concerned about.
if [ -n "$KV_OFFLOADING_SIZE" ] && [ "$KV_OFFLOADING_SIZE" != "0" ]; then
  CMD_ARGS+=( --kv-offloading-size "$KV_OFFLOADING_SIZE"
              --kv-offloading-backend native
              --swap-space "$SWAP_SPACE" )
  echo "KV Offloading enabled: ${KV_OFFLOADING_SIZE} GiB, SWAP_SPACE=${SWAP_SPACE} GiB"
fi

VLLM_SERVER_DEV_MODE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES="$FREE_GPUS" \
  vllm serve "${CMD_ARGS[@]}" 2>&1 | tee "$LOG_FILE"
