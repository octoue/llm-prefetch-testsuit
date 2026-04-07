#!/bin/bash
# 启动 vLLM 用于 Prefetch 消融实验（不依赖 PCIe 调度器 / PP）
#
# 用法: ./start_vllm_prefetch.sh [options]
#   --gpu-blocks N              覆盖 GPU block 数量
#   --prefetch-block-threshold N  准入控制阈值 (default: 150)
#   --max-prefetch-block-ratio F  配额比例 (default: 0.3)
#   --log-file PATH             日志输出路径
#
# 环境变量: 从 ../../run-experiment/config/ 加载

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFETCH_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_EXP_DIR="$(cd "$PREFETCH_ROOT/../run-experiment" && pwd)"

# 保存调用者通过环境变量传入的 API_PORT（防止被 system.env 覆盖）
_SAVED_API_PORT="${API_PORT:-}"

# 加载配置 - 优先使用 prefetch 专用配置
source "$RUN_EXP_DIR/config/system.env"

# 恢复环境变量中的端口覆盖
[[ -n "$_SAVED_API_PORT" ]] && API_PORT="$_SAVED_API_PORT"

PREFETCH_CONFIG="$PREFETCH_ROOT/config/prefetch_experiments.env"
if [[ -f "$PREFETCH_CONFIG" ]]; then
    source "$PREFETCH_CONFIG"
else
    source "$RUN_EXP_DIR/config/experiments.env"
fi

# 默认值（可被命令行参数覆盖）
PREFETCH_BLOCK_THRESHOLD="${PREFETCH_BLOCK_THRESHOLD:-150}"
MAX_PREFETCH_BLOCK_RATIO="${MAX_PREFETCH_BLOCK_RATIO:-0.3}"
LOG_FILE="${VLLM_LOG:-vllm_prefetch.log}"

# 解析参数
PORT_OVERRIDE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu-blocks)                 NUM_GPU_BLOCKS_OVERRIDE="$2"; shift 2 ;;
        --prefetch-block-threshold)   PREFETCH_BLOCK_THRESHOLD="$2"; shift 2 ;;
        --max-prefetch-block-ratio)   MAX_PREFETCH_BLOCK_RATIO="$2"; shift 2 ;;
        --log-file)                   LOG_FILE="$2"; shift 2 ;;
        --port)                       PORT_OVERRIDE="$2"; shift 2 ;;
        *)
            echo "Unknown option: $1"
            exit 1 ;;
    esac
done

# --port 覆盖优先级最高
[[ -n "$PORT_OVERRIDE" ]] && API_PORT="$PORT_OVERRIDE"

# 加载 GPU 锁管理工具，选择空闲且未被其他实验占用的 GPU
source "$RUN_EXP_DIR/scripts/utils/gpu_lock.sh"

NUM_GPUS="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
if ! wait_for_free_gpus "$NUM_GPUS" "${GPU_WAIT_TIMEOUT:-0}"; then
  echo "ERROR: 无法获取 $NUM_GPUS 张空闲 GPU，退出"
  exit 1
fi
FREE_GPUS="$ACQUIRED_GPUS"

echo "============================================"
echo "Prefetch 消融实验: 启动 vLLM"
echo "============================================"
echo "GPU(s): $FREE_GPUS"
echo "Model: $MODEL_PATH"
echo "Tensor Parallel: $VLLM_TENSOR_PARALLEL_SIZE"
echo "GPU blocks: $NUM_GPU_BLOCKS_OVERRIDE"
[[ -n "$MAX_MODEL_LEN" ]] && echo "Max seq len: $MAX_MODEL_LEN (limited)" || echo "Max seq len: default (40960)"
echo "Prefetch threshold: $PREFETCH_BLOCK_THRESHOLD"
echo "Prefetch quota ratio: $MAX_PREFETCH_BLOCK_RATIO"
echo "KV offloading: ${KV_OFFLOADING_SIZE} GiB"
echo "Port: ${API_PORT:-8000}"
echo "============================================"

CMD_ARGS=(
  --model "$MODEL_PATH"
  --host "${VLLM_HOST:-0.0.0.0}"
  --port "${API_PORT:-8000}"
  --max-num-seqs "${VLLM_MAX_NUM_SEQS:-96}"
  --block-size "${VLLM_BLOCK_SIZE:-16}"
  --tensor-parallel-size "${VLLM_TENSOR_PARALLEL_SIZE:-1}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.7}"
  --enable-prefix-caching
  --enable-prompt-tokens-details
  --trust-remote-code
  --disable-hybrid-kv-cache-manager
  --prefetch-block-threshold "$PREFETCH_BLOCK_THRESHOLD"
  --max-prefetch-block-ratio "$MAX_PREFETCH_BLOCK_RATIO"
)

if [ -n "$NUM_GPU_BLOCKS_OVERRIDE" ] && [ "$NUM_GPU_BLOCKS_OVERRIDE" != "auto" ]; then
  CMD_ARGS+=(--num-gpu-blocks-override "$NUM_GPU_BLOCKS_OVERRIDE")
fi

if [ -n "$MAX_MODEL_LEN" ] && [ "$MAX_MODEL_LEN" != "0" ]; then
  CMD_ARGS+=(--max-model-len "$MAX_MODEL_LEN")
fi

if [ -n "$KV_OFFLOADING_SIZE" ] && [ "$KV_OFFLOADING_SIZE" != "0" ]; then
  CMD_ARGS+=(--kv-offloading-size "$KV_OFFLOADING_SIZE")
  CMD_ARGS+=(--kv-offloading-backend "native")
  CMD_ARGS+=(--swap-space "${SWAP_SPACE:-256}")
fi

export VLLM_LOGGING_LEVEL="${VLLM_LOG_LEVEL:-INFO}"

# 日志过滤 (与 pcie 版本一致)
NOISE_FILTER='offload (MISS|HIT)|scheduling CPU->GPU load|Prefetch .+: (CPU hit|GPU hit|NO HIT|CPU load complete|deferred)|Block allocation failed|Delaying request|offloading [0-9]+ blocks'

VLLM_SERVER_DEV_MODE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$FREE_GPUS \
  vllm serve "${CMD_ARGS[@]}" 2>&1 \
  | grep --line-buffered -Ev "$NOISE_FILTER" \
  | tee "$LOG_FILE"
