#!/bin/bash
# vLLM 启动脚本 - 使用本地模型、离线模式、不连接 HuggingFace
# 参考: vllm/tests/start_vllm.sh, vllm-launch-script/config.env
# 模型: Qwen3-8B (单卡), 目标硬件: A100-SXM4-80GB
# 用法: ./start_vllm.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

[ -f "$SCRIPT_DIR/config.env" ] && set -a && source "$SCRIPT_DIR/config.env" && set +a

# 默认值（当 config.env 未定义时）
MODEL_PATH="${MODEL_PATH:-/lpai/models/Qwen__Qwen3-8B/25-07-26-0349}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.5}"
NUM_GPU_BLOCKS_OVERRIDE="${NUM_GPU_BLOCKS_OVERRIDE:-115}"
KV_OFFLOADING_SIZE="${KV_OFFLOADING_SIZE:-5}"
SWAP_SPACE="${SWAP_SPACE:-256}"
API_PORT="${API_PORT:-8000}"
VLLM_LOG="${VLLM_LOG:-vllm_state.log}"
[[ "$VLLM_LOG" != /* ]] && VLLM_LOG="$SCRIPT_DIR/$VLLM_LOG"

# 自动选择显存最空闲的 1 张 GPU（8B 模型单卡即可）
FREE_GPUS=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null | \
  sort -t',' -k2 -rn | head -n 1 | cut -d',' -f1 | tr -d ' ')
if [ -z "$FREE_GPUS" ]; then
  echo "Warning: nvidia-smi failed, using CUDA_VISIBLE_DEVICES=0"
  FREE_GPUS=0
fi
echo "Selected GPU(s): $FREE_GPUS"
echo "Model: $MODEL_PATH (local, HF_HUB_OFFLINE=1)"
echo "GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION, NUM_GPU_BLOCKS_OVERRIDE=$NUM_GPU_BLOCKS_OVERRIDE"
echo "KV_OFFLOADING_SIZE=${KV_OFFLOADING_SIZE:-0}, SWAP_SPACE=${SWAP_SPACE:-256}"
echo "VLLM_SERVER_DEV_MODE=1 (enabled for /reset_prefix_cache)"

# 构建启动参数
CMD_ARGS=(
  --model "$MODEL_PATH"
  --host 0.0.0.0
  --port "$API_PORT"
  --max-num-seqs 256
  --block-size 16
  --tensor-parallel-size 1
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --enable-prefix-caching
  --enable-prompt-tokens-details
  --trust-remote-code
  --disable-hybrid-kv-cache-manager
)

# GPU blocks 覆盖（用于触发 offloading）
if [ -n "$NUM_GPU_BLOCKS_OVERRIDE" ] && [ "$NUM_GPU_BLOCKS_OVERRIDE" != "auto" ]; then
  CMD_ARGS+=(--num-gpu-blocks-override "$NUM_GPU_BLOCKS_OVERRIDE")
fi

# KV offloading（prefetch 可从 CPU 加载 KV 到 GPU）
if [ -n "$KV_OFFLOADING_SIZE" ] && [ "$KV_OFFLOADING_SIZE" != "0" ]; then
  CMD_ARGS+=(--kv-offloading-size "$KV_OFFLOADING_SIZE")
  CMD_ARGS+=(--kv-offloading-backend "native")
  CMD_ARGS+=(--swap-space "$SWAP_SPACE")
  echo "KV Offloading enabled: ${KV_OFFLOADING_SIZE} GiB, SWAP_SPACE=${SWAP_SPACE} GiB"
fi

# VLLM_SERVER_DEV_MODE=1 启用 /reset_prefix_cache 等开发端点（用于 A/B 实验间清空 cache）
# HF_HUB_OFFLINE=1 使用本地模型，不联网
VLLM_SERVER_DEV_MODE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$FREE_GPUS vllm serve "${CMD_ARGS[@]}" | tee "$VLLM_LOG"
