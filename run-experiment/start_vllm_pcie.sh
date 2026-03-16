#!/bin/bash
# vLLM 启动脚本（PCIe Profiling 版）
# 在 start_vllm.sh 基础上增加：禁用 NVLink、轻量 PCIe Profiler（仅 PCIeTracer，无 torch 开销）
# 用法: ./start_vllm_pcie.sh [medium]
#   medium: 仅提示，与默认共用 NUM_GPU_BLOCKS_OVERRIDE（config.env 中统一配置）
# 配合 run_lite_test_pcie.sh 使用，采集 KV Offload / Prefetch 的 PCIe 带宽数据

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

[ -f "$SCRIPT_DIR/config.env" ] || { echo "错误: 缺少 config.env"; exit 1; }
set -a && source "$SCRIPT_DIR/config.env" && set +a

[ "${1:-}" = "medium" ] && echo "中等重度模式 (max_input=2500): NUM_GPU_BLOCKS_OVERRIDE=$NUM_GPU_BLOCKS_OVERRIDE"

[[ "$VLLM_LOG" != /* ]] && VLLM_LOG="$SCRIPT_DIR/$VLLM_LOG"
[[ "$VLLM_SRC" != /* ]] && VLLM_SRC="$SCRIPT_DIR/$VLLM_SRC"
[[ "$PCIE_PROFILER_DIR" != /* ]] && PCIE_PROFILER_DIR="$SCRIPT_DIR/$PCIE_PROFILER_DIR"

mkdir -p "$PCIE_PROFILER_DIR"

# 禁用 NVLink，强制 PCIe 路径（单卡时主要影响 KV offload 的 D2H/H2D）
if [ -f "$VLLM_SRC/tools/profiler/setup_pcie_env.sh" ]; then
  source "$VLLM_SRC/tools/profiler/setup_pcie_env.sh"
else
  echo "Warning: setup_pcie_env.sh 未找到，使用默认 NCCL 设置"
  export NCCL_P2P_DISABLE=1
  export NCCL_NVLS_ENABLE=0
fi

# 启用 PCIeTracer 事件采集
export VLLM_PCIE_TRACE=1

# PP 卡数（默认 2）
PP_SIZE="${VLLM_PIPELINE_PARALLEL_SIZE:-2}"
NUM_GPUS=$PP_SIZE

# 自动选择显存最空闲的 N 张 GPU（N = PP_SIZE）
FREE_GPUS=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null | \
  sort -t',' -k2 -rn | head -n "$NUM_GPUS" | cut -d',' -f1 | tr -d ' ' | paste -sd ',' -)
if [ -z "$FREE_GPUS" ]; then
  echo "Warning: nvidia-smi failed, using CUDA_VISIBLE_DEVICES=0,1"
  FREE_GPUS="0,1"
fi

echo "============================================"
echo "PCIe Profiling 模式启动 vLLM (PP=$PP_SIZE)"
echo "============================================"
echo "Selected GPU(s): $FREE_GPUS"
echo "Model: $MODEL_PATH (local, HF_HUB_OFFLINE=1)"
echo "KV_OFFLOADING_SIZE=${KV_OFFLOADING_SIZE}, SWAP_SPACE=${SWAP_SPACE}"
echo "Profiler 输出: $PCIE_PROFILER_DIR"
echo "VLLM_PCIE_TRACE=1, NCCL_P2P_DISABLE=1"
echo "============================================"

# 构建启动参数（与 start_vllm.sh 一致，额外增加 profiler 和 PP）
CMD_ARGS=(
  --model "$MODEL_PATH"
  --host "$VLLM_HOST"
  --port "$API_PORT"
  --max-num-seqs "$VLLM_MAX_NUM_SEQS"
  --block-size "$VLLM_BLOCK_SIZE"
  --tensor-parallel-size "$VLLM_TENSOR_PARALLEL_SIZE"
  --pipeline-parallel-size "$PP_SIZE"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --enable-prefix-caching
  --enable-prompt-tokens-details
  --trust-remote-code
  --disable-hybrid-kv-cache-manager
  --profiler-config "{\"profiler\": \"pcie\", \"torch_profiler_dir\": \"$PCIE_PROFILER_DIR\"}"
)

if [ -n "$NUM_GPU_BLOCKS_OVERRIDE" ] && [ "$NUM_GPU_BLOCKS_OVERRIDE" != "auto" ]; then
  CMD_ARGS+=(--num-gpu-blocks-override "$NUM_GPU_BLOCKS_OVERRIDE")
fi

if [ -n "$KV_OFFLOADING_SIZE" ] && [ "$KV_OFFLOADING_SIZE" != "0" ]; then
  CMD_ARGS+=(--kv-offloading-size "$KV_OFFLOADING_SIZE")
  CMD_ARGS+=(--kv-offloading-backend "native")
  CMD_ARGS+=(--swap-space "$SWAP_SPACE")
  echo "KV Offloading enabled: ${KV_OFFLOADING_SIZE} GiB"
fi

export VLLM_LOGGING_LEVEL="${VLLM_LOG_LEVEL:-INFO}"
VLLM_SERVER_DEV_MODE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$FREE_GPUS vllm serve "${CMD_ARGS[@]}" | tee "$VLLM_LOG"
