#!/bin/bash

# ============================================
# vLLM CPU Offloading 启动脚本
# ============================================

set -e

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/config.env"

# 加载配置文件
if [ -f "$CONFIG_FILE" ]; then
    echo "Loading configuration from $CONFIG_FILE"
    source "$CONFIG_FILE"
else
    echo "Error: Configuration file not found at $CONFIG_FILE"
    exit 1
fi

# 设置环境变量
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
export HF_ENDPOINT="${HF_ENDPOINT}"

if [ -n "${HF_TOKEN}" ]; then
    export HF_TOKEN="${HF_TOKEN}"
fi

# 启用详细的 offloading 日志
if [ "${ENABLE_OFFLOAD_LOGGING}" = "true" ]; then
    export VLLM_LOGGING_LEVEL="${LOG_LEVEL}"
    # 启用 vLLM 内部调试日志
    export VLLM_TRACE_FUNCTION=1
fi

# 构建启动命令
echo "============================================"
echo "Starting vLLM API Server with KV Offloading"
echo "============================================"
echo "Model: ${MODEL_NAME}"
echo "GPU Device: ${CUDA_VISIBLE_DEVICES}"
echo "GPU Memory Utilization: ${GPU_MEMORY_UTILIZATION}"
echo "GPU Blocks Override: ${NUM_GPU_BLOCKS_OVERRIDE}"
echo "KV Offloading Size: ${KV_OFFLOADING_SIZE:-0} GiB"
echo "KV Offloading Backend: ${KV_OFFLOADING_BACKEND:-disabled}"
echo "Swap Space: ${SWAP_SPACE:-4} GiB"
echo "API Port: ${API_PORT}"
echo "Prefix Caching: ${ENABLE_PREFIX_CACHING}"
echo "============================================"

# 构建命令参数
CMD_ARGS=(
    --model "${MODEL_NAME}"
    --host "${API_HOST}"
    --port "${API_PORT}"
    --max-num-seqs "${MAX_NUM_SEQS}"
    --block-size "${BLOCK_SIZE}"
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
    --pipeline-parallel-size "${PIPELINE_PARALLEL_SIZE}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
)

# 添加 GPU blocks 覆盖配置
if [ "${NUM_GPU_BLOCKS_OVERRIDE}" != "auto" ]; then
    CMD_ARGS+=(--num-gpu-blocks-override "${NUM_GPU_BLOCKS_OVERRIDE}")
fi

# 添加 KV Cache Offloading 配置 (新版vLLM使用 kv-offloading-size 和 kv-offloading-backend)
if [ -n "${KV_OFFLOADING_SIZE}" ] && [ "${KV_OFFLOADING_SIZE}" != "0" ]; then
    CMD_ARGS+=(--kv-offloading-size "${KV_OFFLOADING_SIZE}")
    CMD_ARGS+=(--kv-offloading-backend "${KV_OFFLOADING_BACKEND:-native}")
    
    # # 添加 CPU blocks 配置 (通过 kv-connector-extra-config)
    # if [ -n "${NUM_CPU_BLOCKS}" ]; then
    #     CMD_ARGS+=(--kv-connector-extra-config "{\"num_cpu_blocks\":${NUM_CPU_BLOCKS}}")
    # fi
    
    echo "KV Offloading ENABLED with ${KV_OFFLOADING_SIZE} GiB (backend: ${KV_OFFLOADING_BACKEND:-native}, cpu_blocks: ${NUM_CPU_BLOCKS:-auto})"
else
    echo "KV Offloading DISABLED"
fi

# 添加 swap space 配置
if [ -n "${SWAP_SPACE}" ]; then
    CMD_ARGS+=(--swap-space "${SWAP_SPACE}")
fi

# 添加 prefix caching 配置
if [ "${ENABLE_PREFIX_CACHING}" = "true" ]; then
    CMD_ARGS+=(--enable-prefix-caching)
fi

# 添加 trust remote code
if [ "${TRUST_REMOTE_CODE}" = "true" ]; then
    CMD_ARGS+=(--trust-remote-code)
fi

CMD_ARGS+=(--disable-hybrid-kv-cache-manager)

# 启动服务
echo ""
echo "Starting vLLM server..."
echo "Command: vllm serve ${CMD_ARGS[@]}"
echo ""

# 创建日志目录
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/vllm_offload_$(date +%Y%m%d_%H%M%S).log"

echo "Logs will be saved to: ${LOG_FILE}"
echo ""

# 启动服务并保存日志
# 注意: grep中的方括号需要转义，否则会被解释为字符集
vllm serve "${CMD_ARGS[@]}"

# 2>&1 | tee "${LOG_FILE}" | grep -E "(offload|swap|Offload|Swap|CPU|evict|Evict|\[LHT-DEBUG\])" --line-buffered --color=auto || true

# 如果 grep 过滤导致没有输出，则显示所有日志
# if [ $? -ne 0 ]; then
#     echo "No offloading events detected in filtered output. Showing full logs..."
#     vllm serve "${CMD_ARGS[@]}" 2>&1 | tee "${LOG_FILE}"
# fi
