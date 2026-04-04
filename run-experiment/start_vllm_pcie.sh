#!/bin/bash
# vLLM 启动脚本（PCIe Profiling 版）
# 在 start_vllm.sh 基础上增加：禁用 NVLink、轻量 PCIe Profiler（仅 PCIeTracer，无 torch 开销）
# 用法: ./start_vllm_pcie.sh [--pcie-scheduler] [--gpu-blocks N] [--no-pp-phase-aware] [--log-file PATH]
#   --pcie-scheduler:    启用 PCIe 调度算法 (VLLM_PCIE_SCHEDULER=1)，用于 A/B 实验 Phase 1
#   --gpu-blocks N:      覆盖 system.env 中的 NUM_GPU_BLOCKS_OVERRIDE（不同数据集可能需要不同值）
#   --no-pp-phase-aware: 消融实验：禁用 PP Phase 感知
#   --log-file PATH:     日志输出到指定路径
# 配合 run_pcie_scheduling_ab.sh 使用

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 保存调用者通过环境变量传入的 API_PORT（防止被 system.env 覆盖）
_SAVED_API_PORT="${API_PORT:-}"

# 解析可选参数
PCIE_SCHEDULER=0
NO_PP_PHASE_AWARE=0  # 消融实验：禁用 PP Phase 感知，仅验证双队列+Evict-first
LOG_FILE_OVERRIDE=""
GPU_BLOCKS_OVERRIDE=""
PORT_OVERRIDE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --pcie-scheduler)
            PCIE_SCHEDULER=1
            shift ;;
        --no-pp-phase-aware)
            NO_PP_PHASE_AWARE=1
            shift ;;
        --log-file)
            LOG_FILE_OVERRIDE="$2"
            shift 2 ;;
        --gpu-blocks)
            GPU_BLOCKS_OVERRIDE="$2"
            shift 2 ;;
        --port)
            PORT_OVERRIDE="$2"
            shift 2 ;;
        medium)
            shift ;;
        *)
            shift ;;
    esac
done

# 加载新的配置文件
[ -f "$SCRIPT_DIR/config/system.env" ] || { echo "错误: 缺少 config/system.env"; exit 1; }
[ -f "$SCRIPT_DIR/config/experiments.env" ] || { echo "错误: 缺少 config/experiments.env"; exit 1; }

set -a
source "$SCRIPT_DIR/config/system.env"
source "$SCRIPT_DIR/config/experiments.env"
set +a

# 恢复端口覆盖: --port > 环境变量 > system.env
[[ -n "$PORT_OVERRIDE" ]] && API_PORT="$PORT_OVERRIDE"
[[ -z "$PORT_OVERRIDE" && -n "$_SAVED_API_PORT" ]] && API_PORT="$_SAVED_API_PORT"

# --gpu-blocks 覆盖 system.env 中的 NUM_GPU_BLOCKS_OVERRIDE
if [[ -n "$GPU_BLOCKS_OVERRIDE" ]]; then
    NUM_GPU_BLOCKS_OVERRIDE="$GPU_BLOCKS_OVERRIDE"
fi

[[ "$PCIE_SCHEDULER" -eq 1 ]] && export VLLM_PCIE_SCHEDULER=1 && echo "PCIe Scheduler: enabled (VLLM_PCIE_SCHEDULER=1)"

[[ "$VLLM_LOG" != /* ]] && VLLM_LOG="$SCRIPT_DIR/$VLLM_LOG"
[[ "$VLLM_SRC" != /* ]] && VLLM_SRC="$SCRIPT_DIR/$VLLM_SRC"
[[ "$PCIE_PROFILER_DIR" != /* ]] && PCIE_PROFILER_DIR="$SCRIPT_DIR/$PCIE_PROFILER_DIR"

# --log-file 覆盖默认 VLLM_LOG，用于将日志实时写入指定路径（如实验目录）
if [[ -n "$LOG_FILE_OVERRIDE" ]]; then
    [[ "$LOG_FILE_OVERRIDE" != /* ]] && LOG_FILE_OVERRIDE="$(pwd)/$LOG_FILE_OVERRIDE"
    mkdir -p "$(dirname "$LOG_FILE_OVERRIDE")"
    VLLM_LOG="$LOG_FILE_OVERRIDE"
fi

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

# 加载 GPU 锁管理工具，选择空闲且未被其他实验占用的 GPU
source "$SCRIPT_DIR/scripts/utils/gpu_lock.sh"

if ! wait_for_free_gpus "$NUM_GPUS" 600; then
  echo "ERROR: 无法获取 $NUM_GPUS 张空闲 GPU，退出"
  exit 1
fi
FREE_GPUS="$ACQUIRED_GPUS"

echo "============================================"
echo "PCIe Profiling 模式启动 vLLM (PP=$PP_SIZE)"
echo "============================================"
echo "Selected GPU(s): $FREE_GPUS"
echo "Model: $MODEL_PATH (local, HF_HUB_OFFLINE=1)"
echo "KV_OFFLOADING_SIZE=${KV_OFFLOADING_SIZE}, SWAP_SPACE=${SWAP_SPACE}"
echo "Profiler 输出: $PCIE_PROFILER_DIR"
echo "VLLM_PCIE_TRACE=1, NCCL_P2P_DISABLE=1"
[[ "$PCIE_SCHEDULER" -eq 1 ]] && echo "VLLM_PCIE_SCHEDULER=1 (PCIe scheduling enabled)"
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

if [ "$PCIE_SCHEDULER" -eq 1 ]; then
  CMD_ARGS+=(--enable-pcie-scheduling)
  CMD_ARGS+=(--max-concurrent-h2d "${PCIE_MAX_CONCURRENT_H2D:-2}")
  CMD_ARGS+=(--prefetch-block-threshold "${PCIE_PREFETCH_BLOCK_THRESHOLD:-150}")
  CMD_ARGS+=(--max-queue-wait-ms "${PCIE_MAX_QUEUE_WAIT_MS:-30}")
  [[ "$NO_PP_PHASE_AWARE" -eq 1 ]] && CMD_ARGS+=(--no-enable-pp-phase-aware) && echo "Ablation: PP-phase-aware disabled"
  [[ "${PCIE_NO_PRIORITY_QUEUE:-0}" == "1" ]] && CMD_ARGS+=(--no-priority-queue) && echo "Ablation: priority queue disabled (FIFO mode)"
  [[ "${PCIE_NO_EVICT_FIRST:-0}" == "1" ]] && CMD_ARGS+=(--no-evict-first) && echo "Ablation: evict-first disabled (H2D before D2H)"
  echo "PCIe Scheduling parameters added to vLLM args"
fi

export VLLM_LOGGING_LEVEL="${VLLM_LOG_LEVEL:-INFO}"

# Log noise filter: per-request high-frequency logs that dominate file size
# These are filtered from BOTH the log file and terminal output.
# To capture full unfiltered output, set VLLM_FULL_LOG=1 before running.
#   - "offload MISS" / "offload HIT"        (offloading_connector.py, every request)
#   - "scheduling CPU->GPU load"             (offloading_connector.py, every load)
#   - "Prefetch .*: CPU hit/GPU hit/NO HIT"  (scheduler.py, every prefetch check)
#   - "CPU load complete"                    (scheduler.py, every load finish)
#   - "Block allocation failed"              (scheduler.py, when blocks are tight)
#   - "Delaying request"                     (offloading_connector.py, blocks loading)
#   - "offloading .* blocks"                 (offloading_connector.py, every offload)
NOISE_FILTER='offload (MISS|HIT)|scheduling CPU->GPU load|Prefetch .+: (CPU hit|GPU hit|NO HIT|CPU load complete|deferred)|Block allocation failed|Delaying request|offloading [0-9]+ blocks'

if [[ "${VLLM_FULL_LOG:-0}" == "1" ]]; then
  echo "⚠️  VLLM_FULL_LOG=1: writing unfiltered log to $VLLM_LOG (may be very large)"
  VLLM_SERVER_DEV_MODE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$FREE_GPUS \
    vllm serve "${CMD_ARGS[@]}" 2>&1 \
    | tee "$VLLM_LOG" \
    | grep --line-buffered -Ev "$NOISE_FILTER"
else
  VLLM_SERVER_DEV_MODE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$FREE_GPUS \
    vllm serve "${CMD_ARGS[@]}" 2>&1 \
    | grep --line-buffered -Ev "$NOISE_FILTER" \
    | tee "$VLLM_LOG"
fi
