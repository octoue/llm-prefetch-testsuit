#!/bin/bash
# 输出完整实验配置（需在 source config 之后调用）
# 用于 config_snapshot.env 和报告生成
#
# 用法: 在 run-experiment 目录下执行，或传入 RUN_EXPERIMENT_DIR
#   bash dump_config.sh
#   DATASET=pcie-medium bash dump_config.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="${RUN_EXPERIMENT_DIR:-$SCRIPT_DIR}"
cd "$RUN_DIR"

# 若尚未加载，则加载配置
if [[ -z "${MODEL_PATH:-}" ]] && [[ -f "config/system.env" ]]; then
    source config/system.env 2>/dev/null || true
fi
if [[ -z "${QPS:-}" ]] && [[ -f "config/experiments.env" ]]; then
    source config/experiments.env 2>/dev/null || true
fi
if [[ -z "${DATA_ROOT:-}" ]] && [[ -f "config/datasets.env" ]]; then
    source config/datasets.env 2>/dev/null || true
fi

# 输出关键配置变量（按类别）
echo ""
echo "# ---------- system.env 变量 ----------"
for v in MODEL_PATH GPU_MEMORY_UTILIZATION NUM_GPU_BLOCKS_OVERRIDE SWAP_SPACE \
         KV_OFFLOADING_SIZE VLLM_BLOCK_SIZE VLLM_TENSOR_PARALLEL_SIZE \
         VLLM_PIPELINE_PARALLEL_SIZE VLLM_MAX_NUM_SEQS API_PORT VLLM_HOST \
         VLLM_LOG VLLM_SRC PCIE_PROFILER_DIR; do
    [[ -n "${!v:-}" ]] && echo "${v}=${!v}"
done

echo ""
echo "# ---------- experiments.env 变量 ----------"
for v in QPS SEED PREFETCH_LEAD_TIME SCHEDULE_MODE TIMEOUT REQUEST_TIMEOUT \
         TB_PORT ENABLE_TENSORBOARD RESULTS_ROOT; do
    [[ -n "${!v:-}" ]] && echo "${v}=${!v}"
done

echo ""
echo "# ---------- datasets.env 变量 (DATA_ROOT + 当前数据集) ----------"
[[ -n "${DATA_ROOT:-}" ]] && echo "DATA_ROOT=$DATA_ROOT"
if [[ -n "${DATASET:-}" ]]; then
    echo "DATASET=$DATASET"
    prefix="DATASET_${DATASET//-/_}"
    prefix="${prefix^^}"
    for suffix in TRACE FULL_TRACE MAX_INPUT NUM_CONV GPU_BLOCKS; do
        var="${prefix}_${suffix}"
        [[ -n "${!var:-}" ]] && echo "${var}=${!var}"
    done
fi

echo ""
echo "# ---------- 运行时解析后的值 ----------"
[[ -n "${TRACE:-}" ]] && echo "TRACE=$TRACE"
[[ -n "${FULL_TRACE:-}" ]] && echo "FULL_TRACE=$FULL_TRACE"
[[ -n "${NUM_CONV:-}" ]] && echo "NUM_CONV=$NUM_CONV"
[[ -n "${MAX_INPUT:-}" ]] && echo "MAX_INPUT=$MAX_INPUT"
