#!/bin/bash
# Prefetch A/B 实验主脚本
#
# 用法:
#   ./run_experiment.sh [dataset] [options]
#
# 示例:
#   ./run_experiment.sh pcie-medium
#   ./run_experiment.sh pcie-heavy --qps 1.5
#   ./run_experiment.sh pcie-lite --prefetch-only

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

# 加载配置
source config/system.env
source config/datasets.env
source config/experiments.env
source scripts/utils/common.sh

# 默认数据集
DATASET="${1:-pcie-medium}"
shift 2>/dev/null || true

# 解析选项
NO_TENSORBOARD=0
PREFETCH_ONLY=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --qps)
            QPS="$2"; shift 2 ;;
        --lead-time)
            PREFETCH_LEAD_TIME="$2"; shift 2 ;;
        --gpu-blocks)
            NUM_GPU_BLOCKS_OVERRIDE="$2"; shift 2 ;;
        --no-tensorboard)
            NO_TENSORBOARD=1; shift ;;
        --prefetch-only)
            PREFETCH_ONLY=1; shift ;;
        *)
            echo "❌ Unknown option: $1"
            echo "Usage: $0 [dataset] [--qps N] [--lead-time N] [--gpu-blocks N] [--no-tensorboard] [--prefetch-only]"
            exit 1 ;;
    esac
done

# 加载数据集配置
load_dataset_config "$DATASET"

# 使用数据集推荐的GPU blocks(如果未在命令行指定)
if [[ -z "${NUM_GPU_BLOCKS_OVERRIDE_SET:-}" ]]; then
    NUM_GPU_BLOCKS_OVERRIDE="${DATASET_GPU_BLOCKS:-$NUM_GPU_BLOCKS_OVERRIDE}"
fi

# 检查数据集,如果不存在则生成
generate_dataset_if_needed "$TRACE" "$FULL_TRACE" "$DATASET" || exit 1

# 检查vLLM是否运行
if ! check_vllm_running; then
    echo "❌ Error: vLLM not running."
    echo "Start with: bash scripts/utils/start_vllm.sh"
    exit 1
fi

# 创建结果目录
RESULTS_DIR="$RESULTS_ROOT/${DATASET}_qps${QPS}_lead${PREFETCH_LEAD_TIME}"
mkdir -p "$RESULTS_DIR"

print_separator
echo "Prefetch A/B Experiment"
print_separator
echo "Dataset: $DATASET"
echo "Trace: $TRACE"
echo "Num conversations: $NUM_CONV"
echo "QPS: $QPS"
echo "Prefetch lead time: ${PREFETCH_LEAD_TIME}s"
echo "GPU blocks: $NUM_GPU_BLOCKS_OVERRIDE"
echo "Results: $RESULTS_DIR"
print_separator

# Phase 1: Prefetch
print_phase "[Phase 1/2] Running Prefetch mode..."
echo "Resetting prefix cache..."
curl -s -X POST "http://localhost:$API_PORT/reset_prefix_cache?reset_external=true" >/dev/null || true
sleep 5

TB_ARGS=()
if [[ $ENABLE_TENSORBOARD -eq 1 && $NO_TENSORBOARD -eq 0 ]]; then
    TB_ARGS=(--tensorboard-dir "$RESULTS_DIR/tensorboard/prefetch")
fi

python3 prefetch_ab_runner.py \
    --trace-file "$TRACE" \
    --mode prefetch \
    --qps "$QPS" \
    --num-multi-turn "$NUM_CONV" \
    --model "$MODEL_PATH" \
    --api-base "http://localhost:$API_PORT/v1" \
    --output "$RESULTS_DIR/prefetch.jsonl" \
    --seed "$SEED" \
    --timeout "$TIMEOUT" \
    --request-timeout "$REQUEST_TIMEOUT" \
    --prefetch-lead-time "$PREFETCH_LEAD_TIME" \
    --schedule-mode "$SCHEDULE_MODE" \
    "${TB_ARGS[@]}" \
    &> "$RESULTS_DIR/prefetch.log"

echo "✓ Prefetch phase completed"

if [[ $PREFETCH_ONLY -eq 1 ]]; then
    echo ""
    echo "Prefetch-only mode, skipping baseline."
    echo "Results saved to: $RESULTS_DIR"
    exit 0
fi

# Cooling period
echo ""
echo "Waiting 60s for server to cool down..."
sleep 60

# Phase 2: Baseline
print_phase "[Phase 2/2] Running Baseline mode..."
echo "Resetting prefix cache..."
curl -s -X POST "http://localhost:$API_PORT/reset_prefix_cache?reset_external=true" >/dev/null || true
sleep 5

TB_ARGS=()
if [[ $ENABLE_TENSORBOARD -eq 1 && $NO_TENSORBOARD -eq 0 ]]; then
    TB_ARGS=(--tensorboard-dir "$RESULTS_DIR/tensorboard/baseline")
fi

python3 prefetch_ab_runner.py \
    --trace-file "$TRACE" \
    --mode baseline \
    --qps "$QPS" \
    --num-multi-turn "$NUM_CONV" \
    --model "$MODEL_PATH" \
    --api-base "http://localhost:$API_PORT/v1" \
    --output "$RESULTS_DIR/baseline.jsonl" \
    --seed "$SEED" \
    --timeout "$TIMEOUT" \
    --request-timeout "$REQUEST_TIMEOUT" \
    --schedule-mode "$SCHEDULE_MODE" \
    "${TB_ARGS[@]}" \
    &> "$RESULTS_DIR/baseline.log"

echo "✓ Baseline phase completed"

# Generate report
print_phase "[Phase 3/3] Generating report..."

python3 ../result-analysis/generate_report.py \
    --baseline "$RESULTS_DIR/baseline.jsonl" \
    --prefetch "$RESULTS_DIR/prefetch.jsonl" \
    --output "$RESULTS_DIR/report.md" \
    2>/dev/null || echo "Warning: Failed to generate report"

# Save config snapshot
bash dump_config.sh > "$RESULTS_DIR/config_snapshot.env" 2>/dev/null || true

echo ""
echo "✅ Experiment complete!"
echo ""
echo "Results:"
echo "  Directory: $RESULTS_DIR"
echo "  Report: $RESULTS_DIR/report.md"
echo "  Logs: $RESULTS_DIR/{prefetch,baseline}.log"
if [[ $ENABLE_TENSORBOARD -eq 1 && $NO_TENSORBOARD -eq 0 ]]; then
    echo "  TensorBoard: $RESULTS_DIR/tensorboard/"
fi
print_separator
