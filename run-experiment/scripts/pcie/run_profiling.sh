#!/bin/bash
# PCIe Bandwidth Profiling 脚本
#
# 用法:
#   ./run_profiling.sh [dataset] [options]
#
# 示例:
#   ./run_profiling.sh pcie-medium
#   ./run_profiling.sh pcie-heavy --duration 180 --qps 1.2

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
DURATION=120

while [[ $# -gt 0 ]]; do
    case "$1" in
        --duration)
            DURATION="$2"; shift 2 ;;
        --qps)
            QPS="$2"; shift 2 ;;
        --lead-time)
            PREFETCH_LEAD_TIME="$2"; shift 2 ;;
        *)
            echo "❌ Unknown option: $1"
            exit 1 ;;
    esac
done

# 加载数据集配置
load_dataset_config "$DATASET"
NUM_GPU_BLOCKS_OVERRIDE="${DATASET_GPU_BLOCKS:-$NUM_GPU_BLOCKS_OVERRIDE}"

# 检查数据集
generate_dataset_if_needed "$TRACE" "$FULL_TRACE" "$DATASET" || exit 1

# 检查vLLM是否运行（profiling检查已简化）
if ! check_vllm_running; then
    echo "❌ Error: vLLM not running."
    echo "Start with: bash run-experiment/start_vllm_pcie.sh"
    exit 1
fi

# PCIe profiling模式检查（已简化，只要vLLM运行即可）
if ! check_vllm_profiling; then
    echo "⚠️  Warning: Cannot verify PCIe profiling mode."
    echo "If you started vLLM with start_vllm_pcie.sh, you can ignore this."
    # 不再退出，继续执行
fi

# 提取模型名称（支持手动传参或自动解析）
# 方式1: 支持通过环境变量 MODEL_SIZE 手动设置（如 export MODEL_SIZE=32b）
# 方式2: 自动从MODEL_PATH解析
if [[ -n "$MODEL_SIZE" ]]; then
    MODEL_NAME="$MODEL_SIZE"
    echo "Using manually specified model size: $MODEL_NAME"
else
    # 从MODEL_PATH自动提取，支持多种格式：
    # /lpai/models/Qwen__Qwen3-32B/xxx -> 32b
    # /lpai/models/Qwen3-8B -> 8b
    # 提取包含数字+B的部分，转换为小写
    MODEL_NAME=$(echo "$MODEL_PATH" | grep -oP '\d+[Bb]' | head -1 | tr '[:upper:]' '[:lower:]')

    # 如果提取失败，使用默认值
    if [[ -z "$MODEL_NAME" ]]; then
        MODEL_NAME="model"
        echo "⚠️  Warning: Could not extract model size from MODEL_PATH: $MODEL_PATH"
        echo "Using default: $MODEL_NAME"
        echo "Tip: Set MODEL_SIZE environment variable to override (e.g., export MODEL_SIZE=32b)"
    else
        echo "Detected model size from path: $MODEL_NAME"
    fi
fi

# 创建结果目录（包含模型和override信息）
RESULTS_DIR="$RESULTS_ROOT/${MODEL_NAME}_blk${NUM_GPU_BLOCKS_OVERRIDE}_${DATASET}_qps${QPS}_dur${DURATION}"
mkdir -p "$RESULTS_DIR"

print_separator
echo "PCIe Bandwidth Profiling"
print_separator
echo "Dataset: $DATASET"
echo "Duration: ${DURATION}s"
echo "QPS: $QPS"
echo "Prefetch lead time: ${PREFETCH_LEAD_TIME}s"
echo "Results: $RESULTS_DIR"
print_separator

# 清空cache
echo ""
echo "Resetting cache..."
curl -s -X POST "http://localhost:$API_PORT/reset_prefix_cache?reset_external=true" >/dev/null || true
sleep 5

# 启动profiler
echo "Starting profiler..."
curl -s -X POST "http://localhost:$API_PORT/start_profile" >/dev/null || {
    echo "❌ Error: Failed to start profiler"
    exit 1
}
echo "✓ Profiler started"

# 运行负载
print_phase "Running workload for ${DURATION}s..."

python3 prefetch_ab_runner.py \
    --trace-file "$TRACE" \
    --mode prefetch \
    --qps "$QPS" \
    --num-multi-turn "$NUM_CONV" \
    --model "$MODEL_PATH" \
    --api-base "http://localhost:$API_PORT/v1" \
    --output "$RESULTS_DIR/requests.jsonl" \
    --seed "$SEED" \
    --timeout "$DURATION" \
    --prefetch-lead-time "$PREFETCH_LEAD_TIME" \
    --schedule-mode "$SCHEDULE_MODE" \
    &> "$RESULTS_DIR/run.log"

echo "✓ Workload completed"

# 停止profiler
echo ""
echo "Stopping profiler..."
curl -s -X POST "http://localhost:$API_PORT/stop_profile" >/dev/null || true
sleep 3
echo "✓ Profiler stopped"

# 收集PCIe事件文件
echo ""
echo "Collecting PCIe events..."

PCIE_FILES=$(ls "$PCIE_PROFILER_DIR"/pcie_events_*.json 2>/dev/null || true)

if [[ -z "$PCIE_FILES" ]]; then
    echo "⚠️  Warning: No PCIe event files found in $PCIE_PROFILER_DIR"
    echo "Check if VLLM was started with --pcie-profiling"
    exit 1
fi

# 合并多卡事件
PCIE_COUNT=$(echo "$PCIE_FILES" | wc -l)

if [[ $PCIE_COUNT -gt 1 ]]; then
    echo "Merging $PCIE_COUNT PCIe event files..."
    python3 -c "
import json, glob
events = []
for f in sorted(glob.glob('$PCIE_PROFILER_DIR/pcie_events_*.json')):
    with open(f) as fp:
        events.extend(json.load(fp))
with open('$RESULTS_DIR/pcie_events.json', 'w') as fp:
    json.dump(events, fp, indent=2)
print(f'Merged {len(events)} events')
"
    PCIE_EVENTS="$RESULTS_DIR/pcie_events.json"
else
    cp "$PCIE_PROFILER_DIR/pcie_events_0.json" "$RESULTS_DIR/pcie_events.json"
    PCIE_EVENTS="$RESULTS_DIR/pcie_events.json"
fi

echo "✓ PCIe events saved to: $PCIE_EVENTS"

# 生成甘特图
print_phase "Generating Gantt chart..."

if [[ -f "$VLLM_SRC/tools/profiler/visualize_pcie_gantt.py" ]]; then
    python3 "$VLLM_SRC/tools/profiler/visualize_pcie_gantt.py" \
        --input "$PCIE_EVENTS" \
        --output "$RESULTS_DIR/pcie_gantt.html" 2>/dev/null && \
        echo "✓ Gantt chart saved to: $RESULTS_DIR/pcie_gantt.html" || \
        echo "⚠️  Warning: Failed to generate Gantt chart (需要plotly: pip install plotly)"
else
    echo "⚠️  Warning: visualize_pcie_gantt.py not found in $VLLM_SRC/tools/profiler/"
fi

# PCIe事件摘要
if [[ -f "../result-analysis/summarize_pcie_events.py" ]]; then
    echo ""
    echo "PCIe Event Summary:"
    python3 ../result-analysis/summarize_pcie_events.py "$PCIE_EVENTS" 2>/dev/null || true
fi

# 保存vLLM日志
[[ -f "$VLLM_LOG" ]] && cp "$VLLM_LOG" "$RESULTS_DIR/vllm_state.log" 2>/dev/null || true

# 保存配置
bash dump_config.sh > "$RESULTS_DIR/config_snapshot.env" 2>/dev/null || true

# 自动生成分析报告
print_phase "Generating analysis report..."
ANALYSIS_SCRIPT="$SCRIPT_DIR/../utils/analyze_pcie_experiment.py"

if [[ -f "$ANALYSIS_SCRIPT" ]]; then
    python3 "$ANALYSIS_SCRIPT" "$RESULTS_DIR" && \
        echo "✓ Analysis report saved to: $RESULTS_DIR/analysis_report.md" || \
        echo "⚠️  Warning: Failed to generate analysis report"
else
    echo "⚠️  Warning: Analysis script not found: $ANALYSIS_SCRIPT"
fi

echo ""
echo "✅ Profiling complete!"
echo ""
echo "Results:"
echo "  Directory: $RESULTS_DIR"
echo "  PCIe events: $PCIE_EVENTS"
echo "  Gantt chart: $RESULTS_DIR/pcie_gantt.html"
echo "  Analysis report: $RESULTS_DIR/analysis_report.md"
echo "  Run log: $RESULTS_DIR/run.log"
print_separator
