#!/bin/bash
# PCIe 调度 A/B 实验脚本
#
# 对比 VLLM_PCIE_SCHEDULER=1（启用）与 未启用 两种配置下的 Prefetch 表现。
# 实验流程：Phase 1 启用调度 → Phase 2 未启用 → Phase 3 生成对比报告
#
# 用法:
#   ./run_pcie_scheduling_ab.sh [dataset] [options]
#
# 示例:
#   ./run_pcie_scheduling_ab.sh pcie-medium
#   ./run_pcie_scheduling_ab.sh pcie-heavy --qps 3.0
#
# 注意: 需在 Phase 1 前用 start_vllm_pcie.sh --pcie-scheduler 启动 vLLM；
#       Phase 1 结束后需重启 vLLM（不用 --pcie-scheduler）再继续 Phase 2。

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
NUM_GPU_BLOCKS_OVERRIDE_SET=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --qps)
            QPS="$2"; shift 2 ;;
        --lead-time)
            PREFETCH_LEAD_TIME="$2"; shift 2 ;;
        --gpu-blocks)
            NUM_GPU_BLOCKS_OVERRIDE="$2"; NUM_GPU_BLOCKS_OVERRIDE_SET=1; shift 2 ;;
        --no-tensorboard)
            NO_TENSORBOARD=1; shift ;;
        *)
            echo "❌ Unknown option: $1"
            echo "Usage: $0 [dataset] [--qps N] [--lead-time N] [--gpu-blocks N] [--no-tensorboard]"
            exit 1 ;;
    esac
done

# 加载数据集配置
load_dataset_config "$DATASET"
[[ $NUM_GPU_BLOCKS_OVERRIDE_SET -eq 0 ]] && NUM_GPU_BLOCKS_OVERRIDE="${DATASET_GPU_BLOCKS:-$NUM_GPU_BLOCKS_OVERRIDE}"

# 检查数据集
generate_dataset_if_needed "$TRACE" "$FULL_TRACE" "$DATASET" || exit 1

# 结果目录
RESULTS_DIR="$RESULTS_ROOT/pcie_sched_${DATASET}_qps${QPS}_lead${PREFETCH_LEAD_TIME}"
mkdir -p "$RESULTS_DIR"

print_separator
echo "PCIe Scheduling A/B Experiment"
print_separator
echo "Dataset: $DATASET"
echo "Trace: $TRACE"
echo "Num conversations: $NUM_CONV"
echo "QPS: $QPS"
echo "Prefetch lead time: ${PREFETCH_LEAD_TIME}s"
echo "GPU blocks: $NUM_GPU_BLOCKS_OVERRIDE"
echo "Results: $RESULTS_DIR"
print_separator

# ------------------------------------------------------------------
# Phase 1: Prefetch + PCIe Scheduling (VLLM_PCIE_SCHEDULER=1)
# ------------------------------------------------------------------
print_phase "[Phase 1/3] Prefetch with PCIe Scheduling (VLLM_PCIE_SCHEDULER=1)"

if ! check_vllm_running; then
    echo "❌ Error: vLLM not running."
    echo "Start with: ./start_vllm_pcie.sh --pcie-scheduler"
    exit 1
fi

echo "Resetting prefix cache..."
curl -s -X POST "http://localhost:$API_PORT/reset_prefix_cache?reset_external=true" >/dev/null || true
sleep 5

TB_ARGS=()
[[ $ENABLE_TENSORBOARD -eq 1 && $NO_TENSORBOARD -eq 0 ]] && \
    TB_ARGS=(--tensorboard-dir "$RESULTS_DIR/tensorboard/pcie_sched")

python3 prefetch_ab_runner.py \
    --trace-file "$TRACE" \
    --mode prefetch \
    --qps "$QPS" \
    --num-multi-turn "$NUM_CONV" \
    --model "$MODEL_PATH" \
    --api-base "http://localhost:$API_PORT/v1" \
    --output "$RESULTS_DIR/prefetch_pcie_sched.jsonl" \
    --seed "$SEED" \
    --timeout "$TIMEOUT" \
    --request-timeout "$REQUEST_TIMEOUT" \
    --prefetch-lead-time "$PREFETCH_LEAD_TIME" \
    --schedule-mode "$SCHEDULE_MODE" \
    "${TB_ARGS[@]}" \
    &> "$RESULTS_DIR/prefetch_pcie_sched.log"

echo "✓ Phase 1 completed"

# 收集 PCIe 事件（若有）
PCIE_FILES=$(ls "$PCIE_PROFILER_DIR"/pcie_events_*.json 2>/dev/null || true)
if [[ -n "$PCIE_FILES" ]]; then
    PCIE_COUNT=$(echo "$PCIE_FILES" | wc -l | tr -d ' ')
    if [[ $PCIE_COUNT -gt 1 ]]; then
        python3 -c "
import json, glob
events = []
for f in sorted(glob.glob('$PCIE_PROFILER_DIR/pcie_events_*.json')):
    with open(f) as fp:
        events.extend(json.load(fp))
with open('$RESULTS_DIR/pcie_events_pcie_sched.json', 'w') as fp:
    json.dump(events, fp, indent=2)
print(f'Merged {len(events)} events')
"
    else
        cp "$PCIE_PROFILER_DIR/pcie_events_0.json" "$RESULTS_DIR/pcie_events_pcie_sched.json"
    fi
fi
[[ -f "$VLLM_LOG" ]] && cp "$VLLM_LOG" "$RESULTS_DIR/vllm_state_pcie_sched.log" 2>/dev/null || true

# ------------------------------------------------------------------
# 提示用户重启 vLLM
# ------------------------------------------------------------------
echo ""
print_separator
echo "⚠️  Please RESTART vLLM WITHOUT PCIe scheduler:"
echo "   1. Stop current vLLM (Ctrl+C in the terminal running vLLM)"
echo "   2. Start: ./start_vllm_pcie.sh   (no --pcie-scheduler)"
echo "   3. Press Enter here to continue Phase 2"
print_separator
read -r

# ------------------------------------------------------------------
# Phase 2: Prefetch + No Scheduling (baseline)
# ------------------------------------------------------------------
if ! check_vllm_running; then
    echo "❌ Error: vLLM not running. Start vLLM without --pcie-scheduler and re-run."
    exit 1
fi

print_phase "[Phase 2/3] Prefetch without PCIe Scheduling (baseline)"

echo "Resetting prefix cache..."
curl -s -X POST "http://localhost:$API_PORT/reset_prefix_cache?reset_external=true" >/dev/null || true
sleep 5

TB_ARGS=()
[[ $ENABLE_TENSORBOARD -eq 1 && $NO_TENSORBOARD -eq 0 ]] && \
    TB_ARGS=(--tensorboard-dir "$RESULTS_DIR/tensorboard/baseline")

python3 prefetch_ab_runner.py \
    --trace-file "$TRACE" \
    --mode prefetch \
    --qps "$QPS" \
    --num-multi-turn "$NUM_CONV" \
    --model "$MODEL_PATH" \
    --api-base "http://localhost:$API_PORT/v1" \
    --output "$RESULTS_DIR/prefetch_baseline.jsonl" \
    --seed "$SEED" \
    --timeout "$TIMEOUT" \
    --request-timeout "$REQUEST_TIMEOUT" \
    --prefetch-lead-time "$PREFETCH_LEAD_TIME" \
    --schedule-mode "$SCHEDULE_MODE" \
    "${TB_ARGS[@]}" \
    &> "$RESULTS_DIR/prefetch_baseline.log"

echo "✓ Phase 2 completed"

# 收集 baseline PCIe 事件
PCIE_FILES=$(ls "$PCIE_PROFILER_DIR"/pcie_events_*.json 2>/dev/null || true)
if [[ -n "$PCIE_FILES" ]]; then
    PCIE_COUNT=$(echo "$PCIE_FILES" | wc -l | tr -d ' ')
    if [[ $PCIE_COUNT -gt 1 ]]; then
        python3 -c "
import json, glob
events = []
for f in sorted(glob.glob('$PCIE_PROFILER_DIR/pcie_events_*.json')):
    with open(f) as fp:
        events.extend(json.load(fp))
with open('$RESULTS_DIR/pcie_events_baseline.json', 'w') as fp:
    json.dump(events, fp, indent=2)
"
    else
        cp "$PCIE_PROFILER_DIR/pcie_events_0.json" "$RESULTS_DIR/pcie_events_baseline.json"
    fi
fi
[[ -f "$VLLM_LOG" ]] && cp "$VLLM_LOG" "$RESULTS_DIR/vllm_state_baseline.log" 2>/dev/null || true

# ------------------------------------------------------------------
# Phase 3: 生成对比报告
# ------------------------------------------------------------------
print_phase "[Phase 3/3] Generating comparison report..."

# 先保存完整配置快照（供报告和 generate_report 使用）
{
    echo "# PCIe Scheduling A/B Experiment - 完整配置快照"
    echo ""
    echo "# ========== 实验运行时参数 =========="
    echo "DATASET=$DATASET"
    echo "QPS=$QPS"
    echo "PREFETCH_LEAD_TIME=$PREFETCH_LEAD_TIME"
    echo "NUM_GPU_BLOCKS_OVERRIDE=$NUM_GPU_BLOCKS_OVERRIDE"
    echo "TRACE=$TRACE"
    echo "FULL_TRACE=$FULL_TRACE"
    echo "NUM_CONV=$NUM_CONV"
    echo "MODEL_PATH=$MODEL_PATH"
    echo ""
    echo "# ========== system.env =========="
    cat config/system.env 2>/dev/null || true
    echo ""
    echo "# ========== experiments.env =========="
    cat config/experiments.env 2>/dev/null || true
    echo ""
    echo "# ========== datasets.env (当前数据集: $DATASET) =========="
    DS_PREFIX="DATASET_$(echo ${DATASET//-/_} | tr '[:lower:]' '[:upper:]')_"
    grep -E "^DATA_ROOT=|^${DS_PREFIX}" config/datasets.env 2>/dev/null || cat config/datasets.env 2>/dev/null || true
} > "$RESULTS_DIR/config_snapshot.env"
bash dump_config.sh >> "$RESULTS_DIR/config_snapshot.env" 2>/dev/null || true

# 使用 generate_report 做 TTFT/TPOT 对比（传入配置文件以修复配置详情为空）
python3 ../result-analysis/generate_report.py \
    --baseline "$RESULTS_DIR/prefetch_baseline.jsonl" \
    --prefetch "$RESULTS_DIR/prefetch_pcie_sched.jsonl" \
    --output "$RESULTS_DIR/ttft_report.md" \
    --config-file "$RESULTS_DIR/config_snapshot.env" \
    2>/dev/null && echo "✓ TTFT report: $RESULTS_DIR/ttft_report.md" || true

# 使用 PCIe 调度专用报告生成器（含 config、PCIe 带宽等）
if [[ -f "../result-analysis/generate_pcie_scheduling_report.py" ]]; then
    python3 ../result-analysis/generate_pcie_scheduling_report.py \
        --results-dir "$RESULTS_DIR" \
        --dataset "$DATASET" \
        --qps "$QPS" \
        --lead-time "$PREFETCH_LEAD_TIME" \
        --output "$RESULTS_DIR/pcie_scheduling_report.md" \
        2>/dev/null && echo "✓ PCIe scheduling report: $RESULTS_DIR/pcie_scheduling_report.md" || \
        echo "⚠️  generate_pcie_scheduling_report.py failed (optional)"
else
    echo "⚠️  generate_pcie_scheduling_report.py not found, skipping enhanced report"
fi

echo ""
echo "✅ PCIe Scheduling A/B experiment complete!"
echo ""
echo "Results:"
echo "  Directory: $RESULTS_DIR"
echo "  TTFT report: $RESULTS_DIR/ttft_report.md"
echo "  PCIe scheduling report: $RESULTS_DIR/pcie_scheduling_report.md"
echo "  Logs: prefetch_pcie_sched.log, prefetch_baseline.log"
print_separator
