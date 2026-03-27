#!/bin/bash
# PCIe 调度 A/B 实验脚本
#
# 对比 VLLM_PCIE_SCHEDULER=1（启用）与 未启用 两种配置下的 Prefetch 表现。
# 实验流程：Phase 1 启用调度 → Phase 2 未启用 → Phase 3 生成合并 Markdown + 追加 TSV 汇总表
#
# 产物：
#   - 单次实验目录：<repo>/results/<实验辨识码>/（jsonl、log、pcie json 等，与旧版相同文件名）
#   - 合并报告：pcie_scheduling_ab_report.md（原 ttft_report + pcie_scheduling_report 拼接，中间 ---）
#   - 全实验 TSV：<repo>/results/pcie_scheduling_experiments.txt（制表符分隔，可粘贴 Excel）
#   不在实验目录写入 config_snapshot.env；报告用临时 key=value 文件生成。
#
# 用法:
#   ./run_pcie_scheduling_ab.sh [dataset] [options]
#
# 示例:
#   ./run_pcie_scheduling_ab.sh pcie-medium
#   ./run_pcie_scheduling_ab.sh pcie-heavy --qps 3.0
#   ./run_pcie_scheduling_ab.sh pcie-full   # 直接使用 data/qwen_traceA_blksz_16.jsonl（与 medium 同参数范式）
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

# 仓库根目录（llm-prefetch-testsuit）与固定汇总表路径
RUN_EXP_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
MASTER_TABLE_TSV="$REPO_ROOT/results/pcie_scheduling_experiments.txt"

# 实验辨识码：时间 + 数据集 + 关键参数 + 模型规模（便于区分历次实验）
EXP_TS="$(date +%Y%m%d_%H%M%S)"
MODEL_TAG="$(python3 -c "
import re, sys
p = sys.argv[1].lower()
checks = [
    (r'(?<![0-9.])72b(?![0-9])', '72B'),
    (r'(?<![0-9.])32b(?![0-9])', '32B'),
    (r'(?<![0-9.])8b(?![0-9])', '8B'),
]
for pat, lab in checks:
    if re.search(pat, p):
        print(lab)
        break
else:
    print('unknown')
" "$MODEL_PATH")"

EXP_ID="${EXP_TS}_${DATASET}_q${QPS}_lead${PREFETCH_LEAD_TIME}_${MODEL_TAG}"
EXP_ID="${EXP_ID//\//_}"
EXP_ID="${EXP_ID// /_}"

# 单次实验产物目录：results/<实验辨识码>/
mkdir -p "$REPO_ROOT/results"
RESULTS_DIR="$REPO_ROOT/results/$EXP_ID"
mkdir -p "$RESULTS_DIR"

# Profiler 目录改为绝对路径（与 start_vllm_pcie.sh 一致），便于清空与收集
RUN_EXP_ROOT="$(pwd)"
if [[ "$PCIE_PROFILER_DIR" != /* ]]; then
    PCIE_PROFILER_DIR="$RUN_EXP_ROOT/${PCIE_PROFILER_DIR#./}"
fi
mkdir -p "$PCIE_PROFILER_DIR"

print_separator
echo "PCIe Scheduling A/B Experiment"
print_separator
echo "Experiment ID: $EXP_ID"
echo "Dataset: $DATASET"
echo "Trace: $TRACE"
echo "Num conversations: $NUM_CONV"
echo "QPS: $QPS"
echo "Prefetch lead time: ${PREFETCH_LEAD_TIME}s"
echo "GPU blocks: $NUM_GPU_BLOCKS_OVERRIDE"
echo "Run directory: $RESULTS_DIR"
echo "Master TSV (all runs): $MASTER_TABLE_TSV"
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

# 避免沿用上轮实验的 PCIe 事件文件；与 run_profiling.sh 一致需 start/stop_profile 才能落盘
echo "Clearing stale PCIe event files in $PCIE_PROFILER_DIR..."
rm -f "$PCIE_PROFILER_DIR"/pcie_events_*.json 2>/dev/null || true
echo "Starting PCIe profiler..."
curl -s -X POST "http://localhost:$API_PORT/start_profile" >/dev/null || true

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

echo "Flushing PCIe profiler to disk (stop_profile)..."
curl -s -X POST "http://localhost:$API_PORT/stop_profile" >/dev/null || true
sleep 3

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

# Phase 2 会重新写入 profiler；若不删除，baseline 易重复采集 Phase 1 的 pcie_events_*.json
echo "Removing profiler PCIe event files before Phase 2..."
rm -f "$PCIE_PROFILER_DIR"/pcie_events_*.json 2>/dev/null || true

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

echo "Starting PCIe profiler..."
curl -s -X POST "http://localhost:$API_PORT/start_profile" >/dev/null || true

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

echo "Flushing PCIe profiler to disk (stop_profile)..."
curl -s -X POST "http://localhost:$API_PORT/stop_profile" >/dev/null || true
sleep 3

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
# Phase 3: 生成合并 Markdown 报告 + 追加 TSV 汇总行（不写 results 下的 config_snapshot.env）
# ------------------------------------------------------------------
print_phase "[Phase 3/3] Generating comparison report..."

CONFIG_TMP=$(mktemp)
TTFT_PART=$(mktemp)
PCIE_PART=$(mktemp)
cleanup_phase3_tmp() {
    rm -f "$CONFIG_TMP" "$TTFT_PART" "$PCIE_PART"
}
trap cleanup_phase3_tmp EXIT

# 临时 key=value 配置（仅用于报告生成，不落盘到实验目录）
# grep 在无匹配时返回 1，需吞掉以免 set -e 中断
DATASET="$DATASET" QPS="$QPS" PREFETCH_LEAD_TIME="$PREFETCH_LEAD_TIME" \
    NUM_GPU_BLOCKS_OVERRIDE="$NUM_GPU_BLOCKS_OVERRIDE" TRACE="$TRACE" FULL_TRACE="$FULL_TRACE" \
    NUM_CONV="$NUM_CONV" MODEL_PATH="$MODEL_PATH" \
    RUN_EXPERIMENT_DIR="$RUN_EXP_ROOT" bash "$RUN_EXP_ROOT/dump_config.sh" 2>/dev/null \
    | grep -E '^[A-Za-z_][A-Za-z0-9_]*=' > "$CONFIG_TMP" || true

python3 ../result-analysis/generate_report.py \
    --baseline "$RESULTS_DIR/prefetch_baseline.jsonl" \
    --prefetch "$RESULTS_DIR/prefetch_pcie_sched.jsonl" \
    --output "$TTFT_PART" \
    --config-file "$CONFIG_TMP" \
    2>/dev/null && echo "✓ Prefetch A/B section (temp)" || echo "⚠️  generate_report.py failed"

if [[ -f "../result-analysis/generate_pcie_scheduling_report.py" ]]; then
    python3 ../result-analysis/generate_pcie_scheduling_report.py \
        --results-dir "$RESULTS_DIR" \
        --dataset "$DATASET" \
        --qps "$QPS" \
        --lead-time "$PREFETCH_LEAD_TIME" \
        --config-file "$CONFIG_TMP" \
        --output "$PCIE_PART" \
        2>/dev/null && echo "✓ PCIe scheduling section (temp)" || \
        echo "⚠️  generate_pcie_scheduling_report.py failed"
else
    echo "⚠️  generate_pcie_scheduling_report.py not found, PCIe section omitted"
fi

MERGED_MD="$RESULTS_DIR/pcie_scheduling_ab_report.md"
{
    if [[ -s "$TTFT_PART" ]]; then
        cat "$TTFT_PART"
    fi
    echo ""
    echo "---"
    echo ""
    if [[ -s "$PCIE_PART" ]]; then
        cat "$PCIE_PART"
    fi
} > "$MERGED_MD"
echo "✓ Merged report: $MERGED_MD"

cleanup_phase3_tmp
trap - EXIT

if [[ -f "../result-analysis/append_pcie_scheduling_summary_row.py" ]]; then
    python3 ../result-analysis/append_pcie_scheduling_summary_row.py \
        --results-dir "$RESULTS_DIR" \
        --exp-id "$EXP_ID" \
        --dataset "$DATASET" \
        --qps "$QPS" \
        --lead-time "$PREFETCH_LEAD_TIME" \
        --num-gpu-blocks "$NUM_GPU_BLOCKS_OVERRIDE" \
        --num-conv "$NUM_CONV" \
        --gpu-mem-util "$GPU_MEMORY_UTILIZATION" \
        --vllm-pp "$VLLM_PIPELINE_PARALLEL_SIZE" \
        --vllm-max-num-seqs "$VLLM_MAX_NUM_SEQS" \
        --model-path "$MODEL_PATH" \
        --table "$MASTER_TABLE_TSV" \
        && echo "✓ Appended row to $MASTER_TABLE_TSV" || \
        echo "⚠️  append_pcie_scheduling_summary_row.py failed"
else
    echo "⚠️  append_pcie_scheduling_summary_row.py not found, skip TSV append"
fi

echo ""
echo "✅ PCIe Scheduling A/B experiment complete!"
echo ""
echo "Results:"
echo "  Run directory: $RESULTS_DIR"
echo "  Merged report: $MERGED_MD"
echo "  Master TSV:    $MASTER_TABLE_TSV"
echo "  Logs / jsonl:  prefetch_*.log, prefetch_*.jsonl, pcie_events_*.json, vllm_state_*.log"
print_separator
