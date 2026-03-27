#!/bin/bash
# PCIe 调度 A/B 实验脚本（跳过 Plain baseline，仅两路对比）
#
# 与 run_pcie_scheduling_ab.sh 相同，但不跑 Phase 1（普通 vLLM / 客户端 baseline）。
#
# 1. vLLM + Prefetch + PCIe 调度：--mode prefetch，服务端 start_vllm_pcie.sh --pcie-scheduler
# 2. vLLM + Prefetch：--mode prefetch，服务端无 PCIe 调度
#
# 论文/消融建议三组（须分别启动 vLLM 与脚本）：
#   Exp A（推荐）: start_vllm_pcie.sh --pcie-scheduler [--pp-phase-h2d-policy idle_only]
#   Exp B（对照）: start_vllm_pcie.sh --pcie-scheduler --pp-phase-h2d-policy soft
#   Exp C（无调度）: start_vllm_pcie.sh（勿加 --pcie-scheduler），本脚本 Phase 2 即 Prefetch-only
#
# 流程：Phase 1 在「有 PCIe 调度」的 vLLM 上跑 Prefetch+PCIe → 重启并禁用调度 → Phase 2 Prefetch → Phase 3 生成合并报告并追加汇总 TSV
#       （finalize 在无 plain_vllm.jsonl 时自动输出「+Prefetch+PCIe / +Prefetch」两路报告）
#
# 用法:
#   ./run_pcie_scheduling_ab_no_plain.sh [dataset] [options]
#
# 示例:
#   ./run_pcie_scheduling_ab_no_plain.sh pcie-medium
#   ./run_pcie_scheduling_ab_no_plain.sh pcie-heavy --qps 3.0
#
# 注意: 开始前用 ./start_vllm_pcie.sh --pcie-scheduler；Phase 1 结束后按提示重启并去掉 --pcie-scheduler。

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

# 解析选项（与 start_vllm_pcie.sh 的 --pp-phase-h2d-policy 一致，用于报告/汇总表记录 Phase 1 配置）
NO_TENSORBOARD=0
NUM_GPU_BLOCKS_OVERRIDE_SET=0
PP_PHASE_H2D_POLICY=idle_only

while [[ $# -gt 0 ]]; do
    case "$1" in
        --qps)
            QPS="$2"; shift 2 ;;
        --lead-time)
            PREFETCH_LEAD_TIME="$2"; shift 2 ;;
        --gpu-blocks)
            NUM_GPU_BLOCKS_OVERRIDE="$2"; NUM_GPU_BLOCKS_OVERRIDE_SET=1; shift 2 ;;
        --pp-phase-h2d-policy)
            if [[ $# -lt 2 ]]; then
                echo "❌ --pp-phase-h2d-policy 需要参数: idle_only | soft | hard | restore_only"
                exit 1
            fi
            PP_PHASE_H2D_POLICY="$2"
            case "$PP_PHASE_H2D_POLICY" in
                idle_only|soft|hard|restore_only) ;;
                *)
                    echo "❌ --pp-phase-h2d-policy 必须是 idle_only、soft、hard 或 restore_only，收到: $PP_PHASE_H2D_POLICY"
                    exit 1 ;;
            esac
            shift 2 ;;
        --no-tensorboard)
            NO_TENSORBOARD=1; shift ;;
        *)
            echo "❌ Unknown option: $1"
            echo "Usage: $0 [dataset] [--qps N] [--lead-time N] [--gpu-blocks N] [--pp-phase-h2d-policy idle_only|soft|hard|restore_only] [--no-tensorboard]"
            exit 1 ;;
    esac
done

# 加载数据集配置
load_dataset_config "$DATASET"
[[ $NUM_GPU_BLOCKS_OVERRIDE_SET -eq 0 ]] && NUM_GPU_BLOCKS_OVERRIDE="${DATASET_GPU_BLOCKS:-$NUM_GPU_BLOCKS_OVERRIDE}"

# 检查数据集
generate_dataset_if_needed "$TRACE" "$FULL_TRACE" "$DATASET" || exit 1

# 结果根目录（绝对路径）与实验辨识码：时间 + 数据集 + 关键参数（原始 log/jsonl 均在此目录下）
mkdir -p "$RESULTS_ROOT"
RESULTS_ROOT_ABS="$(cd "$RESULTS_ROOT" && pwd)"
EXP_TS="$(date +%Y%m%d_%H%M%S)"
DS_SAFE="${DATASET//\//_}"
EXP_ID="${EXP_TS}_${DS_SAFE}_q${QPS}_l${PREFETCH_LEAD_TIME}_blk${NUM_GPU_BLOCKS_OVERRIDE}"
RESULTS_DIR="$RESULTS_ROOT_ABS/$EXP_ID"
mkdir -p "$RESULTS_DIR"
TSV_SUMMARY="$RESULTS_ROOT_ABS/pcie_scheduling_ab_table.txt"

# Profiler 目录改为绝对路径（与 start_vllm_pcie.sh 一致），便于清空与收集
RUN_EXP_ROOT="$(pwd)"
if [[ "$PCIE_PROFILER_DIR" != /* ]]; then
    PCIE_PROFILER_DIR="$RUN_EXP_ROOT/${PCIE_PROFILER_DIR#./}"
fi
mkdir -p "$PCIE_PROFILER_DIR"

print_separator
echo "PCIe Scheduling A/B (skip Plain): Prefetch+PCIe / Prefetch only"
print_separator
echo "Dataset: $DATASET"
echo "Trace: $TRACE"
echo "Num conversations: $NUM_CONV"
echo "QPS: $QPS"
echo "Prefetch lead time: ${PREFETCH_LEAD_TIME}s"
echo "GPU blocks: $NUM_GPU_BLOCKS_OVERRIDE"
echo "PP phase H2D policy (Phase 1 须与 start_vllm_pcie.sh 一致): $PP_PHASE_H2D_POLICY"
echo "Experiment ID: $EXP_ID"
echo "Results: $RESULTS_DIR"
echo "Summary TSV: $TSV_SUMMARY"
print_separator

# ------------------------------------------------------------------
# Phase 1/3: Prefetch + PCIe 调度（原脚本的 Phase 2）
# ------------------------------------------------------------------
if ! check_vllm_running; then
    echo "❌ Error: vLLM not running."
    if [[ "$PP_PHASE_H2D_POLICY" == "idle_only" ]]; then
        echo "Start with: ./start_vllm_pcie.sh --pcie-scheduler"
    else
        echo "Start with: ./start_vllm_pcie.sh --pcie-scheduler --pp-phase-h2d-policy $PP_PHASE_H2D_POLICY"
    fi
    exit 1
fi

print_phase "[Phase 1/3] vLLM + Prefetch + PCIe scheduling"

echo "Resetting prefix cache..."
curl -s -X POST "http://localhost:$API_PORT/reset_prefix_cache?reset_external=true" >/dev/null || true
sleep 5

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

echo "Removing profiler PCIe event files before Phase 2..."
rm -f "$PCIE_PROFILER_DIR"/pcie_events_*.json 2>/dev/null || true

# ------------------------------------------------------------------
# 提示用户重启 vLLM 并禁用 PCIe 调度
# ------------------------------------------------------------------
echo ""
print_separator
echo "⚠️  Please RESTART vLLM WITHOUT PCIe scheduler for Phase 2:"
echo "   1. Stop current vLLM (Ctrl+C)"
echo "   2. Start: ./start_vllm_pcie.sh   (do NOT use --pcie-scheduler)"
echo "   3. Press Enter here to continue Phase 2"
print_separator
read -r

# ------------------------------------------------------------------
# Phase 2/3: Prefetch，无 PCIe 调度（原脚本的 Phase 1）
# ------------------------------------------------------------------
if ! check_vllm_running; then
    echo "❌ Error: vLLM not running."
    echo "Start with: ./start_vllm_pcie.sh   (do NOT use --pcie-scheduler)"
    exit 1
fi

print_phase "[Phase 2/3] vLLM + Prefetch (no PCIe scheduling)"

echo "Resetting prefix cache..."
curl -s -X POST "http://localhost:$API_PORT/reset_prefix_cache?reset_external=true" >/dev/null || true
sleep 5

echo "Clearing stale PCIe event files in $PCIE_PROFILER_DIR..."
rm -f "$PCIE_PROFILER_DIR"/pcie_events_*.json 2>/dev/null || true
echo "Starting PCIe profiler..."
curl -s -X POST "http://localhost:$API_PORT/start_profile" >/dev/null || true

TB_ARGS=()
[[ $ENABLE_TENSORBOARD -eq 1 && $NO_TENSORBOARD -eq 0 ]] && \
    TB_ARGS=(--tensorboard-dir "$RESULTS_DIR/tensorboard/prefetch_no_sched")

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
# Phase 3/3: 合并 Markdown 报告 + 追加制表符汇总表（可粘贴 Excel）
# ------------------------------------------------------------------
print_phase "[Phase 3/3] Generating merged report and appending summary TSV..."

if [[ -f "../result-analysis/pcie_scheduling_ab_finalize.py" ]]; then
    python3 ../result-analysis/pcie_scheduling_ab_finalize.py \
        --results-dir "$RESULTS_DIR" \
        --experiment-id "$EXP_ID" \
        --model-path "$MODEL_PATH" \
        --dataset "$DATASET" \
        --qps "$QPS" \
        --lead-time "$PREFETCH_LEAD_TIME" \
        --num-gpu-blocks "$NUM_GPU_BLOCKS_OVERRIDE" \
        --num-conv "$NUM_CONV" \
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-}" \
        --vllm-pipeline-parallel-size "${VLLM_PIPELINE_PARALLEL_SIZE:-}" \
        --vllm-max-num-seqs "${VLLM_MAX_NUM_SEQS:-}" \
        --pp-phase-h2d-policy "$PP_PHASE_H2D_POLICY" \
        --md-output "$RESULTS_DIR/experiment_report.md" \
        --tsv-path "$TSV_SUMMARY" \
        && echo "✓ Merged report: $RESULTS_DIR/experiment_report.md" \
        && echo "✓ Summary table appended: $TSV_SUMMARY" || \
        echo "⚠️  pcie_scheduling_ab_finalize.py failed"
else
    echo "⚠️  pcie_scheduling_ab_finalize.py not found"
fi

echo ""
echo "✅ PCIe Scheduling A/B experiment complete (Plain baseline skipped)!"
echo ""
echo "Results:"
echo "  Experiment ID: $EXP_ID"
echo "  Directory (raw logs & jsonl): $RESULTS_DIR"
echo "  Merged report: $RESULTS_DIR/experiment_report.md"
echo "  Tab-separated summary: $TSV_SUMMARY"
echo "  Logs: prefetch_pcie_sched.log, prefetch_baseline.log"
print_separator