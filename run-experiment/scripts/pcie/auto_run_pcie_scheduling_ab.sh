#!/bin/bash
# 自动化运行 PCIe Scheduling A/B 实验
# 自动管理 vLLM 的启动和重启，无需手动交互
#
# Phase 1: Prefetch + PCIe Scheduling (VLLM_PCIE_SCHEDULER=1)
# Phase 2: Prefetch 无 PCIe Scheduling (baseline)
# Phase 3: 生成报告
#
# 用法: ./auto_run_pcie_scheduling_ab.sh [dataset] [options]
# 示例:
#   ./auto_run_pcie_scheduling_ab.sh pcie-medium
#   ./auto_run_pcie_scheduling_ab.sh pcie-trace-a-light
#   ./auto_run_pcie_scheduling_ab.sh pcie-heavy --qps 3.0

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_EXP_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$RUN_EXP_DIR"

# ============================================================
# vLLM 自动管理
# ============================================================

VLLM_PID=""

start_vllm() {
    local label="$1"; shift
    local args=("$@")

    print_phase "Starting vLLM [$label] args: ${args[*]}"

    # 清理残留进程
    pkill -f "vllm serve" 2>/dev/null || true
    sleep 2

    local startup_log="$RESULTS_DIR/vllm_startup_${label}.log"
    nohup bash "$RUN_EXP_DIR/start_vllm_pcie.sh" "${args[@]}" > "$startup_log" 2>&1 &
    VLLM_PID=$!
    echo "vLLM PID: $VLLM_PID  (log: $startup_log)"

    local max_wait=300
    local waited=0
    echo -n "Waiting for vLLM to be ready"

    while [ $waited -lt $max_wait ]; do
        if curl -s "http://localhost:$API_PORT/health" &>/dev/null; then
            echo ""
            echo "✓ vLLM is ready (${waited}s)"
            sleep 5
            return 0
        fi
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo ""
            echo "❌ vLLM process died during startup. Check: $startup_log"
            return 1
        fi
        echo -n "."
        sleep 3
        waited=$((waited + 3))
    done

    echo ""
    echo "❌ vLLM failed to start within ${max_wait}s. Check: $startup_log"
    return 1
}

stop_vllm() {
    if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "Stopping vLLM (PID: $VLLM_PID)..."
        kill "$VLLM_PID" 2>/dev/null || true

        local waited=0
        while kill -0 "$VLLM_PID" 2>/dev/null && [ $waited -lt 30 ]; do
            sleep 1
            waited=$((waited + 1))
        done

        if kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "Force killing..."
            kill -9 "$VLLM_PID" 2>/dev/null || true
        fi
        VLLM_PID=""
    fi

    # 额外保险
    pkill -f "vllm serve" 2>/dev/null || true
    sleep 3
    echo "✓ vLLM stopped"
}

cleanup() {
    echo ""
    echo "Cleaning up..."
    stop_vllm
}
trap cleanup EXIT INT TERM

# Helper: 收集 PCIe 事件、vLLM 日志，并清理 profiler 文件
collect_pcie_events() {
    local suffix="$1"
    local output_json="$RESULTS_DIR/pcie_events_${suffix}.json"

    PCIE_FILES=$(ls "$PCIE_PROFILER_DIR"/pcie_events_*.json 2>/dev/null || true)
    if [[ -n "$PCIE_FILES" ]]; then
        PCIE_COUNT=$(echo "$PCIE_FILES" | wc -l | tr -d ' ')
        if [[ $PCIE_COUNT -gt 1 ]]; then
            python3 -c "
import json, glob
events = []
for f in sorted(glob.glob('${PCIE_PROFILER_DIR}/pcie_events_*.json')):
    with open(f) as fp:
        events.extend(json.load(fp))
with open('${output_json}', 'w') as fp:
    json.dump(events, fp, indent=2)
print(f'Merged {len(events)} events')
"
        else
            cp "$PCIE_PROFILER_DIR"/pcie_events_*.json "$output_json"
        fi
        echo "✓ PCIe events: $output_json"
    else
        echo "⚠️  No pcie_events_*.json found for $suffix"
    fi

    rm -f "$PCIE_PROFILER_DIR"/pcie_events_*.json 2>/dev/null || true
}

# ============================================================
# 加载配置（与手动脚本一致）
# ============================================================

source config/system.env
source config/datasets.env
source config/experiments.env
source scripts/utils/common.sh

DATASET="${1:-pcie-medium}"
shift 2>/dev/null || true

# 解析选项
NUM_GPU_BLOCKS_OVERRIDE_SET=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --qps)             QPS="$2"; shift 2 ;;
        --lead-time)       PREFETCH_LEAD_TIME="$2"; shift 2 ;;
        --gpu-blocks)      NUM_GPU_BLOCKS_OVERRIDE="$2"; NUM_GPU_BLOCKS_OVERRIDE_SET=1; shift 2 ;;
        *)
            echo "❌ Unknown option: $1"
            echo "Usage: $0 [dataset] [--qps N] [--lead-time N] [--gpu-blocks N]"
            exit 1 ;;
    esac
done

load_dataset_config "$DATASET"
[[ $NUM_GPU_BLOCKS_OVERRIDE_SET -eq 0 ]] && NUM_GPU_BLOCKS_OVERRIDE="${DATASET_GPU_BLOCKS:-$NUM_GPU_BLOCKS_OVERRIDE}"

generate_dataset_if_needed "$TRACE" "$FULL_TRACE" "$DATASET" || exit 1

# Timeout 配置
PCIE_FULL_RUNNER_TIMEOUT_ARGS=(--timeout "$TIMEOUT" --request-timeout "$REQUEST_TIMEOUT")
if [[ "$DATASET" == "pcie-full" || "$DATASET" == "pcie-trace-a-light" ]]; then
    if [[ ! -f "$TRACE" ]]; then
        echo "❌ $DATASET: trace 不存在: $TRACE"
        exit 1
    fi
    NUM_CONV=$(
        awk '
        index($0, "\"parent_chat_id\": -1") > 0 {
            if (match($0, /"chat_id": [0-9]+/)) {
                cid = substr($0, RSTART+11, RLENGTH-11)
                isroot[cid] = 1
            }
        }
        {
            idx = index($0, "\"parent_chat_id\": ")
            if (idx == 0) next
            rest = substr($0, idx + length("\"parent_chat_id\": "))
            if (length(rest) == 0 || substr(rest, 1, 1) == "-") next
            if (match(rest, /^[0-9]+/)) {
                pid = substr(rest, 1, RLENGTH)
                haschild[pid] = 1
            }
        }
        END {
            n = 0
            for (c in isroot) if (c in haschild) n++
            print n
        }
        ' "$TRACE"
    )
    if [[ -z "${NUM_CONV// /}" || ! "$NUM_CONV" =~ ^[0-9]+$ || "$NUM_CONV" -eq 0 ]]; then
        echo "❌ $DATASET: 无法从 trace 统计多轮对话根数量: $TRACE"
        exit 1
    fi
    echo "✓ $DATASET: 全量多轮根数量 NUM_CONV=$NUM_CONV（自 trace 统计）"
    REQUEST_TIMEOUT=360
    TIMEOUT=""
    PCIE_FULL_RUNNER_TIMEOUT_ARGS=(--request-timeout "$REQUEST_TIMEOUT")
fi

# 可选: 限制单请求最大生成 token 数
MAX_OUTPUT_ARGS=()
if [[ -n "$MAX_OUTPUT" ]]; then
    MAX_OUTPUT_ARGS=(--max-output-tokens "$MAX_OUTPUT")
    echo "✓ Max output tokens: $MAX_OUTPUT"
fi

# ============================================================
# 路径设置
# ============================================================

REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
MASTER_TABLE_TSV="$REPO_ROOT/results/pcie_scheduling_experiments.txt"

# VLLM_LOG / PCIE_PROFILER_DIR 转为绝对路径
[[ "$VLLM_LOG" != /* ]] && VLLM_LOG="$RUN_EXP_DIR/$VLLM_LOG"
if [[ "$PCIE_PROFILER_DIR" != /* ]]; then
    PCIE_PROFILER_DIR="$RUN_EXP_DIR/${PCIE_PROFILER_DIR#./}"
fi
mkdir -p "$PCIE_PROFILER_DIR"

# 实验辨识码
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

mkdir -p "$REPO_ROOT/results"
RESULTS_DIR="$REPO_ROOT/results/$EXP_ID"
mkdir -p "$RESULTS_DIR"

# 打印配置
print_separator
echo "PCIe Scheduling A/B Experiment (AUTOMATED)"
print_separator
echo "Experiment ID: $EXP_ID"
echo "Dataset: $DATASET"
echo "Trace: $TRACE"
echo "Num conversations: $NUM_CONV"
echo "QPS: $QPS"
echo "Prefetch lead time: ${PREFETCH_LEAD_TIME}s"
echo "GPU blocks: $NUM_GPU_BLOCKS_OVERRIDE"
if [[ "$DATASET" == "pcie-full" || "$DATASET" == "pcie-trace-a-light" ]]; then
    echo "Runner timeouts: ${PCIE_FULL_RUNNER_TIMEOUT_ARGS[*]} (no global phase timeout)"
else
    echo "Runner timeouts: TIMEOUT=${TIMEOUT}s REQUEST_TIMEOUT=${REQUEST_TIMEOUT}s"
fi
echo "Run directory: $RESULTS_DIR"
echo "Master TSV (all runs): $MASTER_TABLE_TSV"
print_separator

# ============================================================
# Phase 1: Prefetch + PCIe Scheduling
# ============================================================

start_vllm pcie_sched --pcie-scheduler --log-file "$RESULTS_DIR/vllm_log_pcie_sched.log" || exit 1

print_phase "[Phase 1/3] Prefetch with PCIe Scheduling (VLLM_PCIE_SCHEDULER=1)"

echo "Resetting prefix cache..."
curl -s -X POST "http://localhost:$API_PORT/reset_prefix_cache?reset_external=true" >/dev/null || true
sleep 5

echo "Clearing stale PCIe event files in $PCIE_PROFILER_DIR..."
rm -f "$PCIE_PROFILER_DIR"/pcie_events_*.json 2>/dev/null || true
echo "Starting PCIe profiler..."
curl -s -X POST "http://localhost:$API_PORT/start_profile" >/dev/null || true

python3 prefetch_ab_runner.py \
    --trace-file "$TRACE" \
    --mode prefetch \
    --qps "$QPS" \
    --num-multi-turn "$NUM_CONV" \
    --model "$MODEL_PATH" \
    --api-base "http://localhost:$API_PORT/v1" \
    --output "$RESULTS_DIR/prefetch_pcie_sched.jsonl" \
    --seed "$SEED" \
    "${PCIE_FULL_RUNNER_TIMEOUT_ARGS[@]}" \
    "${MAX_OUTPUT_ARGS[@]}" \
    --prefetch-lead-time "$PREFETCH_LEAD_TIME" \
    --schedule-mode "$SCHEDULE_MODE" \
    2>&1 | tee "$RESULTS_DIR/prefetch_pcie_sched.log"

echo "✓ Phase 1 completed"

echo "Flushing PCIe profiler to disk (stop_profile)..."
curl -s -X POST "http://localhost:$API_PORT/stop_profile" >/dev/null || true
sleep 3

collect_pcie_events "pcie_sched"

# ============================================================
# Phase 1 → Phase 2: 重启 vLLM（不带 --pcie-scheduler）
# ============================================================

print_phase "Restarting vLLM for Phase 2 (without PCIe scheduler)..."
stop_vllm
start_vllm baseline --log-file "$RESULTS_DIR/vllm_log_baseline.log" || exit 1

# ============================================================
# Phase 2: Prefetch without PCIe Scheduling (baseline)
# ============================================================

print_phase "[Phase 2/3] Prefetch without PCIe Scheduling (baseline)"

echo "Resetting prefix cache..."
curl -s -X POST "http://localhost:$API_PORT/reset_prefix_cache?reset_external=true" >/dev/null || true
sleep 5

echo "Starting PCIe profiler..."
curl -s -X POST "http://localhost:$API_PORT/start_profile" >/dev/null || true

python3 prefetch_ab_runner.py \
    --trace-file "$TRACE" \
    --mode prefetch \
    --qps "$QPS" \
    --num-multi-turn "$NUM_CONV" \
    --model "$MODEL_PATH" \
    --api-base "http://localhost:$API_PORT/v1" \
    --output "$RESULTS_DIR/prefetch_baseline.jsonl" \
    --seed "$SEED" \
    "${PCIE_FULL_RUNNER_TIMEOUT_ARGS[@]}" \
    "${MAX_OUTPUT_ARGS[@]}" \
    --prefetch-lead-time "$PREFETCH_LEAD_TIME" \
    --schedule-mode "$SCHEDULE_MODE" \
    2>&1 | tee "$RESULTS_DIR/prefetch_baseline.log"

echo "✓ Phase 2 completed"

echo "Flushing PCIe profiler to disk (stop_profile)..."
curl -s -X POST "http://localhost:$API_PORT/stop_profile" >/dev/null || true
sleep 3

collect_pcie_events "baseline"

# 停止 vLLM
stop_vllm

# ============================================================
# Phase 3: 生成报告
# ============================================================

print_phase "[Phase 3/3] Generating comparison report..."

CONFIG_TMP=$(mktemp)

DATASET="$DATASET" QPS="$QPS" PREFETCH_LEAD_TIME="$PREFETCH_LEAD_TIME" \
    NUM_GPU_BLOCKS_OVERRIDE="$NUM_GPU_BLOCKS_OVERRIDE" TRACE="$TRACE" FULL_TRACE="$FULL_TRACE" \
    NUM_CONV="$NUM_CONV" MODEL_PATH="$MODEL_PATH" \
    TIMEOUT="${TIMEOUT:-}" REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-}" \
    RUN_EXPERIMENT_DIR="$RUN_EXP_DIR" bash "$RUN_EXP_DIR/dump_config.sh" 2>/dev/null \
    | grep -E '^[A-Za-z_][A-Za-z0-9_]*=' > "$CONFIG_TMP" || true

python3 ../result-analysis/generate_report.py \
    --baseline "$RESULTS_DIR/prefetch_baseline.jsonl" \
    --prefetch "$RESULTS_DIR/prefetch_pcie_sched.jsonl" \
    --output "$RESULTS_DIR/ttft_report.md" \
    --config-file "$CONFIG_TMP" \
    2>/dev/null && echo "✓ Prefetch A/B section" || echo "⚠️  generate_report.py failed"

PCIE_PART=""
if [[ -f "../result-analysis/generate_pcie_scheduling_report.py" ]]; then
    PCIE_PART=$(mktemp)
    python3 ../result-analysis/generate_pcie_scheduling_report.py \
        --results-dir "$RESULTS_DIR" \
        --dataset "$DATASET" \
        --qps "$QPS" \
        --lead-time "$PREFETCH_LEAD_TIME" \
        --config-file "$CONFIG_TMP" \
        --output "$PCIE_PART" \
        2>/dev/null && echo "✓ PCIe scheduling section" || \
        echo "⚠️  generate_pcie_scheduling_report.py failed"
fi

MERGED_MD="$RESULTS_DIR/pcie_scheduling_ab_report.md"
{
    if [[ -s "$RESULTS_DIR/ttft_report.md" ]]; then
        cat "$RESULTS_DIR/ttft_report.md"
    fi
    echo ""
    echo "---"
    echo ""
    if [[ -n "$PCIE_PART" && -s "$PCIE_PART" ]]; then
        cat "$PCIE_PART"
    fi
} > "$MERGED_MD"
echo "✓ Merged report: $MERGED_MD"

rm -f "$CONFIG_TMP" "$PCIE_PART" 2>/dev/null || true

# 追加到 Master TSV
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

# ============================================================
# 完成
# ============================================================

print_separator
echo "✓ Automated PCIe Scheduling A/B Test Complete!"
echo ""
echo "Results:"
echo "  Experiment ID: $EXP_ID"
echo "  Directory:     $RESULTS_DIR"
echo "  Merged report: $MERGED_MD"
echo "  Master TSV:    $MASTER_TABLE_TSV"
echo "  Logs / jsonl:  prefetch_*.log, prefetch_*.jsonl, pcie_events_*.json, vllm_log_*.log"
print_separator
