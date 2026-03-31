#!/bin/bash
# PCIe Ablation 实验脚本（4 组对比）
#
# G3: 完整调度（VLLM_PCIE_SCHEDULER=1, PP-phase-aware ON）
# G2: 调度器开启，PP 阶段感知关闭（VLLM_PCIE_SCHEDULER=1, --no-pp-phase-aware）
# G1: 无调度器，有 Prefetch（baseline prefetch）
# G0: 无调度器，无 Prefetch（原始 vLLM baseline）
#
# 每个 Phase 之间需要手动重启 vLLM（脚本会提示）。
#
# 用法:
#   ./run_pcie_ablation_ab.sh [dataset] [options]
#
# 示例:
#   ./run_pcie_ablation_ab.sh pcie-medium
#   ./run_pcie_ablation_ab.sh pcie-medium --qps 3.0
#   ./run_pcie_ablation_ab.sh pcie-heavy --qps 1.5 --gpu-blocks 1000

set -eo pipefail

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
NUM_GPU_BLOCKS_OVERRIDE_SET=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --qps)
            QPS="$2"; shift 2 ;;
        --lead-time)
            PREFETCH_LEAD_TIME="$2"; shift 2 ;;
        --gpu-blocks)
            NUM_GPU_BLOCKS_OVERRIDE="$2"; NUM_GPU_BLOCKS_OVERRIDE_SET=1; shift 2 ;;
        *)
            echo "❌ Unknown option: $1"
            echo "Usage: $0 [dataset] [--qps N] [--lead-time N] [--gpu-blocks N]"
            exit 1 ;;
    esac
done

# 加载数据集配置
load_dataset_config "$DATASET"
[[ $NUM_GPU_BLOCKS_OVERRIDE_SET -eq 0 ]] && NUM_GPU_BLOCKS_OVERRIDE="${DATASET_GPU_BLOCKS:-$NUM_GPU_BLOCKS_OVERRIDE}"

# 检查数据集
generate_dataset_if_needed "$TRACE" "$FULL_TRACE" "$DATASET" || exit 1

# pcie-full / pcie-trace-a-light / pcie-multiturn 特殊处理
RUNNER_TIMEOUT_ARGS=(--timeout "$TIMEOUT" --request-timeout "$REQUEST_TIMEOUT")
if [[ "$DATASET" == "pcie-full" || "$DATASET" == "pcie-trace-a-light" || "$DATASET" == "pcie-multiturn" ]]; then
    if [[ ! -f "$TRACE" ]]; then
        echo "❌ $DATASET: trace 不存在: $TRACE"
        exit 1
    fi
    # pcie-multiturn 已在 datasets.env 中设置 NUM_CONV，无需自动统计
    if [[ "$NUM_CONV" -eq 0 ]]; then
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
    else
        echo "✓ $DATASET: NUM_CONV=$NUM_CONV（来自 datasets.env）"
    fi
    REQUEST_TIMEOUT=360
    TIMEOUT=""
    RUNNER_TIMEOUT_ARGS=(--request-timeout "$REQUEST_TIMEOUT")
fi

# 可选: 限制单请求最大生成 token 数
MAX_OUTPUT_ARGS=()
if [[ -n "$MAX_OUTPUT" ]]; then
    MAX_OUTPUT_ARGS=(--max-output-tokens "$MAX_OUTPUT")
    echo "✓ Max output tokens: $MAX_OUTPUT"
fi

# 路径设置
RUN_EXP_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

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

EXP_ID="${EXP_TS}_ablation_${DATASET}_q${QPS}_lead${PREFETCH_LEAD_TIME}_${MODEL_TAG}"
EXP_ID="${EXP_ID//\//_}"
EXP_ID="${EXP_ID// /_}"

mkdir -p "$REPO_ROOT/results"
RESULTS_DIR="$REPO_ROOT/results/$EXP_ID"
mkdir -p "$RESULTS_DIR"

# Profiler 目录
RUN_EXP_ROOT="$(pwd)"
if [[ "$PCIE_PROFILER_DIR" != /* ]]; then
    PCIE_PROFILER_DIR="$RUN_EXP_ROOT/${PCIE_PROFILER_DIR#./}"
fi
mkdir -p "$PCIE_PROFILER_DIR"

print_separator
echo "PCIe Ablation Experiment (4 groups: G3/G2/G1/G0)"
print_separator
echo "Experiment ID: $EXP_ID"
echo "Dataset: $DATASET"
echo "Trace: $TRACE"
echo "Num conversations: $NUM_CONV"
echo "QPS: $QPS"
echo "Prefetch lead time: ${PREFETCH_LEAD_TIME}s"
echo "GPU blocks: $NUM_GPU_BLOCKS_OVERRIDE"
echo "Run directory: $RESULTS_DIR"
print_separator
echo ""
echo "Groups:"
echo "  G3: PCIe Scheduler + PP Phase-Aware   (start_vllm_pcie.sh --pcie-scheduler)"
echo "  G2: PCIe Scheduler, no Phase-Aware    (start_vllm_pcie.sh --pcie-scheduler --no-pp-phase-aware)"
echo "  G1: No Scheduler, with Prefetch       (start_vllm_pcie.sh)"
echo "  G0: No Scheduler, no Prefetch         (start_vllm_pcie.sh, --mode baseline)"
print_separator

# Helper: 后台捕获 vLLM 日志到实验目录（实时镜像）
VLLM_TAIL_PID=""
start_vllm_log_capture() {
    local dest="$1"
    stop_vllm_log_capture
    if [[ -f "$VLLM_LOG" ]]; then
        tail -f "$VLLM_LOG" > "$dest" 2>/dev/null &
        VLLM_TAIL_PID=$!
        echo "✓ Capturing vLLM log → $dest (pid $VLLM_TAIL_PID)"
    fi
}
stop_vllm_log_capture() {
    if [[ -n "$VLLM_TAIL_PID" ]] && kill -0 "$VLLM_TAIL_PID" 2>/dev/null; then
        kill "$VLLM_TAIL_PID" 2>/dev/null || true
        wait "$VLLM_TAIL_PID" 2>/dev/null || true
        VLLM_TAIL_PID=""
    fi
}

# ============================================================
# Helper: run one experiment phase
# ============================================================
run_phase() {
    local GROUP="$1"      # e.g. G3, G2, G1, G0
    local MODE="$2"       # prefetch or baseline
    local SUFFIX="$3"     # output file suffix

    echo "Resetting prefix cache..."
    curl -s -X POST "http://localhost:$API_PORT/reset_prefix_cache?reset_external=true" >/dev/null || true
    sleep 5

    echo "Clearing stale PCIe event files..."
    rm -f "$PCIE_PROFILER_DIR"/pcie_events_*.json 2>/dev/null || true
    echo "Starting PCIe profiler..."
    curl -s -X POST "http://localhost:$API_PORT/start_profile" >/dev/null || true

    python3 prefetch_ab_runner.py \
        --trace-file "$TRACE" \
        --mode "$MODE" \
        --qps "$QPS" \
        --num-multi-turn "$NUM_CONV" \
        --model "$MODEL_PATH" \
        --api-base "http://localhost:$API_PORT/v1" \
        --output "$RESULTS_DIR/prefetch_${SUFFIX}.jsonl" \
        --seed "$SEED" \
        "${RUNNER_TIMEOUT_ARGS[@]}" \
        "${MAX_OUTPUT_ARGS[@]}" \
        --prefetch-lead-time "$PREFETCH_LEAD_TIME" \
        --schedule-mode "$SCHEDULE_MODE" \
        2>&1 | tee "$RESULTS_DIR/prefetch_${SUFFIX}.log"

    echo "✓ $GROUP completed"

    echo "Flushing PCIe profiler..."
    curl -s -X POST "http://localhost:$API_PORT/stop_profile" >/dev/null || true
    sleep 3

    # 收集 PCIe 事件
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
with open('$RESULTS_DIR/pcie_events_${SUFFIX}.json', 'w') as fp:
    json.dump(events, fp, indent=2)
print(f'Merged {len(events)} events')
"
        else
            cp "$PCIE_PROFILER_DIR/pcie_events_0.json" "$RESULTS_DIR/pcie_events_${SUFFIX}.json"
        fi
    fi
    # 清理 profiler 文件
    rm -f "$PCIE_PROFILER_DIR"/pcie_events_*.json 2>/dev/null || true
}

# Helper: prompt user to restart vLLM
prompt_restart() {
    local MSG="$1"
    local LOG_DEST="$2"
    stop_vllm_log_capture
    echo ""
    print_separator
    echo "⚠️  Please RESTART vLLM:"
    echo "   $MSG"
    echo "   Press Enter here to continue..."
    print_separator
    read -r

    if ! check_vllm_running; then
        echo "❌ Error: vLLM not running."
        exit 1
    fi
    [[ -n "$LOG_DEST" ]] && start_vllm_log_capture "$LOG_DEST"
}

# ------------------------------------------------------------------
# Phase 1: G3 — Full scheduling (PCIe Scheduler + PP Phase-Aware)
# ------------------------------------------------------------------
print_phase "[Phase 1/5] G3: Prefetch + PCIe Scheduler + PP Phase-Aware"

if ! check_vllm_running; then
    echo "❌ Error: vLLM not running."
    echo "Start with: ./start_vllm_pcie.sh --pcie-scheduler"
    exit 1
fi

start_vllm_log_capture "$RESULTS_DIR/vllm_log_g3.log"
run_phase "G3" "prefetch" "g3_full_sched"

# ------------------------------------------------------------------
# Phase 2: G2 — Scheduler ON, Phase-Aware OFF
# ------------------------------------------------------------------
prompt_restart "1. Stop current vLLM (Ctrl+C)
   2. Start: ./start_vllm_pcie.sh --pcie-scheduler --no-pp-phase-aware
   3. Press Enter here to continue Phase 2" \
   "$RESULTS_DIR/vllm_log_g2.log"

print_phase "[Phase 2/5] G2: Prefetch + PCIe Scheduler, NO PP Phase-Aware"

run_phase "G2" "prefetch" "g2_sched_no_phase"

# ------------------------------------------------------------------
# Phase 3: G1 — No Scheduler, with Prefetch
# ------------------------------------------------------------------
prompt_restart "1. Stop current vLLM (Ctrl+C)
   2. Start: ./start_vllm_pcie.sh
   3. Press Enter here to continue Phase 3" \
   "$RESULTS_DIR/vllm_log_g1_g0.log"

print_phase "[Phase 3/5] G1: Prefetch, no PCIe Scheduler (baseline prefetch)"

run_phase "G1" "prefetch" "g1_prefetch_only"

# ------------------------------------------------------------------
# Phase 4: G0 — No Scheduler, No Prefetch (original vLLM)
# ------------------------------------------------------------------
# G0 uses the same vLLM instance as G1 (no scheduler), just --mode baseline
print_phase "[Phase 4/5] G0: No Prefetch, no Scheduler (original vLLM baseline)"

echo "Resetting prefix cache before G0..."
curl -s -X POST "http://localhost:$API_PORT/reset_prefix_cache?reset_external=true" >/dev/null || true
sleep 5

run_phase "G0" "baseline" "g0_no_prefetch"

stop_vllm_log_capture

# ------------------------------------------------------------------
# Phase 5: Generate Report
# ------------------------------------------------------------------
print_phase "[Phase 5/5] Generating ablation comparison report..."

CONFIG_TMP=$(mktemp)
cleanup_phase5_tmp() {
    rm -f "$CONFIG_TMP"
}
trap cleanup_phase5_tmp EXIT

DATASET="$DATASET" QPS="$QPS" PREFETCH_LEAD_TIME="$PREFETCH_LEAD_TIME" \
    NUM_GPU_BLOCKS_OVERRIDE="$NUM_GPU_BLOCKS_OVERRIDE" TRACE="$TRACE" FULL_TRACE="$FULL_TRACE" \
    NUM_CONV="$NUM_CONV" MODEL_PATH="$MODEL_PATH" \
    TIMEOUT="${TIMEOUT:-}" REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-}" \
    RUN_EXPERIMENT_DIR="$RUN_EXP_ROOT" bash "$RUN_EXP_ROOT/dump_config.sh" 2>/dev/null \
    | grep -E '^[A-Za-z_][A-Za-z0-9_]*=' > "$CONFIG_TMP" || true

# Generate per-pair TTFT reports (G3 vs G1 as the primary comparison)
if [[ -f "$RESULTS_DIR/prefetch_g3_full_sched.jsonl" && -f "$RESULTS_DIR/prefetch_g1_prefetch_only.jsonl" ]]; then
    python3 ../result-analysis/generate_report.py \
        --baseline "$RESULTS_DIR/prefetch_g1_prefetch_only.jsonl" \
        --prefetch "$RESULTS_DIR/prefetch_g3_full_sched.jsonl" \
        --output "$RESULTS_DIR/ttft_report_g3_vs_g1.md" \
        --config-file "$CONFIG_TMP" \
        2>/dev/null && echo "✓ TTFT report: G3 vs G1" || echo "⚠️  generate_report.py failed"
fi

# Generate ablation summary
ABLATION_MD="$RESULTS_DIR/ablation_report.md"
{
    echo "# PCIe Scheduling Ablation Report"
    echo ""
    echo "**Experiment**: $EXP_ID"
    echo "**Dataset**: $DATASET | **QPS**: $QPS | **Lead Time**: ${PREFETCH_LEAD_TIME}s | **GPU Blocks**: $NUM_GPU_BLOCKS_OVERRIDE"
    echo ""
    echo "## Groups"
    echo ""
    echo "| Group | PCIe Scheduler | PP Phase-Aware | Prefetch | Output |"
    echo "|-------|---------------|----------------|----------|--------|"
    echo "| G3 | ON | ON | ON | prefetch_g3_full_sched.jsonl |"
    echo "| G2 | ON | OFF | ON | prefetch_g2_sched_no_phase.jsonl |"
    echo "| G1 | OFF | - | ON | prefetch_g1_prefetch_only.jsonl |"
    echo "| G0 | OFF | - | OFF | prefetch_g0_no_prefetch.jsonl |"
    echo ""

    # Extract TTFT stats for each group
    echo "## TTFT Summary"
    echo ""
    echo "| Group | Requests | Mean TTFT (ms) | P50 | P95 | P99 | Std |"
    echo "|-------|----------|----------------|-----|-----|-----|-----|"
    for SUFFIX in g3_full_sched g2_sched_no_phase g1_prefetch_only g0_no_prefetch; do
        JSONL="$RESULTS_DIR/prefetch_${SUFFIX}.jsonl"
        if [[ -f "$JSONL" ]]; then
            python3 -c "
import json, sys
import numpy as np

ttfts = []
with open('$JSONL') as f:
    for line in f:
        d = json.loads(line)
        if 'ttft_ms' in d and d['ttft_ms'] is not None:
            ttfts.append(d['ttft_ms'])
if not ttfts:
    print('| $SUFFIX | 0 | - | - | - | - | - |')
else:
    arr = np.array(ttfts)
    print(f'| $SUFFIX | {len(arr)} | {np.mean(arr):.1f} | {np.percentile(arr, 50):.1f} | {np.percentile(arr, 95):.1f} | {np.percentile(arr, 99):.1f} | {np.std(arr):.1f} |')
"
        else
            echo "| $SUFFIX | - | - | - | - | - | - |"
        fi
    done
} > "$ABLATION_MD"
echo "✓ Ablation report: $ABLATION_MD"

cleanup_phase5_tmp
trap - EXIT

# Append 4 rows (G0-G3) to the ablation master TSV table
ABLATION_TABLE_TSV="$REPO_ROOT/results/pcie_ablation_experiments.txt"
if [[ -f "../result-analysis/append_pcie_ablation_summary_row.py" ]]; then
    python3 ../result-analysis/append_pcie_ablation_summary_row.py \
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
        --table "$ABLATION_TABLE_TSV" \
        && echo "✓ Appended 4 rows to $ABLATION_TABLE_TSV" || \
        echo "⚠️  append_pcie_ablation_summary_row.py failed"
else
    echo "⚠️  append_pcie_ablation_summary_row.py not found, skip TSV append"
fi

echo ""
echo "✅ PCIe Ablation experiment complete!"
echo ""
echo "Results:"
echo "  Run directory:    $RESULTS_DIR"
echo "  Ablation report:  $ABLATION_MD"
echo "  Master TSV:       $ABLATION_TABLE_TSV"
echo "  Per-group files:  prefetch_g{0,1,2,3}_*.{jsonl,log}, pcie_events_g{0,1,2,3}_*.json, vllm_log_*.log"
print_separator
