#!/bin/bash
# 机制消融实验：逐一去掉调度器的三个核心子机制
#
# Groups:
#   full:   完整调度（Priority heap + CC=2 + Evict-first, PP-Phase OFF）
#   no-pq:  去掉优先队列（FIFO instead of heap）
#   no-ef:  去掉 Evict-first（H2D before D2H）
#   no-cc:  去掉并发控制（CC=999）
#   g1:     无调度器，有 Prefetch
#   g0:     无调度器，无 Prefetch（原始 vLLM）
#
# 用法: ./auto_run_mechanism_ablation.sh [dataset] [options]
# 示例:
#   ./auto_run_mechanism_ablation.sh pcie-heavy --qps 2.0 --gpu-blocks 1000
#   ./auto_run_mechanism_ablation.sh pcie-heavy --qps 2.0 --gpu-blocks 1000 --groups no-pq,no-ef,no-cc
#   ./auto_run_mechanism_ablation.sh pcie-heavy --qps 2.0 --gpu-blocks 1000 --groups all

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
            echo "vLLM is ready (${waited}s)"
            sleep 5
            return 0
        fi
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo ""
            echo "vLLM process died during startup. Check: $startup_log"
            return 1
        fi
        echo -n "."
        sleep 3
        waited=$((waited + 3))
    done

    echo ""
    echo "vLLM failed to start within ${max_wait}s. Check: $startup_log"
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

    pkill -f "vllm serve" 2>/dev/null || true
    sleep 3
    echo "vLLM stopped"
}

cleanup() {
    echo ""
    echo "Cleaning up..."
    stop_vllm
}
trap cleanup EXIT INT TERM

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
        echo "PCIe events: $output_json"
    else
        echo "No pcie_events_*.json found for $suffix"
    fi

    rm -f "$PCIE_PROFILER_DIR"/pcie_events_*.json 2>/dev/null || true
}

# ============================================================
# 加载配置
# ============================================================

source config/system.env
source config/datasets.env
source config/experiments.env
source scripts/utils/common.sh

DATASET="${1:-pcie-heavy}"
shift 2>/dev/null || true

NUM_GPU_BLOCKS_OVERRIDE_SET=0
RUN_GROUPS="no-pq,no-ef,no-cc"  # 默认只跑 3 个新消融组
OPEN_LOOP=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --qps)             QPS="$2"; shift 2 ;;
        --lead-time)       PREFETCH_LEAD_TIME="$2"; shift 2 ;;
        --gpu-blocks)      NUM_GPU_BLOCKS_OVERRIDE="$2"; NUM_GPU_BLOCKS_OVERRIDE_SET=1; shift 2 ;;
        --open-loop)       OPEN_LOOP=1; shift ;;
        --groups)
            _g="$(echo "$2" | tr '[:upper:]' '[:lower:]')"
            if [[ "$_g" == "all" ]]; then
                RUN_GROUPS="g0,g1,full,no-pq,no-ef,no-cc"
            else
                RUN_GROUPS="$_g"
            fi
            shift 2 ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [dataset] [--qps N] [--lead-time N] [--gpu-blocks N] [--open-loop] [--groups g0,g1,full,no-pq,no-ef,no-cc|all]"
            exit 1 ;;
    esac
done

should_run_group() {
    [[ ",$RUN_GROUPS," == *",$1,"* ]]
}

load_dataset_config "$DATASET"
[[ $NUM_GPU_BLOCKS_OVERRIDE_SET -eq 0 ]] && NUM_GPU_BLOCKS_OVERRIDE="${DATASET_GPU_BLOCKS:-$NUM_GPU_BLOCKS_OVERRIDE}"

generate_dataset_if_needed "$TRACE" "$FULL_TRACE" "$DATASET" || exit 1

# Timeout 配置
RUNNER_TIMEOUT_ARGS=(--timeout "$TIMEOUT" --request-timeout "$REQUEST_TIMEOUT")
if [[ "$DATASET" == "pcie-full" || "$DATASET" == "pcie-trace-a-light" || "$DATASET" == "pcie-multiturn" || "$DATASET" == "pcie-heavy" ]]; then
    if [[ ! -f "$TRACE" ]]; then
        echo "$DATASET: trace does not exist: $TRACE"
        exit 1
    fi
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
            echo "$DATASET: cannot count multi-turn roots from trace: $TRACE"
            exit 1
        fi
        echo "$DATASET: NUM_CONV=$NUM_CONV (from trace)"
    else
        echo "$DATASET: NUM_CONV=$NUM_CONV (from datasets.env)"
    fi
    REQUEST_TIMEOUT=360
    TIMEOUT=""
    RUNNER_TIMEOUT_ARGS=(--request-timeout "$REQUEST_TIMEOUT")
fi

MAX_OUTPUT_ARGS=()
if [[ -n "$MAX_OUTPUT" ]]; then
    MAX_OUTPUT_ARGS=(--max-output-tokens "$MAX_OUTPUT")
fi

# ============================================================
# 路径设置
# ============================================================

REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

[[ "$VLLM_LOG" != /* ]] && VLLM_LOG="$RUN_EXP_DIR/$VLLM_LOG"
if [[ "$PCIE_PROFILER_DIR" != /* ]]; then
    PCIE_PROFILER_DIR="$RUN_EXP_DIR/${PCIE_PROFILER_DIR#./}"
fi
mkdir -p "$PCIE_PROFILER_DIR"

EXP_TS="$(date +%Y%m%d_%H%M%S)"
MODEL_TAG="$(python3 -c "
import re, sys
p = sys.argv[1].lower()
checks = [
    (r'(?<![0-9.])72b(?![0-9])', '72B'),
    (r'(?<![0-9.])32b(?![0-9])', '32B'),
    (r'(?<![0-9.])14b(?![0-9])', '14B'),
    (r'(?<![0-9.])8b(?![0-9])', '8B'),
]
for pat, lab in checks:
    if re.search(pat, p):
        print(lab)
        break
else:
    print('unknown')
" "$MODEL_PATH")"

EXP_ID="${EXP_TS}_mech_ablation_${DATASET}_q${QPS}_blk${NUM_GPU_BLOCKS_OVERRIDE}_${MODEL_TAG}"
EXP_ID="${EXP_ID//\//_}"
EXP_ID="${EXP_ID// /_}"

mkdir -p "$REPO_ROOT/results"
RESULTS_DIR="$REPO_ROOT/results/$EXP_ID"
mkdir -p "$RESULTS_DIR"

# 打印配置
print_separator
echo "Mechanism Ablation Experiment"
print_separator
echo "Experiment ID: $EXP_ID"
LOOP_LABEL="closed-loop"
[[ "${OPEN_LOOP:-0}" -eq 1 ]] && LOOP_LABEL="open-loop"
echo "Dataset: $DATASET | QPS: $QPS | Lead: ${PREFETCH_LEAD_TIME}s | Blocks: $NUM_GPU_BLOCKS_OVERRIDE | Scheduling: $LOOP_LABEL"
echo "Groups: $RUN_GROUPS"
echo "Results: $RESULTS_DIR"
print_separator
echo ""
ACTIVE_GROUP_COUNT=0
echo "Groups:"
should_run_group full   && echo "  full:   Priority heap, CC=2, Evict-first ON    (complete scheduler)" && ((ACTIVE_GROUP_COUNT++)) || true
should_run_group no-pq  && echo "  no-pq:  FIFO queue, CC=2, Evict-first ON      (no priority)"       && ((ACTIVE_GROUP_COUNT++)) || true
should_run_group no-ef  && echo "  no-ef:  Priority heap, CC=2, Evict-first OFF   (no evict-first)"    && ((ACTIVE_GROUP_COUNT++)) || true
should_run_group no-cc  && echo "  no-cc:  Priority heap, CC=999, Evict-first ON  (no concurrency ctrl)" && ((ACTIVE_GROUP_COUNT++)) || true
should_run_group g1     && echo "  g1:     no scheduler, prefetch only"                                 && ((ACTIVE_GROUP_COUNT++)) || true
should_run_group g0     && echo "  g0:     no scheduler, no prefetch (baseline)"                        && ((ACTIVE_GROUP_COUNT++)) || true
echo "Active groups: $ACTIVE_GROUP_COUNT"
print_separator

if [[ "$ACTIVE_GROUP_COUNT" -eq 0 ]]; then
    echo "❌ Error: No groups matched RUN_GROUPS='$RUN_GROUPS'"
    echo "   Valid groups: full, no-pq, no-ef, no-cc, g1, g0 (or 'all')"
    exit 1
fi

# ============================================================
# run_phase: 运行单个实验阶段
# ============================================================

run_phase() {
    local GROUP="$1"
    local MODE="$2"
    local SUFFIX="$3"

    echo "Resetting prefix cache..."
    curl -s -X POST "http://localhost:$API_PORT/reset_prefix_cache?reset_external=true" >/dev/null || true
    sleep 5

    echo "Clearing stale PCIe event files..."
    rm -f "$PCIE_PROFILER_DIR"/pcie_events_*.json 2>/dev/null || true
    echo "Starting PCIe profiler..."
    curl -s -X POST "http://localhost:$API_PORT/start_profile" >/dev/null || true

    OPEN_LOOP_ARGS=()
    [[ "${OPEN_LOOP:-0}" -eq 1 ]] && OPEN_LOOP_ARGS=(--open-loop)

    local run_status=0
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
        "${OPEN_LOOP_ARGS[@]}" \
        2>&1 | tee "$RESULTS_DIR/prefetch_${SUFFIX}.log" || run_status=$?

    if [[ $run_status -ne 0 ]]; then
        echo "WARNING: $GROUP runner exited with status $run_status"
        if grep -qiE "out of memory|CUDA OOM|block.*exhaust" "$RESULTS_DIR/prefetch_${SUFFIX}.log" 2>/dev/null; then
            echo "DETECTED: OOM or block exhaustion in $GROUP"
            echo "OOM_DETECTED=true" >> "$RESULTS_DIR/prefetch_${SUFFIX}.meta"
        fi
    fi

    echo "$GROUP completed (exit=$run_status)"

    echo "Flushing PCIe profiler..."
    curl -s -X POST "http://localhost:$API_PORT/stop_profile" >/dev/null || true
    sleep 3

    collect_pcie_events "$SUFFIX"
    return $run_status
}

# ============================================================
# 启动辅助: 带调度参数启动 vLLM
# ============================================================

start_vllm_scheduler() {
    local label="$1"
    shift
    # 额外 env vars 通过参数传入（KEY=VAL 格式）
    local env_args=("$@")

    # 清除所有消融 env vars
    unset PCIE_MAX_CONCURRENT_H2D PCIE_NO_PRIORITY_QUEUE PCIE_NO_EVICT_FIRST

    # 设置本组需要的 env vars
    for kv in "${env_args[@]}"; do
        export "$kv"
        echo "  env: $kv"
    done

    start_vllm "$label" --pcie-scheduler --no-pp-phase-aware --gpu-blocks "$NUM_GPU_BLOCKS_OVERRIDE" --log-file "$RESULTS_DIR/vllm_log_${label}.log" || return 1

    # 清除 env vars
    unset PCIE_MAX_CONCURRENT_H2D PCIE_NO_PRIORITY_QUEUE PCIE_NO_EVICT_FIRST
}

# ============================================================
# 执行各组（每组独立 vLLM 实例）
# ============================================================

FAILED_GROUPS=()

# --- full: 完整调度 ---
if should_run_group full; then
    start_vllm_scheduler full || exit 1
    print_phase "[full] Complete scheduler"
    run_phase "full" "prefetch" "full_sched" || FAILED_GROUPS+=(full)
    stop_vllm
fi

# --- no-pq: 去掉优先队列 ---
if should_run_group no-pq; then
    start_vllm_scheduler no-pq "PCIE_NO_PRIORITY_QUEUE=1" || exit 1
    print_phase "[no-pq] No priority queue (FIFO mode)"
    run_phase "no-pq" "prefetch" "no_pq" || FAILED_GROUPS+=(no-pq)
    stop_vllm
fi

# --- no-ef: 去掉 Evict-first ---
if should_run_group no-ef; then
    start_vllm_scheduler no-ef "PCIE_NO_EVICT_FIRST=1" || exit 1
    print_phase "[no-ef] No evict-first (H2D before D2H)"
    run_phase "no-ef" "prefetch" "no_ef" || FAILED_GROUPS+=(no-ef)
    stop_vllm
fi

# --- no-cc: 去掉并发控制 ---
if should_run_group no-cc; then
    start_vllm_scheduler no-cc "PCIE_MAX_CONCURRENT_H2D=999" || exit 1
    print_phase "[no-cc] No concurrency control (CC=999)"
    run_phase "no-cc" "prefetch" "no_cc" || FAILED_GROUPS+=(no-cc)
    stop_vllm
fi

# --- g1 / g0: 无调度器 ---
if should_run_group g1 || should_run_group g0; then
    start_vllm g1_g0 --gpu-blocks "$NUM_GPU_BLOCKS_OVERRIDE" --log-file "$RESULTS_DIR/vllm_log_g1_g0.log" || exit 1
fi

if should_run_group g1; then
    print_phase "[g1] Prefetch only, no scheduler"
    run_phase "g1" "prefetch" "g1_prefetch_only" || FAILED_GROUPS+=(g1)
fi

if should_run_group g0; then
    print_phase "[g0] Original vLLM baseline"
    curl -s -X POST "http://localhost:$API_PORT/reset_prefix_cache?reset_external=true" >/dev/null || true
    sleep 5
    run_phase "g0" "baseline" "g0_no_prefetch" || FAILED_GROUPS+=(g0)
fi

if should_run_group g1 || should_run_group g0; then
    stop_vllm
fi

# ============================================================
# 报告生成
# ============================================================

print_phase "Generating mechanism ablation report..."

# Build suffix list based on RUN_GROUPS
REPORT_SUFFIXES=()
should_run_group full   && REPORT_SUFFIXES+=(full_sched)
should_run_group no-pq  && REPORT_SUFFIXES+=(no_pq)
should_run_group no-ef  && REPORT_SUFFIXES+=(no_ef)
should_run_group no-cc  && REPORT_SUFFIXES+=(no_cc)
should_run_group g1     && REPORT_SUFFIXES+=(g1_prefetch_only)
should_run_group g0     && REPORT_SUFFIXES+=(g0_no_prefetch)

ABLATION_MD="$RESULTS_DIR/mechanism_ablation_report.md"
{
    echo "# Mechanism Ablation Report"
    echo ""
    echo "**Experiment**: $EXP_ID"
    echo "**Dataset**: $DATASET | **QPS**: $QPS | **Lead Time**: ${PREFETCH_LEAD_TIME}s | **GPU Blocks**: $NUM_GPU_BLOCKS_OVERRIDE"
    echo ""
    echo "## Groups"
    echo ""
    echo "| Group | Scheduler | Priority | CC (h2d) | Evict-first | Output |"
    echo "|-------|:---------:|:--------:|:--------:|:-----------:|--------|"
    should_run_group full   && echo "| full | ON | heap | 2 | yes | prefetch_full_sched.jsonl |"
    should_run_group no-pq  && echo "| no-pq | ON | FIFO | 2 | yes | prefetch_no_pq.jsonl |"
    should_run_group no-ef  && echo "| no-ef | ON | heap | 2 | no | prefetch_no_ef.jsonl |"
    should_run_group no-cc  && echo "| no-cc | ON | heap | 999 | yes | prefetch_no_cc.jsonl |"
    should_run_group g1     && echo "| g1 | OFF | - | - | - | prefetch_g1_prefetch_only.jsonl |"
    should_run_group g0     && echo "| g0 | OFF | - | - | - | prefetch_g0_no_prefetch.jsonl |"
    echo ""

    echo "## TTFT Summary"
    echo ""
    echo "| Group | Requests | Mean TTFT (ms) | P50 | P95 | P99 | Std |"
    echo "|-------|----------|----------------|-----|-----|-----|-----|"
    for SUFFIX in "${REPORT_SUFFIXES[@]}"; do
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

    if [[ ${#FAILED_GROUPS[@]} -gt 0 ]]; then
        echo ""
        echo "## Failed Groups"
        for g in "${FAILED_GROUPS[@]}"; do
            echo "- **$g**"
        done
    fi
} > "$ABLATION_MD"
echo "Ablation report: $ABLATION_MD"

# 追加到机制消融专用 TSV（列与 pcie_ablation_experiments 一致，group 为 full / no-pq / …）
MECH_ABLATION_TABLE_TSV="$REPO_ROOT/results/pcie_mechanism_ablation_experiments.txt"
if [[ -f "../result-analysis/append_pcie_mechanism_ablation_summary_row.py" ]]; then
    python3 ../result-analysis/append_pcie_mechanism_ablation_summary_row.py \
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
        --groups "$RUN_GROUPS" \
        --table "$MECH_ABLATION_TABLE_TSV" \
        && echo "✓ Appended mechanism ablation rows to $MECH_ABLATION_TABLE_TSV" || \
        echo "⚠️  append_pcie_mechanism_ablation_summary_row.py failed"
else
    echo "⚠️  append_pcie_mechanism_ablation_summary_row.py not found, skip mechanism TSV append"
fi

# ============================================================
# 完成
# ============================================================

print_separator
echo "Mechanism Ablation Experiment Complete!"
echo "  Experiment ID:    $EXP_ID"
echo "  Results:          $RESULTS_DIR"
echo "  Report:           $ABLATION_MD"
echo "  Mechanism TSV:    $MECH_ABLATION_TABLE_TSV"
if [[ ${#FAILED_GROUPS[@]} -gt 0 ]]; then
    echo "  Failed groups:    ${FAILED_GROUPS[*]}"
fi
print_separator
