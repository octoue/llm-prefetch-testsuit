#!/bin/bash
# 参数化机制消融实验脚本
# 支持 32B/72B 模型切换，多轮 QPS 遍历，closed/open loop 选择
#
# Groups:
#   full:   完整调度（Priority heap + CC=2 + Evict-first, PP-Phase OFF）
#   no-pq:  去掉优先队列（FIFO instead of heap）
#   no-ef:  去掉 Evict-first（H2D before D2H）
#   no-cc:  去掉并发控制（CC=999）
#   g1:     无调度器，有 Prefetch
#   g0:     无调度器，无 Prefetch（原始 vLLM）
#
# 用法:
#   ./auto_run_mechanism_ablation_param.sh --model 32b --qps 0.5,1.0,1.5 --rounds 3 --num-blocks 1000 --pp 2 --dataset pcie-heavy --closed-loop
#   ./auto_run_mechanism_ablation_param.sh --model 72b --qps 0.5,1.0 --rounds 2 --num-blocks 750 --pp 4 --dataset pcie-heavy --closed-loop
#   ./auto_run_mechanism_ablation_param.sh --model 72b --qps 1.0 --rounds 1 --num-blocks 750 --pp 4 --dataset pcie-heavy --closed-loop --groups full,g0

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_EXP_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$RUN_EXP_DIR"

# ============================================================
# 默认值
# ============================================================

MODEL_SIZE=""
QPS_LIST=""
ROUNDS=1
NUM_BLOCKS=""
PP_SIZE=""
DATASET="pcie-heavy"
LOOP_MODE="closed"   # closed 或 open
RUN_GROUPS="full,g1,g0"
MAX_REQUESTS=""

# 模型路径配置（按需修改）
MODEL_PATH_32B="/lpai/models/Qwen__Qwen3-32B/25-07-26-0345"
MODEL_PATH_72B="/lpai/models/qwen__qwen2_5-72b/24-09-25-1228"  # TODO: 填写 72B 模型路径

# ============================================================
# 参数解析
# ============================================================

usage() {
    cat <<'USAGE'
Usage: ./auto_run_mechanism_ablation_param.sh [options]

Required:
  --model MODEL          模型大小: 32b 或 72b
  --qps QPS_LIST         QPS 列表, 逗号分隔 (e.g., 0.5,1.0,1.5,2.0)
  --num-blocks N         GPU blocks 数量

Optional:
  --rounds N             测试轮数, 每轮遍历所有 QPS (default: 1)
  --pp N                 Pipeline parallel size (default: 32b→2, 72b→4)
  --dataset NAME         数据集名称 (default: pcie-heavy)
  --closed-loop          使用 closed-loop 调度 (default)
  --open-loop            使用 open-loop 调度
  --groups GROUPS        运行的组, 逗号分隔或 'all' (default: all)
  --max-requests N       最大请求数
  --lead-time N          Prefetch 提前时间(秒) (default: 2.0)
  --model-path PATH      手动指定模型路径 (覆盖内置路径)
  --gpu-mem-util F       GPU 显存利用率 (default: 0.7)
  --max-num-seqs N       最大并发序列数 (default: 96)
  --kv-offloading-size N KV offloading 大小 GiB (default: 10)
USAGE
    exit 1
}

# 可覆盖的额外参数
LEAD_TIME_OVERRIDE=""
MODEL_PATH_OVERRIDE=""
GPU_MEM_UTIL_OVERRIDE=""
MAX_NUM_SEQS_OVERRIDE=""
KV_OFFLOADING_SIZE_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL_SIZE="$(echo "$2" | tr '[:upper:]' '[:lower:]')"
            shift 2 ;;
        --qps)
            QPS_LIST="$2"
            shift 2 ;;
        --rounds)
            ROUNDS="$2"
            shift 2 ;;
        --num-blocks)
            NUM_BLOCKS="$2"
            shift 2 ;;
        --pp)
            PP_SIZE="$2"
            shift 2 ;;
        --dataset)
            DATASET="$2"
            shift 2 ;;
        --closed-loop)
            LOOP_MODE="closed"
            shift ;;
        --open-loop)
            LOOP_MODE="open"
            shift ;;
        --groups)
            _g="$(echo "$2" | tr '[:upper:]' '[:lower:]')"
            if [[ "$_g" == "all" ]]; then
                RUN_GROUPS="full,no-pq,no-ef,no-cc,g1,g0"  # all 仍包含全部组
            else
                RUN_GROUPS="$_g"
            fi
            shift 2 ;;
        --max-requests)
            MAX_REQUESTS="$2"
            shift 2 ;;
        --lead-time)
            LEAD_TIME_OVERRIDE="$2"
            shift 2 ;;
        --model-path)
            MODEL_PATH_OVERRIDE="$2"
            shift 2 ;;
        --gpu-mem-util)
            GPU_MEM_UTIL_OVERRIDE="$2"
            shift 2 ;;
        --max-num-seqs)
            MAX_NUM_SEQS_OVERRIDE="$2"
            shift 2 ;;
        --kv-offloading-size)
            KV_OFFLOADING_SIZE_OVERRIDE="$2"
            shift 2 ;;
        -h|--help)
            usage ;;
        *)
            echo "Unknown option: $1"
            usage ;;
    esac
done

# 参数校验
if [[ -z "$MODEL_SIZE" ]]; then
    echo "Error: --model is required (32b or 72b)"
    usage
fi
if [[ "$MODEL_SIZE" != "32b" && "$MODEL_SIZE" != "72b" ]]; then
    echo "Error: --model must be 32b or 72b, got: $MODEL_SIZE"
    exit 1
fi
if [[ -z "$QPS_LIST" ]]; then
    echo "Error: --qps is required (e.g., 0.5,1.0,1.5)"
    usage
fi
if [[ -z "$NUM_BLOCKS" ]]; then
    echo "Error: --num-blocks is required"
    usage
fi

# 解析 QPS 列表
IFS=',' read -ra QPS_ARRAY <<< "$QPS_LIST"

# ============================================================
# 根据模型设置默认值
# ============================================================

case "$MODEL_SIZE" in
    32b)
        [[ -z "$PP_SIZE" ]] && PP_SIZE=2
        MODEL_PATH="${MODEL_PATH_OVERRIDE:-$MODEL_PATH_32B}"
        MODEL_TAG="32B"
        ;;
    72b)
        [[ -z "$PP_SIZE" ]] && PP_SIZE=4
        MODEL_PATH="${MODEL_PATH_OVERRIDE:-$MODEL_PATH_72B}"
        MODEL_TAG="72B"
        ;;
esac

if [[ -z "$MODEL_PATH" ]]; then
    echo "Error: 模型路径未设置。请使用 --model-path 参数或编辑脚本中的 MODEL_PATH_72B 变量"
    exit 1
fi

# ============================================================
# vLLM 自动管理 (复用自 auto_run_mechanism_ablation.sh)
# ============================================================

VLLM_PID=""
_SAVED_API_PORT="${API_PORT:-}"

start_vllm() {
    local label="$1"; shift
    local args=("$@")

    print_phase "Starting vLLM [$label] args: ${args[*]}"

    if [[ -n "$VLLM_PID" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
        kill "$VLLM_PID" 2>/dev/null || true
        sleep 2
    fi

    local startup_log="$RESULTS_DIR/vllm_startup_${label}.log"
    bash "$RUN_EXP_DIR/start_vllm_pcie.sh" "${args[@]}" > "$startup_log" 2>&1 &
    VLLM_PID=$!
    echo "vLLM PID: $VLLM_PID  (log: $startup_log)"

    local max_wait=600  # 72B 模型加载更慢，增加等待时间
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

_kill_tree() {
    local pid="$1"
    local sig="${2:-TERM}"
    local children
    children=$(pgrep -P "$pid" 2>/dev/null) || true
    for child in $children; do
        _kill_tree "$child" "$sig"
    done
    kill -"$sig" "$pid" 2>/dev/null || true
}

stop_vllm() {
    if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "Stopping vLLM (PID: $VLLM_PID) and all child processes..."
        _kill_tree "$VLLM_PID" TERM

        local waited=0
        while kill -0 "$VLLM_PID" 2>/dev/null && [ $waited -lt 15 ]; do
            sleep 1
            waited=$((waited + 1))
        done

        if kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "Force killing process tree..."
            _kill_tree "$VLLM_PID" 9
        fi
        VLLM_PID=""
    fi

    sleep 3
    echo "vLLM stopped"
}

cleanup() {
    exec 1>&2 2>/dev/null
    echo ""
    echo "Cleaning up..."
    stop_vllm
    local pids
    pids=$(lsof -ti :"$API_PORT" 2>/dev/null) || true
    if [[ -n "$pids" ]]; then
        echo "Killing residual processes on port $API_PORT: $pids"
        echo "$pids" | xargs kill -9 2>/dev/null || true
        sleep 2
    fi
    source "$RUN_EXP_DIR/scripts/utils/gpu_lock.sh" 2>/dev/null && _clean_stale_locks 2>/dev/null || true
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

should_run_group() {
    [[ ",$RUN_GROUPS," == *",$1,"* ]]
}

# ============================================================
# 加载配置
# ============================================================

source config/system.env
source config/datasets.env
source config/experiments.env
source scripts/utils/common.sh

# 恢复端口覆盖
[[ -n "$_SAVED_API_PORT" ]] && API_PORT="$_SAVED_API_PORT"
API_PORT="${API_PORT:-8000}"
export API_PORT

# 用命令行参数覆盖配置
export MODEL_PATH
export VLLM_PIPELINE_PARALLEL_SIZE="$PP_SIZE"
[[ -n "$GPU_MEM_UTIL_OVERRIDE" ]] && GPU_MEMORY_UTILIZATION="$GPU_MEM_UTIL_OVERRIDE"
[[ -n "$MAX_NUM_SEQS_OVERRIDE" ]] && VLLM_MAX_NUM_SEQS="$MAX_NUM_SEQS_OVERRIDE"
[[ -n "$KV_OFFLOADING_SIZE_OVERRIDE" ]] && KV_OFFLOADING_SIZE="$KV_OFFLOADING_SIZE_OVERRIDE"
[[ -n "$LEAD_TIME_OVERRIDE" ]] && PREFETCH_LEAD_TIME="$LEAD_TIME_OVERRIDE"

NUM_GPU_BLOCKS_OVERRIDE="$NUM_BLOCKS"

OPEN_LOOP=0
[[ "$LOOP_MODE" == "open" ]] && OPEN_LOOP=1

# 加载数据集配置
load_dataset_config "$DATASET"
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
mkdir -p "$REPO_ROOT/results"

# ============================================================
# 打印总览
# ============================================================

LOOP_LABEL="closed-loop"
[[ "$OPEN_LOOP" -eq 1 ]] && LOOP_LABEL="open-loop"

print_separator
echo "Parameterized Mechanism Ablation Experiment"
print_separator
echo "Model:    $MODEL_TAG ($MODEL_PATH)"
echo "PP:       $PP_SIZE"
echo "Dataset:  $DATASET"
echo "QPS:      ${QPS_ARRAY[*]}"
echo "Rounds:   $ROUNDS"
echo "Blocks:   $NUM_GPU_BLOCKS_OVERRIDE"
echo "Loop:     $LOOP_LABEL"
echo "Groups:   $RUN_GROUPS"
echo "Mem Util: $GPU_MEMORY_UTILIZATION"
echo "Max Seqs: $VLLM_MAX_NUM_SEQS"
print_separator

ACTIVE_GROUP_COUNT=0
echo "Groups:"
should_run_group full   && echo "  full:   Priority heap, CC=2, Evict-first ON    (complete scheduler)" && ((ACTIVE_GROUP_COUNT++)) || true
should_run_group no-pq  && echo "  no-pq:  FIFO queue, CC=2, Evict-first ON      (no priority)"       && ((ACTIVE_GROUP_COUNT++)) || true
should_run_group no-ef  && echo "  no-ef:  Priority heap, CC=2, Evict-first OFF   (no evict-first)"    && ((ACTIVE_GROUP_COUNT++)) || true
should_run_group no-cc  && echo "  no-cc:  Priority heap, CC=999, Evict-first ON  (no concurrency ctrl)" && ((ACTIVE_GROUP_COUNT++)) || true
should_run_group g1     && echo "  g1:     no scheduler, prefetch only"                                 && ((ACTIVE_GROUP_COUNT++)) || true
should_run_group g0     && echo "  g0:     no scheduler, no prefetch (baseline)"                        && ((ACTIVE_GROUP_COUNT++)) || true
echo "Active groups: $ACTIVE_GROUP_COUNT"
echo "Total runs: $ROUNDS rounds x ${#QPS_ARRAY[@]} QPS values x $ACTIVE_GROUP_COUNT groups = $(( ROUNDS * ${#QPS_ARRAY[@]} * ACTIVE_GROUP_COUNT )) runs"
print_separator

if [[ "$ACTIVE_GROUP_COUNT" -eq 0 ]]; then
    echo "Error: No groups matched RUN_GROUPS='$RUN_GROUPS'"
    echo "  Valid groups: full, no-pq, no-ef, no-cc, g1, g0 (or 'all')"
    exit 1
fi

# ============================================================
# run_phase: 运行单个实验阶段
# ============================================================

run_phase() {
    local GROUP="$1"
    local MODE="$2"
    local SUFFIX="$3"
    local CUR_QPS="$4"

    echo "Resetting prefix cache..."
    curl -s -X POST "http://localhost:$API_PORT/reset_prefix_cache?reset_external=true" >/dev/null || true
    sleep 5

    echo "Clearing stale PCIe event files..."
    rm -f "$PCIE_PROFILER_DIR"/pcie_events_*.json 2>/dev/null || true
    echo "Starting PCIe profiler..."
    curl -s -X POST "http://localhost:$API_PORT/start_profile" >/dev/null || true

    OPEN_LOOP_ARGS=()
    [[ "${OPEN_LOOP:-0}" -eq 1 ]] && OPEN_LOOP_ARGS=(--open-loop)

    MAX_REQUESTS_ARGS=()
    [[ -n "$MAX_REQUESTS" ]] && MAX_REQUESTS_ARGS=(--max-requests "$MAX_REQUESTS")

    local run_status=0
    python3 prefetch_ab_runner.py \
        --trace-file "$TRACE" \
        --mode "$MODE" \
        --qps "$CUR_QPS" \
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
        "${MAX_REQUESTS_ARGS[@]}" \
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
    local env_args=("$@")

    unset PCIE_MAX_CONCURRENT_H2D PCIE_NO_PRIORITY_QUEUE PCIE_NO_EVICT_FIRST

    for kv in "${env_args[@]}"; do
        export "$kv"
        echo "  env: $kv"
    done

    start_vllm "$label" --pcie-scheduler --no-pp-phase-aware --gpu-blocks "$NUM_GPU_BLOCKS_OVERRIDE" --log-file "$RESULTS_DIR/vllm_log_${label}.log" || return 1

    unset PCIE_MAX_CONCURRENT_H2D PCIE_NO_PRIORITY_QUEUE PCIE_NO_EVICT_FIRST
}

# ============================================================
# 主循环: rounds x qps x groups
# ============================================================

ALL_FAILED_GROUPS=()

for (( round=1; round<=ROUNDS; round++ )); do
    for CUR_QPS in "${QPS_ARRAY[@]}"; do
        # 为每个 (round, qps) 组合创建独立结果目录
        EXP_TS="$(date +%Y%m%d_%H%M%S)"
        EXP_ID="${EXP_TS}_mech_ablation_${DATASET}_q${CUR_QPS}_blk${NUM_GPU_BLOCKS_OVERRIDE}_${MODEL_TAG}"
        EXP_ID="${EXP_ID//\//_}"
        EXP_ID="${EXP_ID// /_}"

        RESULTS_DIR="$REPO_ROOT/results/$EXP_ID"
        mkdir -p "$RESULTS_DIR"

        print_separator
        echo "Round $round/$ROUNDS | QPS=$CUR_QPS | Blocks=$NUM_GPU_BLOCKS_OVERRIDE | $MODEL_TAG"
        echo "Experiment ID: $EXP_ID"
        echo "Results: $RESULTS_DIR"
        print_separator

        FAILED_GROUPS=()

        # --- full: 完整调度 ---
        if should_run_group full; then
            start_vllm_scheduler "full_q${CUR_QPS}_r${round}" || exit 1
            print_phase "[full] Complete scheduler (round=$round, qps=$CUR_QPS)"
            run_phase "full" "prefetch" "full_sched" "$CUR_QPS" || FAILED_GROUPS+=(full)
            stop_vllm
        fi

        # --- no-pq: 去掉优先队列 ---
        if should_run_group no-pq; then
            start_vllm_scheduler "no-pq_q${CUR_QPS}_r${round}" "PCIE_NO_PRIORITY_QUEUE=1" || exit 1
            print_phase "[no-pq] No priority queue (round=$round, qps=$CUR_QPS)"
            run_phase "no-pq" "prefetch" "no_pq" "$CUR_QPS" || FAILED_GROUPS+=(no-pq)
            stop_vllm
        fi

        # --- no-ef: 去掉 Evict-first ---
        if should_run_group no-ef; then
            start_vllm_scheduler "no-ef_q${CUR_QPS}_r${round}" "PCIE_NO_EVICT_FIRST=1" || exit 1
            print_phase "[no-ef] No evict-first (round=$round, qps=$CUR_QPS)"
            run_phase "no-ef" "prefetch" "no_ef" "$CUR_QPS" || FAILED_GROUPS+=(no-ef)
            stop_vllm
        fi

        # --- no-cc: 去掉并发控制 ---
        if should_run_group no-cc; then
            start_vllm_scheduler "no-cc_q${CUR_QPS}_r${round}" "PCIE_MAX_CONCURRENT_H2D=999" || exit 1
            print_phase "[no-cc] No concurrency control (round=$round, qps=$CUR_QPS)"
            run_phase "no-cc" "prefetch" "no_cc" "$CUR_QPS" || FAILED_GROUPS+=(no-cc)
            stop_vllm
        fi

        # --- g1 / g0: 无调度器 ---
        if should_run_group g1 || should_run_group g0; then
            start_vllm "g1_g0_q${CUR_QPS}_r${round}" --gpu-blocks "$NUM_GPU_BLOCKS_OVERRIDE" --log-file "$RESULTS_DIR/vllm_log_g1_g0.log" || exit 1
        fi

        if should_run_group g1; then
            print_phase "[g1] Prefetch only, no scheduler (round=$round, qps=$CUR_QPS)"
            run_phase "g1" "prefetch" "g1_prefetch_only" "$CUR_QPS" || FAILED_GROUPS+=(g1)
        fi

        if should_run_group g0; then
            print_phase "[g0] Original vLLM baseline (round=$round, qps=$CUR_QPS)"
            curl -s -X POST "http://localhost:$API_PORT/reset_prefix_cache?reset_external=true" >/dev/null || true
            sleep 5
            run_phase "g0" "baseline" "g0_no_prefetch" "$CUR_QPS" || FAILED_GROUPS+=(g0)
        fi

        if should_run_group g1 || should_run_group g0; then
            stop_vllm
        fi

        # ============================================================
        # 报告生成 (每个 round+qps 组合)
        # ============================================================

        print_phase "Generating report for round=$round, qps=$CUR_QPS..."

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
            echo "**Model**: $MODEL_TAG (PP=$PP_SIZE) | **Dataset**: $DATASET | **QPS**: $CUR_QPS | **Lead Time**: ${PREFETCH_LEAD_TIME}s | **GPU Blocks**: $NUM_GPU_BLOCKS_OVERRIDE"
            echo "**Round**: $round/$ROUNDS | **Loop**: $LOOP_LABEL"
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
        echo "Report: $ABLATION_MD"

        # 追加到机制消融专用 TSV
        MECH_ABLATION_TABLE_TSV="$REPO_ROOT/results/pcie_mechanism_ablation_experiments.txt"
        if [[ -f "../result-analysis/append_pcie_mechanism_ablation_summary_row.py" ]]; then
            python3 ../result-analysis/append_pcie_mechanism_ablation_summary_row.py \
                --results-dir "$RESULTS_DIR" \
                --exp-id "$EXP_ID" \
                --dataset "$DATASET" \
                --qps "$CUR_QPS" \
                --lead-time "$PREFETCH_LEAD_TIME" \
                --num-gpu-blocks "$NUM_GPU_BLOCKS_OVERRIDE" \
                --num-conv "$NUM_CONV" \
                --gpu-mem-util "$GPU_MEMORY_UTILIZATION" \
                --vllm-pp "$PP_SIZE" \
                --vllm-max-num-seqs "$VLLM_MAX_NUM_SEQS" \
                --model-path "$MODEL_PATH" \
                --groups "$RUN_GROUPS" \
                --table "$MECH_ABLATION_TABLE_TSV" \
                && echo "Appended rows to $MECH_ABLATION_TABLE_TSV" || \
                echo "Warning: append_pcie_mechanism_ablation_summary_row.py failed"
        else
            echo "Warning: append_pcie_mechanism_ablation_summary_row.py not found, skip TSV append"
        fi

        ALL_FAILED_GROUPS+=("${FAILED_GROUPS[@]}")

    done  # QPS loop
done  # round loop

# ============================================================
# 总结
# ============================================================

print_separator
echo "All Experiments Complete!"
echo "  Model:       $MODEL_TAG (PP=$PP_SIZE)"
echo "  Dataset:     $DATASET"
echo "  QPS tested:  ${QPS_ARRAY[*]}"
echo "  Rounds:      $ROUNDS"
echo "  Blocks:      $NUM_GPU_BLOCKS_OVERRIDE"
echo "  Loop:        $LOOP_LABEL"
echo "  Results dir: $REPO_ROOT/results/"
if [[ ${#ALL_FAILED_GROUPS[@]} -gt 0 ]]; then
    echo "  Failed runs: ${ALL_FAILED_GROUPS[*]}"
fi
print_separator
