#!/bin/bash
# 自动化运行 PCIe Scheduling A/B 实验
# 自动管理 vLLM 的启动和重启，无需手动交互
# 用法: ./auto_run_pcie_scheduling_ab.sh [dataset] [options]
#   示例: ./auto_run_pcie_scheduling_ab.sh pcie-trace-a-light

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ============================================================
# 辅助函数
# ============================================================

print_separator() {
    echo "============================================================"
}

print_phase() {
    echo ""
    print_separator
    echo "$1"
    print_separator
    echo ""
}

# 启动 vLLM（后台）
start_vllm() {
    local args=("$@")
    print_phase "Starting vLLM with args: ${args[*]}"

    # 调用 start_vllm_pcie.sh，后台运行并记录 PID
    cd "$SCRIPT_DIR/../.."
    nohup bash start_vllm_pcie.sh "${args[@]}" > /dev/null 2>&1 &
    VLLM_PID=$!
    cd "$SCRIPT_DIR"

    echo "vLLM started with PID: $VLLM_PID"

    # 等待 vLLM 启动完成
    local max_wait=180  # 最多等待3分钟
    local waited=0
    echo -n "Waiting for vLLM to be ready"

    while [ $waited -lt $max_wait ]; do
        if curl -s "http://localhost:$API_PORT/health" &>/dev/null; then
            echo ""
            echo "✓ vLLM is ready!"
            sleep 5  # 额外等待5秒确保完全就绪
            return 0
        fi
        echo -n "."
        sleep 2
        waited=$((waited + 2))
    done

    echo ""
    echo "❌ Error: vLLM failed to start within ${max_wait}s"
    return 1
}

# 停止 vLLM
stop_vllm() {
    if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "Stopping vLLM (PID: $VLLM_PID)..."
        kill "$VLLM_PID" 2>/dev/null || true

        # 等待进程结束
        local waited=0
        while kill -0 "$VLLM_PID" 2>/dev/null && [ $waited -lt 30 ]; do
            sleep 1
            waited=$((waited + 1))
        done

        # 如果还在运行，强制结束
        if kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "Force killing vLLM..."
            kill -9 "$VLLM_PID" 2>/dev/null || true
        fi

        echo "✓ vLLM stopped"
        VLLM_PID=""
    else
        echo "No vLLM process to stop"
    fi

    # 额外保险：清理所有 vllm serve 进程
    pkill -f "vllm serve" 2>/dev/null || true
    sleep 2
}

# 清理函数（脚本退出时调用）
cleanup() {
    echo ""
    echo "Cleaning up..."
    stop_vllm
}

trap cleanup EXIT INT TERM

# ============================================================
# 加载配置和工具
# ============================================================

source ../utils/common.sh
source config/vllm.env
source config/runner.env

DATASET="${1:-pcie-medium}"
shift || true

# 解析参数
QPS="$DEFAULT_QPS"
PREFETCH_LEAD_TIME="$DEFAULT_PREFETCH_LEAD_TIME"
NUM_GPU_BLOCKS_OVERRIDE="$DEFAULT_NUM_GPU_BLOCKS_OVERRIDE"
NUM_GPU_BLOCKS_OVERRIDE_SET=0
NO_TENSORBOARD=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --qps)
            QPS="$2"
            shift 2 ;;
        --lead-time)
            PREFETCH_LEAD_TIME="$2"
            shift 2 ;;
        --gpu-blocks)
            NUM_GPU_BLOCKS_OVERRIDE="$2"
            NUM_GPU_BLOCKS_OVERRIDE_SET=1
            shift 2 ;;
        --no-tensorboard)
            NO_TENSORBOARD=1
            shift ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [dataset] [--qps N] [--lead-time N] [--gpu-blocks N] [--no-tensorboard]"
            exit 1 ;;
    esac
done

load_dataset_config "$DATASET"
[[ $NUM_GPU_BLOCKS_OVERRIDE_SET -eq 0 ]] && NUM_GPU_BLOCKS_OVERRIDE="${DATASET_GPU_BLOCKS:-$NUM_GPU_BLOCKS_OVERRIDE}"

generate_dataset_if_needed "$TRACE" "$FULL_TRACE" "$DATASET" || exit 1

# 处理 NUM_CONV（从 trace 统计）
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

# 路径设置
RUN_EXP_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

EXP_TS="$(date +%Y%m%d_%H%M%S)"
MODEL_TAG="$(python3 -c "import os; p=os.environ.get('MODEL_PATH',''); print(os.path.basename(p) if p else 'unknown')")"
EXP_ID="${EXP_TS}_${DATASET}_q${QPS}_lead${PREFETCH_LEAD_TIME}_${MODEL_TAG}"
RESULTS_DIR="$RUN_EXP_ROOT/results/pcie_scheduling/$EXP_ID"
mkdir -p "$RESULTS_DIR"

# Profiler 目录
if [[ "$PCIE_PROFILER_DIR" != /* ]]; then
    PCIE_PROFILER_DIR="$RUN_EXP_ROOT/${PCIE_PROFILER_DIR#./}"
fi
mkdir -p "$PCIE_PROFILER_DIR"

# 打印配置
print_separator
echo "Experiment: PCIe Scheduling A/B Test (AUTOMATED)"
echo "Experiment ID: $EXP_ID"
echo "Dataset: $DATASET"
echo "Num conversations: $NUM_CONV"
echo "QPS: $QPS"
echo "Prefetch Lead Time: ${PREFETCH_LEAD_TIME}s"
echo "GPU Blocks: $NUM_GPU_BLOCKS_OVERRIDE"
echo "Model: $MODEL_PATH"
echo "Results: $RESULTS_DIR"
print_separator

# ============================================================
# Phase 1: Prefetch with PCIe Scheduling
# ============================================================

start_vllm --pcie-scheduler || exit 1

print_phase "[Phase 1/3] Prefetch with PCIe Scheduling (VLLM_PCIE_SCHEDULER=1)"

echo "Resetting prefix cache..."
curl -s -X POST "http://localhost:$API_PORT/reset_prefix_cache?reset_external=true" >/dev/null || true

echo "Clearing profiler events..."
rm -f "$PCIE_PROFILER_DIR"/pcie_events_*.json 2>/dev/null || true

echo "Starting profiler..."
curl -s -X POST "http://localhost:$API_PORT/start_profile" >/dev/null || true

sleep 2

# 运行实验
PREFETCH_MODE_ARG="--prefetch"
JSONL_OUTPUT="$RESULTS_DIR/prefetch_pcie_sched.jsonl"
RUNNER_LOG="$RESULTS_DIR/prefetch_pcie_sched.log"

echo "Running workload (Phase 1)..."
cd "$REPO_ROOT"
python -m src.runner \
    --trace "$TRACE" \
    --model "$MODEL_NAME" \
    --api-base "http://localhost:$API_PORT/v1" \
    --output "$JSONL_OUTPUT" \
    --qps "$QPS" \
    --num-multi-turn "$NUM_CONV" \
    --prefetch-lead-time "$PREFETCH_LEAD_TIME" \
    --max-input-length "$MAX_INPUT" \
    "${PCIE_FULL_RUNNER_TIMEOUT_ARGS[@]}" \
    $PREFETCH_MODE_ARG \
    2>&1 | tee "$RUNNER_LOG"
cd "$SCRIPT_DIR"

echo "Stopping profiler..."
curl -s -X POST "http://localhost:$API_PORT/stop_profile" >/dev/null || true

# 收集结果
PCIE_EVENTS_JSON="$RESULTS_DIR/pcie_events_pcie_sched.json"
if compgen -G "$PCIE_PROFILER_DIR/pcie_events_*.json" >/dev/null; then
    cat "$PCIE_PROFILER_DIR"/pcie_events_*.json > "$PCIE_EVENTS_JSON"
    echo "✓ Collected PCIe events: $PCIE_EVENTS_JSON"
else
    echo "⚠️  No pcie_events_*.json found"
fi

[[ -f "$VLLM_LOG" ]] && cp "$VLLM_LOG" "$RESULTS_DIR/vllm_state_pcie_sched.log" 2>/dev/null || true

echo "Removing profiler PCIe event files before Phase 2..."
rm -f "$PCIE_PROFILER_DIR"/pcie_events_*.json 2>/dev/null || true

sleep 2

# ============================================================
# Phase 1 → Phase 2: 重启 vLLM（不带 --pcie-scheduler）
# ============================================================

print_phase "Restarting vLLM for Phase 2 (without PCIe scheduler)..."
stop_vllm
sleep 3
start_vllm || exit 1

# ============================================================
# Phase 2: Prefetch without PCIe Scheduling (baseline)
# ============================================================

print_phase "[Phase 2/3] Prefetch without PCIe Scheduling (baseline)"

echo "Resetting prefix cache..."
curl -s -X POST "http://localhost:$API_PORT/reset_prefix_cache?reset_external=true" >/dev/null || true

echo "Starting profiler..."
curl -s -X POST "http://localhost:$API_PORT/start_profile" >/dev/null || true

sleep 2

# 运行实验
JSONL_OUTPUT="$RESULTS_DIR/prefetch_baseline.jsonl"
RUNNER_LOG="$RESULTS_DIR/prefetch_baseline.log"

echo "Running workload (Phase 2)..."
cd "$REPO_ROOT"
python -m src.runner \
    --trace "$TRACE" \
    --model "$MODEL_NAME" \
    --api-base "http://localhost:$API_PORT/v1" \
    --output "$JSONL_OUTPUT" \
    --qps "$QPS" \
    --num-multi-turn "$NUM_CONV" \
    --prefetch-lead-time "$PREFETCH_LEAD_TIME" \
    --max-input-length "$MAX_INPUT" \
    "${PCIE_FULL_RUNNER_TIMEOUT_ARGS[@]}" \
    $PREFETCH_MODE_ARG \
    2>&1 | tee "$RUNNER_LOG"
cd "$SCRIPT_DIR"

echo "Stopping profiler..."
curl -s -X POST "http://localhost:$API_PORT/stop_profile" >/dev/null || true

# 收集结果
PCIE_EVENTS_JSON="$RESULTS_DIR/pcie_events_baseline.json"
if compgen -G "$PCIE_PROFILER_DIR/pcie_events_*.json" >/dev/null; then
    cat "$PCIE_PROFILER_DIR"/pcie_events_*.json > "$PCIE_EVENTS_JSON"
    echo "✓ Collected PCIe events: $PCIE_EVENTS_JSON"
else
    echo "⚠️  No pcie_events_*.json found"
fi

[[ -f "$VLLM_LOG" ]] && cp "$VLLM_LOG" "$RESULTS_DIR/vllm_state_baseline.log" 2>/dev/null || true

# 停止 vLLM
stop_vllm

# ============================================================
# Phase 3: 生成报告
# ============================================================

print_phase "[Phase 3/3] Generating comparison report..."

MERGED_MD="$RESULTS_DIR/pcie_scheduling_comparison.md"
cd "$REPO_ROOT"
python -m src.analysis.merge_pcie_scheduling_report \
    --sched-jsonl "$RESULTS_DIR/prefetch_pcie_sched.jsonl" \
    --baseline-jsonl "$RESULTS_DIR/prefetch_baseline.jsonl" \
    --sched-pcie-events "$RESULTS_DIR/pcie_events_pcie_sched.json" \
    --baseline-pcie-events "$RESULTS_DIR/pcie_events_baseline.json" \
    --output "$MERGED_MD"
cd "$SCRIPT_DIR"

# 追加到 Master TSV
MASTER_TABLE_TSV="$RUN_EXP_ROOT/results/pcie_scheduling_experiments.tsv"
if [ ! -f "$MASTER_TABLE_TSV" ]; then
    echo -e "exp_id\tdataset\tqps\tlead_time\tnum_conv\tgpu_mem_util\tvllm_pp\tvllm_max_seqs\tmodel_path\tsched_e2e_avg\tsched_e2e_p50\tsched_e2e_p99\tsched_tpot_avg\tsched_tpot_p50\tsched_tpot_p99\tsched_ttft_avg\tsched_ttft_p50\tsched_ttft_p99\tbaseline_e2e_avg\tbaseline_e2e_p50\tbaseline_e2e_p99\tbaseline_tpot_avg\tbaseline_tpot_p50\tbaseline_tpot_p99\tbaseline_ttft_avg\tbaseline_ttft_p50\tbaseline_ttft_p99\tsched_pcie_h2d_sum\tsched_pcie_h2d_avg\tsched_pcie_h2d_count\tsched_pcie_d2h_sum\tsched_pcie_d2h_avg\tsched_pcie_d2h_count\tbaseline_pcie_h2d_sum\tbaseline_pcie_h2d_avg\tbaseline_pcie_h2d_count\tbaseline_pcie_d2h_sum\tbaseline_pcie_d2h_avg\tbaseline_pcie_d2h_count" > "$MASTER_TABLE_TSV"
fi

if [ -f "$REPO_ROOT/src/analysis/append_pcie_scheduling_summary_row.py" ]; then
    python -m src.analysis.append_pcie_scheduling_summary_row \
        --exp-id "$EXP_ID" \
        --sched-jsonl "$RESULTS_DIR/prefetch_pcie_sched.jsonl" \
        --baseline-jsonl "$RESULTS_DIR/prefetch_baseline.jsonl" \
        --sched-pcie-events "$RESULTS_DIR/pcie_events_pcie_sched.json" \
        --baseline-pcie-events "$RESULTS_DIR/pcie_events_baseline.json" \
        --dataset "$DATASET" \
        --qps "$QPS" \
        --lead-time "$PREFETCH_LEAD_TIME" \
        --num-conv "$NUM_CONV" \
        --gpu-mem-util "$GPU_MEMORY_UTILIZATION" \
        --vllm-pp "$VLLM_PIPELINE_PARALLEL_SIZE" \
        --vllm-max-num-seqs "$VLLM_MAX_NUM_SEQS" \
        --model-path "$MODEL_PATH" \
        --table "$MASTER_TABLE_TSV" \
        && echo "✓ Appended row to $MASTER_TABLE_TSV" || \
        echo "⚠️  append_pcie_scheduling_summary_row.py failed"
else
    echo "⚠️  append_pcie_scheduling_summary_row.py not found, skipping TSV append"
fi

# TensorBoard（可选）
if [[ $NO_TENSORBOARD -eq 0 ]]; then
    cd "$REPO_ROOT"
    DATASET="$DATASET" QPS="$QPS" PREFETCH_LEAD_TIME="$PREFETCH_LEAD_TIME" \
        NUM_CONV="$NUM_CONV" MODEL_PATH="$MODEL_PATH" \
        bash -c '
            python -m src.analysis.log_to_tensorboard_pcie_scheduling \
                --sched-jsonl "'"$RESULTS_DIR"'/prefetch_pcie_sched.jsonl" \
                --baseline-jsonl "'"$RESULTS_DIR"'/prefetch_baseline.jsonl" \
                --logdir "'"$RESULTS_DIR"'/tensorboard" \
                --tag "'"$EXP_ID"'" \
                --hparams dataset="$DATASET" qps="$QPS" lead_time="$PREFETCH_LEAD_TIME" num_conv="$NUM_CONV" model="$MODEL_PATH"
        ' && echo "✓ TensorBoard logs: $RESULTS_DIR/tensorboard" || echo "⚠️  TensorBoard logging failed"
    cd "$SCRIPT_DIR"
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
echo "  Logs / jsonl:  prefetch_*.log, prefetch_*.jsonl, pcie_events_*.json, vllm_state_*.log"
print_separator
