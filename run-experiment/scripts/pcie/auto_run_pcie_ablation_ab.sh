#!/bin/bash
# 自动化运行 PCIe Ablation A/B 实验
# 自动管理 vLLM 的启动和重启，无需手动交互
# 用法: ./auto_run_pcie_ablation_ab.sh [dataset] [options]
#   示例: ./auto_run_pcie_ablation_ab.sh pcie-trace-a-light

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

# 运行单个实验阶段
run_phase() {
    local group_id="$1"      # G0/G1/G2/G3
    local mode="$2"          # prefetch/baseline
    local suffix="$3"        # 文件名后缀

    echo "Resetting prefix cache..."
    curl -s -X POST "http://localhost:$API_PORT/reset_prefix_cache?reset_external=true" >/dev/null || true

    echo "Clearing profiler events..."
    rm -f "$PCIE_PROFILER_DIR"/pcie_events_*.json 2>/dev/null || true

    echo "Starting profiler..."
    curl -s -X POST "http://localhost:$API_PORT/start_profile" >/dev/null || true

    sleep 2

    # 确定模式参数
    local mode_arg=""
    [[ "$mode" == "prefetch" ]] && mode_arg="--prefetch"
    [[ "$mode" == "baseline" ]] && mode_arg=""

    # 输出文件
    local jsonl_out="$RESULTS_DIR/${mode}_${suffix}.jsonl"
    local log_out="$RESULTS_DIR/${mode}_${suffix}.log"

    echo "Running workload ($group_id, mode=$mode)..."
    cd "$REPO_ROOT"
    python -m src.runner \
        --trace "$TRACE" \
        --model "$MODEL_NAME" \
        --api-base "http://localhost:$API_PORT/v1" \
        --output "$jsonl_out" \
        --qps "$QPS" \
        --num-multi-turn "$NUM_CONV" \
        --prefetch-lead-time "$PREFETCH_LEAD_TIME" \
        --max-input-length "$MAX_INPUT" \
        $mode_arg \
        2>&1 | tee "$log_out"
    cd "$SCRIPT_DIR"

    echo "Stopping profiler..."
    curl -s -X POST "http://localhost:$API_PORT/stop_profile" >/dev/null || true

    # 收集 profiler 结果
    local pcie_json="$RESULTS_DIR/pcie_events_${suffix}.json"
    if compgen -G "$PCIE_PROFILER_DIR/pcie_events_*.json" >/dev/null; then
        cat "$PCIE_PROFILER_DIR"/pcie_events_*.json > "$pcie_json"
        echo "✓ Collected PCIe events: $pcie_json"
    else
        echo "⚠️  No pcie_events_*.json found for $group_id"
    fi

    # 复制 vLLM 状态日志
    [[ -f "$VLLM_LOG" ]] && cp "$VLLM_LOG" "$RESULTS_DIR/vllm_state_${suffix}.log" 2>/dev/null || true

    # 清理 profiler 文件
    rm -f "$PCIE_PROFILER_DIR"/pcie_events_*.json 2>/dev/null || true

    sleep 2
}

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
    RUNNER_TIMEOUT_ARGS=(--request-timeout "$REQUEST_TIMEOUT")
fi

# 路径设置
RUN_EXP_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

EXP_TS="$(date +%Y%m%d_%H%M%S)"
MODEL_TAG="$(python3 -c "import os; p=os.environ.get('MODEL_PATH',''); print(os.path.basename(p) if p else 'unknown')")"
EXP_ID="${EXP_TS}_ablation_${DATASET}_q${QPS}_lead${PREFETCH_LEAD_TIME}_${MODEL_TAG}"
RESULTS_DIR="$RUN_EXP_ROOT/results/pcie_ablation/$EXP_ID"
mkdir -p "$RESULTS_DIR"

# Profiler 目录
if [[ "$PCIE_PROFILER_DIR" != /* ]]; then
    PCIE_PROFILER_DIR="$RUN_EXP_ROOT/${PCIE_PROFILER_DIR#./}"
fi
mkdir -p "$PCIE_PROFILER_DIR"

# 打印配置
print_separator
echo "Experiment: PCIe Ablation A/B Test (AUTOMATED)"
echo "Experiment ID: $EXP_ID"
echo ""
echo "Dataset: $DATASET"
echo "Num conversations: $NUM_CONV"
echo "QPS: $QPS"
echo "Prefetch Lead Time: ${PREFETCH_LEAD_TIME}s"
echo "GPU Blocks: $NUM_GPU_BLOCKS_OVERRIDE"
echo "Model: $MODEL_PATH"
echo "Results: $RESULTS_DIR"
echo ""
echo "Groups:"
echo "  G3: PCIe Scheduler + PP Phase-Aware   (start_vllm_pcie.sh --pcie-scheduler)"
echo "  G2: PCIe Scheduler, no Phase-Aware    (start_vllm_pcie.sh --pcie-scheduler --no-pp-phase-aware)"
echo "  G1: No Scheduler, with Prefetch       (start_vllm_pcie.sh)"
echo "  G0: No Scheduler, no Prefetch         (start_vllm_pcie.sh, --mode baseline)"
print_separator

# ============================================================
# Phase 1: G3 — Full scheduling (PCIe Scheduler + PP Phase-Aware)
# ============================================================

start_vllm --pcie-scheduler || exit 1

print_phase "[Phase 1/5] G3: Prefetch + PCIe Scheduler + PP Phase-Aware"
run_phase "G3" "prefetch" "g3_full_sched"

# ============================================================
# Phase 1 → Phase 2: 重启 vLLM（--pcie-scheduler --no-pp-phase-aware）
# ============================================================

print_phase "Restarting vLLM for Phase 2 (--pcie-scheduler --no-pp-phase-aware)..."
stop_vllm
sleep 3
start_vllm --pcie-scheduler --no-pp-phase-aware || exit 1

# ============================================================
# Phase 2: G2 — Scheduler ON, Phase-Aware OFF
# ============================================================

print_phase "[Phase 2/5] G2: Prefetch + PCIe Scheduler, NO PP Phase-Aware"
run_phase "G2" "prefetch" "g2_sched_no_phase"

# ============================================================
# Phase 2 → Phase 3: 重启 vLLM（不带 --pcie-scheduler）
# ============================================================

print_phase "Restarting vLLM for Phase 3 (no scheduler)..."
stop_vllm
sleep 3
start_vllm || exit 1

# ============================================================
# Phase 3: G1 — No Scheduler, with Prefetch
# ============================================================

print_phase "[Phase 3/5] G1: Prefetch, no PCIe Scheduler (baseline prefetch)"
run_phase "G1" "prefetch" "g1_prefetch_only"

# ============================================================
# Phase 4: G0 — No Scheduler, No Prefetch (same vLLM instance)
# ============================================================

print_phase "[Phase 4/5] G0: No Prefetch, no Scheduler (original vLLM baseline)"
echo "Resetting prefix cache before G0..."
curl -s -X POST "http://localhost:$API_PORT/reset_prefix_cache?reset_external=true" >/dev/null || true
sleep 5

run_phase "G0" "baseline" "g0_no_prefetch"

# 停止 vLLM
stop_vllm

# ============================================================
# Phase 5: 生成报告
# ============================================================

print_phase "[Phase 5/5] Generating ablation comparison report..."

MERGED_MD="$RESULTS_DIR/pcie_ablation_comparison.md"
cd "$REPO_ROOT"
python -m src.analysis.merge_pcie_ablation_report \
    --g3-jsonl "$RESULTS_DIR/prefetch_g3_full_sched.jsonl" \
    --g2-jsonl "$RESULTS_DIR/prefetch_g2_sched_no_phase.jsonl" \
    --g1-jsonl "$RESULTS_DIR/prefetch_g1_prefetch_only.jsonl" \
    --g0-jsonl "$RESULTS_DIR/baseline_g0_no_prefetch.jsonl" \
    --g3-pcie-events "$RESULTS_DIR/pcie_events_g3_full_sched.json" \
    --g2-pcie-events "$RESULTS_DIR/pcie_events_g2_sched_no_phase.json" \
    --g1-pcie-events "$RESULTS_DIR/pcie_events_g1_prefetch_only.json" \
    --g0-pcie-events "$RESULTS_DIR/pcie_events_g0_no_prefetch.json" \
    --output "$MERGED_MD"
cd "$SCRIPT_DIR"

# 追加到 Ablation TSV
ABLATION_TABLE_TSV="$RUN_EXP_ROOT/results/pcie_ablation_experiments.tsv"
if [ ! -f "$ABLATION_TABLE_TSV" ]; then
    echo -e "exp_id\tdataset\tqps\tlead_time\tnum_conv\tgpu_mem_util\tvllm_pp\tvllm_max_seqs\tmodel_path\tg3_e2e_avg\tg3_e2e_p50\tg3_e2e_p99\tg3_tpot_avg\tg3_tpot_p50\tg3_tpot_p99\tg3_ttft_avg\tg3_ttft_p50\tg3_ttft_p99\tg3_pcie_h2d_sum\tg3_pcie_h2d_avg\tg3_pcie_d2h_sum\tg3_pcie_d2h_avg\tg2_e2e_avg\tg2_e2e_p50\tg2_e2e_p99\tg2_tpot_avg\tg2_tpot_p50\tg2_tpot_p99\tg2_ttft_avg\tg2_ttft_p50\tg2_ttft_p99\tg2_pcie_h2d_sum\tg2_pcie_h2d_avg\tg2_pcie_d2h_sum\tg2_pcie_d2h_avg\tg1_e2e_avg\tg1_e2e_p50\tg1_e2e_p99\tg1_tpot_avg\tg1_tpot_p50\tg1_tpot_p99\tg1_ttft_avg\tg1_ttft_p50\tg1_ttft_p99\tg1_pcie_h2d_sum\tg1_pcie_h2d_avg\tg1_pcie_d2h_sum\tg1_pcie_d2h_avg\tg0_e2e_avg\tg0_e2e_p50\tg0_e2e_p99\tg0_tpot_avg\tg0_tpot_p50\tg0_tpot_p99\tg0_ttft_avg\tg0_ttft_p50\tg0_ttft_p99\tg0_pcie_h2d_sum\tg0_pcie_h2d_avg\tg0_pcie_d2h_sum\tg0_pcie_d2h_avg" > "$ABLATION_TABLE_TSV"
fi

if [ -f "$REPO_ROOT/src/analysis/append_pcie_ablation_summary_row.py" ]; then
    python -m src.analysis.append_pcie_ablation_summary_row \
        --exp-id "$EXP_ID" \
        --g3-jsonl "$RESULTS_DIR/prefetch_g3_full_sched.jsonl" \
        --g2-jsonl "$RESULTS_DIR/prefetch_g2_sched_no_phase.jsonl" \
        --g1-jsonl "$RESULTS_DIR/prefetch_g1_prefetch_only.jsonl" \
        --g0-jsonl "$RESULTS_DIR/baseline_g0_no_prefetch.jsonl" \
        --g3-pcie-events "$RESULTS_DIR/pcie_events_g3_full_sched.json" \
        --g2-pcie-events "$RESULTS_DIR/pcie_events_g2_sched_no_phase.json" \
        --g1-pcie-events "$RESULTS_DIR/pcie_events_g1_prefetch_only.json" \
        --g0-pcie-events "$RESULTS_DIR/pcie_events_g0_no_prefetch.json" \
        --dataset "$DATASET" \
        --qps "$QPS" \
        --lead-time "$PREFETCH_LEAD_TIME" \
        --num-conv "$NUM_CONV" \
        --gpu-mem-util "$GPU_MEMORY_UTILIZATION" \
        --vllm-pp "$VLLM_PIPELINE_PARALLEL_SIZE" \
        --vllm-max-num-seqs "$VLLM_MAX_NUM_SEQS" \
        --model-path "$MODEL_PATH" \
        --table "$ABLATION_TABLE_TSV" \
        && echo "✓ Appended 4 rows to $ABLATION_TABLE_TSV" || \
        echo "⚠️  append_pcie_ablation_summary_row.py failed"
else
    echo "⚠️  append_pcie_ablation_summary_row.py not found, skipping TSV append"
fi

# TensorBoard（可选）
if [[ $NO_TENSORBOARD -eq 0 ]]; then
    cd "$REPO_ROOT"
    DATASET="$DATASET" QPS="$QPS" PREFETCH_LEAD_TIME="$PREFETCH_LEAD_TIME" \
        NUM_CONV="$NUM_CONV" MODEL_PATH="$MODEL_PATH" \
        bash -c '
            python -m src.analysis.log_to_tensorboard_pcie_ablation \
                --g3-jsonl "'"$RESULTS_DIR"'/prefetch_g3_full_sched.jsonl" \
                --g2-jsonl "'"$RESULTS_DIR"'/prefetch_g2_sched_no_phase.jsonl" \
                --g1-jsonl "'"$RESULTS_DIR"'/prefetch_g1_prefetch_only.jsonl" \
                --g0-jsonl "'"$RESULTS_DIR"'/baseline_g0_no_prefetch.jsonl" \
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
echo "✓ Automated PCIe Ablation A/B Test Complete!"
echo ""
echo "Results:"
echo "  Experiment ID: $EXP_ID"
echo "  Directory:     $RESULTS_DIR"
echo "  Merged report: $MERGED_MD"
echo "  Master TSV:    $ABLATION_TABLE_TSV"
echo "  Logs / jsonl:  prefetch_*.log, baseline_*.log, pcie_events_*.json, vllm_state_*.log"
print_separator
