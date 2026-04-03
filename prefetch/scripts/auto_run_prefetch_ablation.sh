#!/bin/bash
# ============================================================
# Prefetch 消融实验自动化脚本
# ============================================================
# 实验组:
#   A) Baseline vs Prefetch 基础对比
#   B) 准入阈值消融 (prefetch_block_threshold)
#   C) 配额比例消融 (max_prefetch_block_ratio)
#   D) Prefetch 提前量消融 (prefetch_lead_time)
#   E) QPS 负载敏感性
#
# 用法:
#   ./auto_run_prefetch_ablation.sh [dataset] [options]
#   ./auto_run_prefetch_ablation.sh optimal --groups A,B
#   ./auto_run_prefetch_ablation.sh optimal --groups D --qps 1.0
#   ./auto_run_prefetch_ablation.sh optimal --groups A,B,C,D,E
#
# 选项:
#   --qps N              默认 QPS (用于非 E 组实验, default: 0.5)
#   --lead-time N        默认提前量秒 (用于非 D 组, default: 2.0)
#   --gpu-blocks N       覆盖 GPU block 数量
#   --groups A,B,C,D,E   选择要运行的实验组 (default: A,B,C,D)
# ============================================================

set -o pipefail

# 失败计数
SKIP_COUNT=0
FAIL_LOG=""
record_skip() {
    SKIP_COUNT=$((SKIP_COUNT + 1))
    FAIL_LOG="${FAIL_LOG}\n  - $1"
    echo "⚠️  SKIP: $1"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFETCH_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_EXP_DIR="$(cd "$PREFETCH_ROOT/../run-experiment" && pwd)"

cd "$RUN_EXP_DIR"

# ============================================================
# vLLM 自动管理
# ============================================================

VLLM_PID=""
API_PORT="${API_PORT:-8000}"

start_vllm() {
    local label="$1"; shift
    local extra_args=("$@")

    print_phase "Starting vLLM [$label]"
    echo "  Extra args: ${extra_args[*]}"

    pkill -f "vllm serve" 2>/dev/null || true
    sleep 2

    local startup_log="$RESULTS_DIR/vllm_startup_${label}.log"
    nohup bash "$SCRIPT_DIR/start_vllm_prefetch.sh" \
        --gpu-blocks "$NUM_GPU_BLOCKS_OVERRIDE" \
        --log-file "$RESULTS_DIR/vllm_log_${label}.log" \
        "${extra_args[@]}" \
        > "$startup_log" 2>&1 &
    VLLM_PID=$!

    local max_wait=300
    local waited=0
    echo -n "Waiting for vLLM"

    while [ $waited -lt $max_wait ]; do
        if curl -s "http://localhost:$API_PORT/health" &>/dev/null; then
            echo ""
            echo "  vLLM ready (${waited}s)"
            sleep 5
            return 0
        fi
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo ""
            echo "  vLLM process died. Check: $startup_log"
            return 1
        fi
        echo -n "."
        sleep 3
        waited=$((waited + 3))
    done
    echo ""
    echo "  vLLM timeout (${max_wait}s). Check: $startup_log"
    return 1
}

stop_vllm() {
    if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "Stopping vLLM (PID: $VLLM_PID)..."
        kill "$VLLM_PID" 2>/dev/null || true
        local waited=0
        while kill -0 "$VLLM_PID" 2>/dev/null && [ $waited -lt 30 ]; do
            sleep 1; waited=$((waited + 1))
        done
        kill -0 "$VLLM_PID" 2>/dev/null && kill -9 "$VLLM_PID" 2>/dev/null
        VLLM_PID=""
    fi
    pkill -f "vllm serve" 2>/dev/null || true
    sleep 3
}

cleanup() { echo ""; echo "Cleaning up..."; stop_vllm; }
trap cleanup EXIT INT TERM

# ============================================================
# 加载配置
# ============================================================

# 加载 run-experiment 的通用配置
source "$RUN_EXP_DIR/config/system.env"
source "$RUN_EXP_DIR/config/datasets.env"
source "$RUN_EXP_DIR/scripts/utils/common.sh"

# 覆盖为 prefetch 专用配置
PREFETCH_CONFIG="$PREFETCH_ROOT/config/prefetch_experiments.env"
if [[ -f "$PREFETCH_CONFIG" ]]; then
    source "$PREFETCH_CONFIG"
    echo "✓ Loaded prefetch config: $PREFETCH_CONFIG"
else
    echo "⚠️  Prefetch config not found: $PREFETCH_CONFIG"
    echo "   Using run-experiment defaults"
    source "$RUN_EXP_DIR/config/experiments.env"
fi

DATASET="${1:-optimal}"
shift 2>/dev/null || true

# 默认值
DEFAULT_QPS="${QPS:-0.5}"
DEFAULT_LEAD_TIME="${PREFETCH_LEAD_TIME:-2.0}"
RUN_GROUPS="A,B,C,D"
NUM_GPU_BLOCKS_OVERRIDE_SET=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --qps)         DEFAULT_QPS="$2"; shift 2 ;;
        --lead-time)   DEFAULT_LEAD_TIME="$2"; shift 2 ;;
        --gpu-blocks)  NUM_GPU_BLOCKS_OVERRIDE="$2"; NUM_GPU_BLOCKS_OVERRIDE_SET=1; shift 2 ;;
        --groups)      RUN_GROUPS="$(echo "$2" | tr '[:lower:]' '[:upper:]')"; shift 2 ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [dataset] [--qps N] [--lead-time N] [--gpu-blocks N] [--groups A,B,C,D,E]"
            exit 1 ;;
    esac
done

should_run() { [[ ",$RUN_GROUPS," == *",$1,"* ]]; }

load_dataset_config "$DATASET"
[[ $NUM_GPU_BLOCKS_OVERRIDE_SET -eq 0 ]] && NUM_GPU_BLOCKS_OVERRIDE="${DATASET_GPU_BLOCKS:-$NUM_GPU_BLOCKS_OVERRIDE}"

generate_dataset_if_needed "$TRACE" "$FULL_TRACE" "$DATASET" || exit 1

# Timeout 配置 (与原脚本保持一致)
RUNNER_TIMEOUT_ARGS=(--timeout "$TIMEOUT" --request-timeout "$REQUEST_TIMEOUT")
if [[ "$DATASET" == "pcie-full" || "$DATASET" == "pcie-trace-a-light" || "$DATASET" == "pcie-multiturn" || "$DATASET" == "pcie-heavy" ]]; then
    if [[ ! -f "$TRACE" ]]; then
        echo "❌ $DATASET: trace 不存在: $TRACE"
        exit 1
    fi
    REQUEST_TIMEOUT=360
    TIMEOUT=""
    RUNNER_TIMEOUT_ARGS=(--request-timeout "$REQUEST_TIMEOUT")
fi

MAX_OUTPUT_ARGS=()
[[ -n "$MAX_OUTPUT" ]] && MAX_OUTPUT_ARGS=(--max-output-tokens "$MAX_OUTPUT")

# ============================================================
# 路径 & 实验 ID
# ============================================================

EXP_TS="$(date +%Y%m%d_%H%M%S)"
MODEL_TAG="$(python3 -c "
import re, sys
p = sys.argv[1].lower()
for pat, lab in [(r'72b','72B'),(r'32b','32B'),(r'8b','8B')]:
    if re.search(pat, p): print(lab); break
else: print('unknown')
" "$MODEL_PATH")"

EXP_ID="${EXP_TS}_prefetch_ablation_${DATASET}_q${DEFAULT_QPS}_lead${DEFAULT_LEAD_TIME}_${MODEL_TAG}"
RESULTS_DIR="$PREFETCH_ROOT/results/$EXP_ID"
mkdir -p "$RESULTS_DIR"

# 多轮会话数
if [[ "$NUM_CONV" -eq 0 ]]; then
    NUM_CONV=$(python3 -c "
import json
chats={}
with open('$TRACE') as f:
    for line in f:
        r=json.loads(line)
        cid=r['chat_id']; pid=r.get('parent_chat_id',-1)
        if pid==-1: chats.setdefault(cid,{'root':True,'has_child':False})
        else: chats.setdefault(pid,{})['has_child']=True
print(sum(1 for c in chats.values() if c.get('root') and c.get('has_child')))
")
fi

SEED="${SEED:-42}"
SCHEDULE_MODE="${SCHEDULE_MODE:-scaled-timestamp}"

print_separator
echo "Prefetch Ablation Experiment"
print_separator
echo "ID:         $EXP_ID"
echo "Dataset:    $DATASET ($TRACE)"
echo "Num conv:   $NUM_CONV"
echo "GPU:        Single card (TP=$VLLM_TENSOR_PARALLEL_SIZE)"
echo "QPS:        $DEFAULT_QPS"
echo "Lead time:  ${DEFAULT_LEAD_TIME}s"
echo "GPU blocks: $NUM_GPU_BLOCKS_OVERRIDE (from: $([ $NUM_GPU_BLOCKS_OVERRIDE_SET -eq 1 ] && echo "CLI --gpu-blocks" || echo "config"))"
[[ -n "$MAX_MODEL_LEN" ]] && echo "Max seq len: $MAX_MODEL_LEN (limited for single GPU)"
echo "Groups:     $RUN_GROUPS"
echo "Results:    $RESULTS_DIR"
print_separator

# ============================================================
# run_experiment: 通用实验执行器
# ============================================================

run_experiment() {
    local LABEL="$1"
    local MODE="$2"        # baseline | prefetch
    local QPS_VAL="$3"
    local LEAD_TIME_VAL="$4"
    local SUFFIX="$5"

    echo ""
    echo "--- [$LABEL] mode=$MODE qps=$QPS_VAL lead_time=$LEAD_TIME_VAL ---"

    curl -s -X POST "http://localhost:$API_PORT/reset_prefix_cache?reset_external=true" >/dev/null || true
    sleep 3

    python3 prefetch_ab_runner.py \
        --trace-file "$TRACE" \
        --mode "$MODE" \
        --qps "$QPS_VAL" \
        --num-multi-turn "$NUM_CONV" \
        --model "$MODEL_PATH" \
        --api-base "http://localhost:$API_PORT/v1" \
        --output "$RESULTS_DIR/${SUFFIX}.jsonl" \
        --seed "$SEED" \
        --prefetch-lead-time "$LEAD_TIME_VAL" \
        --schedule-mode "$SCHEDULE_MODE" \
        "${RUNNER_TIMEOUT_ARGS[@]}" \
        "${MAX_OUTPUT_ARGS[@]}" \
        2>&1 | tee "$RESULTS_DIR/${SUFFIX}.log"

    echo "  [$LABEL] done -> ${SUFFIX}.jsonl"
}

# ============================================================
# Group A: Baseline vs Prefetch
# ============================================================

if should_run A; then
    print_phase "[Group A] Baseline vs Prefetch (default config)"

    if ! start_vllm "A_default" \
        --prefetch-block-threshold 150 \
        --max-prefetch-block-ratio 0.3; then
        record_skip "[Group A] vLLM 启动失败"
        stop_vllm
    else
        run_experiment "A-baseline" "baseline" "$DEFAULT_QPS" "$DEFAULT_LEAD_TIME" \
            "A_baseline_q${DEFAULT_QPS}" \
            || record_skip "[Group A] baseline 实验失败"

        run_experiment "A-prefetch" "prefetch" "$DEFAULT_QPS" "$DEFAULT_LEAD_TIME" \
            "A_prefetch_q${DEFAULT_QPS}" \
            || record_skip "[Group A] prefetch 实验失败"

        stop_vllm
    fi
fi

# ============================================================
# Group B: 准入阈值消融 (prefetch_block_threshold)
# 固定 QPS, lead_time, ratio=0.3; 扫描 threshold
# ============================================================

if should_run B; then
    print_phase "[Group B] Admission Threshold Ablation"

    for THRESH in 0 50 100 150 200 300; do
        if ! start_vllm "B_thresh${THRESH}" \
            --prefetch-block-threshold "$THRESH" \
            --max-prefetch-block-ratio 0.3; then
            record_skip "[Group B] thresh=${THRESH} vLLM 启动失败"
            stop_vllm
            continue
        fi

        run_experiment "B-thresh${THRESH}" "prefetch" "$DEFAULT_QPS" "$DEFAULT_LEAD_TIME" \
            "B_thresh${THRESH}_q${DEFAULT_QPS}" \
            || record_skip "[Group B] thresh=${THRESH} 实验失败"

        stop_vllm
    done
fi

# ============================================================
# Group C: 配额比例消融 (max_prefetch_block_ratio)
# 固定 QPS, lead_time, threshold=150; 扫描 ratio
# ============================================================

if should_run C; then
    print_phase "[Group C] Prefetch Quota Ratio Ablation"

    for RATIO in 0.0 0.1 0.2 0.3 0.5 1.0; do
        if ! start_vllm "C_ratio${RATIO}" \
            --prefetch-block-threshold 150 \
            --max-prefetch-block-ratio "$RATIO"; then
            record_skip "[Group C] ratio=${RATIO} vLLM 启动失败"
            stop_vllm
            continue
        fi

        run_experiment "C-ratio${RATIO}" "prefetch" "$DEFAULT_QPS" "$DEFAULT_LEAD_TIME" \
            "C_ratio${RATIO}_q${DEFAULT_QPS}" \
            || record_skip "[Group C] ratio=${RATIO} 实验失败"

        stop_vllm
    done
fi

# ============================================================
# Group D: 预取提前量消融 (prefetch_lead_time)
# 固定 QPS, threshold=150, ratio=0.3; 扫描 lead_time
# 不需要重启 vLLM (lead_time 是 runner 侧参数)
# ============================================================

if should_run D; then
    print_phase "[Group D] Prefetch Lead Time Ablation"

    if ! start_vllm "D_leadtime" \
        --prefetch-block-threshold 150 \
        --max-prefetch-block-ratio 0.3; then
        record_skip "[Group D] vLLM 启动失败"
        stop_vllm
    else
        for LT in 0.0 1.0 2.0 5.0 10.0 30.0; do
            run_experiment "D-lead${LT}" "prefetch" "$DEFAULT_QPS" "$LT" \
                "D_lead${LT}_q${DEFAULT_QPS}" \
                || record_skip "[Group D] lead_time=${LT} 实验失败"
        done

        stop_vllm
    fi
fi

# ============================================================
# Group E: QPS 负载敏感性
# 固定 lead_time, threshold=150, ratio=0.3; 扫描 QPS
# Baseline + Prefetch 各跑一遍
# ============================================================

if should_run E; then
    print_phase "[Group E] QPS Load Sensitivity"

    if ! start_vllm "E_qps" \
        --prefetch-block-threshold 150 \
        --max-prefetch-block-ratio 0.3; then
        record_skip "[Group E] vLLM 启动失败"
        stop_vllm
    else
        for Q in 0.2 0.4 0.6 0.8 1.0; do
            run_experiment "E-baseline-q${Q}" "baseline" "$Q" "$DEFAULT_LEAD_TIME" \
                "E_baseline_q${Q}" \
                || record_skip "[Group E] baseline qps=${Q} 实验失败"

            run_experiment "E-prefetch-q${Q}" "prefetch" "$Q" "$DEFAULT_LEAD_TIME" \
                "E_prefetch_q${Q}" \
                || record_skip "[Group E] prefetch qps=${Q} 实验失败"
        done

        stop_vllm
    fi
fi

# ============================================================
# 生成汇总报告
# ============================================================

print_phase "Generating summary report..."

REPORT="$RESULTS_DIR/ablation_report.md"
{
    echo "# Prefetch Ablation Report"
    echo ""
    echo "**Experiment**: \`$EXP_ID\`"
    echo "**Dataset**: $DATASET | **Model**: $MODEL_TAG | **GPU Blocks**: $NUM_GPU_BLOCKS_OVERRIDE"
    echo ""

    for GROUP_PREFIX in A B C D E; do
        FILES=("$RESULTS_DIR"/${GROUP_PREFIX}_*.jsonl)
        [[ ! -f "${FILES[0]}" ]] && continue

        echo "## Group $GROUP_PREFIX"
        echo ""
        echo "| Experiment | Requests | Mean TTFT (ms) | P50 | P95 | P99 | Cache Hit Rate |"
        echo "|------------|----------|----------------|-----|-----|-----|----------------|"

        for JSONL in "$RESULTS_DIR"/${GROUP_PREFIX}_*.jsonl; do
            FNAME="$(basename "$JSONL" .jsonl)"
            python3 -c "
import json, sys
import numpy as np

ttfts, cache_hits, total_reqs = [], 0, 0
with open('$JSONL') as f:
    for line in f:
        d = json.loads(line)
        total_reqs += 1
        if d.get('ttft_ms') is not None:
            ttfts.append(d['ttft_ms'])
        ct = d.get('cached_tokens', 0) or 0
        pt = d.get('prompt_tokens', 1) or 1
        if ct > 0:
            cache_hits += 1

if not ttfts:
    print('| $FNAME | $total_reqs | - | - | - | - | - |')
else:
    arr = np.array(ttfts)
    hit_rate = cache_hits / total_reqs * 100 if total_reqs else 0
    print(f'| $FNAME | {len(arr)} | {np.mean(arr):.1f} | {np.percentile(arr,50):.1f} | {np.percentile(arr,95):.1f} | {np.percentile(arr,99):.1f} | {hit_rate:.1f}% |')
" 2>/dev/null || echo "| $FNAME | error | - | - | - | - | - |"
        done
        echo ""
    done
} > "$REPORT"

echo "Report: $REPORT"

# ============================================================
# 追加到 Prefetch Ablation TSV 表格
# ============================================================

ABLATION_TABLE_TSV="$PREFETCH_ROOT/results/prefetch_ablation_experiments.txt"
if [[ -f "$PREFETCH_ROOT/../result-analysis/append_prefetch_ablation_summary_row.py" ]]; then
    python3 "$PREFETCH_ROOT/../result-analysis/append_prefetch_ablation_summary_row.py" \
        --results-dir "$RESULTS_DIR" \
        --exp-id "$EXP_ID" \
        --dataset "$DATASET" \
        --default-qps "$DEFAULT_QPS" \
        --default-lead-time "$DEFAULT_LEAD_TIME" \
        --num-gpu-blocks "$NUM_GPU_BLOCKS_OVERRIDE" \
        --num-conv "$NUM_CONV" \
        --gpu-mem-util "$GPU_MEMORY_UTILIZATION" \
        --max-num-seqs "$VLLM_MAX_NUM_SEQS" \
        --model-path "$MODEL_PATH" \
        --table "$ABLATION_TABLE_TSV" \
        && echo "✓ Appended results to $ABLATION_TABLE_TSV" || \
        echo "⚠️  append_prefetch_ablation_summary_row.py failed"
else
    echo "⚠️  append_prefetch_ablation_summary_row.py not found, skip TSV append"
fi

# ============================================================
# 完成
# ============================================================

print_separator
echo "Prefetch Ablation Experiment Complete!"
echo ""
echo "  ID:      $EXP_ID"
echo "  Results: $RESULTS_DIR"
echo "  Report:  $REPORT"
echo "  TSV:     $ABLATION_TABLE_TSV"
if [[ $SKIP_COUNT -gt 0 ]]; then
    echo ""
    echo "  ⚠️  Skipped $SKIP_COUNT experiment(s):"
    echo -e "$FAIL_LOG"
fi
print_separator
