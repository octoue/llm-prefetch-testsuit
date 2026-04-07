#!/bin/bash
# EPLB Gradient Experiments: 证明 vllm+prefetch+pcie调度 > vllm+prefetch > vllm
#
# 在 3 个 QPS 梯度下 (1.0, 2.0, 4.0), 各跑 G0/G1/G4 三组, 每组 3 轮取平均.
# 设计为 nohup 挂机运行, 单个实验失败自动跳过.
#
# 用法:
#   nohup ./run_eplb_gradient_experiments.sh > experiment.log 2>&1 &
#   ./run_eplb_gradient_experiments.sh --qps-list "2.0"  # 只跑某个 QPS
#   ./run_eplb_gradient_experiments.sh --groups "g0,g4"   # 只跑某些组
#   ./run_eplb_gradient_experiments.sh --dry-run           # 预览实验矩阵

set -o pipefail
# 注意: 不用 set -e, 单个实验失败不能中止整个流程

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_EXP_DIR="$REPO_ROOT/run-experiment"
DATA_DIR="$REPO_ROOT/data"

# ============================================================
# Defaults
# ============================================================
MODEL_PATH="${MODEL_PATH:-/lpai/models/mistralai__mixtral-8x7b-instruct-v0_1/24-08-19-1318}"
API_PORT="${API_PORT:-8000}"
EP_SIZE=4
GPU_MEM_UTIL=0.5
NUM_GPU_BLOCKS=2500
MAX_NUM_SEQS=32
MAX_MODEL_LEN=4096
KV_OFFLOADING_SIZE=20
EPLB_STEP_INTERVAL=50
PREFETCH_LEAD_TIME=2.0
REQUEST_TIMEOUT=360
DATASET="pcie-heavy"
QPS_LIST="1.0 2.0 4.0"
RUN_GROUPS="g0,g1,g4"
ROUNDS=3
DRY_RUN=0
GPU_WAIT_INTERVAL=60        # 无可用 GPU 时等待间隔 (秒)
GPU_WAIT_MAX=7200            # 最长等待时间 (秒), 2小时
GPU_UTIL_THRESHOLD=30        # GPU 利用率阈值 (%), 越低越严格
GPU_MEM_THRESHOLD=40         # GPU 显存占用阈值 (%), 越低越严格
SERVER_READY_TIMEOUT=600     # vLLM 启动等待超时 (秒)
EXPERIMENT_TIMEOUT=1800      # 单个实验超时 (秒), 30分钟

# ============================================================
# Parse args
# ============================================================
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)            MODEL_PATH="$2";            shift 2 ;;
        --port)             API_PORT="$2";              shift 2 ;;
        --qps-list)         QPS_LIST="$2";              shift 2 ;;
        --groups)           RUN_GROUPS="$2";            shift 2 ;;
        --rounds)           ROUNDS="$2";                shift 2 ;;
        --num-gpu-blocks)   NUM_GPU_BLOCKS="$2";        shift 2 ;;
        --gpu-mem-util)     GPU_MEM_UTIL="$2";          shift 2 ;;
        --step-interval)    EPLB_STEP_INTERVAL="$2";    shift 2 ;;
        --ep-size)          EP_SIZE="$2";               shift 2 ;;
        --gpu-util-thresh)  GPU_UTIL_THRESHOLD="$2";    shift 2 ;;
        --gpu-mem-thresh)   GPU_MEM_THRESHOLD="$2";     shift 2 ;;
        --dry-run)          DRY_RUN=1;                  shift ;;
        -h|--help)
            sed -n '2,/^$/p' "$0" | grep '^#' | sed 's/^# \?//'
            exit 0 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

# ============================================================
# Dataset config
# ============================================================
case "$DATASET" in
    lite)       TRACE_FILE="$DATA_DIR/lite_dataset.jsonl";      NUM_CONV=18 ;;
    pcie-heavy) TRACE_FILE="$DATA_DIR/pcie_stress_heavy.jsonl"; NUM_CONV=40 ;;
    *)          TRACE_FILE="$DATASET";                          NUM_CONV=40 ;;
esac

if [[ ! -f "$TRACE_FILE" ]]; then
    echo "FATAL: trace file not found: $TRACE_FILE"
    exit 1
fi
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "FATAL: model not found: $MODEL_PATH"
    exit 1
fi

API_BASE="http://localhost:$API_PORT"
RESULTS_ROOT="$SCRIPT_DIR/results/gradient_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_ROOT"
PIDFILE="/tmp/vllm_gradient_${API_PORT}_$$.pid"

# ============================================================
# Logging
# ============================================================
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
log_section() {
    echo ""
    echo "================================================================"
    log "$*"
    echo "================================================================"
}

# ============================================================
# Process tracking: only kill OUR vLLM
# ============================================================
# We use setsid to give vLLM its own process group. Track PGID
# so we can kill the entire group (main + EP workers) without
# affecting other users' processes.
# ============================================================
VLLM_PGID=""

stop_our_vllm() {
    # 1) Kill by process group
    if [[ -n "$VLLM_PGID" ]]; then
        log "  Sending TERM to process group $VLLM_PGID..."
        kill -TERM -"$VLLM_PGID" 2>/dev/null || true
        sleep 3
        # Force kill survivors
        kill -9 -"$VLLM_PGID" 2>/dev/null || true
        sleep 2
    fi
    VLLM_PGID=""

    # 2) Fallback: PID file
    if [[ -f "$PIDFILE" ]]; then
        local pid
        pid=$(cat "$PIDFILE" 2>/dev/null)
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            log "  Killing PID $pid from pidfile..."
            kill -9 "$pid" 2>/dev/null || true
        fi
        rm -f "$PIDFILE"
    fi

    # 3) Fallback: anything holding OUR port
    local port_pids
    port_pids=$(lsof -ti :"$API_PORT" 2>/dev/null) || true
    if [[ -n "$port_pids" ]]; then
        log "  Killing remaining processes on port $API_PORT: $(echo $port_pids | tr '\n' ' ')"
        echo "$port_pids" | xargs kill -9 2>/dev/null || true
    fi

    sleep 2
}

cleanup() {
    log "Cleanup triggered (signal or exit)..."
    stop_our_vllm
    rm -f "$PIDFILE"
}
trap cleanup EXIT INT TERM

# ============================================================
# GPU selection: strict criteria for experiment accuracy
# ============================================================
# PCIe 调度实验对 GPU 利用率非常敏感:
#   - 其他进程的 GPU compute 会干扰 CUDA kernel 调度
#   - 其他进程的显存占用会影响 KV cache 可用空间
#   - 其他进程的 PCIe 传输会干扰我们的 H2D/D2H 延迟测量
#
# 策略:
#   Score = 0.7 * gpu_util% + 0.3 * mem_used%
#   拒绝: gpu_util > GPU_UTIL_THRESHOLD 或 mem_used > GPU_MEM_THRESHOLD
#   选择 score 最低的 EP_SIZE 块 GPU
# ============================================================

try_select_gpus() {
    local needed=$1

    if ! command -v nvidia-smi &>/dev/null; then
        echo "NOGPU"
        return
    fi

    local gpu_info
    gpu_info=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
               --format=csv,noheader,nounits 2>/dev/null)
    if [[ -z "$gpu_info" ]]; then
        echo "NOGPU"
        return
    fi

    local -a eligible=()

    while IFS=',' read -r idx util mem_used mem_total; do
        idx=$(echo "$idx" | xargs)
        util=$(echo "$util" | xargs)
        mem_used=$(echo "$mem_used" | xargs)
        mem_total=$(echo "$mem_total" | xargs)

        local mem_pct=0
        if [[ "$mem_total" -gt 0 ]]; then
            mem_pct=$((mem_used * 100 / mem_total))
        fi

        # Strict filtering
        if [[ "$util" -gt "$GPU_UTIL_THRESHOLD" ]] || [[ "$mem_pct" -gt "$GPU_MEM_THRESHOLD" ]]; then
            continue
        fi

        local score=$(( 70 * util + 30 * mem_pct ))
        eligible+=("${score}:${idx}")
    done <<< "$gpu_info"

    if [[ ${#eligible[@]} -lt $needed ]]; then
        echo "NOGPU"
        return
    fi

    IFS=$'\n' sorted=($(printf '%s\n' "${eligible[@]}" | sort -t: -k1 -n))
    unset IFS

    local selected=()
    for ((i = 0; i < needed; i++)); do
        selected+=("${sorted[$i]#*:}")
    done

    echo "$(IFS=,; echo "${selected[*]}")"
}

wait_for_gpus() {
    local needed=$1
    local waited=0

    while true; do
        local result
        result=$(try_select_gpus "$needed")

        if [[ "$result" != "NOGPU" ]]; then
            export CUDA_VISIBLE_DEVICES="$result"
            log "GPU acquired: CUDA_VISIBLE_DEVICES=$result"
            return 0
        fi

        if [[ $waited -ge $GPU_WAIT_MAX ]]; then
            log "ERROR: waited ${GPU_WAIT_MAX}s for GPUs, giving up"
            return 1
        fi

        if [[ $((waited % 300)) -eq 0 ]]; then
            log "Waiting for $needed idle GPUs (util<=${GPU_UTIL_THRESHOLD}%, mem<=${GPU_MEM_THRESHOLD}%)... ${waited}s elapsed"
            nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
                       --format=csv,noheader 2>/dev/null | while read -r line; do
                log "  GPU $line"
            done
        fi

        sleep "$GPU_WAIT_INTERVAL"
        waited=$((waited + GPU_WAIT_INTERVAL))
    done
}

# ============================================================
# Start vLLM
# ============================================================
start_vllm() {
    local label="$1"
    local enable_eplb="$2"
    local enable_pcie_sched="$3"
    local enable_eplb_phase="$4"
    local log_file="$5"

    log "Starting vLLM [$label] eplb=$enable_eplb sched=$enable_pcie_sched phase=$enable_eplb_phase"

    stop_our_vllm

    local CMD_ARGS=(
        --model "$MODEL_PATH"
        --host 0.0.0.0
        --port "$API_PORT"
        --dtype float16
        --tensor-parallel-size "$EP_SIZE"
        --gpu-memory-utilization "$GPU_MEM_UTIL"
        --max-num-seqs "$MAX_NUM_SEQS"
        --max-model-len "$MAX_MODEL_LEN"
        --trust-remote-code
        --enforce-eager
        --enable-expert-parallel
        --kv-offloading-size "$KV_OFFLOADING_SIZE"
        --kv-offloading-backend native
        --swap-space 64
        --enable-prefix-caching
        --disable-hybrid-kv-cache-manager
        --num-gpu-blocks-override "$NUM_GPU_BLOCKS"
    )

    if [[ "$enable_eplb" -eq 1 ]]; then
        CMD_ARGS+=(
            --enable-eplb
            --eplb-config "{\"step_interval\": $EPLB_STEP_INTERVAL, \"num_redundant_experts\": 0}"
        )
    fi

    export NCCL_P2P_DISABLE=1 NCCL_NVLS_ENABLE=0 VLLM_TEST_ENABLE_EP=1 HF_HUB_OFFLINE=1
    unset VLLM_PCIE_SCHEDULER VLLM_EPLB_PHASE_AWARE
    [[ "$enable_pcie_sched" -eq 1 ]] && export VLLM_PCIE_SCHEDULER=1
    [[ "$enable_eplb_phase" -eq 1 ]] && export VLLM_EPLB_PHASE_AWARE=1

    setsid vllm serve "${CMD_ARGS[@]}" > "$log_file" 2>&1 &
    local vllm_pid=$!
    VLLM_PGID=$vllm_pid
    echo "$vllm_pid" > "$PIDFILE"
    log "  PID=$vllm_pid PGID=$VLLM_PGID"

    local waited=0
    while [[ $waited -lt $SERVER_READY_TIMEOUT ]]; do
        if curl -s "$API_BASE/health" &>/dev/null; then
            log "  Server ready (${waited}s)"
            sleep 5
            return 0
        fi
        if ! kill -0 "$vllm_pid" 2>/dev/null; then
            log "  ERROR: server died. Log: $log_file"
            stop_our_vllm
            return 1
        fi
        sleep 3
        waited=$((waited + 3))
    done
    log "  ERROR: server start timeout (${SERVER_READY_TIMEOUT}s)"
    stop_our_vllm
    return 1
}

# ============================================================
# Run workload with timeout
# ============================================================
run_workload() {
    local mode="$1"
    local qps="$2"
    local label="$3"
    local round="$4"
    local output_dir="$5"
    local output="$output_dir/${label}_r${round}.jsonl"

    log "  Workload: mode=$mode qps=$qps label=$label round=$round"

    curl -s -X POST "$API_BASE/reset_prefix_cache?reset_external=true" >/dev/null 2>&1 || true
    sleep 3

    timeout "$EXPERIMENT_TIMEOUT" \
        python3 "$RUN_EXP_DIR/prefetch_ab_runner.py" \
            --trace-file "$TRACE_FILE" \
            --mode "$mode" \
            --qps "$qps" \
            --num-multi-turn "$NUM_CONV" \
            --model "$MODEL_PATH" \
            --api-base "$API_BASE/v1" \
            --output "$output" \
            --seed $((42 + round)) \
            --request-timeout "$REQUEST_TIMEOUT" \
            --prefetch-lead-time "$PREFETCH_LEAD_TIME" \
            --schedule-mode uniform \
            > "$output_dir/${label}_r${round}.log" 2>&1
    local rc=$?

    if [[ $rc -eq 124 ]]; then
        log "  WARNING: workload timed out after ${EXPERIMENT_TIMEOUT}s"
        return 1
    elif [[ $rc -ne 0 ]]; then
        log "  WARNING: workload failed (rc=$rc)"
        return 1
    fi

    # Validate output has data
    if [[ ! -s "$output" ]]; then
        log "  WARNING: output is empty"
        return 1
    fi
    local lines
    lines=$(wc -l < "$output")
    log "  Done: $lines results -> $output"
    return 0
}

# ============================================================
# Group definitions
# ============================================================
# G0: vllm baseline        (EP + Offload, no prefetch, no EPLB, no scheduler)
# G1: vllm + prefetch       (EP + Offload + Prefetch, no EPLB, no scheduler)
# G4: vllm + prefetch + pcie调度 (EP + Offload + Prefetch + EPLB + Scheduler + Phase)
# ============================================================
should_run_group() {
    [[ "$RUN_GROUPS" == "all" ]] || [[ ",$RUN_GROUPS," == *",$1,"* ]]
}

# Returns: mode enable_eplb enable_sched enable_phase
get_group_config() {
    case "$1" in
        g0) echo "baseline 0 0 0" ;;
        g1) echo "prefetch 0 0 0" ;;
        g4) echo "prefetch 1 1 1" ;;
        *)  echo ""; return 1 ;;
    esac
}

get_group_desc() {
    case "$1" in
        g0) echo "vllm (baseline)" ;;
        g1) echo "vllm+prefetch" ;;
        g4) echo "vllm+prefetch+pcie_sched" ;;
    esac
}

# ============================================================
# Report generation
# ============================================================
generate_report() {
    local report="$RESULTS_ROOT/report.md"
    log "Generating report -> $report"

    {
        echo "# EPLB Gradient Experiment Report"
        echo ""
        echo "Generated: $(date '+%Y-%m-%d %H:%M:%S')"
        echo ""
        echo "## Configuration"
        echo ""
        echo "| Parameter | Value |"
        echo "|-----------|-------|"
        echo "| Model | $(basename "$MODEL_PATH") |"
        echo "| EP size | $EP_SIZE |"
        echo "| GPU blocks | $NUM_GPU_BLOCKS |"
        echo "| GPU mem util | $GPU_MEM_UTIL |"
        echo "| Dataset | $DATASET |"
        echo "| EPLB step | $EPLB_STEP_INTERVAL |"
        echo "| Rounds | $ROUNDS |"
        echo ""

        for qps in $QPS_LIST; do
            local qps_tag="${qps//./_}"
            echo "## QPS = $qps"
            echo ""
            echo "| Group | Description | Requests | Mean TTFT | P50 | P95 | P99 | Cache Hit |"
            echo "|-------|-------------|----------|-----------|-----|-----|-----|-----------|"

            for group in g0 g1 g4; do
                if ! should_run_group "$group"; then continue; fi
                local desc
                desc=$(get_group_desc "$group")

                local combined=""
                for r in $(seq 1 "$ROUNDS"); do
                    local f="$RESULTS_ROOT/qps_${qps_tag}/${group}_r${r}.jsonl"
                    [[ -f "$f" ]] && combined="$combined $f"
                done

                if [[ -z "$combined" ]]; then
                    echo "| $group | $desc | - | - | - | - | - | - |"
                    continue
                fi

                python3 -c "
import json, sys, os
import numpy as np

ttfts, cached_list = [], []
for fpath in '''$combined'''.split():
    if not os.path.isfile(fpath): continue
    with open(fpath) as f:
        for line in f:
            d = json.loads(line)
            if d.get('ttft_ms') is not None:
                ttfts.append(d['ttft_ms'])
            cached = d.get('cached_tokens', 0) or 0
            prompt = d.get('prompt_tokens', 1) or 1
            if cached > 0:
                cached_list.append(cached / prompt)

if not ttfts:
    print('| $group | $desc | 0 | - | - | - | - | - |')
else:
    arr = np.array(ttfts)
    hit = f'{np.mean(cached_list)*100:.1f}%' if cached_list else '0%'
    print(f'| $group | $desc | {len(arr)} | {np.mean(arr):.0f} | {np.percentile(arr,50):.0f} | {np.percentile(arr,95):.0f} | {np.percentile(arr,99):.0f} | {hit} |')
" 2>/dev/null || echo "| $group | $desc | ERROR | - | - | - | - | - |"
            done
            echo ""
        done

        # Improvement summary
        echo "## Improvement Summary"
        echo ""
        echo "| QPS | G1 vs G0 (Mean) | G4 vs G1 (Mean) | G4 vs G0 (Mean) | G4 vs G1 (P95) |"
        echo "|-----|-----------------|-----------------|-----------------|----------------|"

        for qps in $QPS_LIST; do
            local qps_tag="${qps//./_}"
            python3 -c "
import json, sys, os
import numpy as np

def load_ttfts(pattern, rounds):
    ttfts = []
    for r in range(1, rounds+1):
        fpath = pattern.replace('RR', str(r))
        if not os.path.isfile(fpath): continue
        with open(fpath) as f:
            for line in f:
                d = json.loads(line)
                if d.get('ttft_ms') is not None:
                    ttfts.append(d['ttft_ms'])
    return np.array(ttfts) if ttfts else None

base = '$RESULTS_ROOT/qps_${qps_tag}'
g0 = load_ttfts(f'{base}/g0_rRR.jsonl', $ROUNDS)
g1 = load_ttfts(f'{base}/g1_rRR.jsonl', $ROUNDS)
g4 = load_ttfts(f'{base}/g4_rRR.jsonl', $ROUNDS)

def pct(old, new):
    if old is None or new is None or len(old)==0: return '-'
    return f'{(np.mean(new)/np.mean(old)-1)*100:+.1f}%'

def pct95(old, new):
    if old is None or new is None or len(old)==0: return '-'
    return f'{(np.percentile(new,95)/np.percentile(old,95)-1)*100:+.1f}%'

print(f'| $qps | {pct(g0,g1)} | {pct(g1,g4)} | {pct(g0,g4)} | {pct95(g1,g4)} |')
" 2>/dev/null || echo "| $qps | ERROR | ERROR | ERROR | ERROR |"
        done
        echo ""
    } > "$report"

    cat "$report"
}

# ============================================================
# Dry run: show experiment matrix
# ============================================================
if [[ $DRY_RUN -eq 1 ]]; then
    echo "========================================"
    echo "DRY RUN — Experiment Matrix"
    echo "========================================"
    echo ""
    echo "Config: blk=$NUM_GPU_BLOCKS gpu_mem=$GPU_MEM_UTIL rounds=$ROUNDS"
    echo "GPU selection: util<=${GPU_UTIL_THRESHOLD}% mem<=${GPU_MEM_THRESHOLD}%"
    echo ""
    total=0
    for qps in $QPS_LIST; do
        echo "QPS=$qps:"
        for group in g0 g1 g4; do
            if should_run_group "$group"; then
                config=$(get_group_config "$group")
                desc=$(get_group_desc "$group")
                echo "  $group ($desc): $ROUNDS rounds x mode=${config%% *}"
                total=$((total + ROUNDS))
            fi
        done
    done
    echo ""
    echo "Total experiments: $total"
    echo "Each experiment: start fresh vLLM + run workload + stop vLLM"
    echo "vLLM restarts every round to ensure clean state."
    exit 0
fi

# ============================================================
# Main experiment loop
# ============================================================
log_section "EPLB Gradient Experiments"
log "Model:      $MODEL_PATH"
log "EP size:    $EP_SIZE"
log "GPU blocks: $NUM_GPU_BLOCKS"
log "GPU mem:    $GPU_MEM_UTIL"
log "Dataset:    $DATASET"
log "QPS list:   $QPS_LIST"
log "Groups:     $RUN_GROUPS"
log "Rounds:     $ROUNDS"
log "Results:    $RESULTS_ROOT"

TOTAL_EXPERIMENTS=0
PASSED_EXPERIMENTS=0
FAILED_EXPERIMENTS=0
SKIPPED_EXPERIMENTS=0

# Loop order: round → qps → group
# Each round completes all QPS x group combinations before the next round
# starts, so that multi-round averages spread over time and reduce
# temporal bias (e.g. cluster load variation).
# vLLM restarts every single experiment for clean state.
for round in $(seq 1 "$ROUNDS"); do
    log_section "ROUND $round / $ROUNDS"

    for qps in $QPS_LIST; do
        qps_tag="${qps//./_}"
        qps_dir="$RESULTS_ROOT/qps_${qps_tag}"
        mkdir -p "$qps_dir"

        log_section "QPS = $qps  (Round $round)"

        for group in g0 g1 g4; do
            if ! should_run_group "$group"; then continue; fi

            TOTAL_EXPERIMENTS=$((TOTAL_EXPERIMENTS + 1))
            desc=$(get_group_desc "$group")
            config=$(get_group_config "$group")
            read -r mode enable_eplb enable_sched enable_phase <<< "$config"

            log_section "[$group] $desc @ QPS=$qps  Round $round/$ROUNDS"

            # --- Acquire GPUs ---
            if ! wait_for_gpus "$EP_SIZE"; then
                log "SKIP [$group @ QPS=$qps r$round]: no GPUs available"
                SKIPPED_EXPERIMENTS=$((SKIPPED_EXPERIMENTS + 1))
                continue
            fi

            # --- Fresh vLLM instance per experiment ---
            vllm_log="$qps_dir/vllm_${group}_r${round}.log"
            if ! start_vllm "$group" "$enable_eplb" "$enable_sched" "$enable_phase" "$vllm_log"; then
                log "FAIL [$group @ QPS=$qps r$round]: vLLM failed to start"
                FAILED_EXPERIMENTS=$((FAILED_EXPERIMENTS + 1))
                stop_our_vllm
                continue
            fi

            # --- Run workload ---
            if run_workload "$mode" "$qps" "$group" "$round" "$qps_dir"; then
                PASSED_EXPERIMENTS=$((PASSED_EXPERIMENTS + 1))
            else
                FAILED_EXPERIMENTS=$((FAILED_EXPERIMENTS + 1))
                log "  Round $round failed, continuing..."
            fi

            # --- Always stop vLLM ---
            stop_our_vllm
        done
    done
done

# ============================================================
# Summary
# ============================================================
log_section "Experiment Complete"
log "Total: $TOTAL_EXPERIMENTS | Passed: $PASSED_EXPERIMENTS | Failed: $FAILED_EXPERIMENTS | Skipped: $SKIPPED_EXPERIMENTS"
log "Results: $RESULTS_ROOT"

generate_report

log "All done."
