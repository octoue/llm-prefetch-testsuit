#!/bin/bash
# DeepSeek-V2-Lite (16B) EPLB Sync + Async 梯度实验
#
# 在 DeepSeek-V2-Lite 上同时测试 EPLB sync 和 async 两种模式,
# 一次运行产出全部对比数据. 实验组与论文一致.
#
# 实验组:
#   G1: Reference — EP + Offload + Prefetch (无 EPLB)
#   G2: Baseline  — EP + Offload + Prefetch + EPLB (无调度)
#   G4: Full      — EP + Offload + Prefetch + EPLB + PCIe Scheduler + EPLB Phase
#
# EPLB 模式:
#   sync  — 同步重排, 阻塞推理流水线 (默认模式)
#   async — 异步迁移, 后台逐层传输, 推理不中断
#
# 注意: G1 不启用 EPLB, 因此 sync/async 对 G1 无影响, 只跑一次.
#
# 用法:
#   nohup ./run_deepseek_v2_lite_gradient.sh > experiment.log 2>&1 &
#   ./run_deepseek_v2_lite_gradient.sh --model /path/to/deepseek-v2-lite
#   ./run_deepseek_v2_lite_gradient.sh --eplb-modes "async"  # 只跑 async
#   ./run_deepseek_v2_lite_gradient.sh --dry-run

set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_EXP_DIR="$REPO_ROOT/run-experiment"
DATA_DIR="$REPO_ROOT/data"

# ============================================================
# Defaults — DeepSeek-V2-Lite specific
# ============================================================
MODEL_PATH="${MODEL_PATH:-/lpai/models/deepseek-ai/DeepSeek-V2-Lite}"
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
QPS_LIST="0.5 1.0 1.5 2.0 2.5"
RUN_GROUPS="g1,g2,g4"
EPLB_MODES="sync async"
ROUNDS=3
DRY_RUN=0
GPU_WAIT_INTERVAL=60
GPU_WAIT_MAX=7200
GPU_UTIL_THRESHOLD=30
GPU_MEM_THRESHOLD=40
SERVER_READY_TIMEOUT=600
EXPERIMENT_TIMEOUT=1800

# ============================================================
# Parse args
# ============================================================
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)            MODEL_PATH="$2";            shift 2 ;;
        --port)             API_PORT="$2";              shift 2 ;;
        --qps-list)         QPS_LIST="$2";              shift 2 ;;
        --groups)           RUN_GROUPS="$2";            shift 2 ;;
        --eplb-modes)       EPLB_MODES="$2";            shift 2 ;;
        --rounds)           ROUNDS="$2";                shift 2 ;;
        --num-gpu-blocks)   NUM_GPU_BLOCKS="$2";        shift 2 ;;
        --gpu-mem-util)     GPU_MEM_UTIL="$2";          shift 2 ;;
        --step-interval)    EPLB_STEP_INTERVAL="$2";    shift 2 ;;
        --ep-size)          EP_SIZE="$2";               shift 2 ;;
        --kv-offloading)    KV_OFFLOADING_SIZE="$2";    shift 2 ;;
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
    echo "  Set MODEL_PATH env var or use --model <path>"
    exit 1
fi

API_BASE="http://localhost:$API_PORT"
RESULTS_ROOT="$SCRIPT_DIR/results/deepseek_v2_lite_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_ROOT"
PIDFILE="/tmp/vllm_deepseek_ep_${API_PORT}_$$.pid"

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
# Process management
# ============================================================
VLLM_PGID=""

stop_our_vllm() {
    if [[ -n "$VLLM_PGID" ]]; then
        log "  Sending TERM to process group $VLLM_PGID..."
        kill -TERM -"$VLLM_PGID" 2>/dev/null || true
        sleep 3
        kill -9 -"$VLLM_PGID" 2>/dev/null || true
        sleep 2
    fi
    VLLM_PGID=""

    if [[ -f "$PIDFILE" ]]; then
        local pid; pid=$(cat "$PIDFILE" 2>/dev/null)
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null || true
        fi
        rm -f "$PIDFILE"
    fi

    local port_pids
    port_pids=$(lsof -ti :"$API_PORT" 2>/dev/null) || true
    if [[ -n "$port_pids" ]]; then
        echo "$port_pids" | xargs kill -9 2>/dev/null || true
    fi
    sleep 2
}

cleanup() { log "Cleanup triggered..."; stop_our_vllm; rm -f "$PIDFILE"; }
trap cleanup EXIT INT TERM

# ============================================================
# GPU selection
# ============================================================
try_select_gpus() {
    local needed=$1
    if ! command -v nvidia-smi &>/dev/null; then echo "NOGPU"; return; fi

    local gpu_info
    gpu_info=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
               --format=csv,noheader,nounits 2>/dev/null)
    if [[ -z "$gpu_info" ]]; then echo "NOGPU"; return; fi

    local -a eligible=()
    while IFS=',' read -r idx util mem_used mem_total; do
        idx=$(echo "$idx" | xargs); util=$(echo "$util" | xargs)
        mem_used=$(echo "$mem_used" | xargs); mem_total=$(echo "$mem_total" | xargs)
        local mem_pct=0
        [[ "$mem_total" -gt 0 ]] && mem_pct=$((mem_used * 100 / mem_total))
        if [[ "$util" -gt "$GPU_UTIL_THRESHOLD" ]] || [[ "$mem_pct" -gt "$GPU_MEM_THRESHOLD" ]]; then
            continue
        fi
        eligible+=("$(( 70 * util + 30 * mem_pct )):${idx}")
    done <<< "$gpu_info"

    if [[ ${#eligible[@]} -lt $needed ]]; then echo "NOGPU"; return; fi
    IFS=$'\n' sorted=($(printf '%s\n' "${eligible[@]}" | sort -t: -k1 -n)); unset IFS
    local selected=()
    for ((i = 0; i < needed; i++)); do selected+=("${sorted[$i]#*:}"); done
    echo "$(IFS=,; echo "${selected[*]}")"
}

wait_for_gpus() {
    local needed=$1 waited=0
    while true; do
        local result; result=$(try_select_gpus "$needed")
        if [[ "$result" != "NOGPU" ]]; then
            export CUDA_VISIBLE_DEVICES="$result"
            log "GPU acquired: CUDA_VISIBLE_DEVICES=$result"
            return 0
        fi
        if [[ $waited -ge $GPU_WAIT_MAX ]]; then
            log "ERROR: waited ${GPU_WAIT_MAX}s for GPUs, giving up"; return 1
        fi
        if [[ $((waited % 300)) -eq 0 ]]; then
            log "Waiting for $needed idle GPUs... ${waited}s elapsed"
        fi
        sleep "$GPU_WAIT_INTERVAL"
        waited=$((waited + GPU_WAIT_INTERVAL))
    done
}

# ============================================================
# Start vLLM — supports sync/async EPLB mode
# ============================================================
start_vllm() {
    local label="$1"
    local enable_eplb="$2"
    local enable_pcie_sched="$3"
    local enable_eplb_phase="$4"
    local eplb_async="$5"   # "true" or "false"
    local log_file="$6"

    local async_label="n/a"
    [[ "$enable_eplb" -eq 1 ]] && async_label="$eplb_async"

    log "Starting vLLM [$label] eplb=$enable_eplb(async=$async_label) sched=$enable_pcie_sched phase=$enable_eplb_phase"
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
            --eplb-config "{\"step_interval\": $EPLB_STEP_INTERVAL, \"num_redundant_experts\": 0, \"use_async\": $eplb_async}"
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
# Run workload
# ============================================================
run_workload() {
    local mode="$1" qps="$2" label="$3" round="$4" output_dir="$5"
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
        log "  WARNING: timed out after ${EXPERIMENT_TIMEOUT}s"; return 1
    elif [[ $rc -ne 0 ]]; then
        log "  WARNING: failed (rc=$rc)"; return 1
    fi
    if [[ ! -s "$output" ]]; then
        log "  WARNING: output is empty"; return 1
    fi
    log "  Done: $(wc -l < "$output") results -> $output"
    return 0
}

# ============================================================
# Group definitions
# ============================================================
should_run_group() {
    [[ "$RUN_GROUPS" == "all" ]] || [[ ",$RUN_GROUPS," == *",$1,"* ]]
}

# Returns: mode enable_eplb enable_sched enable_phase
get_group_config() {
    case "$1" in
        g1) echo "prefetch 0 0 0" ;;  # Reference: no EPLB
        g2) echo "prefetch 1 0 0" ;;  # Baseline: +EPLB, no scheduler
        g4) echo "prefetch 1 1 1" ;;  # Full: +EPLB + Scheduler + Phase
        *)  echo ""; return 1 ;;
    esac
}

get_group_desc() {
    local mode_label="$2"  # sync or async, optional
    case "$1" in
        g1) echo "Reference (no EPLB)" ;;
        g2) echo "Baseline (+EPLB ${mode_label:-})" ;;
        g4) echo "+PCIe Sched+Phase (${mode_label:-})" ;;
    esac
}

# ============================================================
# Report generation
# ============================================================
generate_report() {
    local report="$RESULTS_ROOT/report.md"
    log "Generating report -> $report"

    {
        echo "# DeepSeek-V2-Lite EPLB Sync+Async Gradient Report"
        echo ""
        echo "Generated: $(date '+%Y-%m-%d %H:%M:%S')"
        echo ""
        echo "## Configuration"
        echo ""
        echo "| Parameter | Value |"
        echo "|-----------|-------|"
        echo "| Model | $(basename "$MODEL_PATH") |"
        echo "| EP size | $EP_SIZE |"
        echo "| EPLB modes | $EPLB_MODES |"
        echo "| GPU blocks | $NUM_GPU_BLOCKS |"
        echo "| GPU mem util | $GPU_MEM_UTIL |"
        echo "| Offload | ${KV_OFFLOADING_SIZE}GiB |"
        echo "| Dataset | $DATASET |"
        echo "| EPLB step | $EPLB_STEP_INTERVAL |"
        echo "| Rounds | $ROUNDS |"
        echo ""

        for eplb_mode in $EPLB_MODES; do
            echo "---"
            echo ""
            echo "# EPLB Mode: ${eplb_mode^^}"
            echo ""

            for qps in $QPS_LIST; do
                local qps_tag="${qps//./_}"
                echo "## QPS = $qps (${eplb_mode})"
                echo ""
                echo "| Group | Description | Requests | Mean TTFT | P50 | P95 | P99 | Cache Hit |"
                echo "|-------|-------------|----------|-----------|-----|-----|-----|-----------|"

                for group in g1 g2 g4; do
                    if ! should_run_group "$group"; then continue; fi
                    local desc; desc=$(get_group_desc "$group" "$eplb_mode")

                    # G1 不受 EPLB 模式影响, 结果存在 sync 目录下
                    local result_mode="$eplb_mode"
                    [[ "$group" == "g1" ]] && result_mode="sync"

                    local combined=""
                    for r in $(seq 1 "$ROUNDS"); do
                        local f="$RESULTS_ROOT/${result_mode}/qps_${qps_tag}/${group}_r${r}.jsonl"
                        [[ -f "$f" ]] && combined="$combined $f"
                    done

                    if [[ -z "$combined" ]]; then
                        echo "| $group | $desc | - | - | - | - | - | - |"
                        continue
                    fi

                    python3 -c "
import json, os
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
        done

        # Sync vs Async comparison
        echo "---"
        echo ""
        echo "# Sync vs Async Comparison"
        echo ""
        echo "| QPS | G2 Sync Mean | G2 Async Mean | Async vs Sync | G4 Sync Mean | G4 Async Mean | Async vs Sync |"
        echo "|-----|-------------|---------------|---------------|-------------|---------------|---------------|"

        for qps in $QPS_LIST; do
            local qps_tag="${qps//./_}"
            python3 -c "
import json, os
import numpy as np

def load_ttfts(base_dir, group, rounds):
    ttfts = []
    for r in range(1, rounds+1):
        fpath = f'{base_dir}/{group}_r{r}.jsonl'
        if not os.path.isfile(fpath): continue
        with open(fpath) as f:
            for line in f:
                d = json.loads(line)
                if d.get('ttft_ms') is not None:
                    ttfts.append(d['ttft_ms'])
    return np.array(ttfts) if ttfts else None

root = '$RESULTS_ROOT'
qps_tag = '${qps_tag}'

g2_sync  = load_ttfts(f'{root}/sync/qps_{qps_tag}', 'g2', $ROUNDS)
g2_async = load_ttfts(f'{root}/async/qps_{qps_tag}', 'g2', $ROUNDS)
g4_sync  = load_ttfts(f'{root}/sync/qps_{qps_tag}', 'g4', $ROUNDS)
g4_async = load_ttfts(f'{root}/async/qps_{qps_tag}', 'g4', $ROUNDS)

def fmt(arr):
    return f'{np.mean(arr):.0f}' if arr is not None and len(arr)>0 else '-'
def cmp(s, a):
    if s is None or a is None or len(s)==0 or len(a)==0: return '-'
    return f'{(np.mean(a)/np.mean(s)-1)*100:+.1f}%'

print(f'| $qps | {fmt(g2_sync)} | {fmt(g2_async)} | {cmp(g2_sync,g2_async)} | {fmt(g4_sync)} | {fmt(g4_async)} | {cmp(g4_sync,g4_async)} |')
" 2>/dev/null || echo "| $qps | - | - | - | - | - | - |"
        done
        echo ""
    } > "$report"

    cat "$report"
}

# ============================================================
# Dry run
# ============================================================
if [[ $DRY_RUN -eq 1 ]]; then
    echo "========================================"
    echo "DRY RUN — DeepSeek-V2-Lite Experiment Matrix"
    echo "========================================"
    echo ""
    echo "Model: $MODEL_PATH"
    echo "EP=$EP_SIZE  blk=$NUM_GPU_BLOCKS  gpu_mem=$GPU_MEM_UTIL  step=$EPLB_STEP_INTERVAL  rounds=$ROUNDS"
    echo "EPLB modes: $EPLB_MODES"
    echo ""
    total=0
    for eplb_mode in $EPLB_MODES; do
        echo "=== EPLB mode: $eplb_mode ==="
        for qps in $QPS_LIST; do
            echo "  QPS=$qps:"
            for group in g1 g2 g4; do
                if should_run_group "$group"; then
                    # G1 只跑一次 (在 sync 阶段)
                    if [[ "$group" == "g1" && "$eplb_mode" == "async" ]]; then
                        echo "    $group (Reference, no EPLB): SKIP (already run in sync)"
                        continue
                    fi
                    desc=$(get_group_desc "$group" "$eplb_mode")
                    echo "    $group ($desc): $ROUNDS rounds"
                    total=$((total + ROUNDS))
                fi
            done
        done
    done
    echo ""
    echo "Total experiments: $total"
    exit 0
fi

# ============================================================
# Main loop: eplb_mode → round → qps → group
# ============================================================
log_section "DeepSeek-V2-Lite EPLB Gradient Experiments"
log "Model:      $MODEL_PATH"
log "EP size:    $EP_SIZE"
log "EPLB modes: $EPLB_MODES"
log "GPU blocks: $NUM_GPU_BLOCKS"
log "GPU mem:    $GPU_MEM_UTIL"
log "Offload:    ${KV_OFFLOADING_SIZE}GiB"
log "Dataset:    $DATASET"
log "QPS list:   $QPS_LIST"
log "Groups:     $RUN_GROUPS"
log "Rounds:     $ROUNDS"
log "Results:    $RESULTS_ROOT"

TOTAL=0; PASSED=0; FAILED=0; SKIPPED=0

for eplb_mode in $EPLB_MODES; do
    if [[ "$eplb_mode" == "sync" ]]; then
        eplb_async_val="false"
    else
        eplb_async_val="true"
    fi

    log_section "EPLB MODE: ${eplb_mode^^} (use_async=$eplb_async_val)"

    mode_dir="$RESULTS_ROOT/$eplb_mode"
    mkdir -p "$mode_dir"

    for round in $(seq 1 "$ROUNDS"); do
        log_section "ROUND $round / $ROUNDS  [${eplb_mode^^}]"

        for qps in $QPS_LIST; do
            qps_tag="${qps//./_}"
            qps_dir="$mode_dir/qps_${qps_tag}"
            mkdir -p "$qps_dir"

            log_section "QPS = $qps  [${eplb_mode^^}]  Round $round"

            for group in g1 g2 g4; do
                if ! should_run_group "$group"; then continue; fi

                # G1 不使用 EPLB, 在 async 阶段跳过 (复用 sync 结果)
                if [[ "$group" == "g1" && "$eplb_mode" == "async" ]]; then
                    log "SKIP [$group @ QPS=$qps ${eplb_mode} r$round]: G1 already run in sync"
                    continue
                fi

                TOTAL=$((TOTAL + 1))
                desc=$(get_group_desc "$group" "$eplb_mode")
                config=$(get_group_config "$group")
                read -r mode enable_eplb enable_sched enable_phase <<< "$config"

                log_section "[$group] $desc @ QPS=$qps  [${eplb_mode^^}]  Round $round/$ROUNDS"

                if ! wait_for_gpus "$EP_SIZE"; then
                    log "SKIP: no GPUs"
                    SKIPPED=$((SKIPPED + 1))
                    continue
                fi

                vllm_log="$qps_dir/vllm_${group}_r${round}.log"
                if ! start_vllm "$group" "$enable_eplb" "$enable_sched" "$enable_phase" "$eplb_async_val" "$vllm_log"; then
                    log "FAIL: vLLM failed to start"
                    FAILED=$((FAILED + 1))
                    stop_our_vllm
                    continue
                fi

                if run_workload "$mode" "$qps" "$group" "$round" "$qps_dir"; then
                    PASSED=$((PASSED + 1))
                else
                    FAILED=$((FAILED + 1))
                fi

                stop_our_vllm
            done
        done
    done
done

log_section "Experiment Complete"
log "Total: $TOTAL | Passed: $PASSED | Failed: $FAILED | Skipped: $SKIPPED"
log "Results: $RESULTS_ROOT"

generate_report
log "All done."
