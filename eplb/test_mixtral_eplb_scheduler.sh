#!/bin/bash
# Mixtral-8x7B EP=4 + EPLB + PCIe Scheduler 联合实验
#
# 目的: 在 Mixtral-8x7B 上验证 EPLB-Phase-Aware 调度的效果,
#       并尝试触发 KV Offloading 以测试完整 5 种 PCIe 流量叠加.
#
# 实验组:
#   G0: EP + Offload (baseline)
#   G1: EP + Offload + Prefetch
#   G2: EP + Offload + Prefetch + EPLB (无调度)
#   G3: EP + Offload + Prefetch + EPLB + PCIe Scheduler (无 EPLB Phase)
#   G4: EP + Offload + Prefetch + EPLB + PCIe Scheduler + EPLB Phase
#
# 用法:
#   ./test_mixtral_eplb_scheduler.sh [options]
#
# 选项:
#   --model PATH          Mixtral-8x7B 模型路径
#   --port PORT           API 端口 (default: 8000)
#   --qps QPS             请求速率 (default: 1.0)
#   --dataset NAME        数据集: lite / pcie-heavy (default: pcie-heavy)
#   --groups GROUPS       运行的组, 逗号分隔或 'all' (default: all)
#   --step-interval N     EPLB step interval (default: 100)
#   --gpu-mem-util FLOAT  GPU 显存利用率 (default: 0.5)
#   --rounds N            重复轮次 (default: 1)

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_EXP_DIR="$REPO_ROOT/run-experiment"
DATA_DIR="$REPO_ROOT/data"

# ============================================================
# Defaults
# ============================================================
MODEL_PATH="${MODEL_PATH:-/lpai/models/mistralai__mixtral-8x7b-instruct-v0_1/24-08-19-1318}"
API_PORT="${API_PORT:-8000}"
QPS=1.0
DATASET="pcie-heavy"
RUN_GROUPS="all"
EPLB_STEP_INTERVAL=100
EP_SIZE=4
GPU_MEM_UTIL=0.4
MAX_NUM_SEQS=32
MAX_MODEL_LEN=4096
KV_OFFLOADING_SIZE=20
PREFETCH_LEAD_TIME=2.0
REQUEST_TIMEOUT=360
ROUNDS=1

# ============================================================
# Parse args
# ============================================================
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)           MODEL_PATH="$2";              shift 2 ;;
        --port)            API_PORT="$2";                shift 2 ;;
        --qps)             QPS="$2";                     shift 2 ;;
        --dataset)         DATASET="$2";                 shift 2 ;;
        --groups)          RUN_GROUPS="$2";              shift 2 ;;
        --step-interval)   EPLB_STEP_INTERVAL="$2";     shift 2 ;;
        --ep-size)         EP_SIZE="$2";                 shift 2 ;;
        --gpu-mem-util)    GPU_MEM_UTIL="$2";            shift 2 ;;
        --lead-time)       PREFETCH_LEAD_TIME="$2";      shift 2 ;;
        --kv-offloading)   KV_OFFLOADING_SIZE="$2";      shift 2 ;;
        --rounds)          ROUNDS="$2";                  shift 2 ;;
        -h|--help)
            sed -n '2,/^$/p' "$0" | grep '^#' | sed 's/^# \?//'
            exit 0 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

# ============================================================
# Validate
# ============================================================
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "Error: 模型路径不存在: $MODEL_PATH"
    exit 1
fi

# ============================================================
# Dataset config
# ============================================================
case "$DATASET" in
    lite)
        TRACE_FILE="$DATA_DIR/lite_dataset.jsonl"
        NUM_CONV=18
        ;;
    pcie-heavy)
        TRACE_FILE="$DATA_DIR/pcie_stress_heavy.jsonl"
        NUM_CONV=40
        ;;
    *)
        TRACE_FILE="$DATASET"
        NUM_CONV=20
        ;;
esac

if [[ ! -f "$TRACE_FILE" ]]; then
    echo "Error: trace file not found: $TRACE_FILE"
    exit 1
fi

API_BASE="http://localhost:$API_PORT"
RESULTS_DIR="$SCRIPT_DIR/results/mixtral_eplb_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"
PIDFILE="/tmp/vllm_mixtral_ep_${API_PORT}.pid"

should_run_group() {
    [[ "$RUN_GROUPS" == "all" ]] || [[ ",$RUN_GROUPS," == *",$1,"* ]]
}

# ============================================================
# Robust cleanup: kill vLLM process tree + release GPU
# ============================================================
kill_process_tree() {
    local pid="$1"
    # Find all descendants (children, grandchildren, etc.)
    local children
    children=$(pgrep -P "$pid" 2>/dev/null) || true
    for child in $children; do
        kill_process_tree "$child"
    done
    kill -9 "$pid" 2>/dev/null || true
}

stop_all_vllm() {
    echo ""
    echo "Cleaning up vLLM processes..."

    # 1) Kill by PID file (main process + entire tree)
    if [[ -f "$PIDFILE" ]]; then
        local pid
        pid=$(cat "$PIDFILE" 2>/dev/null)
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            echo "  Killing process tree rooted at PID $pid..."
            # Try graceful TERM first
            kill -TERM "$pid" 2>/dev/null || true
            sleep 2
            # Then force-kill entire tree
            kill_process_tree "$pid"
        fi
        rm -f "$PIDFILE"
    fi

    # 2) Kill anything still holding the port
    local port_pids
    port_pids=$(lsof -ti :"$API_PORT" 2>/dev/null) || true
    if [[ -n "$port_pids" ]]; then
        echo "  Killing remaining processes on port $API_PORT: $port_pids"
        echo "$port_pids" | xargs kill -9 2>/dev/null || true
    fi

    # 3) Kill any orphaned vllm worker processes from our model
    local orphans
    orphans=$(pgrep -f "vllm.entrypoints.*--port $API_PORT" 2>/dev/null) || true
    if [[ -n "$orphans" ]]; then
        echo "  Killing orphaned vllm entrypoint processes: $orphans"
        echo "$orphans" | xargs kill -9 2>/dev/null || true
    fi
    orphans=$(pgrep -f "vllm.v1.worker.*mixtral" 2>/dev/null) || true
    if [[ -n "$orphans" ]]; then
        echo "  Killing orphaned vllm worker processes: $orphans"
        echo "$orphans" | xargs kill -9 2>/dev/null || true
    fi

    sleep 3
    echo "  Cleanup done."
}

cleanup() {
    stop_all_vllm
}
trap cleanup EXIT INT TERM

# ============================================================
# Start/stop helpers
# ============================================================
start_vllm() {
    local label="$1"
    local enable_eplb="$2"      # 0 or 1
    local enable_pcie_sched="$3" # 0 or 1
    local enable_eplb_phase="$4" # 0 or 1

    echo ""
    echo "========================================"
    echo "Starting vLLM [$label]"
    echo "  EP=$EP_SIZE, EPLB=$([ "$enable_eplb" -eq 1 ] && echo ON || echo OFF)"
    echo "  PCIe Scheduler=$([ "$enable_pcie_sched" -eq 1 ] && echo ON || echo OFF)"
    echo "  EPLB Phase=$([ "$enable_eplb_phase" -eq 1 ] && echo ON || echo OFF)"
    echo "  Offloading=${KV_OFFLOADING_SIZE}GiB, gpu_mem_util=$GPU_MEM_UTIL"
    echo "========================================"

    # Kill previous vLLM instance
    stop_all_vllm

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
    )

    if [[ "$enable_eplb" -eq 1 ]]; then
        CMD_ARGS+=(
            --enable-eplb
            --eplb-config "{\"step_interval\": $EPLB_STEP_INTERVAL, \"num_redundant_experts\": 0}"
        )
    fi

    local ENV_VARS="NCCL_P2P_DISABLE=1 NCCL_NVLS_ENABLE=0 VLLM_TEST_ENABLE_EP=1 HF_HUB_OFFLINE=1"
    if [[ "$enable_pcie_sched" -eq 1 ]]; then
        ENV_VARS="$ENV_VARS VLLM_PCIE_SCHEDULER=1"
    fi
    if [[ "$enable_eplb_phase" -eq 1 ]]; then
        ENV_VARS="$ENV_VARS VLLM_EPLB_PHASE_AWARE=1"
    fi

    eval "$ENV_VARS vllm serve ${CMD_ARGS[*]}" > "$RESULTS_DIR/vllm_${label}.log" 2>&1 &
    local vllm_pid=$!
    echo "$vllm_pid" > "$PIDFILE"
    echo "vLLM PID: $vllm_pid"

    # Wait for ready
    local max_wait=600
    local waited=0
    echo -n "Waiting for server"
    while [[ $waited -lt $max_wait ]]; do
        if curl -s "$API_BASE/health" &>/dev/null; then
            echo ""
            echo "Server ready (${waited}s)"
            sleep 5
            return 0
        fi
        if ! kill -0 "$vllm_pid" 2>/dev/null; then
            echo ""
            echo "Server died. Check: $RESULTS_DIR/vllm_${label}.log"
            return 1
        fi
        echo -n "."
        sleep 3
        waited=$((waited + 3))
    done
    echo ""
    echo "Timeout waiting for server"
    return 1
}

stop_vllm() {
    stop_all_vllm
}

run_workload() {
    local mode="$1"   # baseline or prefetch
    local label="$2"
    local round="$3"
    local output="$RESULTS_DIR/${label}_r${round}.jsonl"

    echo ""
    echo "--- Running [$label] round=$round mode=$mode qps=$QPS ---"

    curl -s -X POST "$API_BASE/reset_prefix_cache?reset_external=true" >/dev/null || true
    sleep 3

    python3 "$RUN_EXP_DIR/prefetch_ab_runner.py" \
        --trace-file "$TRACE_FILE" \
        --mode "$mode" \
        --qps "$QPS" \
        --num-multi-turn "$NUM_CONV" \
        --model "$MODEL_PATH" \
        --api-base "$API_BASE/v1" \
        --output "$output" \
        --seed $((42 + round)) \
        --request-timeout "$REQUEST_TIMEOUT" \
        --prefetch-lead-time "$PREFETCH_LEAD_TIME" \
        --schedule-mode uniform \
        2>&1 | tee "$RESULTS_DIR/${label}_r${round}.log"

    echo "Output: $output"
}

# ============================================================
# Print overview
# ============================================================
echo "========================================"
echo "Mixtral-8x7B EPLB + PCIe Scheduler Test"
echo "========================================"
echo "Model:      $MODEL_PATH"
echo "EP size:    $EP_SIZE"
echo "Dataset:    $DATASET ($TRACE_FILE)"
echo "QPS:        $QPS"
echo "Offload:    ${KV_OFFLOADING_SIZE}GiB"
echo "GPU Mem:    $GPU_MEM_UTIL"
echo "Lead time:  ${PREFETCH_LEAD_TIME}s"
echo "EPLB step:  $EPLB_STEP_INTERVAL"
echo "Groups:     $RUN_GROUPS"
echo "Rounds:     $ROUNDS"
echo "Results:    $RESULTS_DIR"
echo "========================================"

# ============================================================
# Run experiments
# ============================================================

for round in $(seq 1 "$ROUNDS"); do
    echo ""
    echo "============ ROUND $round / $ROUNDS ============"

    # G0: EP + Offload (baseline, no prefetch, no EPLB)
    if should_run_group g0; then
        start_vllm "g0" 0 0 0 || exit 1
        run_workload "baseline" "g0_baseline" "$round"
        stop_vllm
    fi

    # G1: EP + Offload + Prefetch (no EPLB)
    if should_run_group g1; then
        start_vllm "g1" 0 0 0 || exit 1
        run_workload "prefetch" "g1_prefetch" "$round"
        stop_vllm
    fi

    # G2: EP + Offload + Prefetch + EPLB (no scheduler)
    if should_run_group g2; then
        start_vllm "g2" 1 0 0 || exit 1
        run_workload "prefetch" "g2_prefetch_eplb" "$round"
        stop_vllm
    fi

    # G3: EP + Offload + Prefetch + EPLB + PCIe Scheduler (no EPLB Phase)
    if should_run_group g3; then
        start_vllm "g3" 1 1 0 || exit 1
        run_workload "prefetch" "g3_sched_no_phase" "$round"
        stop_vllm
    fi

    # G4: EP + Offload + Prefetch + EPLB + PCIe Scheduler + EPLB Phase
    if should_run_group g4; then
        start_vllm "g4" 1 1 1 || exit 1
        run_workload "prefetch" "g4_sched_eplb_phase" "$round"
        stop_vllm
    fi
done

# ============================================================
# Generate report
# ============================================================
echo ""
echo "========================================"
echo "Generating report..."
echo "========================================"

REPORT="$RESULTS_DIR/report.md"
{
    echo "# Mixtral-8x7B EPLB + PCIe Scheduler Report"
    echo ""
    echo "| Config | Value |"
    echo "|--------|-------|"
    echo "| Model | $(basename "$MODEL_PATH") |"
    echo "| EP size | $EP_SIZE |"
    echo "| QPS | $QPS |"
    echo "| Dataset | $DATASET |"
    echo "| Offload | ${KV_OFFLOADING_SIZE}GiB |"
    echo "| GPU Mem Util | $GPU_MEM_UTIL |"
    echo "| EPLB step | $EPLB_STEP_INTERVAL |"
    echo "| Rounds | $ROUNDS |"
    echo ""
    echo "## TTFT Summary (all rounds)"
    echo ""
    echo "| Group | Description | Requests | Mean TTFT | P50 | P95 | P99 | Cache Hit |"
    echo "|-------|-------------|----------|-----------|-----|-----|-----|-----------|"

    for LABEL in g0_baseline g1_prefetch g2_prefetch_eplb g3_sched_no_phase g4_sched_eplb_phase; do
        # Combine all rounds
        COMBINED=""
        for round in $(seq 1 "$ROUNDS"); do
            f="$RESULTS_DIR/${LABEL}_r${round}.jsonl"
            if [[ -f "$f" ]]; then
                COMBINED="$COMBINED $f"
            fi
        done
        if [[ -n "$COMBINED" ]]; then
            python3 -c "
import json, sys
import numpy as np

ttfts, cached_list = [], []
for fpath in '$COMBINED'.split():
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
    print('| $LABEL | - | 0 | - | - | - | - | - |')
else:
    arr = np.array(ttfts)
    hit_rate = f'{np.mean(cached_list)*100:.1f}%' if cached_list else '0%'
    desc = {
        'g0_baseline': 'EP+Offload',
        'g1_prefetch': 'EP+Offload+Prefetch',
        'g2_prefetch_eplb': 'EP+Offload+Prefetch+EPLB',
        'g3_sched_no_phase': '+PCIe Sched (no Phase)',
        'g4_sched_eplb_phase': '+PCIe Sched+EPLB Phase',
    }
    print(f'| $LABEL | {desc.get(\"$LABEL\", \"-\")} | {len(arr)} | {np.mean(arr):.0f} | {np.percentile(arr,50):.0f} | {np.percentile(arr,95):.0f} | {np.percentile(arr,99):.0f} | {hit_rate} |')
"
        fi
    done
} > "$REPORT"

echo ""
cat "$REPORT"
echo ""
echo "Report: $REPORT"
echo "Results: $RESULTS_DIR"
echo "Done."
