#!/bin/bash
# Mixtral-8x7B EPLB Async 模式固定配置实验脚本
# (替代 run_mixtral_async_gradient.sh)
#
# 目的: 在可复现的硬件/软件配置下，验证 async EPLB + PCIe Scheduler 的增量收益
#
# 与旧脚本的关键差异:
#   1. **固定 GPU**: 强制使用 GPU0-3 (同一 NUMA 节点, 同一 PCIe root complex),
#      消除 PCIe 拓扑带来的方差
#   2. **参数归一化**: step_interval=200 (与 sync 脚本一致, 旧: 50 过频繁)
#                    num_gpu_blocks=4000 (旧: 2500 block 不足)
#                    log_balancedness=true (探针)
#   3. **依赖已修复的 pcie_scheduler.py**:
#      async migration 期间从 "降并发到 1" 改为 "仅挂起 Prefetch, Restore 全速",
#      让 critical path (Restore) 不被误伤
#   4. **G3 对照**: 增加一个 "sched 但无 EPLB phase" 组, 分离调度器与 eplb_phase 的贡献
#   5. **多轮默认 3 次**: 便于看方差
#
# 实验组:
#   G1: Reference: EP + Offload + Prefetch (no EPLB)
#   G2: Baseline:  EP + Offload + Prefetch + EPLB (async, no sched)
#   G3: Sched:     +PCIe Sched (no EPLB phase)
#   G4: Full:      +PCIe Sched + EPLB Phase  <-- 新版 EPLB Phase 是 "suppress Prefetch only"
#
# 用法:
#   nohup ./run_mixtral_async_fixed.sh > experiment.log 2>&1 &
#   ./run_mixtral_async_fixed.sh --qps-list "2.0"
#   ./run_mixtral_async_fixed.sh --groups "g2,g4"
#   ./run_mixtral_async_fixed.sh --rounds 1
#   ./run_mixtral_async_fixed.sh --gpus 4,5,6,7
#   ./run_mixtral_async_fixed.sh --dry-run

set -o pipefail

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
GPU_LIST="0,1,2,3"                # 同 NUMA 节点
GPU_MEM_UTIL=0.5
NUM_GPU_BLOCKS=4000
MAX_NUM_SEQS=32
MAX_MODEL_LEN=4096
KV_OFFLOADING_SIZE=20
EPLB_STEP_INTERVAL=200
PREFETCH_LEAD_TIME=2.0
REQUEST_TIMEOUT=360
DATASET="pcie-heavy"
QPS_LIST="0.5 1.0 1.5 2.0 2.5"
RUN_GROUPS="g1,g2,g3,g4"
ROUNDS=3
DRY_RUN=0
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
        --rounds)           ROUNDS="$2";                shift 2 ;;
        --num-gpu-blocks)   NUM_GPU_BLOCKS="$2";        shift 2 ;;
        --gpu-mem-util)     GPU_MEM_UTIL="$2";          shift 2 ;;
        --step-interval)    EPLB_STEP_INTERVAL="$2";    shift 2 ;;
        --ep-size)          EP_SIZE="$2";               shift 2 ;;
        --gpus)             GPU_LIST="$2";              shift 2 ;;
        --dry-run)          DRY_RUN=1;                  shift ;;
        -h|--help)
            sed -n '2,/^$/p' "$0" | grep '^#' | sed 's/^# \?//'
            exit 0 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

# ============================================================
# Hardcoded GPU pinning (NO auto-select)
# ============================================================
export CUDA_VISIBLE_DEVICES="$GPU_LIST"
echo "========================================"
echo "Pinned GPUs: CUDA_VISIBLE_DEVICES=$GPU_LIST"
echo "  (若需要换另一组, 用 --gpus '4,5,6,7')"
echo "========================================"

# ============================================================
# Dataset
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
RESULTS_ROOT="$SCRIPT_DIR/results/mixtral_async_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_ROOT"
PIDFILE="/tmp/vllm_mixtral_async_${API_PORT}_$$.pid"

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
        local pid
        pid=$(cat "$PIDFILE" 2>/dev/null)
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

cleanup() {
    log "Cleanup triggered..."
    stop_our_vllm
    rm -f "$PIDFILE"
}
trap cleanup EXIT INT TERM

# ============================================================
# Start vLLM — ASYNC 模式 (use_async=true)
# ============================================================
start_vllm() {
    local label="$1"
    local enable_eplb="$2"
    local enable_pcie_sched="$3"
    local enable_eplb_phase="$4"
    local log_file="$5"

    log "Starting vLLM [$label] eplb=$enable_eplb(async) sched=$enable_pcie_sched phase=$enable_eplb_phase"
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
        # 关键: use_async=true, log_balancedness=true
        CMD_ARGS+=(
            --enable-eplb
            --eplb-config "{\"step_interval\": $EPLB_STEP_INTERVAL, \"num_redundant_experts\": 0, \"use_async\": true, \"log_balancedness\": true, \"log_balancedness_interval\": 10}"
        )
    fi

    export NCCL_P2P_DISABLE=1 NCCL_NVLS_ENABLE=0 VLLM_TEST_ENABLE_EP=1 HF_HUB_OFFLINE=1
    # 关键: 这是 EP-only 实验, 没有 PP. PP phase aware 的 IDLE-window 计数器
    # 在没有 PP phase 切换时永远不会 reset, 一旦耗尽 H2D 就会被永久 block.
    # 即使有同名代码守卫, 显式关掉作为双保险.
    export VLLM_PCIE_PP_PHASE_AWARE=0
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
# Verify scheduler hooks were registered
# ============================================================
verify_hook_registration() {
    local log_file="$1"
    local label="$2"
    local enable_eplb="$3"
    local enable_pcie_sched="$4"
    local enable_eplb_phase="$5"

    if [[ "$enable_eplb" -ne 1 ]] || [[ "$enable_pcie_sched" -ne 1 ]] \
       || [[ "$enable_eplb_phase" -ne 1 ]]; then
        return 0
    fi

    if grep -q "EPLB PCIe scheduler hooks registered" "$log_file"; then
        log "  ✓ [$label] EPLB PCIe hooks 已注册"
        return 0
    else
        log "  ✗ [$label] WARNING: 未发现 EPLB PCIe hooks registered 日志！"
        return 1
    fi
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
        log "  WARNING: workload timed out after ${EXPERIMENT_TIMEOUT}s"; return 1
    elif [[ $rc -ne 0 ]]; then
        log "  WARNING: workload failed (rc=$rc)"; return 1
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

get_group_config() {
    case "$1" in
        g1) echo "prefetch 0 0 0" ;;  # Reference: no EPLB
        g2) echo "prefetch 1 0 0" ;;  # Baseline: +EPLB async
        g3) echo "prefetch 1 1 0" ;;  # +Sched, no EPLB phase
        g4) echo "prefetch 1 1 1" ;;  # Full: +Sched+EPLB Phase
        *)  echo ""; return 1 ;;
    esac
}

get_group_desc() {
    case "$1" in
        g1) echo "Reference (no EPLB)" ;;
        g2) echo "Baseline (+EPLB async)" ;;
        g3) echo "+PCIe Sched (no EPLB Phase)" ;;
        g4) echo "+PCIe Sched+EPLB Phase" ;;
    esac
}

# ============================================================
# Health inspection
# ============================================================
inspect_run_health() {
    local vllm_log="$1"
    local label="$2"

    local block_fails
    block_fails=$(grep "Block allocation failed" "$vllm_log" 2>/dev/null | wc -l | tr -d ' ')
    local rearrange_count
    rearrange_count=$(grep "Rearranging experts" "$vllm_log" 2>/dev/null | wc -l | tr -d ' ')
    local balancedness
    balancedness=$(grep "balancedness=" "$vllm_log" 2>/dev/null | tail -1 \
        | sed -n 's/.*balancedness=\([0-9.]*\).*/\1/p')

    log "  health[$label] block_fail=$block_fails  rearrange=$rearrange_count  bal=${balancedness:-NA}"

    if [[ "$block_fails" -gt 500 ]]; then
        log "  ✗ [$label] 过多 Block allocation failed ($block_fails), 数据不可信"
    fi
}

# ============================================================
# Report generation
# ============================================================
generate_report() {
    local report="$RESULTS_ROOT/report.md"
    log "Generating report -> $report"

    {
        echo "# Mixtral-8x7B EPLB Async (Fixed) Report"
        echo ""
        echo "Generated: $(date '+%Y-%m-%d %H:%M:%S')"
        echo ""
        echo "## Configuration"
        echo ""
        echo "| Parameter | Value |"
        echo "|-----------|-------|"
        echo "| Model | $(basename "$MODEL_PATH") |"
        echo "| EP size | $EP_SIZE |"
        echo "| Pinned GPUs | $GPU_LIST |"
        echo "| EPLB mode | **async** (use_async=true, suppress-Prefetch gating) |"
        echo "| EPLB step | $EPLB_STEP_INTERVAL |"
        echo "| GPU blocks | $NUM_GPU_BLOCKS |"
        echo "| GPU mem util | $GPU_MEM_UTIL |"
        echo "| Offload | ${KV_OFFLOADING_SIZE}GiB |"
        echo "| Dataset | $DATASET |"
        echo "| Rounds | $ROUNDS |"
        echo ""

        for qps in $QPS_LIST; do
            local qps_tag="${qps//./_}"
            echo "## QPS = $qps"
            echo ""
            echo "| Group | Description | Requests | Mean TTFT | Std% | P50 | P95 | P99 | Cache Hit |"
            echo "|-------|-------------|----------|-----------|------|-----|-----|-----|-----------|"

            for group in g1 g2 g3 g4; do
                if ! should_run_group "$group"; then continue; fi
                local desc; desc=$(get_group_desc "$group")
                local combined=""
                for r in $(seq 1 "$ROUNDS"); do
                    local f="$RESULTS_ROOT/qps_${qps_tag}/${group}_r${r}.jsonl"
                    [[ -f "$f" ]] && combined="$combined $f"
                done
                if [[ -z "$combined" ]]; then
                    echo "| $group | $desc | - | - | - | - | - | - | - |"
                    continue
                fi
                python3 -c "
import json, os
import numpy as np
per_round_means = []
all_ttfts, cached_list = [], []
for fpath in '''$combined'''.split():
    if not os.path.isfile(fpath): continue
    ttfts = []
    with open(fpath) as f:
        for line in f:
            d = json.loads(line)
            if d.get('ttft_ms') is not None:
                ttfts.append(d['ttft_ms'])
            cached = d.get('cached_tokens', 0) or 0
            prompt = d.get('prompt_tokens', 1) or 1
            if cached > 0:
                cached_list.append(cached / prompt)
    if ttfts:
        per_round_means.append(float(np.mean(ttfts)))
        all_ttfts.extend(ttfts)
if not all_ttfts:
    print('| $group | $desc | 0 | - | - | - | - | - | - |')
else:
    arr = np.array(all_ttfts)
    hit = f'{np.mean(cached_list)*100:.1f}%' if cached_list else '0%'
    if len(per_round_means) > 1:
        mean_of_means = np.mean(per_round_means)
        std_pct = f'{np.std(per_round_means)/mean_of_means*100:.1f}' if mean_of_means > 0 else '-'
    else:
        std_pct = '-'
    print(f'| $group | $desc | {len(arr)} | {np.mean(arr):.0f} | {std_pct} | {np.percentile(arr,50):.0f} | {np.percentile(arr,95):.0f} | {np.percentile(arr,99):.0f} | {hit} |')
" 2>/dev/null || echo "| $group | $desc | ERROR | - | - | - | - | - | - |"
            done
            echo ""
        done

        echo "## Improvement Summary (g4 vs g2)"
        echo ""
        echo "| QPS | Mean Δ | P50 Δ | P95 Δ | P99 Δ |"
        echo "|-----|--------|-------|-------|-------|"

        for qps in $QPS_LIST; do
            local qps_tag="${qps//./_}"
            python3 -c "
import json, os
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
g2 = load_ttfts(f'{base}/g2_rRR.jsonl', $ROUNDS)
g4 = load_ttfts(f'{base}/g4_rRR.jsonl', $ROUNDS)
def delta(old, new, fn):
    if old is None or new is None: return '-'
    return f'{(fn(new)/fn(old)-1)*100:+.1f}%'
print(f'| $qps | {delta(g2,g4,np.mean)} | {delta(g2,g4,lambda x: np.percentile(x,50))} | {delta(g2,g4,lambda x: np.percentile(x,95))} | {delta(g2,g4,lambda x: np.percentile(x,99))} |')
" 2>/dev/null || echo "| $qps | ERROR | ERROR | ERROR | ERROR |"
        done
        echo ""

        echo "## Per-run health"
        echo ""
        echo "| QPS | Group | Round | Rearrange | Block Fail | Balancedness |"
        echo "|-----|-------|-------|-----------|------------|--------------|"
        for qps in $QPS_LIST; do
            local qps_tag="${qps//./_}"
            local qps_dir="$RESULTS_ROOT/qps_${qps_tag}"
            for group in g1 g2 g3 g4; do
                if ! should_run_group "$group"; then continue; fi
                for r in $(seq 1 "$ROUNDS"); do
                    local vlog="$qps_dir/vllm_${group}_r${r}.log"
                    [[ -f "$vlog" ]] || continue
                    local bf rc bal
                    bf=$(grep "Block allocation failed" "$vlog" 2>/dev/null | wc -l | tr -d ' ')
                    rc=$(grep "Rearranging experts" "$vlog" 2>/dev/null | wc -l | tr -d ' ')
                    bal=$(grep "balancedness=" "$vlog" 2>/dev/null | tail -1 | sed -n 's/.*balancedness=\([0-9.]*\).*/\1/p')
                    echo "| $qps | $group | $r | ${rc:-0} | ${bf:-0} | ${bal:-NA} |"
                done
            done
        done
        echo ""

        echo "## Hook registration check (g4 only)"
        echo ""
        echo "| QPS | Group | Round | Hook Registered |"
        echo "|-----|-------|-------|-----------------|"
        for qps in $QPS_LIST; do
            local qps_tag="${qps//./_}"
            local qps_dir="$RESULTS_ROOT/qps_${qps_tag}"
            for group in g4; do
                if ! should_run_group "$group"; then continue; fi
                for r in $(seq 1 "$ROUNDS"); do
                    local vlog="$qps_dir/vllm_${group}_r${r}.log"
                    [[ -f "$vlog" ]] || continue
                    if grep -q "EPLB PCIe scheduler hooks registered" "$vlog"; then
                        echo "| $qps | $group | $r | ✓ |"
                    else
                        echo "| $qps | $group | $r | ✗ FAIL |"
                    fi
                done
            done
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
    echo "DRY RUN — Mixtral Async Experiment Matrix"
    echo "========================================"
    echo "EPLB mode: ASYNC (use_async=true, suppress-Prefetch gating)"
    echo "GPUs: $GPU_LIST"
    echo "Config: blk=$NUM_GPU_BLOCKS gpu_mem=$GPU_MEM_UTIL step=$EPLB_STEP_INTERVAL rounds=$ROUNDS"
    echo ""
    total=0
    for qps in $QPS_LIST; do
        echo "QPS=$qps:"
        for group in g1 g2 g3 g4; do
            if should_run_group "$group"; then
                desc=$(get_group_desc "$group")
                echo "  $group ($desc): $ROUNDS rounds"
                total=$((total + ROUNDS))
            fi
        done
    done
    echo ""
    echo "Total experiments: $total"
    exit 0
fi

# ============================================================
# Main loop
# ============================================================
log_section "Mixtral-8x7B EPLB ASYNC Fixed Experiments"
log "Model:      $MODEL_PATH"
log "EP size:    $EP_SIZE"
log "EPLB mode:  ASYNC (use_async=true)"
log "GPUs:       $GPU_LIST (pinned, NO auto-select)"
log "GPU blocks: $NUM_GPU_BLOCKS"
log "Step:       $EPLB_STEP_INTERVAL"
log "QPS list:   $QPS_LIST"
log "Groups:     $RUN_GROUPS"
log "Rounds:     $ROUNDS"
log "Results:    $RESULTS_ROOT"

TOTAL=0; PASSED=0; FAILED=0

for round in $(seq 1 "$ROUNDS"); do
    log_section "ROUND $round / $ROUNDS"

    for qps in $QPS_LIST; do
        qps_tag="${qps//./_}"
        qps_dir="$RESULTS_ROOT/qps_${qps_tag}"
        mkdir -p "$qps_dir"
        log_section "QPS = $qps  (Round $round)"

        for group in g1 g2 g3 g4; do
            if ! should_run_group "$group"; then continue; fi
            TOTAL=$((TOTAL + 1))
            desc=$(get_group_desc "$group")
            config=$(get_group_config "$group")
            read -r mode enable_eplb enable_sched enable_phase <<< "$config"

            log_section "[$group] $desc @ QPS=$qps  Round $round/$ROUNDS"

            vllm_log="$qps_dir/vllm_${group}_r${round}.log"
            if ! start_vllm "$group" "$enable_eplb" "$enable_sched" "$enable_phase" "$vllm_log"; then
                log "FAIL [$group @ QPS=$qps r$round]: vLLM failed to start"
                FAILED=$((FAILED + 1))
                stop_our_vllm
                continue
            fi

            sleep 2
            verify_hook_registration "$vllm_log" "$group" "$enable_eplb" "$enable_sched" "$enable_phase" || true

            if run_workload "$mode" "$qps" "$group" "$round" "$qps_dir"; then
                PASSED=$((PASSED + 1))
            else
                FAILED=$((FAILED + 1))
            fi

            inspect_run_health "$vllm_log" "$group @ QPS=$qps r$round"
            stop_our_vllm
        done
    done
done

log_section "Experiment Complete"
log "Total: $TOTAL | Passed: $PASSED | Failed: $FAILED"
log "Results: $RESULTS_ROOT"

generate_report
log "All done."
