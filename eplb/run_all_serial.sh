#!/bin/bash
# ============================================================================
# Serial experiment runner: sync/async × (Mixtral-8x7B | DeepSeek-V2-Lite)
#
# NOTE: 原本计划用 Phi-3.5-MoE 作为第二模型, 但 vLLM 里的 PhiMoE 实现未绑定
# MixtureOfExperts 接口 (`is_mixture_of_experts(model)` assert 失败), EPLB
# 无法在 Phi 上运行. 替换为 DeepSeek-V2-Lite (16B / 2.4B activated, 64 routed
# experts + 2 shared, top-6, MLA KV). 结构上跟 Mixtral 差异较大但能跑, 作为
# "跨 MoE 架构泛化性"对照.
#
# 依次执行 4 组实验, 每组内部的 vLLM start/stop 由内层脚本
# (run_mixtral_{sync,async}_fixed.sh) 自行管理. 本脚本负责:
#   - 串行编排 4 组实验
#   - 一组失败不阻塞后续
#   - 实验间额外做一次 port cleanup 防止残留
#   - 顶级 summary log + 每组结果目录
#
# 为 nohup 设计:
#   nohup ./run_all_serial.sh > /tmp/run_all.log 2>&1 &
#   tail -f /tmp/run_all.log
#
# 可选参数:
#   --only <list>     只跑指定实验, 逗号分隔:
#                     sync_mixtral,sync_deepseek,async_mixtral,async_deepseek
#                     例: --only "sync_mixtral,async_mixtral"
#   --qps-list "..."  覆盖 QPS 列表 (default: "0.5 1.0 1.5 2.0 2.5")
#   --rounds N        覆盖轮数 (default: 1)
# ============================================================================

# 关键: 不要 set -e, 出错也要继续
set +e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYNC_SCRIPT="$SCRIPT_DIR/run_mixtral_sync_fixed.sh"
ASYNC_SCRIPT="$SCRIPT_DIR/run_mixtral_async_fixed.sh"

# ---- Config ---------------------------------------------------------------
DATASET="pcie-heavy-x5"
QPS_LIST="0.5 1.0 1.5 2.0 2.5"
ROUNDS=1
API_PORT=8000
RUN_ONLY="all"

MIXTRAL_MODEL="/lpai/models/mistralai__mixtral-8x7b-instruct-v0_1/24-08-19-1318"
DEEPSEEK_MODEL="/lpai/models/deepseek-ai__deepseek-v2-lite/24-05-17-0658"

# DeepSeek-V2-Lite 专用参数 (A100-80GB)
# 为什么跟 Mixtral 一致的 mem=0.5, step=200, 但 blocks 只有 2000:
#   MLA 压缩 KV 后每 block 约 432 KB (Mixtral MHA 约 2 MB).
#   把 blocks 减到 2000 以强制 evict/restore 活动, 补偿 MLA 的压缩,
#   给 scheduler 制造足够的 KV 流量 vs EPLB rearrange 的竞争.
DEEPSEEK_GPU_MEM_UTIL=0.5
DEEPSEEK_NUM_GPU_BLOCKS=2000
DEEPSEEK_STEP_INTERVAL=200

# ---- Parse args -----------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --only)      RUN_ONLY="$2";  shift 2 ;;
        --qps-list)  QPS_LIST="$2";  shift 2 ;;
        --rounds)    ROUNDS="$2";    shift 2 ;;
        --dataset)   DATASET="$2";   shift 2 ;;
        -h|--help)
            sed -n '2,/^$/p' "$0" | grep '^#' | sed 's/^# \?//'
            exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

should_run() {
    [[ "$RUN_ONLY" == "all" ]] || [[ ",$RUN_ONLY," == *",$1,"* ]]
}

# ---- Setup ----------------------------------------------------------------
SERIAL_ROOT="$SCRIPT_DIR/results/serial_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$SERIAL_ROOT"
GLOBAL_LOG="$SERIAL_ROOT/serial.log"

declare -a RESULTS=()
declare -a RESULT_DIRS=()

log() {
    local ts
    ts=$(date '+%Y-%m-%d %H:%M:%S')
    printf '[%s] %s\n' "$ts" "$*" | tee -a "$GLOBAL_LOG"
}

log_section() {
    echo "" | tee -a "$GLOBAL_LOG"
    echo "================================================================" | tee -a "$GLOBAL_LOG"
    log "$*"
    echo "================================================================" | tee -a "$GLOBAL_LOG"
}

# 每组实验之间额外清理一次端口, 防止内层 trap 没完全释放
cleanup_between_experiments() {
    log "Cross-experiment cleanup: checking port $API_PORT..."
    local port_pids
    port_pids=$(lsof -ti :"$API_PORT" 2>/dev/null) || true
    if [[ -n "$port_pids" ]]; then
        log "  Found stale PIDs on port $API_PORT: $port_pids — killing"
        echo "$port_pids" | xargs kill -9 2>/dev/null || true
    fi
    sleep 15
    log "  Cleanup done."
}

interrupted=0
trap 'log "!!! Serial runner interrupted (SIGINT/SIGTERM)"; interrupted=1; cleanup_between_experiments; exit 130' INT TERM

run_experiment() {
    local label="$1"
    shift

    if ! should_run "$label"; then
        log "Skipping $label (not in --only list)"
        RESULTS+=("- $label (skipped)")
        return 0
    fi

    log_section "Experiment: $label"
    log "MODEL_PATH=$MODEL_PATH"
    log "CMD: $*"

    local start_ts end_ts dur rc
    start_ts=$(date +%s)

    "$@" 2>&1 | tee -a "$GLOBAL_LOG"
    rc=${PIPESTATUS[0]}

    end_ts=$(date +%s)
    dur=$((end_ts - start_ts))

    if [[ $rc -eq 0 ]]; then
        log "✓ [$label] completed in ${dur}s"
        RESULTS+=("✓ $label (${dur}s)")
    else
        log "✗ [$label] FAILED (rc=$rc, dur=${dur}s) — continuing"
        RESULTS+=("✗ $label (rc=$rc, ${dur}s)")
    fi

    local last_result_dir
    last_result_dir=$(grep -oE 'Results:[[:space:]]+[^ ]+' "$GLOBAL_LOG" \
        | tail -1 | awk '{print $NF}')
    [[ -n "$last_result_dir" ]] && RESULT_DIRS+=("$label -> $last_result_dir")

    cleanup_between_experiments

    [[ $interrupted -eq 1 ]] && exit 130
}

# ---- Main -----------------------------------------------------------------
log_section "Serial experiment runner START"
log "Dataset:        $DATASET"
log "QPS list:       $QPS_LIST"
log "Rounds:         $ROUNDS"
log "Run filter:     $RUN_ONLY"
log "Results root:   $SERIAL_ROOT"
log "Mixtral model:  $MIXTRAL_MODEL"
log "DeepSeek model: $DEEPSEEK_MODEL"
log ""
log "Planned experiments:"
log "  1. sync  + Mixtral-8x7B     (default params)"
log "  2. sync  + DeepSeek-V2-Lite (mem=$DEEPSEEK_GPU_MEM_UTIL blocks=$DEEPSEEK_NUM_GPU_BLOCKS step=$DEEPSEEK_STEP_INTERVAL)"
log "  3. async + Mixtral-8x7B     (default params)"
log "  4. async + DeepSeek-V2-Lite (mem=$DEEPSEEK_GPU_MEM_UTIL blocks=$DEEPSEEK_NUM_GPU_BLOCKS step=$DEEPSEEK_STEP_INTERVAL)"

OVERALL_START=$(date +%s)

export MODEL_PATH="$MIXTRAL_MODEL"
run_experiment "sync_mixtral" \
    "$SYNC_SCRIPT" \
        --dataset "$DATASET" \
        --qps-list "$QPS_LIST" \
        --rounds "$ROUNDS"

export MODEL_PATH="$DEEPSEEK_MODEL"
run_experiment "sync_deepseek" \
    "$SYNC_SCRIPT" \
        --dataset "$DATASET" \
        --qps-list "$QPS_LIST" \
        --rounds "$ROUNDS" \
        --gpu-mem-util "$DEEPSEEK_GPU_MEM_UTIL" \
        --num-gpu-blocks "$DEEPSEEK_NUM_GPU_BLOCKS" \
        --step-interval "$DEEPSEEK_STEP_INTERVAL"

export MODEL_PATH="$MIXTRAL_MODEL"
run_experiment "async_mixtral" \
    "$ASYNC_SCRIPT" \
        --dataset "$DATASET" \
        --qps-list "$QPS_LIST" \
        --rounds "$ROUNDS"

export MODEL_PATH="$DEEPSEEK_MODEL"
run_experiment "async_deepseek" \
    "$ASYNC_SCRIPT" \
        --dataset "$DATASET" \
        --qps-list "$QPS_LIST" \
        --rounds "$ROUNDS" \
        --gpu-mem-util "$DEEPSEEK_GPU_MEM_UTIL" \
        --num-gpu-blocks "$DEEPSEEK_NUM_GPU_BLOCKS" \
        --step-interval "$DEEPSEEK_STEP_INTERVAL"

# ---- Summary --------------------------------------------------------------
OVERALL_END=$(date +%s)
TOTAL_DUR=$((OVERALL_END - OVERALL_START))

log_section "Serial runner COMPLETE (total $((TOTAL_DUR/3600))h$(((TOTAL_DUR%3600)/60))m)"
log ""
log "Per-experiment status:"
for r in "${RESULTS[@]}"; do
    log "  $r"
done
log ""
log "Result directories:"
if [[ ${#RESULT_DIRS[@]} -eq 0 ]]; then
    log "  (none captured — check inner scripts' output for paths)"
else
    for d in "${RESULT_DIRS[@]}"; do
        log "  $d"
    done
fi
log ""
log "Global log: $GLOBAL_LOG"
log "Done."
