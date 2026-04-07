#!/bin/bash
# ============================================================
# GPU 锁管理工具
# ============================================================
# 防止多个实验脚本同时使用同一张 GPU
#
# 用法: source gpu_lock.sh
#   acquire_gpus N          选择 N 张空闲 GPU 并锁定，设置 CUDA_VISIBLE_DEVICES
#   release_gpus            释放当前脚本持有的所有 GPU 锁
#   wait_for_free_gpus N [timeout]  等待 N 张空闲 GPU 可用（默认超时 600s）
#
# 锁文件: /tmp/gpu_locks/gpu_<INDEX>.lock (内含 PID)
# 过期锁（PID 已不存在）会自动清理

GPU_LOCK_DIR="/tmp/gpu_locks"
_MY_LOCKED_GPUS=()

# 空闲判定阈值
GPU_UTIL_THRESHOLD="${GPU_UTIL_THRESHOLD:-10}"    # utilization.gpu < 10%
GPU_MEM_THRESHOLD_MB="${GPU_MEM_THRESHOLD_MB:-2000}"  # memory.used < 2000 MiB

_init_lock_dir() {
    mkdir -p "$GPU_LOCK_DIR"
    chmod 1777 "$GPU_LOCK_DIR" 2>/dev/null || true
}

# 清理 PID 已不存在的过期锁
_clean_stale_locks() {
    local _lockfiles=("$GPU_LOCK_DIR"/gpu_*.lock)
    for lockfile in "${_lockfiles[@]}"; do
        [[ -f "$lockfile" ]] || continue
        local lock_pid
        lock_pid=$(cat "$lockfile" 2>/dev/null | head -1)
        if [[ -n "$lock_pid" ]] && ! kill -0 "$lock_pid" 2>/dev/null; then
            rm -f "$lockfile"
        fi
    done
}

# 获取当前被锁定的 GPU 索引列表（逗号分隔）
_get_locked_gpus() {
    _clean_stale_locks
    local locked=()
    local _lockfiles=("$GPU_LOCK_DIR"/gpu_*.lock)
    for lockfile in "${_lockfiles[@]}"; do
        [[ -f "$lockfile" ]] || continue
        local idx
        idx=$(basename "$lockfile" .lock | sed 's/gpu_//')
        locked+=("$idx")
    done
    echo "${locked[*]}" | tr ' ' ','
}

# 查询空闲且未锁定的 GPU
# 输出: 每行一个 "index,util,mem_used"，按 memory.used 升序
_list_available_gpus() {
    local locked
    locked=$(_get_locked_gpus)

    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null \
    | while IFS=',' read -r idx util mem_used; do
        idx=$(echo "$idx" | tr -d ' ')
        util=$(echo "$util" | tr -d ' ')
        mem_used=$(echo "$mem_used" | tr -d ' ')

        # 跳过已锁定的 GPU
        if [[ ",$locked," == *",$idx,"* ]]; then
            continue
        fi

        # 跳过不空闲的 GPU (utilization 或 memory 过高)
        if (( util >= GPU_UTIL_THRESHOLD )); then
            continue
        fi
        if (( mem_used >= GPU_MEM_THRESHOLD_MB )); then
            continue
        fi

        echo "$idx,$util,$mem_used"
    done | sort -t',' -k3 -n
}

# 锁定一张 GPU
_lock_gpu() {
    local idx="$1"
    local lockfile="$GPU_LOCK_DIR/gpu_${idx}.lock"

    # 原子创建锁文件（用 noclobber 防止竞争）
    if ( set -C; echo "$$" > "$lockfile" ) 2>/dev/null; then
        _MY_LOCKED_GPUS+=("$idx")
        return 0
    else
        return 1
    fi
}

# ============================================================
# 公开 API
# ============================================================

# acquire_gpus N
#   选择 N 张空闲 GPU 并锁定
#   成功后设置 ACQUIRED_GPUS (逗号分隔) 和 CUDA_VISIBLE_DEVICES
#   返回 0 成功，1 失败
acquire_gpus() {
    local needed="${1:-1}"
    _init_lock_dir
    _clean_stale_locks

    local available
    available=$(_list_available_gpus)
    local count
    count=$(echo "$available" | grep -c '[0-9]' 2>/dev/null || echo 0)

    if (( count < needed )); then
        echo "GPU Lock: 需要 $needed 张空闲 GPU，但只找到 $count 张可用"
        echo "  (阈值: utilization < ${GPU_UTIL_THRESHOLD}%, memory < ${GPU_MEM_THRESHOLD_MB} MiB)"
        if [[ -n "$(_get_locked_gpus)" ]]; then
            echo "  已锁定的 GPU: $(_get_locked_gpus)"
        fi
        echo "  nvidia-smi 当前状态:"
        nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/    /'
        return 1
    fi

    local acquired=()
    while IFS=',' read -r idx util mem_used; do
        [[ -z "$idx" ]] && continue
        if _lock_gpu "$idx"; then
            acquired+=("$idx")
            (( ${#acquired[@]} >= needed )) && break
        fi
    done <<< "$available"

    if (( ${#acquired[@]} < needed )); then
        # 竞争失败，释放已获取的锁
        for idx in "${acquired[@]}"; do
            rm -f "$GPU_LOCK_DIR/gpu_${idx}.lock"
        done
        _MY_LOCKED_GPUS=()
        echo "GPU Lock: 锁定竞争失败，请重试"
        return 1
    fi

    ACQUIRED_GPUS=$(IFS=','; echo "${acquired[*]}")
    export CUDA_VISIBLE_DEVICES="$ACQUIRED_GPUS"
    echo "GPU Lock: 已锁定 GPU [$ACQUIRED_GPUS] (PID $$)"
    return 0
}

# release_gpus
#   释放当前脚本持有的所有 GPU 锁
release_gpus() {
    for idx in "${_MY_LOCKED_GPUS[@]}"; do
        local lockfile="$GPU_LOCK_DIR/gpu_${idx}.lock"
        if [[ -f "$lockfile" ]]; then
            local lock_pid
            lock_pid=$(cat "$lockfile" 2>/dev/null | head -1)
            if [[ "$lock_pid" == "$$" ]]; then
                rm -f "$lockfile"
            fi
        fi
    done
    _MY_LOCKED_GPUS=()
    echo "GPU Lock: 已释放所有 GPU 锁 (PID $$)"
}

# wait_for_free_gpus N [timeout_seconds]
#   等待直到有 N 张空闲 GPU 可用，默认超时 600 秒
#   成功后调用 acquire_gpus 锁定
wait_for_free_gpus() {
    local needed="${1:-1}"
    local timeout="${2:-600}"
    local waited=0

    _init_lock_dir

    if (( timeout == 0 )); then
        echo "GPU Lock: 等待 $needed 张空闲 GPU (无超时限制)..."
    else
        echo "GPU Lock: 等待 $needed 张空闲 GPU (超时 ${timeout}s)..."
    fi

    while true; do
        if acquire_gpus "$needed"; then
            return 0
        fi
        # timeout=0 表示无限等待
        if (( timeout > 0 && waited >= timeout )); then
            echo "GPU Lock: 超时！${timeout}s 内未找到 $needed 张空闲 GPU"
            return 1
        fi
        echo "  等待中... (${waited}s$([ "$timeout" -gt 0 ] && echo "/${timeout}s" || echo " / ∞"))"
        sleep 10
        waited=$((waited + 10))
    done
}
