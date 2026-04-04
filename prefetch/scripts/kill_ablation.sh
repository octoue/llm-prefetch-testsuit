#!/bin/bash
# ============================================================
# 安全终止 prefetch 消融实验的所有相关进程
# ============================================================
# 用法:
#   ./kill_ablation.sh          # 交互模式, 先列出再确认
#   ./kill_ablation.sh -y       # 直接终止, 不确认
#   ./kill_ablation.sh --dry    # 仅列出, 不终止
# ============================================================

set -o pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

MODE="interactive"
[[ "$1" == "-y" || "$1" == "--yes" ]]  && MODE="force"
[[ "$1" == "--dry" || "$1" == "-n" ]]  && MODE="dry"

# 按终止顺序定义进程匹配模式 (先子后父)
PATTERNS=(
    "prefetch_ab_runner.py"
    "vllm serve"
    "start_vllm_prefetch.sh"
    "auto_run_prefetch_ablation.sh"
)

collect_pids() {
    local pattern="$1"
    # 排除 grep 自身和本脚本
    pgrep -f "$pattern" 2>/dev/null | while read pid; do
        # 排除自身
        [[ "$pid" == "$$" ]] && continue
        # 排除父 shell (如果从脚本内调用)
        [[ "$pid" == "$PPID" ]] && continue
        echo "$pid"
    done
}

echo "============================================"
echo "Prefetch Ablation 进程清理"
echo "============================================"
echo ""

TOTAL_FOUND=0
ALL_PIDS=()

for pattern in "${PATTERNS[@]}"; do
    pids=($(collect_pids "$pattern"))
    if [[ ${#pids[@]} -gt 0 ]]; then
        echo -e "${YELLOW}[$pattern]${NC}"
        for pid in "${pids[@]}"; do
            cmdline=$(ps -p "$pid" -o pid=,ppid=,etime=,command= 2>/dev/null || echo "$pid (already gone)")
            echo "  PID $cmdline"
            ALL_PIDS+=("$pid")
            TOTAL_FOUND=$((TOTAL_FOUND + 1))
        done
        echo ""
    fi
done

# 检查是否有残余 GPU 进程
GPU_PIDS=($(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | sort -u))
if [[ ${#GPU_PIDS[@]} -gt 0 ]]; then
    echo -e "${YELLOW}[GPU 占用进程]${NC}"
    for pid in "${GPU_PIDS[@]}"; do
        [[ -z "$pid" ]] && continue
        cmdline=$(ps -p "$pid" -o pid=,command= 2>/dev/null || echo "$pid (gone)")
        # 只显示 vllm 相关的
        if echo "$cmdline" | grep -qi "vllm\|python.*model"; then
            echo "  PID $cmdline"
            # 如果不在已收集列表中, 加入
            if [[ ! " ${ALL_PIDS[*]} " =~ " $pid " ]]; then
                ALL_PIDS+=("$pid")
                TOTAL_FOUND=$((TOTAL_FOUND + 1))
            fi
        fi
    done
    echo ""
fi

if [[ $TOTAL_FOUND -eq 0 ]]; then
    echo -e "${GREEN}没有发现相关进程, 环境干净。${NC}"
    exit 0
fi

echo "--------------------------------------------"
echo -e "共发现 ${RED}${TOTAL_FOUND}${NC} 个相关进程"
echo ""

if [[ "$MODE" == "dry" ]]; then
    echo "(--dry 模式, 不执行终止)"
    exit 0
fi

if [[ "$MODE" == "interactive" ]]; then
    read -p "确认终止以上所有进程? [y/N] " answer
    [[ "$answer" != "y" && "$answer" != "Y" ]] && { echo "取消。"; exit 0; }
fi

# 第一轮: SIGTERM (优雅终止)
echo ""
echo "发送 SIGTERM..."
for pid in "${ALL_PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null && echo "  TERM -> $pid"
    fi
done

# 等待最多 10 秒
echo "等待进程退出 (最多 10s)..."
for i in $(seq 1 10); do
    still_alive=0
    for pid in "${ALL_PIDS[@]}"; do
        kill -0 "$pid" 2>/dev/null && still_alive=1 && break
    done
    [[ $still_alive -eq 0 ]] && break
    sleep 1
done

# 第二轮: SIGKILL (强制终止残留)
KILLED_FORCE=0
for pid in "${ALL_PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
        kill -9 "$pid" 2>/dev/null && echo "  KILL -9 -> $pid"
        KILLED_FORCE=$((KILLED_FORCE + 1))
    fi
done

sleep 1

echo ""
echo "============================================"
if [[ $KILLED_FORCE -gt 0 ]]; then
    echo -e "${YELLOW}完成。${KILLED_FORCE} 个进程需要强制终止。${NC}"
else
    echo -e "${GREEN}完成。所有进程已优雅退出。${NC}"
fi

# 最终检查
remaining=$(collect_pids "auto_run_prefetch_ablation.sh" | wc -l)
remaining=$((remaining + $(collect_pids "vllm serve" | wc -l)))
if [[ $remaining -gt 0 ]]; then
    echo -e "${RED}警告: 仍有 $remaining 个进程残留, 请手动检查: ps aux | grep -E 'vllm|ablation'${NC}"
fi
echo "============================================"
