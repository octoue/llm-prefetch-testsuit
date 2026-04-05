#!/bin/bash
# 72B 过夜批量测试
# 按风险从低到高排列，即使后面的配置崩溃，前面的数据也已保存
#
# 预计耗时: 每个 (qps, group) 约 10-15 分钟, 3 groups x N 个 qps x rounds
# 下面总共约 30-40 小时工作量，按实际 GPU 时间约 8-12 小时（取决于 QPS 和是否崩溃）

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_PCIE="$SCRIPT_DIR/run_pcie.sh"

LOG_FILE="$SCRIPT_DIR/overnight_72b_$(date +%Y%m%d_%H%M%S).log"

run_config() {
    local desc="$1"; shift
    echo ""
    echo "========================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $desc"
    echo "  Command: $RUN_PCIE $*"
    echo "========================================================"

    if "$RUN_PCIE" "$@"; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] SUCCESS: $desc"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAILED: $desc (exit=$?), continuing..."
    fi
    echo ""
}

{

echo "Overnight 72B batch started at $(date)"
echo ""

# ============================================================
# Phase 1: blk=750, 安全配置, 3 轮 (补充已有的 qps=0.5 数据)
# ============================================================

run_config "Phase1: blk=750 qps=0.8,1.0 x3 rounds" \
    --model 72b --qps 0.8,1.0 --num-blocks 750 --pp 4 --rounds 3 --closed-loop

run_config "Phase1: blk=750 qps=1.5 x3 rounds" \
    --model 72b --qps 1.5 --num-blocks 750 --pp 4 --rounds 3 --closed-loop

# ============================================================
# Phase 2: blk=500, 新配置验证, 先跑 1 轮
# ============================================================

run_config "Phase2: blk=500 qps=0.8 x1 (验证)" \
    --model 72b --qps 0.8 --num-blocks 500 --pp 4 --rounds 1 --closed-loop

run_config "Phase2: blk=500 qps=1.0 x1 (验证)" \
    --model 72b --qps 1.0 --num-blocks 500 --pp 4 --rounds 1 --closed-loop

run_config "Phase2: blk=500 qps=1.5 x1 (验证)" \
    --model 72b --qps 1.5 --num-blocks 500 --pp 4 --rounds 1 --closed-loop

# ============================================================
# Phase 3: blk=500 补轮 (如果 Phase2 通过，这里加 2 轮)
# ============================================================

run_config "Phase3: blk=500 qps=0.8,1.0 x2 rounds (补轮)" \
    --model 72b --qps 0.8,1.0 --num-blocks 500 --pp 4 --rounds 2 --closed-loop

run_config "Phase3: blk=500 qps=1.5 x2 rounds (补轮)" \
    --model 72b --qps 1.5 --num-blocks 500 --pp 4 --rounds 2 --closed-loop

# ============================================================
# Phase 4: blk=500 + max-num-seqs=128, 高压测试
# ============================================================

run_config "Phase4: blk=500 seqs=128 qps=0.8,1.0 x1 (高压验证)" \
    --model 72b --qps 0.8,1.0 --num-blocks 500 --pp 4 --rounds 1 --closed-loop --max-num-seqs 128

run_config "Phase4: blk=500 seqs=128 qps=1.5 x1 (高压验证)" \
    --model 72b --qps 1.5 --num-blocks 500 --pp 4 --rounds 1 --closed-loop --max-num-seqs 128

echo ""
echo "========================================================"
echo "Overnight batch completed at $(date)"
echo "========================================================"

} 2>&1 | tee "$LOG_FILE"
