#!/bin/bash

# ============================================
# 实时监控 CPU Offloading 事件
# ============================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║   实时监控 CPU Offloading 事件                             ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

if [ ! -d "$LOG_DIR" ]; then
    echo "❌ 日志目录不存在: $LOG_DIR"
    echo "   请先启动 vLLM 服务"
    exit 1
fi

# 查找最新的日志文件
LATEST_LOG=$(ls -t "$LOG_DIR"/vllm_offload_*.log 2>/dev/null | head -1)

if [ -z "$LATEST_LOG" ]; then
    echo "❌ 未找到日志文件"
    echo "   请先启动 vLLM 服务"
    exit 1
fi

echo "📋 监控日志文件: $LATEST_LOG"
echo ""
echo "🔍 监控以下关键事件："
echo "   - CPU Offloading (offload, swap)"
echo "   - Block 管理 (allocate, evict, free)"
echo "   - Transfer 事件 (GPU->CPU, CPU->GPU)"
echo "   - LRU 管理 (lru, touch, prepare)"
echo ""
echo "按 Ctrl+C 停止监控"
echo "$(printf '─%.0s' {1..60})"
echo ""

# 实时监控并高亮关键词
tail -f "$LATEST_LOG" | grep -i --line-buffered -E \
    "(offload|swap|evict|allocat|transfer|lru|touch|prepare_store|prepare_load|complete_store|complete_load|cpu_to_gpu|gpu_to_cpu|block.*free|block.*cache|kv.*cache|[LHT-DEBUG])" \
    --color=always
