#!/bin/bash
# 快速 sanity check：验证 vLLM 与 prefetch 环境是否正常
# 用法: ./sanity_check.sh

set -e

API_HOST="${API_HOST:-localhost:8000}"
API_BASE="http://${API_HOST}/v1"

echo "============================================"
echo "Prefetch 环境 Sanity Check"
echo "============================================"
echo "API: $API_BASE"
echo ""

# 1. 检查 vLLM 是否运行
echo "[1] 检查 vLLM 是否运行..."
if curl -s -o /dev/null -w "%{http_code}" "$API_BASE/models" | grep -q 200; then
    echo "  OK: vLLM 可访问"
else
    echo "  FAIL: vLLM 不可访问，请先运行 ./start_vllm.sh"
    exit 1
fi

# 2. 检查 reset_prefix_cache 端点（需 VLLM_SERVER_DEV_MODE=1）
echo ""
echo "[2] 检查 reset_prefix_cache 端点..."
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "http://${API_HOST}/reset_prefix_cache?reset_external=true" || echo "000")
if [ "$HTTP_CODE" = "200" ]; then
    echo "  OK: reset_prefix_cache 可用 (VLLM_SERVER_DEV_MODE=1)"
else
    echo "  WARN: reset_prefix_cache 返回 $HTTP_CODE，请确保 start_vllm.sh 使用 VLLM_SERVER_DEV_MODE=1"
fi

# 3. 可选：运行 observable_prefetch_test 实验 1（需 GPU）
echo ""
echo "[3] 可选：运行 observable_prefetch_test 验证 CPU→GPU 回迁..."
if command -v python &>/dev/null; then
    if python -c "import openai" 2>/dev/null; then
        echo "  执行: python observable_prefetch_test.py --experiment 1"
        python observable_prefetch_test.py --base-url "$API_BASE" --experiment 1 2>/dev/null || echo "  (跳过或失败，需 GPU 环境)"
    else
        echo "  跳过: 未安装 openai"
    fi
else
    echo "  跳过: 未找到 python"
fi

echo ""
echo "============================================"
echo "Sanity check 完成"
echo "============================================"
