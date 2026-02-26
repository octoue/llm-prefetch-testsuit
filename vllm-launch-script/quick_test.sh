#!/bin/bash

# ============================================
# 快速测试脚本 - 一键启动测试流程
# ============================================

echo "╔════════════════════════════════════════════════════════════╗"
echo "║   vLLM CPU Offloading 快速测试                             ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 检查服务是否已经运行
echo "1. 检查 vLLM 服务状态..."
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "   ✓ vLLM 服务已在运行"
    NEED_START=false
else
    echo "   ✗ vLLM 服务未运行，需要启动"
    NEED_START=true
fi

if [ "$NEED_START" = true ]; then
    echo ""
    echo "2. 启动 vLLM 服务..."
    echo "   请在新终端中运行以下命令："
    echo "   cd $SCRIPT_DIR"
    echo "   ./start_vllm_with_offload.sh"
    echo ""
    echo "   按任意键继续（确保服务已启动）..."
    read -n 1 -s
    
    # 等待服务启动
    echo "   等待服务就绪..."
    for i in {1..30}; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo "   ✓ 服务已就绪！"
            break
        fi
        sleep 2
        echo "   等待中... ($i/30)"
    done
    
    if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "   ✗ 服务启动超时，请检查启动日志"
        exit 1
    fi
fi

echo ""
echo "3. 运行压力测试..."
echo "   测试配置："
echo "   - 请求数: 30"
echo "   - 并发数: 10"
echo "   - 模式: unique (更容易触发 offload)"
echo "   - 生成 tokens: 100"
echo ""

# 检查 Python 环境
if ! python3 -c "import aiohttp" 2>/dev/null; then
    echo "   安装依赖..."
    pip install aiohttp -q
fi

python3 "$SCRIPT_DIR/test_offload_client.py" \
    --num-requests 30 \
    --concurrency 10 \
    --pattern unique \
    --max-tokens 100 \
    --output "$SCRIPT_DIR/logs/test_result_$(date +%Y%m%d_%H%M%S).json"

echo ""
echo "4. 测试完成！"
echo ""
echo "💡 查看提示："
echo "   - 服务器日志: $SCRIPT_DIR/logs/"
echo "   - 搜索关键词: grep -i 'offload\|swap\|evict' logs/*.log"
echo "   - 查看最新日志: tail -f logs/*.log | grep -i offload"
echo ""
echo "5. 如需更多测试，可以运行："
echo "   python3 test_offload_client.py --help"
echo ""
