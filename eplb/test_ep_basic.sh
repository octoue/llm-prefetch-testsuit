#!/bin/bash
# EP Smoke Test: 启动 vLLM EP 模式,发送请求,验证基本功能
# 用法: ./test_ep_basic.sh [--model PATH] [--port PORT]

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
API_PORT="${API_PORT:-8000}"
MODEL_PATH="${MODEL_PATH:-/lpai/models/deepseek-ai__deepseek-v2-lite-chat/24-05-17-0658}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL_PATH="$2"; shift 2 ;;
        --port)  API_PORT="$2";   shift 2 ;;
        *) shift ;;
    esac
done

API_BASE="http://localhost:$API_PORT"
RESULTS_DIR="$SCRIPT_DIR/results/basic_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

# ============================================================
# Cleanup
# ============================================================
VLLM_PID=""
cleanup() {
    if [[ -n "$VLLM_PID" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "Stopping vLLM (PID: $VLLM_PID)..."
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# ============================================================
# Start vLLM
# ============================================================
echo "=== Starting vLLM EP server ==="
bash "$SCRIPT_DIR/start_vllm_ep.sh" \
    --model "$MODEL_PATH" \
    --ep-size 2 \
    --port "$API_PORT" \
    --log-file "$RESULTS_DIR/vllm.log" &
VLLM_PID=$!

echo "Waiting for server (PID: $VLLM_PID)..."
MAX_WAIT=300
WAITED=0
while [[ $WAITED -lt $MAX_WAIT ]]; do
    if curl -s "$API_BASE/health" &>/dev/null; then
        echo "Server ready (${WAITED}s)"
        break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "Server died. Check $RESULTS_DIR/vllm.log"
        exit 1
    fi
    sleep 3
    WAITED=$((WAITED + 3))
done

if [[ $WAITED -ge $MAX_WAIT ]]; then
    echo "Timeout waiting for server"
    exit 1
fi

# ============================================================
# Test 1: Single request
# ============================================================
echo ""
echo "=== Test 1: Single completion request ==="
RESP=$(curl -s "$API_BASE/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'"$MODEL_PATH"'",
        "messages": [{"role": "user", "content": "What is 2+2? Answer briefly."}],
        "max_tokens": 64,
        "temperature": 0
    }')

echo "$RESP" | python3 -c "
import json, sys
r = json.load(sys.stdin)
if 'choices' in r:
    print(f'  OK: {r[\"choices\"][0][\"message\"][\"content\"][:100]}')
    print(f'  Usage: {r.get(\"usage\", {})}')
else:
    print(f'  FAIL: {r}')
    sys.exit(1)
"
echo "$RESP" > "$RESULTS_DIR/test1_single.json"

# ============================================================
# Test 2: Concurrent requests (验证 EP all-to-all 在并发下正常)
# ============================================================
echo ""
echo "=== Test 2: 5 concurrent requests ==="
PIDS=()
for i in $(seq 1 5); do
    curl -s "$API_BASE/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "'"$MODEL_PATH"'",
            "messages": [{"role": "user", "content": "Count from 1 to '"$i"'0."}],
            "max_tokens": 128,
            "temperature": 0
        }' > "$RESULTS_DIR/test2_concurrent_$i.json" &
    PIDS+=($!)
done

FAIL=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || FAIL=$((FAIL + 1))
done

if [[ $FAIL -gt 0 ]]; then
    echo "  WARN: $FAIL/$((${#PIDS[@]})) requests failed"
else
    echo "  OK: All 5 requests completed"
fi

# ============================================================
# Test 3: Multi-turn (验证 prefix cache + EP)
# ============================================================
echo ""
echo "=== Test 3: Multi-turn conversation ==="
TURN1=$(curl -s "$API_BASE/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'"$MODEL_PATH"'",
        "messages": [
            {"role": "user", "content": "My name is Alice. Remember it."}
        ],
        "max_tokens": 64,
        "temperature": 0
    }')

TURN2=$(curl -s "$API_BASE/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'"$MODEL_PATH"'",
        "messages": [
            {"role": "user", "content": "My name is Alice. Remember it."},
            {"role": "assistant", "content": "'"$(echo "$TURN1" | python3 -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['message']['content'][:200])" 2>/dev/null || echo "OK")"'"},
            {"role": "user", "content": "What is my name?"}
        ],
        "max_tokens": 64,
        "temperature": 0
    }')

echo "$TURN2" | python3 -c "
import json, sys
r = json.load(sys.stdin)
if 'choices' in r:
    content = r['choices'][0]['message']['content']
    print(f'  Turn 2 reply: {content[:100]}')
    prompt_tokens = r.get('usage', {}).get('prompt_tokens_details', {})
    cached = prompt_tokens.get('cached_tokens', 0)
    print(f'  Cached tokens: {cached}')
else:
    print(f'  FAIL: {r}')
" 2>/dev/null || echo "  Parse error (non-fatal)"

echo "$TURN2" > "$RESULTS_DIR/test3_multiturn.json"

# ============================================================
# Summary
# ============================================================
echo ""
echo "=== EP Smoke Test Complete ==="
echo "Results: $RESULTS_DIR"
echo "Server log: $RESULTS_DIR/vllm.log"
