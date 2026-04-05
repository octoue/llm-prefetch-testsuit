#!/bin/bash
# EPLB Trigger Test: 启动 EP+EPLB, 持续发请求直到触发 expert rearrangement
# 通过观察 vLLM log 中的 EPLB rearrangement 日志来验证
#
# 用法: ./test_ep_eplb.sh [--model PATH] [--port PORT] [--num-requests N]

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
API_PORT="${API_PORT:-8000}"
MODEL_PATH="${MODEL_PATH:-deepseek-ai/DeepSeek-V2-Lite-Chat}"
NUM_REQUESTS=200
STEP_INTERVAL=100  # 调低 step_interval 以更容易触发 rearrangement

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)         MODEL_PATH="$2";   shift 2 ;;
        --port)          API_PORT="$2";      shift 2 ;;
        --num-requests)  NUM_REQUESTS="$2";  shift 2 ;;
        --step-interval) STEP_INTERVAL="$2"; shift 2 ;;
        *) shift ;;
    esac
done

API_BASE="http://localhost:$API_PORT"
RESULTS_DIR="$SCRIPT_DIR/results/eplb_$(date +%Y%m%d_%H%M%S)"
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
# Start vLLM with EPLB enabled
# ============================================================
echo "=== Starting vLLM EP+EPLB server ==="
echo "EPLB step_interval=$STEP_INTERVAL (lower = trigger sooner)"

bash "$SCRIPT_DIR/start_vllm_ep.sh" \
    --model "$MODEL_PATH" \
    --ep-size 2 \
    --eplb \
    --step-interval "$STEP_INTERVAL" \
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
    echo "Timeout"
    exit 1
fi

# ============================================================
# Send sustained requests to trigger EPLB rearrangement
# ============================================================
echo ""
echo "=== Sending $NUM_REQUESTS requests to trigger EPLB ==="

PROMPTS=(
    "Explain machine learning in simple terms."
    "Write a Python function to sort a list."
    "What causes earthquakes?"
    "Summarize the history of the internet."
    "How do vaccines work?"
    "Explain quantum computing to a child."
    "What is the Pythagorean theorem?"
    "Describe photosynthesis step by step."
    "What are the benefits of exercise?"
    "How does a computer CPU work?"
)

COMPLETED=0
FAILED=0

for i in $(seq 1 "$NUM_REQUESTS"); do
    PROMPT="${PROMPTS[$((i % ${#PROMPTS[@]}))]}"

    HTTP_CODE=$(curl -s -o "$RESULTS_DIR/resp_$i.json" -w "%{http_code}" \
        "$API_BASE/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "'"$MODEL_PATH"'",
            "messages": [{"role": "user", "content": "'"$PROMPT"'"}],
            "max_tokens": 64,
            "temperature": 0.7
        }' 2>/dev/null) || HTTP_CODE=000

    if [[ "$HTTP_CODE" == "200" ]]; then
        COMPLETED=$((COMPLETED + 1))
    else
        FAILED=$((FAILED + 1))
    fi

    # Progress
    if (( i % 20 == 0 )); then
        echo "  Progress: $i/$NUM_REQUESTS (ok=$COMPLETED, fail=$FAILED)"

        # Check for EPLB rearrangement in log
        EPLB_COUNT=$(grep -ci "rearrange\|eplb\|expert.*balance\|expert.*move" "$RESULTS_DIR/vllm.log" 2>/dev/null || echo 0)
        echo "  EPLB log hits: $EPLB_COUNT"
    fi
done

# ============================================================
# Analyze EPLB activity from log
# ============================================================
echo ""
echo "=== EPLB Analysis ==="
echo "Requests: $COMPLETED completed, $FAILED failed"

echo ""
echo "--- EPLB-related log entries ---"
grep -i "rearrange\|eplb\|expert.*balance\|expert.*move\|expert.*migrat\|P2P" "$RESULTS_DIR/vllm.log" 2>/dev/null \
    | tail -30 || echo "(no EPLB entries found)"

echo ""
echo "--- Expert load stats ---"
grep -i "expert.*load\|balancedness\|load_level" "$RESULTS_DIR/vllm.log" 2>/dev/null \
    | tail -10 || echo "(no load stats found)"

echo ""
echo "=== EPLB Test Complete ==="
echo "Results: $RESULTS_DIR"
echo "Full log: $RESULTS_DIR/vllm.log"
echo ""
echo "Manual check:"
echo "  grep -i 'rearrange' $RESULTS_DIR/vllm.log"
echo "  grep -i 'P2P' $RESULTS_DIR/vllm.log"
