#!/bin/bash
# 批量运行 mechanism ablation 实验
# num_blocks=1000, qps={0.5,1.0,1.5,2.0,2.5}, groups=g0,g1,full
# num_blocks=750,  qps={0.5,1.5,2.5},           groups=g0,g1,full

set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ABLATION_SCRIPT="$SCRIPT_DIR/auto_run_mechanism_ablation.sh"

DATASET="pcie-heavy"
GROUPS="g0,g1,full"

FAILED=()
SUCCEEDED=()

run_one() {
    local blocks="$1"
    local qps="$2"
    local label="blk${blocks}_qps${qps}"

    echo ""
    echo "============================================"
    echo "  Running: blocks=$blocks qps=$qps groups=$GROUPS"
    echo "  $(date)"
    echo "============================================"

    if bash "$ABLATION_SCRIPT" "$DATASET" \
        --gpu-blocks "$blocks" \
        --qps "$qps" \
        --groups "$GROUPS"; then
        SUCCEEDED+=("$label")
        echo "SUCCESS: $label"
    else
        FAILED+=("$label")
        echo "FAILED: $label (skipping to next)"
    fi
}

# --- num_blocks=1000 ---
for qps in 0.5 1.0 1.5 2.0 2.5; do
    run_one 1000 "$qps"
done

# --- num_blocks=750 ---
for qps in 0.5 1.5 2.5; do
    run_one 750 "$qps"
done

echo ""
echo "============================================"
echo "  Batch Complete! $(date)"
echo "============================================"
echo "  Succeeded: ${#SUCCEEDED[@]} - ${SUCCEEDED[*]}"
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "  Failed:    ${#FAILED[@]} - ${FAILED[*]}"
fi
echo "============================================"
