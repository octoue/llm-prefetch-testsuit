#!/bin/bash
# Sample per-GPU power.draw at 100ms via nvidia-smi.
# Output is a CSV with columns: timestamp, gpu_index, power_w
#
# Usage:
#   sample_power.sh <output_csv> <duration_sec> [gpu_ids]
#
# gpu_ids is an optional comma-separated list of nvidia-smi indices (e.g.
# "4,5"). When set, sampling is restricted to those GPUs via nvidia-smi -i.
# This is essential when other experiments share the node, otherwise the
# CSV will contain unrelated power draw and pollute the energy aggregate.

set -e

OUT="${1:?usage: sample_power.sh <output_csv> <duration_sec> [gpu_ids]}"
DUR="${2:?usage: sample_power.sh <output_csv> <duration_sec> [gpu_ids]}"
GPU_IDS="${3:-}"

mkdir -p "$(dirname "$OUT")"

NVSMI_FILTER=()
if [ -n "$GPU_IDS" ]; then
  NVSMI_FILTER=( -i "$GPU_IDS" )
fi

# nvidia-smi -lms emits one row per GPU per sample, prefixed with timestamp.
# We keep it raw and let the analyzer aggregate across GPUs.
{
  echo "timestamp,gpu_index,power_w"
  timeout "${DUR}s" nvidia-smi \
      "${NVSMI_FILTER[@]}" \
      --query-gpu=timestamp,index,power.draw \
      --format=csv,noheader,nounits \
      -lms 100 || true
} > "$OUT"
