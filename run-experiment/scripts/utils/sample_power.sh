#!/bin/bash
# Sample per-GPU power.draw at 100ms via nvidia-smi.
# Output is a CSV with columns: timestamp, gpu_index, power_w
#
# Usage: sample_power.sh <output_csv> <duration_sec>

set -e

OUT="${1:?usage: sample_power.sh <output_csv> <duration_sec>}"
DUR="${2:?usage: sample_power.sh <output_csv> <duration_sec>}"

mkdir -p "$(dirname "$OUT")"

# nvidia-smi -lms emits one row per GPU per sample, prefixed with timestamp.
# We keep it raw and let the analyzer aggregate across GPUs.
{
  echo "timestamp,gpu_index,power_w"
  timeout "${DUR}s" nvidia-smi \
      --query-gpu=timestamp,index,power.draw \
      --format=csv,noheader,nounits \
      -lms 100 || true
} > "$OUT"
