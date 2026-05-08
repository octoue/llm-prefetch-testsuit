#!/bin/bash
# Sample CPU package energy for the experiment window via perf.
# perf stat -e power/energy-pkg/ requires CAP_SYS_ADMIN or
# /proc/sys/kernel/perf_event_paranoid <= 0. If unavailable, this script
# logs a stub and exits 0 so the experiment continues.
#
# Usage: sample_perf.sh <output_txt> <duration_sec>

set -e

OUT="${1:?usage: sample_perf.sh <output_txt> <duration_sec>}"
DUR="${2:?usage: sample_perf.sh <output_txt> <duration_sec>}"

mkdir -p "$(dirname "$OUT")"

if ! command -v perf >/dev/null 2>&1; then
  echo "perf binary not found; skipping CPU energy sampling" > "$OUT"
  exit 0
fi

if ! perf stat -e power/energy-pkg/ -a sleep 0.1 >/dev/null 2>&1; then
  echo "perf power/energy-pkg/ not accessible (perf_event_paranoid?)" > "$OUT"
  exit 0
fi

perf stat -e power/energy-pkg/ -a -- sleep "$DUR" 2> "$OUT" || true
